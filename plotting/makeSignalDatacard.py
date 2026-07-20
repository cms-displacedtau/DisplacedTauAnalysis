"""
make_datacard.py
================
Generates one Combine datacard per Stau (mass, ctau) signal point
for a one-bin (counting experiment) limit extraction.

Computes JES/JER/PU κ values directly from Up/Down histograms in the
ROOT file — no external kappa_values.json needed.

Systematics included:
  - lumi_13p6TeV       lnN  (1.4%, all processes)
  - xsec_<group>       lnN  (one row per background group, from group_xsec_uncertainties.json)
  - xsec_<signal>      lnN  (signal theory xsec uncertainty, named per signal point)
  - CMS_scale_j        lnN  (JES, computed from jesUp/jesDown histograms)
  - CMS_res_j          lnN  (JER, computed from jerUp/jerDown histograms)
  - CMS_pileup         lnN  (PU,  computed from puUp/puDown histograms)

Expected histogram naming convention in the ROOT file:
  Nominal:   {PLOT_VAR}__{process}          e.g. Jet_pt__DY
  JES up:    {PLOT_VAR}_jesUp__{process}    e.g. Jet_pt_jesUp__DY
  JES down:  {PLOT_VAR}_jesDown__{process}  e.g. Jet_pt_jesDown__DY
  JER up:    {PLOT_VAR}_jerUp__{process}    e.g. Jet_pt_jerUp__DY
  JER down:  {PLOT_VAR}_jerDown__{process}  e.g. Jet_pt_jerDown__DY
  PU up:     {PLOT_VAR}_puUp__{process}     e.g. Jet_pt_puUp__DY
  PU down:   {PLOT_VAR}_puDown__{process}   e.g. Jet_pt_puDown__DY

If your histograms use a different naming convention, adjust SYST_HISTNAME_MAP below.

κ handling rules:
  - If both κ_up and κ_down < 1: keep the one furthest from 1, mirror it
  - If both κ_up and κ_down > 1: keep the one furthest from 1, mirror it
  - If κ_down > κ_up: swap them (down should always be ≤ up)
  - Normal case: κ_down < 1 < κ_up → use as-is

Usage:
  python make_datacard.py
  python make_datacard.py --masses 100 200 --ctaus 1 10
  python make_datacard.py --xsec-unc-file ./plots_config/group_xsec_uncertainties.json

Then run Combine:
  for f in datacards/datacard_Stau_*.txt; do
      combine -M AsymptoticLimits $f -t -1 --run blind \\
              --rMin 0 --rMax 10 -n _$(basename $f .txt) &
  done
  wait
"""

import os
import json
import argparse
import uproot
import numpy as np

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
MASSES      = [100, 200, 300, 400, 500]
CTAUS       = [1, 5, 10, 50, 100, 1000]
ROOTFILE    = "histograms_0p99.root"
PLOT_VAR    = "Jet_pt"
BIN_NAME    = "SR"
LUMI        = 324.75   # fb^-1
LUMI_UNC    = 1.014    # 1.4% — CMS Run3 2022 recommendation
BACKGROUNDS = ["DY", "QCD", "WJets", "Top","EWK"]

## ---------------------------------------------------------------------------
## Zero-rate background handling
## ---------------------------------------------------------------------------
## If a background has zero yield in the SR, set its rate to this tiny floor
## instead of dropping it. This keeps the process in the datacard so that
## rateParam lines can connect it across SR and CRs.
## Set to None to drop zero-rate backgrounds entirely.
## ---------------------------------------------------------------------------
ZERO_RATE_FLOOR = 1e-6

## ---------------------------------------------------------------------------
## Systematic histogram name mapping
## ---------------------------------------------------------------------------
## Maps (datacard_syst_name) → (up_suffix, down_suffix)
## These suffixes are inserted between PLOT_VAR and the process name:
##   {PLOT_VAR}_{suffix}__{process}
##
## Adjust these if your ROOT file uses different naming.
## Set to None or empty dict to skip all shape-based systematics.
## ---------------------------------------------------------------------------
SYST_HISTNAME_MAP = {
    "CMS_scale_j": ("jesUp",  "jesDown"),
    "CMS_res_j":   ("jerUp",  "jerDown"),
    "CMS_pileup":  ("puUp",   "puDown"),
}

## ---------------------------------------------------------------------------
## rateParam configuration
## ---------------------------------------------------------------------------
## Each entry: (param_name, channel_pattern, process_name, initial_value, range)
##
## channel_pattern = "*" means the same parameter scales this process
## in ALL channels (SR + CRs). This is what ties the CR constraint
## to the SR prediction.
##
## IMPORTANT: If a process has a rateParam, REMOVE its xsec lnN uncertainty —
## the CR data replaces the prior.
## ---------------------------------------------------------------------------
RATE_PARAMS = [
    ## Uncomment the processes you want constrained by CR data:
     ("rate_QCD",   "*", "QCD",      "1.0",),
    ## ("rate_DY",    "*", "DY",       "1.0", "[0.2, 5.0]"),
     ("rate_Top",   "*", "Top",       "1.0",),
    ## ("rate_Top",   "*", "singleT",  "1.0", "[0.2, 5.0]"),
     ("rate_WJets", "*", "WJets",   "1.0",),
]

## Processes that have a rateParam should NOT have a cross-section lnN.
## List them here so they are automatically excluded from xsec rows.
RATEPARAM_PROCESSES = set()
for _, _, proc, _, in RATE_PARAMS:
    RATEPARAM_PROCESSES.add(proc)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def signal_name(mass, ctau):
    return f"Stau_{mass}_{ctau}mm"


def get_integral(rootfile, hist_key):
    """Read the total yield (integral) from a histogram in the ROOT file."""
    if hist_key not in rootfile:
        raise KeyError(f"Histogram '{hist_key}' not found in ROOT file.")
    return float(np.sum(rootfile[hist_key].values()))


def get_integral_safe(rootfile, hist_key):
    """
    Read the total yield from a histogram. Returns None if the histogram
    does not exist (instead of raising).
    """
    if hist_key not in rootfile:
        return None
    return float(np.sum(rootfile[hist_key].values()))


def compute_kappa(nominal, varied):
    """
    Compute κ = varied / nominal.
    Returns None if nominal is zero or if varied is None.
    """
    if varied is None or nominal is None or nominal == 0:
        return None
    return varied / nominal


def fix_kappa_pair(kappa_down, kappa_up):
    """
    Fix κ pairs that don't properly bracket 1.0.

    Rules:
      1. Both < 1: keep the one furthest from 1, mirror it
         e.g. (0.92, 0.97) → (0.92, 1.08)
      2. Both > 1: keep the one furthest from 1, mirror it
         e.g. (1.03, 1.08) → (0.92, 1.08)
      3. κ_down > κ_up: swap them
         e.g. (1.05, 0.95) → (0.95, 1.05)
      4. Normal (κ_down < 1 < κ_up): return as-is

    Returns (kappa_down_fixed, kappa_up_fixed)
    """
    if kappa_down is None or kappa_up is None:
        return kappa_down, kappa_up

    ## Rule 3: swap if down > up
    if kappa_down > kappa_up:
        kappa_down, kappa_up = kappa_up, kappa_down

    ## Rule 1: both below 1
    if kappa_down < 1.0 and kappa_up < 1.0:
        ## Keep the one furthest from 1 (smallest), mirror it
        delta = 1.0 - kappa_down
        kappa_up = 1.0 + delta

    ## Rule 2: both above 1
    elif kappa_down > 1.0 and kappa_up > 1.0:
        ## Keep the one furthest from 1 (largest), mirror it
        delta = kappa_up - 1.0
        kappa_down = 1.0 - delta

    ## Rule 4: normal case — already brackets 1.0
    return kappa_down, kappa_up


def format_kappa_lnN(kappa_down, kappa_up):
    """
    Format κ values for the datacard lnN line.

    First fixes the κ pair (swap, mirror if both same side of 1),
    then formats as symmetric or asymmetric.

    Returns "-" if neither κ is available.
    """
    if kappa_down is None and kappa_up is None:
        return "-"

    ## If only one is available, mirror it
    if kappa_down is None:
        kappa_down = 2.0 - kappa_up
    if kappa_up is None:
        kappa_up = 2.0 - kappa_down

    ## Apply fixing rules
    kappa_down, kappa_up = fix_kappa_pair(kappa_down, kappa_up)

    delta_up   = abs(kappa_up - 1.0)
    delta_down = abs(kappa_down - 1.0)

    ## If both are essentially 1.0 (no effect), skip
    if delta_up < 1e-4 and delta_down < 1e-4:
        return "-"

    ## Check asymmetry — use asymmetric format if they differ by > 20%
    if delta_up > 0 and delta_down > 0:
        ratio = max(delta_up, delta_down) / min(delta_up, delta_down)
        if ratio > 1.2:
            return f"{kappa_down:.4f}/{kappa_up:.4f}"

    ## Symmetric: average
    avg_kappa = 1.0 + (delta_up + delta_down) / 2.0
    return f"{avg_kappa:.4f}"


def resolve_signal_xsec_unc(signal_xsec_unc_dict, sname, mass):
    """
    Look up the signal cross-section uncertainty for a given signal point.

    Tries three key styles in order:
      1.  Full signal name, e.g. "Stau_100_1mm"
      2.  Mass as string,   e.g. "100"
      3.  Mass as int,      e.g. 100

    Returns the relative uncertainty (float) or None if not found.
    """
    if signal_xsec_unc_dict is None:
        return None

    if sname in signal_xsec_unc_dict:
        return float(signal_xsec_unc_dict[sname])

    if str(mass) in signal_xsec_unc_dict:
        return float(signal_xsec_unc_dict[str(mass)])

    if mass in signal_xsec_unc_dict:
        return float(signal_xsec_unc_dict[mass])

    return None


# ---------------------------------------------------------------------------
# Datacard writer
# ---------------------------------------------------------------------------

def make_single_datacard(mass, ctau, args, rf, xsec_unc, signal_xsec_unc_dict):
    """
    Write one datacard for a single (mass, ctau) signal point.
    """
    sname    = signal_name(mass, ctau)
    sig_rate = get_integral(rf, f"{PLOT_VAR}__{sname}")

    ## Get background rates — apply floor for zero-rate backgrounds
    bkg_rates = {}
    for bkg in args.backgrounds:
        try:
            rate = get_integral(rf, f"{PLOT_VAR}__{bkg}")
        except KeyError:
            rate = 0.0

        if rate <= 0 and ZERO_RATE_FLOOR is not None:
            if args.verbose:
                print(f"    [{bkg}] zero yield → using floor {ZERO_RATE_FLOOR}")
            rate = ZERO_RATE_FLOOR
        bkg_rates[bkg] = rate

    ## Keep all backgrounds (including floor-value ones) if ZERO_RATE_FLOOR is set
    ## Otherwise drop backgrounds with zero yield
    if ZERO_RATE_FLOOR is not None:
        active_backgrounds = list(args.backgrounds)
    else:
        active_backgrounds = [b for b in args.backgrounds if bkg_rates[b] > 0]

    total_bkg = sum(bkg_rates[b] for b in active_backgrounds)

    ## Column ordering: signal (index 0) first, then backgrounds (1, 2, ...)
    all_processes = [sname]    + active_backgrounds
    all_proc_idx  = [0]        + list(range(1, len(active_backgrounds) + 1))
    all_rates     = [sig_rate] + [bkg_rates[b] for b in active_backgrounds]

    ## Column width for alignment
    w = max(len(p) for p in all_processes) + 4

    # ------------------------------------------------------------------
    # Build systematics rows
    # ------------------------------------------------------------------
    syst_rows = []

    ## --- Luminosity ---
    syst_rows.append((
        "lumi_13p6TeV",
        "lnN",
        [str(LUMI_UNC)] * len(all_processes),
    ))

    ## --- Signal cross-section uncertainty ---
    sig_rel_unc = None
    if args.signal_xsec_unc is not None:
        sig_rel_unc = args.signal_xsec_unc
    else:
        sig_rel_unc = resolve_signal_xsec_unc(signal_xsec_unc_dict, sname, mass)

    if sig_rel_unc is not None:
        lnN_val = f"{1.0 + sig_rel_unc:.4f}"
        vals = [lnN_val] + ["-"] * len(active_backgrounds)
        syst_rows.append((f"xsec_{sname}", "lnN", vals))

    ## --- Background cross-section uncertainties ---
    ## Skip processes that have a rateParam (CR data replaces the prior)
    if xsec_unc:
        for bkg in active_backgrounds:
            print(bkg)
            print(RATEPARAM_PROCESSES)
            if bkg in RATEPARAM_PROCESSES:
                if args.verbose:
                    print(f"    [{bkg}] has rateParam — skipping xsec lnN")
                continue
            if bkg not in xsec_unc:
                if args.verbose:
                    print(f"    [WARNING] No xsec uncertainty for {bkg}")
                continue
            rel_unc  = xsec_unc[bkg]
            lnN_val  = f"{1.0 + rel_unc:.4f}"
            vals = ["-"] + [
                lnN_val if b == bkg else "-"
                for b in active_backgrounds
            ]
            syst_rows.append((f"xsec_{bkg}", "lnN", vals))

    ## --- JES / JER / PU systematics (computed from histograms) ---
    if SYST_HISTNAME_MAP:
        for syst_name, (up_suffix, down_suffix) in SYST_HISTNAME_MAP.items():
            vals = []
            any_nontriv = False

            for proc in all_processes:
                nom_key  = f"{PLOT_VAR}__{proc}"
                up_key   = f"{PLOT_VAR}_{up_suffix}__{proc}"
                down_key = f"{PLOT_VAR}_{down_suffix}__{proc}"

                nom_yield  = get_integral_safe(rf, nom_key)
                up_yield   = get_integral_safe(rf, up_key)
                down_yield = get_integral_safe(rf, down_key)

                if args.verbose:
                    print(f"    {syst_name:15s}  {proc:20s}  "
                          f"nom={nom_yield}  up={up_yield}  down={down_yield}")

                kappa_up   = compute_kappa(nom_yield, up_yield)
                kappa_down = compute_kappa(nom_yield, down_yield)

                lnN_str = format_kappa_lnN(kappa_down, kappa_up)

                if args.verbose and kappa_up is not None and kappa_down is not None:
                    kd_fix, ku_fix = fix_kappa_pair(kappa_down, kappa_up)
                    print(f"      → κ_down={kappa_down:.4f}  κ_up={kappa_up:.4f}"
                          f"  fixed=({kd_fix:.4f},{ku_fix:.4f})  → {lnN_str}")

                if lnN_str != "-":
                    any_nontriv = True
                vals.append(lnN_str)

            if any_nontriv:
                syst_rows.append((syst_name, "lnN", vals))

    # ------------------------------------------------------------------
    # Write the datacard
    # ------------------------------------------------------------------
    lines = []

    lines += [
        f"# Combine datacard — one-bin counting experiment",
        f"# Signal : {sname}",
        f"# Bin    : {args.bin_name}",
        f"# Lumi   : {LUMI} fb^-1 at 13.6 TeV",
        f"#",
        f"imax 1",
        f"jmax *",
        f"kmax *",
        f"",
        "-" * 70,
        f"",
    ]

    lines += [
        f"bin          {args.bin_name}",
        f"observation  -1",
        f"",
        "-" * 70,
        f"",
    ]

    lines += [
        "bin      " + "".join(f"{args.bin_name:>{w}}" for _ in all_processes),
        "process  " + "".join(f"{p:>{w}}"             for p in all_processes),
        "process  " + "".join(f"{str(i):>{w}}"        for i in all_proc_idx),
        "rate     " + "".join(f"{r:>{w}.4f}"          for r in all_rates),
        f"",
        "-" * 70,
        f"",
    ]

    if syst_rows:
        lines.append("# Systematics")
        for name, stype, vals in syst_rows:
            row = f"{name:<22} {stype:<6} " + "".join(f"{v:>{w}}" for v in vals)
            lines.append(row)
    else:
        lines.append("# No systematics.")

    ## --- rateParam lines ---
    has_rateparams = False
    for param_name, ch_pattern, proc, init in RATE_PARAMS:
        if proc in active_backgrounds:
            if not has_rateparams:
                lines.append("")
                lines.append("# --- rateParam: normalisation constrained by CRs ---")
                has_rateparams = True
            lines.append(
                f"{param_name}  rateParam  {ch_pattern}  {proc}  {init}"
            )
    lines.append("* autoMCStats 0")
    lines.append("")

    outpath = os.path.join(args.outdir, args.region, sname, f"datacard_{sname}.txt")
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath, "w") as f:
        f.write("\n".join(lines))

    return outpath, sig_rate, total_bkg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Generate one-bin Combine datacards per Stau signal point"
    )
    p.add_argument("--masses",        nargs="+", type=int, default=MASSES)
    p.add_argument("--ctaus",         nargs="+", type=int, default=CTAUS)
    p.add_argument("--rootfile",      default=ROOTFILE)
    p.add_argument("--outdir",        default="datacards")
    p.add_argument("--region",        default="SR_score")
    p.add_argument("--bin-name",      default=BIN_NAME)
    p.add_argument("--backgrounds",   nargs="+", default=BACKGROUNDS)
    p.add_argument("--xsec-unc-file", default=None,
                   help="JSON with per-group relative xsec uncertainties")
    p.add_argument("--signal-xsec-unc", type=float, default=None,
                   help="Flat relative signal xsec uncertainty for all points")
    p.add_argument("--signal-xsec-unc-file", default=None,
                   help="JSON with per-signal-point relative xsec uncertainties")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="Print κ values and histogram lookup details")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(f"{args.outdir}/{args.region}", exist_ok=True)

    ## Load group-level xsec uncertainties if provided
    xsec_unc = None
    if args.xsec_unc_file:
        with open(args.xsec_unc_file, "r") as f:
            xsec_unc = json.load(f)
        print(f"Loaded background xsec uncertainties: {xsec_unc}")

    ## Load signal xsec uncertainties if provided via JSON
    signal_xsec_unc_dict = None
    if args.signal_xsec_unc_file:
        with open(args.signal_xsec_unc_file, "r") as f:
            signal_xsec_unc_dict = json.load(f)
        print(f"Loaded signal xsec uncertainties from: {args.signal_xsec_unc_file}")

    if args.signal_xsec_unc is not None:
        print(f"Flat signal xsec uncertainty: {args.signal_xsec_unc*100:.1f}%")

    print(f"Reading ROOT file : {args.rootfile}")
    print(f"Signal grid       : masses={args.masses}, ctaus={args.ctaus}")
    print(f"Backgrounds       : {args.backgrounds}")
    print(f"Zero-rate floor   : {ZERO_RATE_FLOOR}")
    print(f"Lumi uncertainty  : {(LUMI_UNC - 1)*100:.1f}%")
    if SYST_HISTNAME_MAP:
        print(f"Histogram systs   : {list(SYST_HISTNAME_MAP.keys())}")
    if RATE_PARAMS:
        print(f"rateParams        : {[(p, proc) for p, _, proc, _, in RATE_PARAMS]}")
        print(f"  (xsec lnN excluded for: {RATEPARAM_PROCESSES})")
    print()

    rf      = uproot.open(args.rootfile)
    written = []

    if args.verbose:
        print("Available histograms in ROOT file:")
        for k in sorted(rf.keys()):
            print(f"  {k}")
        print()

    for mass in args.masses:
        for ctau in args.ctaus:
            sname = signal_name(mass, ctau)
            os.makedirs(f"{args.outdir}/{args.region}/{sname}", exist_ok=True)
            try:
                path, sig_rate, total_bkg = make_single_datacard(
                    mass, ctau, args, rf, xsec_unc, signal_xsec_unc_dict
                )
                written.append((mass, ctau, path))
                print(f"  {sname:25s}  sig={sig_rate:.2f}  "
                      f"total_bkg={total_bkg:.2f}  -> {path}")
            except KeyError as e:
                print(f"  [WARNING] Skipping {sname}: {e}")

    rf.close()

    print(f"\n{len(written)} datacard(s) written to {args.outdir}/")
    print("\nTo run Combine over all cards (blinded):")
    print("  for f in datacards/SR_score/Stau_*/datacard_Stau_*.txt; do")
    print("    combine -M AsymptoticLimits $f -t -1 --run blind \\")
    print("            --rMin 0 --rMax 10 -n _$(basename $f .txt) &")
    print("  done")
    print("  wait")


if __name__ == "__main__":
    main()

