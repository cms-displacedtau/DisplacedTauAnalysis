"""
make_datacard.py
================
Generates one Combine datacard per Stau (mass, ctau) signal point
for a one-bin (counting experiment) limit extraction.

Systematics included:
  - lumi_13p6TeV       lnN  (1.4%, all processes)
  - xsec_<group>       lnN  (one row per background group, from group_xsec_uncertainties.json)
  - xsec_<signal>      lnN  (signal theory xsec uncertainty, named per signal point)

Usage:
  python make_datacard.py
  python make_datacard.py --masses 100 200 --ctaus 1 10
  python make_datacard.py --xsec-unc-file ./plots_config/group_xsec_uncertainties.json
  python make_datacard.py --signal-xsec-unc-file ./plots_config/signal_xsec_uncertainties.json

Signal xsec uncertainty JSON format (either key style works):
  Keyed by signal name:   {"Stau_100_1mm": 0.07, "Stau_200_10mm": 0.10, ...}
  Keyed by mass only:     {"100": 0.07, "200": 0.10, ...}
  Flat value for all:     pass --signal-xsec-unc 0.10 instead of a file.

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
ROOTFILE    = "SR_score_QCDFix.root"
PLOT_VAR    = "Jet_pt"
BIN_NAME    = "CR"
LUMI        = 324.75   # fb^-1
LUMI_UNC    = 1.014  # 1.4% — CMS Run3 2022 recommendation

BACKGROUNDS = ["DY", "QCD", "WtoLNu", "TT", "singleT"]


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

    # Try full signal name first (most specific)
    if sname in signal_xsec_unc_dict:
        return float(signal_xsec_unc_dict[sname])

    # Try mass as string
    if str(mass) in signal_xsec_unc_dict:
        return float(signal_xsec_unc_dict[str(mass)])

    # Try mass as int (JSON keys are always strings, but be safe)
    if mass in signal_xsec_unc_dict:
        return float(signal_xsec_unc_dict[mass])

    return None


# ---------------------------------------------------------------------------
# Datacard writer
# ---------------------------------------------------------------------------

def make_single_datacard(mass, ctau, args, rf, xsec_unc, signal_xsec_unc_dict):
    """
    Write one datacard for a single (mass, ctau) signal point.

    Parameters
    ----------
    mass, ctau            : signal grid point
    args                  : parsed CLI arguments
    rf                    : open uproot file handle
    xsec_unc              : dict of {group: relative_uncertainty} or None
                            e.g. {"TT": 0.05, "QCD": 0.50, ...}
    signal_xsec_unc_dict  : dict with signal xsec uncertainties, or None
    """
    sname    = signal_name(mass, ctau)
    sig_rate = get_integral(rf, f"{PLOT_VAR}__{sname}")

    bkg_rates = {bkg: get_integral(rf, f"{PLOT_VAR}__{bkg}")
                 for bkg in args.backgrounds}
    total_bkg = sum(bkg_rates.values())

    # Column ordering: signal (index 0) first, then backgrounds (1, 2, ...)
    all_processes = [sname]    + args.backgrounds
    all_proc_idx  = [0]        + list(range(1, len(args.backgrounds) + 1))
    all_rates     = [sig_rate] + [bkg_rates[b] for b in args.backgrounds]

    # Column width for alignment
    w = max(len(p) for p in all_processes) + 4

    # ------------------------------------------------------------------
    # Build systematics rows
    # Each row is a tuple: (name, type, [value per process column])
    # "-" means this systematic does not affect that process.
    # ------------------------------------------------------------------
    syst_rows = []

    # --- Luminosity ---
    # lnN: affects every process (signal and all backgrounds) equally.
    # Value = 1 + relative uncertainty = 1.014 for 1.4%.
    syst_rows.append((
        "lumi_13p6TeV",
        "lnN",
        [str(LUMI_UNC)] * len(all_processes),
    ))

    # --- Signal cross-section uncertainty ---
    # Named xsec_<signal_name> (e.g. xsec_Stau_100_1mm) so that Combine
    # treats each signal point as an independent nuisance parameter.
    # This matters if datacards are ever combined across mass points or
    # channels — a shared name would incorrectly correlate them.
    # Follows the same convention as background rows (xsec_TT, xsec_QCD, ...).
    sig_rel_unc = None

    # Priority: flat CLI value > per-point JSON lookup
    if args.signal_xsec_unc is not None:
        sig_rel_unc = args.signal_xsec_unc
    else:
        sig_rel_unc = resolve_signal_xsec_unc(signal_xsec_unc_dict, sname, mass)

    if sig_rel_unc is not None:
        lnN_val = f"{1.0 + sig_rel_unc:.4f}"
        vals = [lnN_val] + ["-"] * len(args.backgrounds)
        syst_rows.append((f"xsec_{sname}", "lnN", vals))

    # --- Background cross-section uncertainties ---
    # One lnN row per background group.
    # Signal gets "-"; each background only has a non-"-" value in its own column.
    if xsec_unc:
        for bkg in args.backgrounds:
            if bkg not in xsec_unc:
                print(f"  [WARNING] No xsec uncertainty for {bkg} — skipping row.")
                continue
            rel_unc  = xsec_unc[bkg]
            lnN_val  = f"{1.0 + rel_unc:.4f}"
            # "-" for signal, lnN for this bkg, "-" for all others
            vals = ["-"] + [
                lnN_val if b == bkg else "-"
                for b in args.backgrounds
            ]
            syst_rows.append((f"xsec_{bkg}", "lnN", vals))

    lines = []

    # ------------------------------------------------------------------
    # Header
    # imax: number of bins/channels — always 1 for a counting experiment
    # jmax: number of background processes (not counting signal)
    # kmax: number of systematic rows — * means let Combine count
    # ------------------------------------------------------------------
    lines += [
        f"# Combine datacard — one-bin counting experiment",
        f"# Signal : {sname}",
        f"# Bin    : {args.bin_name}  (control region)",
        f"# Lumi   : {LUMI} fb^-1 at 13.6 TeV",
        f"#",
        f"imax 1",
        f"jmax {len(args.backgrounds)}",
        f"kmax *",
        f"",
        "-" * 70,
        f"",
    ]

    # ------------------------------------------------------------------
    # Observation block
    # bin         : channel name
    # observation : number of observed data events.
    #               -1 tells Combine to use the Asimov dataset
    #               (= sum of all background rates).
    #               Replace with the real count when unblinding.
    # ------------------------------------------------------------------
    lines += [
        f"bin          {args.bin_name}",
        f"observation  -1",
        f"",
        "-" * 70,
        f"",
    ]

    # ------------------------------------------------------------------
    # Rates block
    # bin     : channel name, repeated once per (bin, process) column
    # process : process name — one column per process
    # process : process index — <=0 signal, >=1 background
    # rate    : expected yield for each process in this bin
    # ------------------------------------------------------------------
    lines += [
        "bin      " + "".join(f"{args.bin_name:>{w}}" for _ in all_processes),
        "process  " + "".join(f"{p:>{w}}"             for p in all_processes),
        "process  " + "".join(f"{str(i):>{w}}"        for i in all_proc_idx),
        "rate     " + "".join(f"{r:>{w}.4f}"          for r in all_rates),
        f"",
        "-" * 70,
        f"",
    ]

    # ------------------------------------------------------------------
    # Systematics block
    # Format: <name>  <type>  <value per process column>
    #
    # lnN  : log-normal normalisation uncertainty
    #         value = 1 + relative uncertainty (e.g. 1.05 = 5%)
    #         use - if this process is unaffected
    #
    # shape: shape uncertainty (not used here yet)
    #         requires mT__<process>_<syst>Up/Down histograms in ROOT file
    #         value = 1.0
    # ------------------------------------------------------------------
    if syst_rows:
        lines.append("# Systematics")
        for name, stype, vals in syst_rows:
            row = f"{name:<22} {stype:<6} " + "".join(f"{v:>{w}}" for v in vals)
            lines.append(row)
    else:
        lines.append("# No systematics.")

    lines.append("")

    outpath = os.path.join(args.outdir, args.region, sname, f"datacard_{sname}.txt")
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
    p.add_argument("--outdir",        default="/eos/uscms/store/user/dally/CMSSW_15_0_15_patch4/src/HiggsAnalysis/CombinedLimit/data/tutorials/longexercise/displacedTaus/datacards")
    p.add_argument("--region",        default="SR_score")
    p.add_argument("--bin-name",      default=BIN_NAME)
    p.add_argument("--backgrounds",   nargs="+", default=BACKGROUNDS)
    p.add_argument("--xsec-unc-file", default=None,
                   help="JSON with per-group relative xsec uncertainties, "
                        "e.g. {\"TT\": 0.05, \"QCD\": 0.50}. "
                        "Produced by compute_xsec_uncertainties.py.")
    p.add_argument("--signal-xsec-unc", type=float, default=None,
                   help="Flat relative signal xsec uncertainty applied to all "
                        "signal points, e.g. 0.10 for 10%%. Takes priority "
                        "over --signal-xsec-unc-file.")
    p.add_argument("--signal-xsec-unc-file", default=None,
                   help="JSON with per-signal-point relative xsec uncertainties. "
                        "Keys can be full signal names (Stau_100_1mm) or mass "
                        "strings (\"100\").")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(f"{args.outdir}/{args.region}", exist_ok=True)

    # Load group-level xsec uncertainties if provided
    xsec_unc = None
    if args.xsec_unc_file:
        with open(args.xsec_unc_file, "r") as f:
            xsec_unc = json.load(f)
        print(f"Loaded background xsec uncertainties: {xsec_unc}")

    # Load signal xsec uncertainties if provided via JSON
    signal_xsec_unc_dict = None
    if args.signal_xsec_unc_file:
        with open(args.signal_xsec_unc_file, "r") as f:
            signal_xsec_unc_dict = json.load(f)
        print(f"Loaded signal xsec uncertainties from: {args.signal_xsec_unc_file}")

    if args.signal_xsec_unc is not None:
        print(f"Flat signal xsec uncertainty: {args.signal_xsec_unc*100:.1f}% (overrides JSON)")

    print(f"Reading ROOT file : {args.rootfile}")
    print(f"Signal grid       : masses={args.masses}, ctaus={args.ctaus}")
    print(f"Backgrounds       : {args.backgrounds}")
    print(f"Lumi uncertainty  : {(LUMI_UNC - 1)*100:.1f}%")
    print()

    rf      = uproot.open(args.rootfile)
    written = []

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
    print("  for f in datacards/datacard_Stau_*.txt; do")
    print("    combine -M AsymptoticLimits $f -t -1 --run blind \\")
    print("            --rMin 0 --rMax 10 -n _$(basename $f .txt) &")
    print("  done")
    print("  wait")


if __name__ == "__main__":
    main()

