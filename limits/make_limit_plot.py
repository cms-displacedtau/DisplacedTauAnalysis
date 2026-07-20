"""
make_limit_plot.py
==================
Collects AsymptoticLimits output from Combine and produces limit plots
in the group's CMS style.

Two plots are produced:
  1. Limit vs mass — one curve per ctau value
  2. 2D exclusion map in (mass, ctau) space

Usage:
  python make_limit_plot.py
  python make_limit_plot.py --masses 100 200 300 400 --ctaus 1 10 100
  python make_limit_plot.py --inputdir datacards/ --outdir plots/limits/
"""

import os
import glob
import json
import argparse
import numpy as np
import uproot
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.ticker import MultipleLocator
import mplhep as hep
from scipy.interpolate import RegularGridInterpolator

hep.style.use("CMS")

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
MASSES  = [100, 200, 300, 400, 500]
CTAUS   = [1, 5, 10, 50, 100, 1000]
LUMI    = 324.75   # fb^-1

# One colour per ctau, one linestyle per mass — matches makePlotsSignal.py
CTAU_COLOURS = ["#5790fc", "#e42536", "#f89c20", "#964a8b", "#9c9ca1", "#7a21dd"]
LINESTYLES   = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def signal_name(mass, ctau):
    return f"Stau_{mass}_{ctau}mm"


def load_xsec(xsec_file):
    """
    Load xsec and br from the group's xsec_and_br.json.
    Returns a dict: xsec_pb[mass] = xsec * br in pb, keyed by integer mass.
    Assumes sample names like Stau_100_1mm — takes xsec from the first ctau
    entry found for each mass since xsec doesn't depend on ctau.
    """
    if xsec_file is None:
        return None
    with open(xsec_file, "r") as f:
        data = json.load(f)
    xsec = data["xsec"]
    br   = data["br"]
    return xsec, br


def get_xsec_pb(xsec, br, mass, ctaus):
    """
    Return xsec * br in pb for a given mass, trying each ctau until one matches.
    Signal sample names in the JSON are expected to be Stau_{mass}_{ctau}mm.
    """
    for ctau in ctaus:
        key = signal_name(mass, ctau)
        if key in xsec and key in br:
            return xsec[key] * br[key]
    return None


def scale_limits(lims, xsec_pb):
    """
    Convert a limit dict from signal strength r to excluded cross-section in fb.
    Returns a new dict with the same keys but values in fb.
    """
    if xsec_pb is None:
        return lims
    factor = xsec_pb * 1000.0   # pb -> fb
    return {k: (v * factor if v is not None else None) for k, v in lims.items()}


def find_combine_output(inputdir, mass, ctau):
    """
    Find the Combine output ROOT file for a given signal point.
    Tries a few common naming patterns.
    """
    sname    = signal_name(mass, ctau)
    patterns = [
        os.path.join(inputdir, f"higgsCombine_datacard_{sname}.AsymptoticLimits.mH*.root"),
        os.path.join(inputdir, f"higgsCombine_{sname}.AsymptoticLimits.mH*.root"),
        os.path.join(inputdir, f"higgsCombine*{sname}*.AsymptoticLimits*.root"),
    ]
    for pattern in patterns:
        files = glob.glob(pattern)
        if files:
            return files[0]
    return None


def read_limits(filepath):
    """
    Read the 6 limit entries from a Combine AsymptoticLimits output file.
    Returns a dict with keys: minus2, minus1, exp, plus1, plus2, obs.
    Returns None if the file cannot be read.
    """
    try:
        with uproot.open(filepath) as f:
            tree   = f["limit"]
            limits = tree["limit"].array(library="np")
        if len(limits) < 5:
            print(f"  [WARNING] Only {len(limits)} entries in {filepath} — skipping.")
            return None
        return {
            "minus2" : limits[0],
            "minus1" : limits[1],
            "exp"    : limits[2],
            "plus1"  : limits[3],
            "plus2"  : limits[4],
            "obs"    : limits[5] if len(limits) > 5 else None,
        }
    except Exception as e:
        print(f"  [WARNING] Could not read {filepath}: {e}")
        return None


def collect_results(args):
    """
    Walk the input directory and collect limits for all (mass, ctau) points.
    If --xsec-file is provided, limits are scaled to excluded cross-section in fb.
    Returns a dict: results[mass][ctau] = limit dict, or None if missing.
    """
    # Load xsec if provided
    xsec, br = load_xsec(args.xsec_file) if args.xsec_file else (None, None)

    results = {}
    print("Collecting limits ...")
    for mass in args.masses:
        results[mass] = {}
        for ctau in args.ctaus:
            fpath = find_combine_output(args.inputdir, mass, ctau)
            if fpath is None:
                print(f"  [WARNING] No Combine output found for {signal_name(mass, ctau)} — skipping.")
                results[mass][ctau] = None
                continue
            lims = read_limits(fpath)
            if lims and xsec is not None:
                xsec_pb = get_xsec_pb(xsec, br, mass, args.ctaus)
                if xsec_pb is None:
                    print(f"  [WARNING] No xsec found for mass={mass} — keeping r units.")
                else:
                    lims = scale_limits(lims, xsec_pb)
            results[mass][ctau] = lims
            if lims:
                print(f"  {signal_name(mass, ctau):25s}  exp={lims['exp']:.3f}")
    return results


# ---------------------------------------------------------------------------
# Plot 1: Limit vs mass, one curve per ctau
# ---------------------------------------------------------------------------

def plot_limit_vs_mass(results, args):
    ctau_colour_map = {c: CTAU_COLOURS[i % len(CTAU_COLOURS)] for i, c in enumerate(args.ctaus)}

    # Y-axis label depends on whether we're showing r or cross-section
    use_xsec = args.xsec_file is not None
    ylabel   = (r"95% CL upper limit on $\sigma \cdot \mathcal{B}$ [fb]"
                if use_xsec
                else r"95% CL upper limit on $\sigma/\sigma_{\rm th}$")

    fig, ax = plt.subplots(figsize=(10, 6))

    for ctau, color in ctau_colour_map.items():
        masses_ok = [m for m in args.masses if results[m][ctau] is not None]
        if not masses_ok:
            continue

        exp    = np.array([results[m][ctau]["exp"]    for m in masses_ok])
        plus1  = np.array([results[m][ctau]["plus1"]  for m in masses_ok])
        minus1 = np.array([results[m][ctau]["minus1"] for m in masses_ok])
        plus2  = np.array([results[m][ctau]["plus2"]  for m in masses_ok])
        minus2 = np.array([results[m][ctau]["minus2"] for m in masses_ok])

        # ±2σ and ±1σ bands
        ax.fill_between(masses_ok, minus2, plus2, color=color, alpha=0.15)
        ax.fill_between(masses_ok, minus1, plus1, color=color, alpha=0.30)

        # Expected limit line
        ax.plot(masses_ok, exp, "--", color=color,
                label=rf"$c\tau = {ctau}$ mm (exp.)", linewidth=2)

        # Theoretical cross-section line — only meaningful when showing xsec
        if use_xsec and args.xsec_file:
            try:
                xsec, br = load_xsec(args.xsec_file)
                th_xsec = []
                for m in masses_ok:
                    xpb = get_xsec_pb(xsec, br, m, args.ctaus)
                    th_xsec.append(xpb * 1000.0 if xpb else np.nan)  # pb -> fb
                ax.plot(masses_ok, th_xsec, "-", color=color, linewidth=1.5, alpha=0.5,
                        label=rf"$c\tau = {ctau}$ mm (theory)")
            except Exception:
                pass

    # r=1 reference line only relevant when y-axis is signal strength
    if not use_xsec:
        ax.axhline(1, color="gray", linestyle=":", linewidth=1.5, label="r = 1")

    ax.set_xlabel(r"Stau mass [GeV]", fontsize=13)
    ax.set_ylabel(ylabel,             fontsize=13)
    ax.set_yscale("log")
    ax.xaxis.set_major_locator(MultipleLocator(50))
    ax.xaxis.set_minor_locator(MultipleLocator(10))
    ax.legend(fontsize=10, ncol=2)

    hep.cms.label(
        data=False, loc=0,
        label="Private Work",
        com=13.6,
        lumi=round(args.lumi, 1),
        ax=ax,
    )

    os.makedirs(args.outdir, exist_ok=True)
    base = os.path.join(args.outdir, "limit_vs_mass")
    for ext in (".pdf", ".png", ".eps"):
        plt.savefig(base + ext, bbox_inches="tight")
        print(f"  Saved: {base}{ext}")

    plt.clf()
    plt.close()


# ---------------------------------------------------------------------------
# Plot 2: 2D exclusion map in (mass, ctau) space
# ---------------------------------------------------------------------------

def plot_2d_exclusion(results, args):
    """
    Fill a 2D grid of expected limits and draw the r=1 exclusion contour.
    Points above the contour (r < 1) are excluded.
    """
    masses = np.array(args.masses, dtype=float)
    ctaus  = np.array(args.ctaus,  dtype=float)

    # Build grid — NaN where result is missing
    Z = np.full((len(masses), len(ctaus)), np.nan)
    for i, mass in enumerate(masses):
        for j, ctau in enumerate(ctaus):
            lims = results[mass][ctau]
            if lims is not None:
                Z[i, j] = lims["exp"]

    if np.all(np.isnan(Z)):
        print("[WARNING] No results available for 2D plot — skipping.")
        return

    fig, ax = plt.subplots(figsize=(9, 6))

    # Filled colour map of expected limit values
    M, C = np.meshgrid(masses, np.log10(ctaus), indexing="ij")
    cf = ax.contourf(M, 10**C, Z, levels=20, cmap="RdYlGn_r")
    plt.colorbar(cf, ax=ax, label=r"Expected 95% CL limit on $r$")

    # r = 1 exclusion boundary
    # Only draw if we have enough non-NaN points
    valid = ~np.isnan(Z)
    if valid.sum() >= 4:
        try:
            cs = ax.contour(M, 10**C, Z, levels=[1.0],
                            colors="black", linewidths=2.5)
            ax.clabel(cs, fmt="r = 1", fontsize=10)
        except Exception:
            pass  # contour may fail if not enough points straddle r=1

    ax.set_xlabel(r"Stau mass [GeV]",   fontsize=13)
    ax.set_ylabel(r"$c\tau$ [mm]",      fontsize=13)
    ax.set_yscale("log")
    ax.xaxis.set_major_locator(MultipleLocator(50))
    ax.xaxis.set_minor_locator(MultipleLocator(10))

    # Annotate excluded region
    ax.text(0.05, 0.05, "Excluded (r < 1)",
            transform=ax.transAxes, fontsize=10,
            color="black", style="italic")

    hep.cms.label(
        data=False, loc=0,
        label="Private Work",
        com=13.6,
        lumi=round(args.lumi, 1),
        ax=ax,
    )

    os.makedirs(args.outdir, exist_ok=True)
    base = os.path.join(args.outdir, "limit_2d_exclusion")
    for ext in (".pdf", ".png", ".eps"):
        plt.savefig(base + ext, bbox_inches="tight")
        print(f"  Saved: {base}{ext}")

    plt.clf()
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Collect Combine limits and make limit plots")
    p.add_argument("--masses",   nargs="+", type=int,   default=MASSES)
    p.add_argument("--ctaus",    nargs="+", type=int,   default=CTAUS)
    p.add_argument("--inputdir", default="./",
                   help="Directory containing Combine higgsCombine*.root output files")
    p.add_argument("--outdir",   default="plots/limits/")
    p.add_argument("--lumi",     type=float, default=LUMI)
    p.add_argument("--xsec-file", default=None,
                   help="Path to xsec_and_br.json — if provided, y-axis shows "
                        "excluded cross-section in fb instead of signal strength r")
    return p.parse_args()


def main():
    args = parse_args()

    results = collect_results(args)

    n_found = sum(
        1 for m in args.masses for c in args.ctaus
        if results[m][c] is not None
    )
    print(f"\nFound {n_found} / {len(args.masses) * len(args.ctaus)} signal points.")

    if n_found == 0:
        print("ERROR: No results found — check --inputdir and that Combine ran successfully.")
        return

    print("\nMaking limit vs mass plot ...")
    plot_limit_vs_mass(results, args)

    print("\nMaking 2D exclusion plot ...")
    plot_2d_exclusion(results, args)

    print("\nDone.")


if __name__ == "__main__":
    main()
