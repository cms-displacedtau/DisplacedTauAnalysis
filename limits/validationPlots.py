"""
CR Fit Validation Plots
-----------------------
Produces 4 validation plots from FitDiagnostics output:
  1. Pre/Post-fit yields per CR
  2. Nuisance pulls and constraints
  3. Correlation matrix
  4. Rate param likelihood scans

Requirements:
  - fitDiagnosticsMyFit.root         (from FitDiagnostics)
  - higgsCombineScanRate*.root       (from MultiDimFit --algo grid, one per rate param)

Usage:
  python3 cr_validation_plots.py
"""

import ROOT
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import TwoSlopeNorm
import os

ROOT.gROOT.SetBatch(True)
# Colors per process
PROCESS_COLORS = {
    "QCD":   "#5B9BD5",   # blue
    "Top":   "#ED7D31",   # orange
    "WJets": "#70AD47",   # green
}
#---------------------------
# USER CONFIGURATION  fill these in
#---------------------------

FIT_DIAG_FILE = "fitDiagnostics.FitDiagnostics_Stau_400_50mm_0p99.root"

# Nominal MC yields per process per CR (from your datacard)
# Format: YIELDS[cr][process] = N events
YIELDS = {
    "QCD CR": {
        "QCD":   815779.9080,   # <-- fill in
        "Top":   40568.4416,   # <-- fill in
        "WJets": 29261.9707,   # <-- fill in
    },
    "Top CR": {
        "QCD":   1249.6236,
        "Top":   44937.1854,
        "WJets": 3670.4193,
    },
    "W CR": {
        "QCD":   11132.1156,
        "Top":   68292.6224,
        "WJets": 248374.6920,
    },
}

# Observed data per CR (from your datacard observations)
OBSERVED = {
    "QCD CR": 779991,   # <-- fill in
    "Top CR":  42929,   # <-- fill in
    "W CR":   365914,   # <-- fill in
}

# MultiDimFit scan output files (produced by --algo grid)
SCAN_FILES = {
    "rate_QCD":   "higgsCombineScanRateQCD.MultiDimFit.mH300.root",
    "rate_Top":   "higgsCombineScanRateTop.MultiDimFit.mH300.root",
    "rate_WJets": "higgsCombineScanRateWJets.MultiDimFit.mH300.root",
}

# Output directory
OUTDIR = "validation_plots"
os.makedirs(OUTDIR, exist_ok=True)

#---------------------------
# HELPER: read fit_b from FitDiagnostics
#---------------------------

def get_fit_b(filename):
    f = ROOT.TFile(filename)
    fit_b = f.Get("fit_b")
    if not fit_b:
        raise RuntimeError(f"Could not find fit_b in {filename}")
    fit_b = fit_b.Clone()
    f.Close()
    return fit_b

def get_param(fit_result, name):
    """Return (value, error) for a parameter in a RooFitResult."""
    pars = fit_result.floatParsFinal()
    par = pars.find(name)
    if par:
        return par.getVal(), par.getError()
    # check constants
    pars_c = fit_result.constPars()
    par = pars_c.find(name)
    if par:
        return par.getVal(), 0.0
    return None, None

#---------------------------
# PLOT 1: Pre/Post-Fit Yields — per process per CR
#---------------------------
def plot_prepostfit_yields(fit_b):
    print("Plotting pre/post-fit yields (stacked per process)...")

    rate_QCD,   _ = get_param(fit_b, "rate_QCD")
    rate_Top,   _ = get_param(fit_b, "rate_Top")
    rate_WJets, _ = get_param(fit_b, "rate_WJets")

    rates = {"QCD": rate_QCD, "Top": rate_Top, "WJets": rate_WJets}

    channels  = list(YIELDS.keys())   # ["QCD CR", "Top CR", "W CR"]
    processes = ["QCD", "Top", "WJets"]
    n_ch      = len(channels)

    bar_width = 0.30
    gap       = 0.10                          # gap between pre/post pair
    cr_gap    = 0.60                          # gap between CRs

    # x positions: for each CR, two bars (pre, post) side by side
    x_pre  = []
    x_post = []
    x_cursor = 0.0
    cr_centers = []

    for i_ch in range(n_ch):
        xp  = x_cursor
        xpo = x_cursor + bar_width + gap
        x_pre.append(xp)
        x_post.append(xpo)
        cr_centers.append((xp + xpo) / 2)
        x_cursor += 2 * bar_width + gap + cr_gap

    fig, (ax, rax) = plt.subplots(
        2, 1, figsize=(10, 7),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=False
    )

    # ── Stacked bars ──
    for i_ch, ch in enumerate(channels):
        pre_bottom  = 0.0
        post_bottom = 0.0

        for proc in processes:
            color    = PROCESS_COLORS[proc]
            pre_val  = YIELDS[ch][proc]
            post_val = YIELDS[ch][proc] * rates[proc]

            # Pre-fit — faded
            ax.bar(x_pre[i_ch],  pre_val,  bar_width,
                   bottom=pre_bottom,
                   color=color, alpha=0.4, edgecolor="black", linewidth=0.6)

            # Post-fit — solid
            ax.bar(x_post[i_ch], post_val, bar_width,
                   bottom=post_bottom,
                   color=color, alpha=1.0, edgecolor="black", linewidth=0.6)

            pre_bottom  += pre_val
            post_bottom += post_val

        # Data point at CR center
        obs     = OBSERVED[ch]
        obs_err = np.sqrt(obs)
        ax.errorbar(cr_centers[i_ch], obs, yerr=obs_err,
                    fmt="ko", markersize=7, capsize=4, zorder=10,
                    label="Data" if i_ch == 0 else "")

    # ── Legend ──
    legend_handles = []
    for proc in processes:
        legend_handles.append(
            mpatches.Patch(facecolor=PROCESS_COLORS[proc], alpha=0.4,
                           edgecolor="black", label=f"{proc} pre-fit")
        )
        legend_handles.append(
            mpatches.Patch(facecolor=PROCESS_COLORS[proc], alpha=1.0,
                           edgecolor="black", label=f"{proc} post-fit")
        )
    legend_handles.append(
        plt.Line2D([0], [0], marker="o", color="black",
                   markersize=7, linestyle="None", label="Data")
    )
    ax.legend(handles=legend_handles, fontsize=9,
              ncol=2, loc="upper right", framealpha=0.9)

    ax.set_ylabel("Events", fontsize=13)
    ax.set_title("Pre/Post-Fit Yields per Control Region", fontsize=14)

    # x-tick labels: "Pre" / "Post" under each bar pair
    xtick_pos    = []
    xtick_labels = []
    for i_ch in range(n_ch):
        xtick_pos.append(x_pre[i_ch])
        xtick_labels.append("Pre")
        xtick_pos.append(x_post[i_ch])
        xtick_labels.append("Post")

    ax.set_xticks(xtick_pos)
    ax.set_xticklabels(xtick_labels, fontsize=10)

    # ── Ratio panel ──
    ratio_vals, ratio_errs = [], []
    for i_ch, ch in enumerate(channels):
        obs      = OBSERVED[ch]
        post_tot = sum(YIELDS[ch][p] * rates[p] for p in processes)
        obs_err  = np.sqrt(obs)
        ratio    = obs / post_tot if post_tot > 0 else 0
        r_err    = obs_err / post_tot if post_tot > 0 else 0
        ratio_vals.append(ratio)
        ratio_errs.append(r_err)

    # CR-labeled ticks for ratio panel
    cr_labels = ["QCD CR", "Top CR", "W CR"]
    rax.errorbar(cr_centers, ratio_vals, yerr=ratio_errs,
                 fmt="ko", markersize=6, capsize=4)
    rax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    rax.set_ylim(0.8, 1.2)
    rax.set_ylabel("Data / Post-fit", fontsize=11)
    rax.set_xticks(cr_centers)
    rax.set_xticklabels(cr_labels, fontsize=11, fontweight="bold")
    rax.set_xlim(-0.3, x_cursor - cr_gap + 0.3)

    plt.tight_layout()
    outpath = f"{OUTDIR}/prepostfit_yields.pdf"
    plt.savefig(outpath, bbox_inches="tight")
    outpath = f"{OUTDIR}/prepostfit_yields.png"
    plt.savefig(outpath, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {outpath}")


#---------------------------
# PLOT 2: Nuisance Pulls and Constraints
#---------------------------

def plot_pulls(fit_b):
    print("Plotting nuisance pulls and constraints...")

    # Parameters to show — reorder as you like
    param_names = [
        "rate_QCD", "rate_Top", "rate_WJets",
        "CMS_pileup", "CMS_scale_j", "CMS_res_j",
        "lumi_13p6TeV",
        "xsec_DY", "xsec_EWK",
    ]

    pulls       = []
    constraints = []
    labels      = []

    pars = fit_b.floatParsFinal()

    for name in param_names:
        par = pars.find(name)
        if not par:
            continue

        val = par.getVal()
        err = par.getError()

        # Rate params: pull = (val - 1) / 1, constraint = err / 1
        # Nuisances:   pull = val (already in sigma units), constraint = err
        if name.startswith("rate_"):
            pull       = val - 1.0
            constraint = err
        else:
            pull       = val
            constraint = err

        pulls.append(pull)
        constraints.append(constraint)
        labels.append(name)

    y = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(8, max(5, len(labels) * 0.5 + 1)))

    # Constraint bands
    ax.barh(y, [2 * c for c in constraints], left=[-c for c in constraints],
            height=0.5, color="yellow", alpha=0.8, label="Post-fit constraint (±1σ)")

    # Pre-fit band
    ax.axvspan(-1, 1, color="gray", alpha=0.15, label="Pre-fit ±1σ")
    ax.axvspan(-2, 2, color="gray", alpha=0.08, label="Pre-fit ±2σ")

    # Pulls
    ax.errorbar(pulls, y, xerr=constraints,
                fmt="ko", markersize=6, capsize=4, zorder=5, label="Pull ± post-fit error")

    ax.axvline(0, color="black", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("Pull (σ)", fontsize=13)
    ax.set_title("Nuisance Parameter Pulls and Constraints (b-only fit)", fontsize=13)
    ax.set_xlim(-2.5, 2.5)
    ax.legend(fontsize=9, loc="lower right")

    plt.tight_layout()
    outpath = f"{OUTDIR}/nuisance_pulls.pdf"
    plt.savefig(outpath)
    plt.close()
    print(f"  Saved: {outpath}")

#---------------------------
# PLOT 3: Correlation Matrix
#---------------------------

def plot_correlation(fit_b):
    print("Plotting correlation matrix...")

    pars = fit_b.floatParsFinal()
    n    = pars.getSize()

    names  = [pars[i].GetName() for i in range(n)]
    matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            matrix[i, j] = fit_b.correlation(names[i], names[j])

    # Clean up long names for display
    display_names = [
        n.replace("xsec_Stau_300_100mm", "xsec_Stau")
         .replace("lumi_13p6TeV", "lumi")
        for n in names
    ]

    fig, ax = plt.subplots(figsize=(max(8, n * 0.7), max(6, n * 0.6)))

    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    im   = ax.imshow(matrix, cmap="RdBu_r", norm=norm, aspect="auto")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(display_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(display_names, fontsize=9)

    # Annotate cells
    for i in range(n):
        for j in range(n):
            val = matrix[i, j]
            color = "white" if abs(val) > 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=7, color=color)

    plt.colorbar(im, ax=ax, label="Correlation")
    ax.set_title("Post-Fit Parameter Correlation Matrix (b-only fit)", fontsize=13)

    plt.tight_layout()
    outpath = f"{OUTDIR}/correlation_matrix.pdf"
    plt.savefig(outpath)
    plt.close()
    print(f"  Saved: {outpath}")

#---------------------------
# PLOT 4: Rate Param Likelihood Scans
#---------------------------

def plot_likelihood_scans():
    print("Plotting likelihood scans...")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)

    for ax, (param, scanfile) in zip(axes, SCAN_FILES.items()):
        if not os.path.exists(scanfile):
            print(f"  WARNING: {scanfile} not found, skipping {param}")
            ax.set_title(param)
            ax.text(0.5, 0.5, "File not found",
                    ha="center", va="center", transform=ax.transAxes, color="red")
            continue

        f = ROOT.TFile(scanfile)
        t = f.Get("limit")

        vals  = []
        dnlls = []

        for entry in t:
            vals.append(getattr(entry, param))
            dnlls.append(2 * entry.deltaNLL)

        f.Close()

        vals  = np.array(vals)
        dnlls = np.array(dnlls)

        # Sort by parameter value
        order = np.argsort(vals)
        vals  = vals[order]
        dnlls = dnlls[order]

        # Shift minimum to zero
        dnlls -= dnlls.min()

        ax.plot(vals, dnlls, "b-", linewidth=2)
        ax.axhline(1.0, color="red",    linestyle="--", linewidth=1, label="68% CL")
        ax.axhline(3.84, color="orange", linestyle="--", linewidth=1, label="95% CL")
        ax.set_xlabel(param, fontsize=12)
        ax.set_ylabel(r"$-2\Delta\ln L$", fontsize=12)
        ax.set_title(f"Likelihood Scan: {param}", fontsize=12)
        ax.set_ylim(0, 6)
        ax.legend(fontsize=9)

    plt.suptitle("Rate Parameter Likelihood Scans (CR-only fit)", fontsize=13, y=1.02)
    plt.tight_layout()
    outpath = f"{OUTDIR}/likelihood_scans.pdf"
    plt.savefig(outpath, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {outpath}")

#---------------------------
# MAIN
#---------------------------

if __name__ == "__main__":

    print(f"\nReading {FIT_DIAG_FILE}...")
    fit_b = get_fit_b(FIT_DIAG_FILE)

    plot_prepostfit_yields(fit_b)
    plot_pulls(fit_b)
    plot_correlation(fit_b)
    plot_likelihood_scans()

    print(f"\nDone. All plots saved to ./{OUTDIR}/")

