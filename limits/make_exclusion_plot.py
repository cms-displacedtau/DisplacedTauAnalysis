"""
make_exclusion_plot.py
======================
Collects the expected (and observed) upper limits on signal strength r
from AsymptoticLimits output across a (mass, ctau) grid, and produces
a 2D exclusion contour plot.

A point is "excluded" if the upper limit on r is < 1.0
(meaning the predicted signal cross-section is ruled out).

Features:
  - Median expected exclusion contour (r = 1, solid red)
  - ±1σ expected exclusion contours (r = 1 on the +1σ/-1σ limits, dashed red)
  - Shaded band between the ±1σ contours
  - Optional 1D Brazilian flag plot for a single ctau slice

Usage:
  python make_exclusion_plot.py
  python make_exclusion_plot.py --limitdir limits/ --masses 100 200 300 400 --ctaus 1 10 100
  python make_exclusion_plot.py -o exclusion_2D.pdf

Expected input:
  One ROOT file per signal point from Combine's AsymptoticLimits:
    limits/higgsCombine_datacard_Stau_{mass}_{ctau}mm.AsymptoticLimits.mH120.root

  Each contains a TTree "limit" with branches:
    - quantileExpected: -1 (obs), 0.025, 0.16, 0.5 (median exp), 0.84, 0.975
    - limit           : upper limit on r at that quantile
"""

import os
import glob
import argparse
import numpy as np
import uproot
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LogNorm
from scipy import interpolate


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
MASSES = [100, 200, 300, 400, 500]
CTAUS  = [1, 5, 10, 50, 100, 1000]


def signal_name(mass, ctau):
    return f"Stau_{mass}_{ctau}mm"


def read_limit_file(filepath):
    """
    Read one AsymptoticLimits output file.

    Returns dict: {quantile_label: limit_value}
      Keys: 'obs', 'exp-2', 'exp-1', 'exp', 'exp+1', 'exp+2'
    """
    f = uproot.open(filepath)
    tree = f["limit"]
    quantiles = tree["quantileExpected"].array(library="np")
    limits    = tree["limit"].array(library="np")
    f.close()

    result = {}
    for q, l in zip(quantiles, limits):
        if q < 0:
            result["obs"] = l
        elif abs(q - 0.025) < 0.01:
            result["exp-2"] = l
        elif abs(q - 0.16) < 0.01:
            result["exp-1"] = l
        elif abs(q - 0.5) < 0.01:
            result["exp"] = l
        elif abs(q - 0.84) < 0.01:
            result["exp+1"] = l
        elif abs(q - 0.975) < 0.01:
            result["exp+2"] = l

    return result


def collect_limits(masses, ctaus, limitdir, pattern):
    """
    Scan the limit directory for all (mass, ctau) points.

    Returns:
      results : dict of { (mass, ctau): {quantile: value} }
    """
    results = {}
    for mass in masses:
        for ctau in ctaus:
            sname = signal_name(mass, ctau)
            fname = pattern.format(sname=sname)
            fpath = os.path.join(limitdir, fname)

            if not os.path.exists(fpath):
                print(f"  [WARNING] Missing: {fpath}")
                continue

            try:
                lims = read_limit_file(fpath)
                results[(mass, ctau)] = lims
                print(f"  {sname:25s}  exp r < {lims.get('exp', -1):.4f}"
                      f"  [{lims.get('exp-1', -1):.4f}, {lims.get('exp+1', -1):.4f}]")
            except Exception as e:
                print(f"  [ERROR] {fpath}: {e}")

    return results


def _interpolate_contour(results, mass_arr, ctau_arr, quantile_key,
                         fine_mass, fine_logctau):
    """
    Interpolate the limit values for a given quantile onto a fine grid.

    Parameters
    ----------
    results       : dict of {(mass, ctau): {quantile: value}}
    mass_arr      : 1D array of mass grid points
    ctau_arr      : 1D array of ctau grid points
    quantile_key  : which quantile to interpolate ('exp', 'exp-1', 'exp+1', etc.)
    fine_mass     : 1D array of fine mass values for interpolation
    fine_logctau  : 1D array of fine log10(ctau) values for interpolation

    Returns
    -------
    fine_M    : 2D meshgrid of mass values
    fine_C    : 2D meshgrid of ctau values (10**log10(ctau))
    grid_fine : 2D array of interpolated limit values (or None if not enough points)
    """
    fine_M, fine_LC = np.meshgrid(fine_mass, fine_logctau)
    fine_C = 10**fine_LC

    pts = np.array([
        (m, np.log10(c))
        for (m, c) in results.keys()
        if quantile_key in results[(m, c)]
    ])
    vals = np.array([
        results[(m, c)][quantile_key]
        for (m, c) in results.keys()
        if quantile_key in results[(m, c)]
    ])

    if len(pts) < 4:
        return fine_M, fine_C, None

    grid_fine = interpolate.griddata(pts, vals, (fine_M, fine_LC), method="cubic")

    return fine_M, fine_C, grid_fine


def make_exclusion_plot(results, masses, ctaus, output="exclusion_2D.pdf",
                        title="Expected 95% CL Upper Limit on $r$, Private Work",
                        show_obs=False):
    """
    2D colour map of the expected upper limit on r, with exclusion
    contours at r = 1 for the median and ±1σ expected limits.
    """
    # Build 2D grid
    mass_arr = np.array(sorted(set(masses)))
    ctau_arr = np.array(sorted(set(ctaus)))

    # Matrix: rows = ctau, cols = mass (for pcolormesh orientation)
    limit_grid = np.full((len(ctau_arr), len(mass_arr)), np.nan)

    quantile_key = "obs" if show_obs else "exp"

    for (mass, ctau), lims in results.items():
        if quantile_key not in lims:
            continue
        i_ctau = np.where(ctau_arr == ctau)[0][0]
        i_mass = np.where(mass_arr == mass)[0][0]
        limit_grid[i_ctau, i_mass] = lims[quantile_key]

    # --- Figure ---
    fig, ax = plt.subplots(figsize=(9, 7))

    # Colour mesh
    mass_edges = _make_edges(mass_arr, logscale=False)
    ctau_edges = _make_edges(ctau_arr, logscale=True)

    #vmin = np.nanmin(limit_grid[limit_grid > 0]) if np.any(limit_grid > 0) else 0.01
    #vmax = np.nanmax(limit_grid) if np.any(np.isfinite(limit_grid)) else 100

    vmin = 1E-1
    vmax = 1E3

    pcm = ax.pcolormesh(
        mass_edges, ctau_edges, limit_grid,
        norm=LogNorm(vmin=max(vmin, 0.01), vmax=max(vmax, 10)),
        cmap="viridis_r",
        shading="flat",
    )
    cbar = fig.colorbar(pcm, ax=ax, label="95% CL upper limit on $r$")

    # --- Interpolation setup (shared fine grid for all contours) ---
    log_ctau = np.log10(ctau_arr)
    fine_mass    = np.linspace(mass_arr.min(), mass_arr.max(), 200)
    fine_logctau = np.linspace(log_ctau.min(), log_ctau.max(), 200)

    # --- Median expected contour (r = 1) ---
    if not np.all(np.isnan(limit_grid)):
        try:
            fine_M, fine_C, grid_median = _interpolate_contour(
                results, mass_arr, ctau_arr, quantile_key,
                fine_mass, fine_logctau
            )

            if grid_median is not None:
                # Median expected: solid red line
                cs_median = ax.contour(
                    fine_M, fine_C, grid_median,
                    levels=[1.0],
                    colors=["red"],
                    linewidths=[2.5],
                )
                ax.clabel(cs_median, fmt="$r = %.0f$", fontsize=10)

                # Fill excluded region (r < 1) with light shading
                #ax.contourf(
                #    fine_M, fine_C, grid_median,
                #    levels=[0, 1.0],
                #    colors=["red"],
                #    alpha=0.15,
                #)

        except Exception as e:
            print(f"  [NOTE] Median contour interpolation skipped: {e}")

    # --- ±1σ expected contours (r = 1 on the exp-1 / exp+1 surfaces) ---
    # The +1σ limit is *higher* than the median at every point, so
    # the r=1 contour on it is *larger* (more excluded) → outer boundary.
    # The -1σ limit is *lower* → inner boundary.
    for sigma_key, sigma_label, linestyle in [
        ("exp-1", r"Expected $-1\sigma$", "--"),
        ("exp+1", r"Expected $+1\sigma$", "--"),
    ]:
        try:
            fine_M_s, fine_C_s, grid_sigma = _interpolate_contour(
                results, mass_arr, ctau_arr, sigma_key,
                fine_mass, fine_logctau
            )

            if grid_sigma is not None:
                ax.contour(
                    fine_M_s, fine_C_s, grid_sigma,
                    levels=[1.0],
                    colors=["red"],
                    linewidths=[1.5],
                    linestyles=[linestyle],
                )

        except Exception as e:
            print(f"  [NOTE] {sigma_key} contour interpolation skipped: {e}")

    # --- ±1σ band shading ---
    # Shade the region between the -1σ and +1σ r=1 contours.
    # We do this by shading "excluded at +1σ but not at -1σ".
    try:
        fine_M_p, fine_C_p, grid_plus = _interpolate_contour(
            results, mass_arr, ctau_arr, "exp+1",
            fine_mass, fine_logctau
        )
        fine_M_m, fine_C_m, grid_minus = _interpolate_contour(
            results, mass_arr, ctau_arr, "exp-1",
            fine_mass, fine_logctau
        )

        if grid_plus is not None and grid_minus is not None:
            # Region excluded at +1σ (r < 1 on the +1σ surface)
            # but NOT excluded at -1σ (r >= 1 on the -1σ surface)
            band_mask = (grid_plus < 1.0).astype(float) - (grid_minus < 1.0).astype(float)
            band_mask = np.clip(band_mask, 0, 1)
            ax.contourf(
                fine_M_p, fine_C_p, band_mask,
                levels=[0.5, 1.5],
                colors=["red"],
                alpha=0.10,
            )
    except Exception as e:
        print(f"  [NOTE] ±1σ band shading skipped: {e}")

    # --- Grid-point markers + labels ---
    #for (mass, ctau), lims in results.items():
    #    if quantile_key not in lims:
    #        continue
    #    r_val = lims[quantile_key]
    #    excluded = r_val < 1.0
    #    marker   = "v" if excluded else "^"
    #    color    = "red" if excluded else "white"
    #    edge     = "darkred" if excluded else "black"

    #    ax.plot(mass, ctau, marker=marker, markersize=10,
    #            color=color, markeredgecolor=edge, markeredgewidth=1.2, zorder=5)
    #    ax.annotate(
    #        f"{r_val:.2f}",
    #        (mass, ctau),
    #        textcoords="offset points",
    #        xytext=(8, 8),
    #        fontsize=8,
    #        bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7),
    #        zorder=6,
    #    )

    # --- Axes ---
    ax.set_yscale("log")
    ax.set_xlabel(r"$m_{\tilde{\tau}}$ [GeV]", fontsize=13)
    ax.set_ylabel(r"$c\tau$ [mm]", fontsize=13)
    ax.set_title(title, fontsize=14)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="red", linewidth=2.5, label=r"Exclusion contour ($r = 1$)"),
        Line2D([0], [0], color="red", linewidth=1.5, linestyle="--",
               label=r"Expected $\pm 1\sigma$ ($r = 1$)"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9,
              framealpha=0.9)

    fig.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved 2D exclusion plot: {output}")
    plt.close(fig)


def _make_edges(arr, logscale=False):
    """
    From bin centres, build bin edges for pcolormesh.
    """
    if logscale:
        log_arr  = np.log10(arr)
        log_edges = np.zeros(len(arr) + 1)
        log_edges[1:-1] = 0.5 * (log_arr[:-1] + log_arr[1:])
        log_edges[0]    = log_arr[0]  - (log_edges[1] - log_arr[0])
        log_edges[-1]   = log_arr[-1] + (log_arr[-1] - log_edges[-2])
        return 10**log_edges
    else:
        edges = np.zeros(len(arr) + 1)
        edges[1:-1] = 0.5 * (arr[:-1] + arr[1:])
        edges[0]    = arr[0]  - (edges[1] - arr[0])
        edges[-1]   = arr[-1] + (arr[-1] - edges[-2])
        return edges


# ---------------------------------------------------------------------------
# Optional: 1D Brazilian flag plot for one ctau slice
# ---------------------------------------------------------------------------
def make_brazil_plot(results, masses, ctau_slice, output="brazil_1D.pdf",
                     title=None):
    """
    Standard 'Brazilian flag' plot: expected limit on r vs mass
    at a fixed ctau, with ±1σ and ±2σ bands.
    """
    mass_vals = sorted(set(masses))
    exp, exp_m1, exp_p1, exp_m2, exp_p2 = [], [], [], [], []
    obs_vals = []
    valid_masses = []

    for mass in mass_vals:
        key = (mass, ctau_slice)
        if key not in results:
            continue
        lims = results[key]
        if "exp" not in lims:
            continue
        valid_masses.append(mass)
        exp.append(lims["exp"])
        exp_m1.append(lims.get("exp-1", lims["exp"]))
        exp_p1.append(lims.get("exp+1", lims["exp"]))
        exp_m2.append(lims.get("exp-2", lims["exp"]))
        exp_p2.append(lims.get("exp+2", lims["exp"]))
        if "obs" in lims:
            obs_vals.append(lims["obs"])

    if not valid_masses:
        print(f"  [WARNING] No valid points for ctau={ctau_slice} — skipping Brazil plot.")
        return

    m = np.array(valid_masses)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.fill_between(m, exp_m2, exp_p2, color="#FFCC00", label=r"Expected $\pm 2\sigma$")
    ax.fill_between(m, exp_m1, exp_p1, color="#00CC00", label=r"Expected $\pm 1\sigma$")
    ax.plot(m, exp, "k--", linewidth=2, label="Median expected")
    if obs_vals:
        ax.plot(m, obs_vals, "ko-", linewidth=2, label="Observed")

    ax.axhline(1.0, color="red", linewidth=1.5, linestyle=":")
    ax.set_xlabel(r"$m_{\tilde{\tau}}$ [GeV]", fontsize=13)
    ax.set_ylabel(r"95% CL upper limit on $r$", fontsize=13)
    ax.set_yscale("log")
    ax.legend(fontsize=10)
    ax.set_title(title or rf"Expected Limit vs Mass at $c\tau = {ctau_slice}$ mm",
                 fontsize=13)

    fig.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved Brazil plot: {output}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="2D exclusion plot from AsymptoticLimits output")
    p.add_argument("--limitdir", default=".",
                   help="Directory containing higgsCombine*.root files")
    p.add_argument("--masses", nargs="+", type=int, default=MASSES)
    p.add_argument("--ctaus",  nargs="+", type=int, default=CTAUS)
    p.add_argument("-o", "--output", default="exclusion_2D.pdf")
    p.add_argument("--pattern",
                   default="higgsCombine_datacard_{sname}.AsymptoticLimits.mH120.root",
                   help="Filename pattern with {sname} placeholder")
    p.add_argument("--obs", action="store_true",
                   help="Show observed instead of expected limits")
    p.add_argument("--brazil-ctau", type=int, default=None,
                   help="Also make a 1D Brazil plot at this ctau [mm]")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"Scanning: {args.limitdir}")
    print(f"Grid    : masses={args.masses}, ctaus={args.ctaus}")
    print()

    results = collect_limits(args.masses, args.ctaus, args.limitdir, args.pattern)

    if not results:
        print("ERROR: No limit files found. Check --limitdir and --pattern.")
        return

    make_exclusion_plot(results, args.masses, args.ctaus,
                        output=args.output, show_obs=args.obs)

    if args.brazil_ctau is not None:
        brazil_out = args.output.replace(".pdf", f"_brazil_ctau{args.brazil_ctau}.pdf")
        make_brazil_plot(results, args.masses, args.brazil_ctau,
                         output=brazil_out)


if __name__ == "__main__":
    main()

