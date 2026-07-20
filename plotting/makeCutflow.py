"""
Multi-Sample Signal Region Cutflow Plot using Coffea & mplhep
==============================================================
Reads multiple NanoAOD ROOT files **per signal sample**,
applies a cumulative cutflow, and produces:
  - One cutflow bar chart per sample
  - An overlay comparison of all samples on a single plot

Each sample can point to multiple files via:
  • Comma-separated paths :  name:file1.root,file2.root,file3.root
  • Glob / wildcard       :  name:/data/stop500/*.root
  • A mix of both         :  name:/data/part1/*.root,/data/extra.root

Requires: coffea, mplhep, matplotlib, numpy, awkward, uproot
Install:  pip install coffea mplhep matplotlib numpy awkward uproot
"""

import argparse
import glob, os
import json
import pathlib
from collections import OrderedDict

import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
import mplhep as hep
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema, PFNanoAODSchema
from coffea.analysis_tools import PackedSelection

PFNanoAODSchema.warn_missing_crossrefs = False
# ──────────────────────────────────────────────
# 0.  Style
# ──────────────────────────────────────────────
plt.style.use(hep.style.CMS)


# ──────────────────────────────────────────────
# 1.  Resolve file paths (globs + comma-sep)
# ──────────────────────────────────────────────
def resolve_paths(raw_path_str):
    """
    Accept a string that may contain comma-separated paths and/or
    shell-style globs.  Return a sorted list of resolved file paths.
    """
    resolved = []
    for token in raw_path_str.split(","):
        token = token.strip()
        if not token:
            continue
        matches = sorted(glob.glob(token))
        if matches:
            resolved.extend(matches)
        else:
            # No glob match — treat as literal path (will error later
            # if the file doesn't exist, with a clear message).
            resolved.append(token)
    return resolved


# ──────────────────────────────────────────────
# 2.  Define signal-region cuts
#     Edit this function to match your analysis
# ──────────────────────────────────────────────
def build_selection(events):
    """
    Register all signal-region cuts on a PackedSelection.
    Adjust variable names / thresholds to match your NanoAOD branches.
    """
    sel = PackedSelection()

    sel.add("PFMET pt",           events.CorrectedPFMET.pt > 105)
    sel.add("Muon pt",          ak.flatten(events.DisMuon.pt) > 30)
    sel.add("Muon iso",           ak.flatten(events.DisMuon.pfRelIso03_all) < 0.18)
    sel.add("Jet pt",             ak.flatten(events.CorrectedJet.pt) > 32)
    sel.add("Jet score",          ak.flatten(events.CorrectedJet.disTauTag_score1)> 0.9)
    sel.add("Muon dxy",             ((ak.flatten(events.DisMuon.dxy) > 0.1) & (ak.flatten(events.DisMuon.dxy) < 10)))
    sel.add("Jet dxy",            ak.flatten(events.CorrectedJet.dxy) > 0.1)

    return sel

def load_original_counts(json_path):
    """
    Read a JSON file mapping sample names to their original
    event counts (e.g. total generated before skimming).

    Expected format:
    {
        "Stop_500": 1000000,
        "Stop_800": 500000
    }
    """
    with open(json_path, "r") as f:
        counts = json.load(f)
    return counts


# ──────────────────────────────────────────────
# 3.  Load & concatenate events from many files
# ──────────────────────────────────────────────
def load_events(file_list, treename="Events", entry_stop=None):
    """
    Read each ROOT file with NanoEventsFactory and concatenate
    the resulting awkward arrays into one events collection.
    """
    all_events = []
    for fpath in file_list:
        print(f"    Reading {fpath} …")
        ev = NanoEventsFactory.from_root(
            {fpath:treename},
            schemaclass=PFNanoAODSchema,
        ).events()
        all_events.append(ev)

    if len(all_events) == 1:
        return all_events[0]

    return ak.concatenate(all_events, axis=0)


# ──────────────────────────────────────────────
# 4.  Compute cumulative cutflow for one sample
# ──────────────────────────────────────────────
def compute_cutflow(events, n_original):
    """
    Return (labels, cumulative_yields, efficiencies).

    The first two entries are:
      - "Original events"  : from the JSON (pre-skim total)
      - "Input file events": number of events in the loaded ROOT files
    followed by the cumulative cut yields.

    Efficiencies are computed relative to n_original so that
    the full processing chain (skim → selection) is visible.
    """
    sel = build_selection(events)
    cut_names = list(sel.names)

    n_input = len(events)

    # Build the full yield list: original → input → no-selection → cuts
    labels = ["No cuts", "Preselection"] + cut_names
    cumulative_yields = [n_original, n_input]

    for i in range(1, len(cut_names) + 1):
        mask = sel.all(*cut_names[:i])
        cumulative_yields.append(int(ak.sum(mask)))

    cumulative_yields = np.array(cumulative_yields, dtype=float)
    efficiencies = cumulative_yields / n_original * 100
# Relative (N-1) efficiency: yield_i / yield_{i-1}
    rel_efficiencies = np.ones_like(cumulative_yields) * 100.0
    for i in range(1, len(cumulative_yields)):
        if cumulative_yields[i - 1] > 0:
            rel_efficiencies[i] = cumulative_yields[i] / cumulative_yields[i - 1] * 100.0
        else:
            rel_efficiencies[i] = 0.0

    return labels, cumulative_yields, efficiencies, rel_efficiencies

# ──────────────────────────────────────────────
# 5.  Per-sample cutflow bar chart
# ──────────────────────────────────────────────
def plot_single_cutflow(labels, yields, efficiencies, sample_name, n_files, outdir):
    """Two-panel step histogram: yields (log) + cumulative efficiency."""
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1,
        figsize=(14, 9),
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05},
        sharex=True,
    )

    n_cuts = len(labels)
    bin_edges = np.arange(n_cuts + 1) - 0.5

    # -- top: absolute yields (step histogram) --
    ax_bot.stairs(yields, bin_edges, linewidth=2.2, color="#31AFD4",
                  label=sample_name)
    for i, val in enumerate(yields):
        ax_bot.text(i, val * 1.15, f"{int(val):,}", ha="center", va="bottom",
                    fontsize=9, fontweight="bold")

    ax_bot.set_ylabel("Events", fontsize = 20)
    ax_bot.set_yscale("log")
    ax_bot.set_ylim(0.5, yields[0] * 10)
    ax_bot.set_xlim(bin_edges[0], bin_edges[-1])
    hep.cms.label(f"{sample_name}", ax=ax_top, data=False, loc=0, com =13.6)

    # -- bottom: efficiency (step histogram) --
    ax_top.stairs(efficiencies, bin_edges, linewidth=2.2, color="#31AFD4")
    for i, eff in enumerate(efficiencies):
        ax_top.text(i, eff + 1.5, f"{eff:.1f}%", ha="center", va="bottom", fontsize=8)

    ax_top.set_ylabel("Efficiency [%]", fontsize = 18)
    ax_top.set_ylim(0, 120)
    ax_bot.set_xticks(np.arange(n_cuts))
    ax_bot.set_xticklabels(labels, rotation=35, ha="right", fontsize=11)

    fig.tight_layout()
    fname = f"{outdir}/cutflow_{sample_name.replace(' ', '_')}.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    fname = f"{outdir}/cutflow_{sample_name.replace(' ', '_')}.pdf"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved {fname}")


# ──────────────────────────────────────────────
# 6b. Per-sample relative efficiency step histogram
# ──────────────────────────────────────────────
def plot_single_relative_efficiency(labels,yields, rel_efficiencies, sample_name, n_files, outdir):
    """Two-panel step histogram: yields (log) + cumulative efficiency."""
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1,
        figsize=(14, 9),
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05},
        sharex=True,
    )

    n_cuts = len(labels)
    bin_edges = np.arange(n_cuts + 1) - 0.5

    # -- top: absolute yields (step histogram) --
    ax_top.stairs(yields, bin_edges, linewidth=2.2, color="#902D41",
                  label=sample_name)
    for i, val in enumerate(yields):
        ax_top.text(i, val * 1.15, f"{int(val):,}", ha="center", va="bottom",
                    fontsize=9, fontweight="bold")

    ax_top.set_ylabel("Events", fontsize = 20)
    ax_top.set_yscale("log")
    ax_top.set_ylim(0.5, yields[0] * 10)
    ax_top.set_xlim(bin_edges[0], bin_edges[-1])
    hep.cms.label(f"{sample_name}", ax=ax_top, data=False, loc=0, com = 13.6)

    # -- bottom: efficiency (step histogram) --
    ax_bot.stairs(rel_efficiencies, bin_edges, linewidth=2.2, color="#902D41")
    for i, eff in enumerate(rel_efficiencies):
        ax_bot.text(i, eff + 1.5, f"{eff:.1f}%", ha="center", va="bottom", fontsize=8)

    ax_bot.set_ylabel("Relative Efficiency [%]", fontsize = 14)
    ax_bot.set_ylim(0, 120)
    ax_bot.set_xticks(np.arange(n_cuts))
    ax_bot.set_xticklabels(labels, rotation=35, ha="right", fontsize=11)

    fig.tight_layout()
    outpath = f"{outdir}/rel_eff_{sample_name}.png"
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    outpath = f"{outdir}/rel_eff_{sample_name}.pdf"
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved {outpath}", flush=True)

# ──────────────────────────────────────────────
# 6.  Overlay comparison across all samples
# ──────────────────────────────────────────────
def plot_overlay(all_results, outdir):
    """
    Step-histogram overlay for every sample on a single canvas.
    all_results: OrderedDict { sample_name: (labels, yields, effs) }
    """
    sample_names = list(all_results.keys())
    labels = list(all_results.values())[0][0]
    n_cuts = len(labels)
    n_samples = len(sample_names)

    bin_edges = np.arange(n_cuts + 1) - 0.5
    cmap = plt.cm.tab10

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1,
        figsize=(14, 10),
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05},
        sharex=True,
    )

    for idx, name in enumerate(sample_names):
        _, yields, effs = all_results[name]
        color = cmap(idx / max(n_samples - 1, 1))

        ax_top.stairs(yields, bin_edges, linewidth=2.2, color=color, label=name)
        ax_bot.stairs(effs, bin_edges, linewidth=2.2, color=color)

    ax_top.set_ylabel("Events")
    ax_top.set_yscale("log")
    ax_top.set_xlim(bin_edges[0], bin_edges[-1])
    ax_top.legend(fontsize=11, ncol=min(n_samples, 3))
    ax_top.set_title("Cutflow Comparison — All Signal Samples", fontsize=16)
    hep.cms.label(ax=ax_top, data=False, label="Simulation Preliminary", loc=0, com=13.6)

    ax_bot.set_ylabel("Efficiency [%]")
    ax_bot.set_ylim(0, 120)
    ax_bot.set_xticks(np.arange(n_cuts))
    ax_bot.set_xticklabels(labels, rotation=35, ha="right", fontsize=11)

    fig.tight_layout()
    os.makedirs(os.path.join(outdir, "cutflow"), exist_ok = True)
    outdir = os.path.join(outdir, "cutflow")
    fname_png = f"{outdir}/cutflow_overlay.png"
    fname_pdf = f"{outdir}/cutflow_overlay.pdf"
    fig.savefig(fname_png, dpi=150, bbox_inches="tight")
    fig.savefig(fname_pdf, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved {fname}")


# ──────────────────────────────────────────────
# 8.  Main — CLI entry point
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Multi-sample signal-region cutflow with coffea + mplhep",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Two samples with globs + a JSON for original event counts
  python %(prog)s \\
    -s 'Stop_500:/data/stop500/*.root' \\
       'Stop_800:/data/stop800/*.root' \\
    -j original_counts.json

  # Explicit file lists
  python %(prog)s \\
    -s 'Gluino_1400:g1400_1.root,g1400_2.root' \\
       'Gluino_2000:g2000_1.root,g2000_2.root' \\
    -j original_counts.json \\
    --entry-stop 10000
""",
    )
    parser.add_argument(
        "-s", "--samples", nargs="+", required=True,
        help="Sample specifications as  name:path[,path,...]. "
             "Paths may contain shell-style globs.",
    )
    parser.add_argument(
        "-j", "--json", required=True, type=str,
        help="Path to a JSON file mapping sample names to original "
             "event counts (before skimming/filtering).",
    )
    parser.add_argument(
        "-o", "--outdir", default="plots", type=str,
        help="Output directory for plots (default: plots/).",
    )
    parser.add_argument(
        "--entry-stop", type=int, default=None,
        help="Max events to read *per file* (for quick tests).",
    )
    parser.add_argument(
        "--tree", default="Events", type=str,
        help="TTree name inside the ROOT files (default: Events).",
    )
    args = parser.parse_args()

    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load original event counts from JSON
    original_counts = load_original_counts(args.json)

    # Parse sample specs
    samples = OrderedDict()
    for spec in args.samples:
        if ":" not in spec:
            parser.error(f"Bad sample spec '{spec}' — expected  name:path[,path,...]")
        name, raw_paths = spec.split(":", 1)
        file_list = resolve_paths(raw_paths)
        if not file_list:
            parser.error(f"No files resolved for sample '{name}' from '{raw_paths}'")
        samples[name] = file_list

    # Validate that every sample has an entry in the JSON
    for name in samples:
        if name not in original_counts:
            parser.error(
                f"Sample '{name}' not found in JSON file '{args.json}'. "
                f"Available keys: {list(original_counts.keys())}"
            )

    # Process each sample
    all_results = OrderedDict()

    for name, file_list in samples.items():
        n_files = len(file_list)
        n_original = original_counts[name]
        print(f"\n▶ Processing sample: {name}  ({n_files} file{'s' if n_files != 1 else ''},"
              f" {n_original:,} original events from JSON)")

        events = load_events(file_list, treename=args.tree, entry_stop=args.entry_stop)
        labels, yields, effs, rel_effs = compute_cutflow(events, n_original)
        all_results[name] = {
            "labels": labels,
            "yields": yields,
            "efficiencies": effs,
            "rel_efficiencies": rel_effs,
            "n_files": len(file_list),
        }
        plot_single_cutflow(labels, yields, effs, name, n_files, args.outdir)
        plot_single_relative_efficiency(labels, yields, rel_effs, name,len(file_list), args.outdir)
    # Overlay plot & summary
        print("", flush=True)
        header = f"  {'Cut':<30s} {'Yield':>12s} {'Cumul.Eff':>10s} {'Rel.Eff':>10s}"
        print(header, flush=True)
        print("  " + "─" * (len(header) - 2), flush=True)
        for lab, yld, eff, reff in zip(labels, yields, effs, rel_effs):
            print(f"  {lab:<30s} {int(yld):>12,} {eff:>9.2f}% {reff:>9.2f}%", flush=True)


if __name__ == "__main__":
    main()

