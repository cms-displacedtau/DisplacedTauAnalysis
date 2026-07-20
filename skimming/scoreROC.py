#!/usr/bin/env python3
"""
ROC Curve Calculator for events.Jet.disTauTag_score1  (Coffea edition)
======================================================================
Signal efficiency : fraction of jets in Stau MC that are dR-matched to a
                    GenVisTau and pass a score threshold.
Fake rate         : fraction of ALL jets in background MC that pass the
                    same score threshold.

Supports **multiple background files** — pass them as separate arguments
or use shell globs.

Uses coffea.nanoevents for clean, vectorised dR matching via the built-in
`nearest` helper on LorentzVector-like objects.

Usage:
    python roc_disTauTag.py \
        --signal   /path/to/stau_mc.root \
        --background /path/to/qcd*.root /path/to/wjets*.root \
        --tree Events \
        --schema NanoAODSchema \
        --dr 0.4 \
        --npoints 200 \
        --output roc_disTauTag.pdf

    # You can also use shell globs directly:
    python roc_disTauTag.py \
        --signal signal.root \
        --background /bkg_dir/*.root
"""

import argparse
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import awkward as ak

from coffea.nanoevents import NanoEventsFactory, NanoAODSchema, BaseSchema


# ──────────────────────────────────────────────────────────────────────
# Schema lookup helper
# ──────────────────────────────────────────────────────────────────────
SCHEMA_MAP = {
    "NanoAODSchema": NanoAODSchema,
    "BaseSchema":    BaseSchema,
}


def get_schema(name: str):
    if name not in SCHEMA_MAP:
        raise ValueError(
            f"Unknown schema '{name}'. Choose from: {list(SCHEMA_MAP.keys())}"
        )
    return SCHEMA_MAP[name]


# ──────────────────────────────────────────────────────────────────────
# Resolve file paths (expand globs)
# ──────────────────────────────────────────────────────────────────────
def resolve_paths(raw_paths):
    """
    Expand shell-style globs and verify every path exists.
    Returns a flat, deduplicated list of file paths.
    """
    resolved = []
    for p in raw_paths:
        expanded = sorted(glob.glob(p))
        if not expanded:
            # Not a glob — treat as literal path
            if not os.path.isfile(p):
                raise FileNotFoundError(f"File not found: {p}")
            resolved.append(p)
        else:
            resolved.extend(expanded)
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for f in resolved:
        if f not in seen:
            seen.add(f)
            unique.append(f)
    return unique


# ──────────────────────────────────────────────────────────────────────
# Load events via coffea
# ──────────────────────────────────────────────────────────────────────
def load_events(filepath, tree_name, schema_cls):
    """Return a NanoEvents object for the given ROOT file."""
    events = NanoEventsFactory.from_root(
        filepath,
        treepath=tree_name,
        schemaclass=schema_cls,
    ).events()
    return events


# ──────────────────────────────────────────────────────────────────────
# Build the matched-signal score array
# ──────────────────────────────────────────────────────────────────────
def get_signal_scores(filepath, tree_name, schema_cls, dr_cut):
    """
    For every jet in the Stau MC, check whether there is a GenVisTau
    within dR < dr_cut.  Return the disTauTag_score1 values of the
    matched jets.

    Efficiency denominator = all dR-matched jets (regardless of score).
    """
    events = load_events(filepath, tree_name, schema_cls)

    jets     = events.Jet
    gen_taus = events.GenVisTau

    # ── Vectorised dR matching using coffea's nearest() ───────────
    nearest_gen, dr = jets.nearest(gen_taus, axis=1, metric="delta_r",
                                   return_metric=True, threshold=dr_cut)

    matched_mask   = ~ak.is_none(nearest_gen, axis=1)
    matched_scores = jets.disTauTag_score1[matched_mask]
    matched_scores = np.asarray(ak.flatten(matched_scores))

    print(f"[Signal] {filepath}")
    print(f"         Total jets: {int(ak.sum(ak.num(jets)))},  "
          f"dR-matched jets: {len(matched_scores)}")
    return matched_scores


# ──────────────────────────────────────────────────────────────────────
# Build the background (fake) score array — multiple files
# ──────────────────────────────────────────────────────────────────────
def get_background_scores(filepaths, tree_name, schema_cls):
    """
    Return the concatenated flat array of disTauTag_score1 for ALL jets
    across every background file.
    """
    all_scores = []
    total_jets = 0

    for fpath in filepaths:
        events = load_events(fpath, tree_name, schema_cls)
        scores = np.asarray(ak.flatten(events.Jet.disTauTag_score1))
        all_scores.append(scores)
        total_jets += len(scores)
        print(f"[Background] {fpath}  —  {len(scores)} jets")

    combined = np.concatenate(all_scores)
    print(f"[Background] Total jets across {len(filepaths)} file(s): "
          f"{total_jets}")
    return combined


# ──────────────────────────────────────────────────────────────────────
# Compute ROC points
# ──────────────────────────────────────────────────────────────────────
def compute_roc(signal_scores, bkg_scores, n_points=200):
    """Scan thresholds and return (signal_efficiency, fake_rate) arrays."""
    all_scores = np.concatenate([signal_scores, bkg_scores])
    lo, hi = np.min(all_scores), np.max(all_scores)
    thresholds = np.linspace(lo, hi, n_points)

    sig_eff   = np.array([np.mean(signal_scores >= t) for t in thresholds])
    fake_rate = np.array([np.mean(bkg_scores   >= t) for t in thresholds])

    return sig_eff, fake_rate, thresholds


# ──────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────
def plot_roc(sig_eff, fake_rate, output_path):
    """Draw the ROC curve: signal efficiency vs fake rate (linear)."""
    fig, ax = plt.subplots(figsize=(7, 6))

    order      = np.argsort(fake_rate)
    fr_sorted  = fake_rate[order]
    se_sorted  = sig_eff[order]

    auc = np.trapz(se_sorted, fr_sorted)

    ax.plot(fr_sorted, se_sorted, linewidth=2, color="royalblue",
            label=f"disTauTag_score1  (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1, label="Random")

    ax.set_xlabel("Fake rate (background jet efficiency)", fontsize=13)
    ax.set_ylabel("Signal efficiency (dR-matched to GenVisTau)", fontsize=13)
    ax.set_title("ROC — Jet.disTauTag_score1", fontsize=14)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"\n[Done] ROC curve saved to {output_path}")
    plt.show()


def plot_roc_log(sig_eff, fake_rate, output_path):
    """Draw the ROC curve with a log-scale x-axis."""
    fig, ax = plt.subplots(figsize=(7, 6))

    order     = np.argsort(fake_rate)
    fr_sorted = fake_rate[order]
    se_sorted = sig_eff[order]

    mask = fr_sorted > 0
    ax.plot(fr_sorted[mask], se_sorted[mask], linewidth=2, color="crimson",
            label="disTauTag_score1")

    ax.set_xscale("log")
    ax.set_xlabel("Fake rate (log scale)", fontsize=13)
    ax.set_ylabel("Signal efficiency", fontsize=13)
    ax.set_title("ROC (log x) — Jet.disTauTag_score1", fontsize=14)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, which="both", alpha=0.3)

    log_output = output_path.replace(".pdf", "_log.pdf") \
                             .replace(".png", "_log.png")
    fig.tight_layout()
    fig.savefig(log_output, dpi=150)
    print(f"[Done] ROC curve (log) saved to {log_output}")
    plt.show()


# ──────────────────────────────────────────────────────────────────────
# Working-point summary
# ──────────────────────────────────────────────────────────────────────
def print_working_points(sig_eff, fake_rate, thresholds):
    """Print the fake rate at a few standard signal-efficiency targets."""
    targets = [0.90, 0.80, 0.70, 0.50]
    print("\n{:<20s} {:<20s} {:<20s}".format(
        "Sig. Eff. target", "Achieved Sig. Eff.", "Fake rate"))
    print("-" * 60)
    for target in targets:
        idx = np.argmin(np.abs(sig_eff - target))
        print(f"{target:<20.2f} {sig_eff[idx]:<20.4f} {fake_rate[idx]:<20.6f}"
              f"   (threshold = {thresholds[idx]:.4f})")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="ROC curve for Jet.disTauTag_score1 using Coffea "
                    "(supports multiple background files)")
    parser.add_argument("--signal", required=True,
                        help="Path to Stau signal MC ROOT file")
    parser.add_argument("--background", required=True, nargs="+",
                        help="Path(s) to background MC ROOT file(s). "
                             "Accepts multiple files and shell globs.")
    parser.add_argument("--tree", default="Events",
                        help="TTree name (default: Events)")
    parser.add_argument("--schema", default="NanoAODSchema",
                        choices=list(SCHEMA_MAP.keys()),
                        help="Coffea schema class (default: NanoAODSchema)")
    parser.add_argument("--dr", type=float, default=0.4,
                        help="dR matching cone (default: 0.4)")
    parser.add_argument("--npoints", type=int, default=200,
                        help="Number of threshold scan points (default: 200)")
    parser.add_argument("--output", default="roc_disTauTag.pdf",
                        help="Output plot filename (default: roc_disTauTag.pdf)")

    args = parser.parse_args()
    schema_cls = get_schema(args.schema)

    # ── Resolve background file list (expand globs) ───────────────
    bkg_files = resolve_paths(args.background)
    print(f"\nBackground files ({len(bkg_files)}):")
    for f in bkg_files:
        print(f"  • {f}")
    print()

    # ── Collect scores ────────────────────────────────────────────
    sig_scores = get_signal_scores(args.signal, args.tree,
                                   schema_cls, args.dr)
    bkg_scores = get_background_scores(bkg_files, args.tree, schema_cls)

    # ── Compute & plot ROC ────────────────────────────────────────
    sig_eff, fake_rate, thresholds = compute_roc(sig_scores, bkg_scores,
                                                  args.npoints)
    print_working_points(sig_eff, fake_rate, thresholds)
    plot_roc(sig_eff, fake_rate, args.output)
    plot_roc_log(sig_eff, fake_rate, args.output)


if __name__ == "__main__":
    main()

