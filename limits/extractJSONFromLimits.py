"""
HybridNew Limits JSON Extractor
--------------------------------
Reads HybridNew ROOT files and extracts expected limit quantiles
into JSON files named limits_Stau_{mass}_{lifetime}.json,
one file per unique (mass, lifetime) combination.

Expected filename format:
  higgsCombine.HybridNew_Stau_{mass}_{lifetime}_{eff}.HybridNew.mH{mass}.quant{q}.root

Example:
  higgsCombine.HybridNew_Stau_500_1000mm_0p99.HybridNew.mH500.quant0.975.root
  → limits_Stau_500_1000mm.json

Quantile mapping:
  quant0.025  -> exp-2
  quant0.160  -> exp-1
  quant0.500  -> exp0
  quant0.840  -> exp+1
  quant0.975  -> exp+2
  (no quant)  -> obs   [observed — skipped if blinded]

Usage:
  python3 extract_hybridnew_limits.py --inputdir .
  python3 extract_hybridnew_limits.py --inputdir /path/to/files --blinded
  python3 extract_hybridnew_limits.py --inputdir . --pattern "*.root"
"""

import ROOT
import os
import re
import json
import glob
import argparse
from collections import defaultdict

ROOT.gROOT.SetBatch(True)

# ─────────────────────────────────────────────
# Quantile suffix → JSON key mapping
# ─────────────────────────────────────────────
QUANT_MAP = {
    "0.025": "exp-2",
    "0.160": "exp-1",
    "0.500": "exp0",
    "0.840": "exp+1",
    "0.975": "exp+2",
}

# ─────────────────────────────────────────────
# Parse mass, lifetime, and quantile from filename
# ─────────────────────────────────────────────

def parse_filename(fname):
    """
    Returns (mass_str, lifetime_str, quant_key) from a HybridNew filename.

    Example:
      higgsCombine.HybridNew_Stau_500_1000mm_0p99.HybridNew.mH500.quant0.975.root
      → ("500.0", "1000mm", "exp+2")

    Returns (None, None, None) if the file doesn't match.
    """
    base = os.path.basename(fname)

    # Extract mass from mH{mass} token
    mass_match = re.search(r'\.mH([\d.]+)\.', base)
    if not mass_match:
        return None, None, None
    mass = mass_match.group(1)
    mass_str = str(mass)

    # Extract lifetime from HybridNew_Stau_{mass}_{lifetime}_{eff}
    lifetime_match = re.search(r'HybridNew_Stau_[\d]+_([\w]+)_[\w]+\.HybridNew', base)
    if not lifetime_match:
        print(f"  [WARN] Could not extract lifetime from: {base}")
        return None, None, None
    lifetime_str = lifetime_match.group(1)

    # Extract quantile
    quant_match = re.search(r'quant([\d]+(?:\.[\d]+)?)', base)
    if quant_match:
        quant_val = quant_match.group(1)
        quant_key = QUANT_MAP.get(quant_val)
        if quant_key is None:
            print(f"  [WARN] Unknown quantile {quant_val} in {base} — skipping")
            return None, None, None
    else:
        quant_key = "obs"

    return mass_str, lifetime_str, quant_key

# ─────────────────────────────────────────────
# Read limit value from ROOT file
# ─────────────────────────────────────────────

def read_limit(fname):
    """
    Opens a HybridNew ROOT file and returns the limit value
    from the first entry of the 'limit' tree.
    """
    f = ROOT.TFile.Open(fname)
    if not f or f.IsZombie():
        print(f"  [ERROR] Could not open {fname}")
        return None

    tree = f.Get("limit")
    if not tree:
        print(f"  [ERROR] No 'limit' tree in {fname}")
        f.Close()
        return None

    if tree.GetEntries() == 0:
        print(f"  [ERROR] Empty 'limit' tree in {fname}")
        f.Close()
        return None

    tree.GetEntry(0)
    val = float(tree.limit)
    f.Close()
    return val

# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Extract HybridNew limits to JSON")
    parser.add_argument("--inputdir", default=".",
                        help="Directory containing HybridNew ROOT files")
    parser.add_argument("--pattern",  default="higgsCombine*.HybridNew*.root",
                        help="Glob pattern for ROOT files")
    parser.add_argument("--outdir",   default=".",
                        help="Output directory for JSON files (default: same as inputdir)")
    parser.add_argument("--blinded",  action="store_true",
                        help="Skip observed limits (files without quant suffix)")
    args = parser.parse_args()

    # Collect files
    search = os.path.join(args.inputdir, args.pattern)
    files  = sorted(glob.glob(search))

    if not files:
        print(f"[ERROR] No files found matching: {search}")
        return

    print(f"Found {len(files)} file(s) matching '{args.pattern}'")

    # Group limits by (lifetime) → { mass_str: { quant_key: value } }
    # Each unique lifetime gets its own JSON file
    limits = defaultdict(lambda: defaultdict(dict))

    for fpath in files:
        mass_str, lifetime_str, quant_key = parse_filename(fpath)

        if mass_str is None:
            print(f"  [SKIP] Could not parse: {os.path.basename(fpath)}")
            continue

        if quant_key == "obs" and args.blinded:
            print(f"  [SKIP] Observed limit skipped (--blinded): {os.path.basename(fpath)}")
            continue

        val = read_limit(fpath)
        if val is None:
            continue

        limits[lifetime_str][mass_str][quant_key] = val
        print(f"  {os.path.basename(fpath)} → mass={mass_str}, lifetime={lifetime_str}, {quant_key} = {val}")

    if not limits:
        print("[ERROR] No limits extracted — check your file names and pattern")
        return

    # Write one JSON per lifetime
    os.makedirs(args.outdir, exist_ok=True)

    for fpath in files:
        
        mass_str, lifetime_str, quant_key = parse_filename(fpath)
        mass_sorted = {mass_str: limits[lifetime_str][mass_str]}

        # Extract mass from the first (and usually only) entry for the filename
        mass_val = list(mass_sorted.keys())[0]
        mass_int = int(float(mass_val))

        outfile = os.path.join(args.outdir, f"limits_Stau_{mass_int}_{lifetime_str}.json")

        with open(outfile, "w") as jf:
            json.dump(mass_sorted, jf, indent=2)

        print(f"\nWrote {len(mass_sorted)} mass point(s) to: {outfile}")
        print(json.dumps(mass_sorted, indent=2))

if __name__ == "__main__":
    main()

