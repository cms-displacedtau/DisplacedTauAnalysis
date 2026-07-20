"""
Datacard Generator for Control Regions
=======================================

Reads kappa_values.json (produced by getUncertainties.py) and generates
Combine-ready counting experiment datacards for each control region.

Usage:
    python make_datacards.py

    Optionally specify a different JSON file or output directory:
    python make_datacards.py --input kappa_values.json --outdir datacards/
"""

import json
import os
import argparse


## ============================================================
##  CONFIGURATION
## ============================================================

## Luminosity uncertainty (fractional), applied to all MC processes
LUMI_UNCERTAINTY = 1.014  # 2.5%

## Which channels to generate datacards for
CHANNELS = ["QCD_CR", "W_CR", "TT_CR"]

## Background processes in the order they should appear in the datacard
## "Data" is used for the observation line and excluded from the process columns
## Adjust this list to match what's in your JSON
BG_ORDER = ["QCD", "DY", "Top", "WJets", "EWK", "VBF"]

## Systematic names to look for in the JSON (must match keys in datacard_lnN)
SYSTEMATICS = ["CMS_scale_j", "CMS_res_j", "CMS_pileup"]

## Cross-section uncertainties for processes NOT constrained by a rateParam
## Set to None or omit processes that ARE constrained by a CR
XSEC_UNCERTAINTIES = {
    ## "QCD":   1.50,   # uncomment if QCD is not constrained by a rateParam
    "DY":    1.003463,
    ## "Top":   1.05,
    ## "WJets": 1.15,
    "EWK":   1.000661,
    "VBF":   1.004869,
}

## rateParam definitions
## These tie the normalisation of a process across channels.
## Format: (param_name, channel_pattern, process_name, initial_value, range)
## channel_pattern = "*" means all channels; you can also specify a single channel.
## Processes with a rateParam should NOT have a cross-section lnN.
RATE_PARAMS = [
     ("rate_QCD",   "*", "QCD",    "1.0", "[0.2, 5.0]"),
    ## ("rate_DY",    "*", "DY",     "1.0", "[0.2, 5.0]"),
     ("rate_Top",   "*", "Top",    "1.0", "[0.2, 5.0]"),
     ("rate_WJets", "*", "WJets",  "1.0", "[0.2, 5.0]"),
]


## ============================================================
##  DATACARD GENERATION
## ============================================================

def make_datacard(channel_name, channel_data, outdir="."):
    """
    Generate a Combine counting experiment datacard for one channel.

    Parameters
    ----------
    channel_name : str
        e.g., "QCD_CR", "W_CR", "TT_CR"
    channel_data : dict
        The JSON entry for this channel, containing:
          - "event_counts": {"jet_pt": {"QCD": ..., "Data": ..., ...}, ...}
          - "datacard_lnN": {"QCD": {"CMS_scale_j": "0.98/1.02", ...}, ...}
          - "backgrounds":  {"QCD": {"nominal_yield": ..., ...}, ...}
    outdir : str
        Output directory for the datacard file.
    """

    ## ── Extract observation and nominal yields ──────────────
    event_counts = channel_data["event_counts"]
    datacard_lnN = channel_data["datacard_lnN"]
    backgrounds  = channel_data["backgrounds"]

    ## Nominal yields come from the "jet_pt" key in event_counts
    nominal_counts = event_counts.get("jet_pt", {})
    observation = channel_data.get("observation", 0)

    ## Filter BG_ORDER to only include processes present in this channel
    ## with non-zero yields
    bg_present = []
    for bg in BG_ORDER:
        yield_val = nominal_counts.get(bg, 0)
        if yield_val > 0:
            bg_present.append(bg)

    if not bg_present:
        print(f"  WARNING: No backgrounds with non-zero yields for {channel_name}, skipping.")
        return None

    n_bg = len(bg_present)

    ## ── Column width for alignment ──────────────────────────
    col_w = 18

    ## ── Build the datacard string ───────────────────────────
    lines = []
    lines.append(f"## Datacard for {channel_name}")
    lines.append(f"## Auto-generated from kappa_values.json")
    lines.append(f"## Luminosity: {channel_data.get('luminosity_pb', 0):.0f} pb⁻¹")
    lines.append("")
    lines.append(f"imax 1          # one channel: {channel_name}")
    lines.append(f"jmax *          # {n_bg} background processes")
    lines.append(f"kmax *          # auto-count systematics")
    lines.append("")
    lines.append("-" * 60)
    lines.append("")

    ## ── Observation ─────────────────────────────────────────
    lines.append(f"bin           {channel_name}")
    lines.append(f"observation   {observation:.0f}")
    lines.append("")
    lines.append("-" * 60)
    lines.append("")

    ## ── Process table ───────────────────────────────────────
    ## bin row
    line = f"{'bin':{col_w}s}"
    for bg in bg_present:
        line += f"{channel_name:>{col_w}s}"
    lines.append(line)

    ## process name row
    line = f"{'process':{col_w}s}"
    for bg in bg_present:
        line += f"{bg:>{col_w}s}"
    lines.append(line)

    ## process number row (all >= 1, backgrounds only)
    line = f"{'process':{col_w}s}"
    for i, bg in enumerate(bg_present):
        line += f"{i + 1:>{col_w}d}"
    lines.append(line)

    ## rate row
    line = f"{'rate':{col_w}s}"
    for bg in bg_present:
        rate = nominal_counts.get(bg, 0)
        line += f"{rate:>{col_w}.4f}"
    lines.append(line)

    lines.append("")
    lines.append("-" * 60)
    lines.append("")

    ## ── Luminosity ──────────────────────────────────────────
    line = f"{'lumi_13p6TeV':{col_w}s}{'lnN':>{col_w}s}"
    for bg in bg_present:
        line += f"{LUMI_UNCERTAINTY:>{col_w}.3f}"
    lines.append(line)

    ## ── Cross-section uncertainties ─────────────────────────
    for bg in bg_present:
        if bg in XSEC_UNCERTAINTIES:
            syst_name = f"xsec_{bg}"
            line = f"{syst_name:{col_w}s}{'lnN':>{col_w}s}"
            for bg2 in bg_present:
                if bg2 == bg:
                    line += f"{XSEC_UNCERTAINTIES[bg]:>{col_w}.3f}"
                else:
                    line += f"{'-':>{col_w}s}"
            lines.append(line)

    ## ── JES / JER / PU systematics from JSON ────────────────
    for syst_name in SYSTEMATICS:
        ## Check if any process has this systematic
        any_has_syst = any(syst_name in datacard_lnN.get(bg, {}) for bg in bg_present)
        if not any_has_syst:
            continue

        line = f"{syst_name:{col_w}s}{'lnN':>{col_w}s}"
        for bg in bg_present:
            lnN_val = datacard_lnN.get(bg, {}).get(syst_name, None)
            if lnN_val is not None:
                line += f"{lnN_val:>{col_w}s}"
            else:
                line += f"{'-':>{col_w}s}"
        lines.append(line)

    ## ── rateParam lines ─────────────────────────────────────
    has_rateparams = False
    for param_name, ch_pattern, proc, init, rng in RATE_PARAMS:
        if ch_pattern == "*" or ch_pattern == channel_name:
            if proc in bg_present:
                if not has_rateparams:
                    lines.append("")
                    has_rateparams = True
                lines.append(f"{param_name}  rateParam  {channel_name}  {proc}  {init}")

    lines.append("")

    ## ── Write to file ───────────────────────────────────────
    os.makedirs(outdir, exist_ok=True)
    outfile = os.path.join(outdir, f"datacard_{channel_name}.txt")
    with open(outfile, "w") as f:
        f.write("\n".join(lines))

    print(f"  ✓ Written: {outfile}")
    print(f"    Observation: {observation:.0f}")
    print(f"    Backgrounds: {bg_present}")
    print(f"    Rates: {[round(nominal_counts.get(bg, 0), 1) for bg in bg_present]}")

    return outfile


## ============================================================
##  COMBINE CARDS SCRIPT GENERATION
## ============================================================

def write_combine_script(datacard_files, outdir="."):
    """
    Write a helper script that combines all CR datacards,
    adds rateParams, and builds the workspace.
    """

    script_lines = []
    script_lines.append("#!/bin/bash")
    script_lines.append("## Auto-generated script to combine CR datacards")
    script_lines.append("")

    ## combineCards.py command
    script_lines.append("## Step 1: Combine datacards")
    combine_cmd = "combineCards.py \\\n"
    for f in datacard_files:
        channel = os.path.basename(f).replace("datacard_", "").replace(".txt", "")
        combine_cmd += f"    {channel}={f} \\\n"
    combine_cmd += "    > combined_CRs.txt"
    script_lines.append(combine_cmd)
    script_lines.append("")

    ## rateParam additions
    if RATE_PARAMS:
        script_lines.append("## Step 2: Add rateParams (if using wildcard * pattern)")
        for param_name, ch_pattern, proc, init, rng in RATE_PARAMS:
            if ch_pattern == "*":
                script_lines.append(
                    f'echo "{param_name}  rateParam  *  {proc}  {init}  {rng}" >> combined_CRs.txt'
                )
        script_lines.append("")

    ## text2workspace.py
    script_lines.append("## Step 3: Build workspace with channel masks")
    script_lines.append("text2workspace.py combined_CRs.txt \\")
    script_lines.append("    -o workspace_CRs.root \\")
    script_lines.append("    --channel-masks")
    script_lines.append("")
    script_lines.append("echo '==> Done. Workspace: workspace_CRs.root'")

    outfile = os.path.join(outdir, "combine_datacards.sh")
    with open(outfile, "w") as f:
        f.write("\n".join(script_lines))
    os.chmod(outfile, 0o755)
    print(f"\n  ✓ Combine script written: {outfile}")

    return outfile


## ============================================================
##  MAIN
## ============================================================

def main():
    parser = argparse.ArgumentParser(description="Generate Combine datacards from kappa JSON")
    parser.add_argument("--input", default="kappa_values.json",
                        help="Input JSON file with kappa values")
    parser.add_argument("--outdir", default="datacards",
                        help="Output directory for datacard files")
    args = parser.parse_args()

    ## Load the JSON
    print(f"\n→ Loading {args.input}")
    with open(args.input, "r") as f:
        all_data = json.load(f)

    print(f"  Channels found: {list(all_data.keys())}")

    ## Generate a datacard for each requested channel
    datacard_files = []
    for ch in CHANNELS:
        if ch not in all_data:
            print(f"\n  WARNING: Channel '{ch}' not found in JSON, skipping.")
            print(f"  Available channels: {list(all_data.keys())}")
            continue

        print(f"\n→ Generating datacard for {ch}")
        outfile = make_datacard(ch, all_data[ch], outdir=args.outdir)
        if outfile:
            datacard_files.append(outfile)

    ## Generate the combine script
    if datacard_files:
        write_combine_script(datacard_files, outdir=args.outdir)

    ## Print summary
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  Datacards generated: {len(datacard_files)}")
    for f in datacard_files:
        print(f"    {f}")
    print(f"\n  Next steps:")
    print(f"    1. Review the datacards")
    print(f"    2. Edit RATE_PARAMS and XSEC_UNCERTAINTIES as needed")
    print(f"    3. Run: bash {args.outdir}/combine_datacards.sh")
    print(f"    4. Then follow the blinded analysis workflow")


if __name__ == "__main__":
    main()

