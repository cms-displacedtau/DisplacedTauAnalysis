"""
compute_xsec_uncertainties.py
==============================
Reads per-subsample relative xsec uncertainties from a JSON file,
computes the yield-weighted group uncertainty for each background group,
and writes a new JSON file ready to be passed to make_datacard.py.

Input JSON format (per-subsample uncertainties):
  {
      "TTto2L2Nu"              : 0.05,
      "TTtoLNu2Q"              : 0.08,
      "TTto4Q"                 : 0.12,
      "DYJetsToLL_M-50"        : 0.04,
      "DYto2Tau-2Jets_MLL-50_0J": 0.04,
      ...
  }

Output JSON format (per-group uncertainties, ready for make_datacard.py):
  {
      "TT"     : 0.073,
      "DY"     : 0.041,
      "QCD"    : 0.143,
      ...
  }

Usage:
  python compute_xsec_uncertainties.py
  python compute_xsec_uncertainties.py
      --subsample-unc-file ./plots_config/xsec_uncertainties.json
      --xsec-file          ./plots_config/xsec_and_br.json
      --sumw-file          ./plots_config/Summer22_CHS_v10/v8/all_total_sumw.json
      --outfile            ./plots_config/group_xsec_uncertainties.json
"""

import os
import json
import argparse

# ---------------------------------------------------------------------------
# Background grouping — mirrors ALL_SAMPLES_DICT in make_datacard.py
# ---------------------------------------------------------------------------
ALL_SAMPLES_DICT = {
    "DY" : [
        "DYJetsToLL_M-50",
        "DYto2Tau-2Jets_MLL-50_0J",
        "DYto2Tau-2Jets_MLL-50_1J",
        "DYto2Tau-2Jets_MLL-50_2J",
        "VBFto2L_MLL-50", 
    ],
    "QCD" : [
        "QCD_PT-50to80",
        "QCD_PT-80to120",
        "QCD_PT-120to170",
        "QCD_PT-170to300",
        "QCD_PT-300to470",
        "QCD_PT-470to600",
        "QCD_PT-600to800",
        "QCD_PT-800to1000",
        "QCD_PT-1000to1400",
        "QCD_PT-1400to1800",
        "QCD_PT-1800to2400",
        "QCD_PT-3200",
     ],
    "WJets" : [
        "Wto2Q-2Jets_PTQQ-100to200_1J",
        "Wto2Q-2Jets_PTQQ-100to200_2J",
        "Wto2Q-2Jets_PTQQ-200to400_1J",
        "Wto2Q-2Jets_PTQQ-200to400_2J",
        "Wto2Q-2Jets_PTQQ-400to600_1J",
        "Wto2Q-2Jets_PTQQ-400to600_2J",
        "Wto2Q-2Jets_PTQQ-600_1J",
        "Wto2Q-2Jets_PTQQ-600_2J",
        "WtoLNu-4Jets",
      ],
    "Top" : [
      "TTto2L2Nu", 
      "TTtoLNu2Q", 
      "TTto4Q"
      "TbarWplustoLNu2Q",
      "TbarWplusto2L2Nu",
      "TWminusto2L2Nu",
      "TBbarQ_t-channel_4FS",
      "TWminustoLNu2Q",
      "TbarBQ_t-channel_4FS",
      ],  
    "EWK" : [
      "WW", 
      "WZ", 
      "ZZ", 
      ],
}

LUMI = 26.7  # fb^-1
#LUMI = 17.78

# ---------------------------------------------------------------------------
# Core calculation
# ---------------------------------------------------------------------------

def compute_subsample_yield(subsample, xsec, br, sum_gen_w, lumi):
    """
    Compute the lumi-weighted yield for a single subsample.
    yield = lumi * xsec * br * 1000 / sum_gen_w
    Returns None if any ingredient is missing.
    """
    if subsample not in xsec:
        return None
    if subsample not in br:
        return None
    if subsample not in sum_gen_w:
        return None
    return lumi * xsec[subsample] * br[subsample] * 1000.0 / sum_gen_w[subsample]


def compute_group_uncertainty(group, subsamples, xsec, br, sum_gen_w,
                               subsample_uncs, lumi):
    """
    Compute the yield-weighted relative xsec uncertainty for one group.

    Steps:
      1. Compute the lumi-weighted yield for each subsample
      2. Compute the total yield across all subsamples
      3. Weight each subsample's uncertainty by its fraction of the total yield
      4. Sum the weighted uncertainties to get the group uncertainty

    Returns a dict with full breakdown for transparency.
    """
    yields   = {}
    skipped  = []

    for sub in subsamples:
        y = compute_subsample_yield(sub, xsec, br, sum_gen_w, lumi)
        if y is None:
            print(f"    [WARNING] {sub}: missing from xsec/br/sumw — skipping")
            skipped.append(sub)
            continue
        if sub not in subsample_uncs:
            print(f"    [WARNING] {sub}: no uncertainty in input JSON — skipping")
            skipped.append(sub)
            continue
        yields[sub] = y

    if not yields:
        print(f"    [ERROR] No valid subsamples for group {group}")
        return None

    total_yield = sum(yields.values())

    breakdown = []
    group_unc = 0.0
    for sub, y in yields.items():
        weight    = y / total_yield
        sub_unc   = subsample_uncs[sub]
        contrib   = weight * sub_unc
        group_unc += contrib
        breakdown.append({
            "subsample" : sub,
            "yield"     : round(y, 4),
            "weight"    : round(weight, 4),
            "unc"       : sub_unc,
            "contrib"   : round(contrib, 6),
        })

    return {
        "group_unc"   : round(group_unc, 6),
        "lnN_value"   : round(1.0 + group_unc, 6),
        "total_yield" : round(total_yield, 4),
        "skipped"     : skipped,
        "breakdown"   : breakdown,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_group_report(group, result):
    """Print a human-readable breakdown of the group uncertainty calculation."""
    print(f"\n  Group: {group}")
    print(f"  {'Subsample':<40} {'Yield':>10} {'Weight':>8} {'Unc':>8} {'Contrib':>10}")
    print(f"  {'-'*40} {'-'*10} {'-'*8} {'-'*8} {'-'*10}")
    for row in result["breakdown"]:
        print(
            f"  {row['subsample']:<40} "
            f"{row['yield']:>10.3f} "
            f"{row['weight']:>8.4f} "
            f"{row['unc']:>7.1%} "
            f"{row['contrib']:>10.4%}"
        )
    if result["skipped"]:
        print(f"  Skipped: {result['skipped']}")
    print(f"  {'-'*78}")
    print(f"  Total yield    : {result['total_yield']:.3f}")
    print(f"  Group unc      : {result['group_unc']:.4%}")
    print(f"  lnN value      : {result['lnN_value']:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Compute yield-weighted xsec uncertainties per background group"
    )
    p.add_argument("--subsample-unc-file",
                   default="./plots_config/xsec_unc.json",
                   help="JSON with per-subsample relative xsec uncertainties")
    p.add_argument("--xsec-file",
                   default="./plots_config/xsec_and_br.json",
                   help="JSON with xsec and br per subsample")
    p.add_argument("--sumw-file",
                   default="./plots_config/Summer22_CHS_v19/mutau/v8/all_total_sumw.json",
                   help="JSON with sum of generator weights per subsample")
    p.add_argument("--outfile",
                   default="./plots_config/group_xsec_uncertainties.json",
                   help="Output JSON with per-group uncertainties")
    p.add_argument("--lumi", type=float, default=LUMI)
    return p.parse_args()


def main():
    args = parse_args()

    # Load inputs
    print(f"Loading subsample uncertainties : {args.subsample_unc_file}")
    with open(args.subsample_unc_file, "r") as f:
        subsample_uncs = json.load(f)
    subsample_uncs = subsample_uncs["xsec_unc"]

    print(f"Loading xsec/br                 : {args.xsec_file}")
    with open(args.xsec_file, "r") as f:
        xsec_and_br = json.load(f)
    xsec = xsec_and_br["xsec"]
    br   = xsec_and_br["br"]

    print(f"Loading sum gen weights         : {args.sumw_file}")
    with open(args.sumw_file, "r") as f:
        sum_gen_w = json.load(f)

    print(f"\nLuminosity: {args.lumi} fb^-1")
    print("=" * 80)
    print("Computing yield-weighted xsec uncertainties per background group ...")
    print("=" * 80)

    group_results = {}
    output        = {}   # simple group -> unc dict for make_datacard.py

    for group, subsamples in ALL_SAMPLES_DICT.items():
        result = compute_group_uncertainty(
            group, subsamples, xsec, br, sum_gen_w,
            subsample_uncs, args.lumi
        )
        if result is None:
            continue
        group_results[group] = result
        output[group]        = result["group_unc"]
        print_group_report(group, result)

    # Write output JSON
    os.makedirs(os.path.dirname(args.outfile), exist_ok=True)
    with open(args.outfile, "w") as f:
        json.dump(output, f, indent=4)

    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"  {'Group':<12} {'Group unc':>12} {'lnN value':>12}")
    print(f"  {'-'*12} {'-'*12} {'-'*12}")
    for group, unc in output.items():
        print(f"  {group:<12} {unc:>11.4%} {1+unc:>12.4f}")

    print(f"\nOutput written to: {args.outfile}")
    print(f"\nPass to make_datacard.py with:")
    print(f"  --xsec-unc-file {args.outfile}")


if __name__ == "__main__":
    main()
