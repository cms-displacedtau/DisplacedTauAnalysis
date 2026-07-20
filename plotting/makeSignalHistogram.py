"""
makePlotsSignal.py
==================
Produces jet pt histograms for Stau signal samples (varying mass and ctau)
AND grouped background MC samples from pre-selected NanoAOD ROOT files.

Now includes JES/JER/PU systematic variation histograms.

Output: histograms.root  ready to point a Combine datacard at.

Combine naming convention in the ROOT file:
  Nominal:    Jet_pt__Stau_200_10mm
  JES up:     Jet_pt_jesUp__Stau_200_10mm
  JES down:   Jet_pt_jesDown__Stau_200_10mm
  JER up:     Jet_pt_jerUp__Stau_200_10mm
  JER down:   Jet_pt_jerDown__Stau_200_10mm
  PU up:      Jet_pt_puUp__Stau_200_10mm
  PU down:    Jet_pt_puDown__Stau_200_10mm

  Same pattern for backgrounds:
  Jet_pt__DY,  Jet_pt_jesUp__DY,  Jet_pt_puDown__TT,  etc.

The make_datacard.py script reads these to compute κ values automatically.

Directory structure expected:
  {basedir}/faster_trial_Stau_{mass}_{ctau}mm/*.root   <- signal
  {bkg_basedir}/faster_trial_{subsample}/*.root        <- background subsamples

Usage:
  python makePlotsSignal.py
  python makePlotsSignal.py --masses 100 200 --ctaus 1 10
  python makePlotsSignal.py --outroot histograms.root
"""

import os
import glob
import json
import argparse
import numpy as np
import awkward as ak
import uproot
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
BASE_DIR = (
    "/eos/uscms/store/group/lpcdisptau/dally/displacedTaus/selected/"
    "Summer22_CHS_v19/mutau/v8/SR_score_wUncertainties/faster_trial/"
)

# Background base directory — subfolders named after each subsample
BKG_BASE_DIR = (
    "/eos/uscms/store/group/lpcdisptau/dally/displacedTaus/selected/"
    "Summer22_CHS_v19/mutau/v8/SR_score_wUncertainties/faster_trial/"
)

MASSES      = [100, 200, 300, 400, 500]
CTAUS       = [1, 5, 10, 50, 100, 500, 1000]
TARGET_LUMI = 324.75   # fb^-1

# ---------------------------------------------------------------------------
# Background sample grouping
# ---------------------------------------------------------------------------
ALL_SAMPLES_DICT = {
    "DY": [
        "DYJetsToLL_M-50",
        "DYto2Tau-2Jets_MLL-50_0J",
        "DYto2Tau-2Jets_MLL-50_1J",
        "DYto2Tau-2Jets_MLL-50_2J",
      "VBFto2L_MLL-50", 
    ],
    "QCD": [
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
    "WJets": [
        "WtoLNu-4Jets",
        "Wto2Q-2Jets_PTQQ-100to200_1J",
        "Wto2Q-2Jets_PTQQ-100to200_2J",
        "Wto2Q-2Jets_PTQQ-200to400_1J",
        "Wto2Q-2Jets_PTQQ-200to400_2J",
        "Wto2Q-2Jets_PTQQ-400to600_1J",
        "Wto2Q-2Jets_PTQQ-400to600_2J",
        "Wto2Q-2Jets_PTQQ-600_1J",
        "Wto2Q-2Jets_PTQQ-600_2J",
    ],
    "Top": [
        "TTto2L2Nu",
        "TTtoLNu2Q",
        "TTto4Q",
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

# Build reverse lookup: subsample -> group key
REVERSE_LOOKUP = {
    subsample: group
    for group, subsamples in ALL_SAMPLES_DICT.items()
    for subsample in subsamples
}

# ---------------------------------------------------------------------------
# Histogram settings
# ---------------------------------------------------------------------------
PLOT_SETTINGS = {
    "Jet_pt": {
        "field"            : "CorrectedJet",
        "variable"         : "pt",
        "binning_linspace" : (0, 500, 101),   # 100 bins, 5 GeV each
        "per_event"        : False,
    },
}

# ---------------------------------------------------------------------------
# Systematic variation definitions
# ---------------------------------------------------------------------------
# Each variation specifies:
#   - jet_branch: which branch to use for jet pt (None = use nominal)
#   - weight_key: which weight to use (None = use nominal weight)
#
# For JES/JER: vary the jet branch, keep nominal weights
# For PU:      keep nominal jets, vary the weight
#
# The ROOT histogram key will be:  {plot_name}_{variation_suffix}__{process}
# e.g.  Jet_pt_jesUp__DY
#
# Note: weight_up/weight_down in the ROOT files are SWAPPED:
#   weight_down = puWeightUp   (higher minBias xsec → lower PU weights)
#   weight_up   = puWeightDown (lower minBias xsec  → higher PU weights)
# We correct for this here so that puUp in the datacard means "yield goes up"
# ---------------------------------------------------------------------------
VARIATIONS = {
    "nominal": {
        "jet_branch": None,                    # use default from PLOT_SETTINGS
        "weight_key": None,                    # use nominal weights
        "suffix":     None,                    # no suffix for nominal
    },
    "jesUp": {
        "jet_branch": "CorrectedJet_pt_jesUp",
        "weight_key": None,
        "suffix":     "jesUp",
    },
    "jesDown": {
        "jet_branch": "CorrectedJet_pt_jesDown",
        "weight_key": None,
        "suffix":     "jesDown",
    },
    "jerUp": {
        "jet_branch": "CorrectedJet_pt_jerUp",
        "weight_key": None,
        "suffix":     "jerUp",
    },
    "jerDown": {
        "jet_branch": "CorrectedJet_pt_jerDown",
        "weight_key": None,
        "suffix":     "jerDown",
    },
    "puUp": {
        "jet_branch": None,
        "weight_key": "puUp",                  # uses corrected PU up weight
        "suffix":     "puUp",
    },
    "puDown": {
        "jet_branch": None,
        "weight_key": "puDown",                # uses corrected PU down weight
        "suffix":     "puDown",
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def signal_name(mass, ctau):
    return f"Stau_{mass}_{ctau}mm"


def find_files(directory):
    files = sorted(glob.glob(os.path.join(directory, "*.root")))
    if not files:
        raise FileNotFoundError(f"No ROOT files found in {directory}")
    return files


def _fold_underflow_overflow(values, weights, binning):
    """
    Histogram *values* with *weights* into *binning*, folding underflow
    into the first bin and overflow into the last bin.

    Returns (counts, sumw2, edges) where counts and sumw2 have length
    len(binning)-1 and already include the under/overflow contributions.
    """
    values  = np.asarray(ak.to_numpy(values),  dtype=float)
    weights = np.asarray(ak.to_numpy(weights), dtype=float)

    lo, hi = binning[0], binning[-1]

    underflow_mask = values < lo
    overflow_mask  = values >= hi
    inrange_mask   = ~(underflow_mask | overflow_mask)

    counts, edges = np.histogram(values[inrange_mask],
                                 weights=weights[inrange_mask],
                                 bins=binning)
    sumw2, _      = np.histogram(values[inrange_mask],
                                 weights=weights[inrange_mask] ** 2,
                                 bins=binning)

    if np.any(underflow_mask):
        counts[0] += np.sum(weights[underflow_mask])
        sumw2[0]  += np.sum(weights[underflow_mask] ** 2)

    if np.any(overflow_mask):
        counts[-1] += np.sum(weights[overflow_mask])
        sumw2[-1]  += np.sum(weights[overflow_mask] ** 2)

    return counts.astype(float), sumw2.astype(float), edges


def load_varied_branches(root_files):
    """
    Load jet pt variation branches and PU weight branches using uproot,
    bypassing coffea's NanoEvents name mangling.

    Returns a dict of {branch_name: awkward_array} for each variation branch.
    """
    branch_names = [
        "CorrectedJet_pt_jesUp",
        "CorrectedJet_pt_jesDown",
        "CorrectedJet_pt_jerUp",
        "CorrectedJet_pt_jerDown",
        "weight_up",
        "weight_down",
    ]

    ## Accumulate arrays across all files
    all_arrays = {name: [] for name in branch_names}
    found_branches = set()

    for fpath in root_files:
        tree = uproot.open(f"{fpath}:Events")
        available = tree.keys()

        for bname in branch_names:
            if bname in available:
                all_arrays[bname].append(tree[bname].array())
                found_branches.add(bname)

    ## Concatenate across files
    result = {}
    for bname in branch_names:
        if bname in found_branches and all_arrays[bname]:
            result[bname] = ak.concatenate(all_arrays[bname])

    return result


def fill_histogram_variation(jet_pt, weights, binning):
    """
    Fill a histogram from flat jet pt values and broadcast weights.
    Returns (counts, sumw2, edges).
    """
    flat_pt = ak.flatten(jet_pt)
    w_bc    = ak.broadcast_arrays(jet_pt, weights)[1]
    flat_w  = ak.flatten(w_bc)

    counts, sumw2, edges = _fold_underflow_overflow(flat_pt, flat_w, binning)
    return counts, sumw2, edges


def fill_all_variations(events, varied_branches, nominal_weights, lumi_weight,
                        settings, is_MC=True):
    """
    Fill histograms for nominal + all systematic variations for one sample.

    Parameters
    ----------
    events          : NanoEvents (coffea) — used for nominal jet pt
    varied_branches : dict from load_varied_branches() — jet pt variations + PU weights
    nominal_weights : awkward array of nominal per-event weights
    lumi_weight     : float, the lumi × xsec × BR / sumGenW factor
    settings        : dict from PLOT_SETTINGS for this variable
    is_MC           : bool, skip variations for data

    Returns
    -------
    results : dict {variation_name: (counts, sumw2, edges)}
    """
    binning = np.linspace(*settings["binning_linspace"])
    results = {}

    ## Get nominal jet pt from coffea events
    nominal_jet_pt = getattr(
        getattr(events, settings["field"]),
        settings["variable"]
    )

    ## Build PU-varied weights
    ## IMPORTANT: weight_up and weight_down are SWAPPED in the ROOT files
    ##   weight_down in ROOT = genWeight × puWeightUp   → yields go DOWN → this is "puDown" for datacard
    ##   weight_up   in ROOT = genWeight × puWeightDown → yields go UP   → this is "puUp" for datacard
    pu_weights = {}
    if is_MC and "weight_down" in varied_branches:
        ## weight_down in file → puWeightUp → yield goes down → datacard "puDown"
        ## But we already swapped at read time to correct, so:
        ## "puUp" weight = makes yield go up
        ## "puDown" weight = makes yield go down
        pass

    if is_MC:
        if "weight_down" in varied_branches:
            ## ROOT file's weight_down = genWeight × puWeightUp (higher minBias xsec)
            ## This REDUCES yields → use as puDown for the datacard
            ## ROOT file's weight_up = genWeight × puWeightDown (lower minBias xsec)
            ## This INCREASES yields → use as puUp for the datacard
            pu_weights["puUp"]   = varied_branches["weight_up"]   * lumi_weight
            pu_weights["puDown"] = varied_branches["weight_down"] * lumi_weight

    for var_name, var_config in VARIATIONS.items():
        ## Skip variations for data
        if not is_MC and var_name != "nominal":
            continue

        ## Determine which jet pt to use
        if var_config["jet_branch"] is not None:
            branch_key = var_config["jet_branch"]
            if branch_key not in varied_branches:
                print(f"      [WARNING] Branch {branch_key} not found — skipping {var_name}")
                continue
            jet_pt = varied_branches[branch_key]
        else:
            jet_pt = nominal_jet_pt

        ## Determine which weights to use
        if var_config["weight_key"] is not None:
            weight_key = var_config["weight_key"]
            if weight_key not in pu_weights:
                print(f"      [WARNING] PU weight {weight_key} not available — skipping {var_name}")
                continue
            w = pu_weights[weight_key]
        else:
            w = nominal_weights

        ## Apply jet pt cut using THIS variation's jet pt
        cut_mask = ak.flatten(jet_pt > 32)
        jet_pt = jet_pt[cut_mask]
        w = w[cut_mask]

        ## Fill histogram
        counts, sumw2, edges = fill_histogram_variation(jet_pt, w, binning)
        results[var_name] = (counts, sumw2, edges)

    return results


# ---------------------------------------------------------------------------
# ROOT output
# ---------------------------------------------------------------------------

def save_to_root(histogram_dict, outfile):
    """
    Write one TH1D per (plot_name, process, variation) to a ROOT file.

    Naming convention for make_datacard.py compatibility:
      Nominal:   {plot_name}__{process}              e.g. Jet_pt__DY
      JES up:    {plot_name}_jesUp__{process}        e.g. Jet_pt_jesUp__DY
      JES down:  {plot_name}_jesDown__{process}      e.g. Jet_pt_jesDown__DY
      JER up:    {plot_name}_jerUp__{process}        e.g. Jet_pt_jerUp__DY
      PU up:     {plot_name}_puUp__{process}         e.g. Jet_pt_puUp__DY
      etc.
    """
    import boost_histogram as bh

    #with uproot.recreate(outfile) as f:
    #    for plot_name, processes in histogram_dict.items():
    #        for proc_name, variations in processes.items():
    #            for var_name, (counts, sumw2, edges) in variations.items():
    #                ## Build the ROOT histogram key
    #                var_config = VARIATIONS[var_name]
    #                suffix = var_config["suffix"]

    #                if suffix is None:
    #                    ## Nominal: Jet_pt__DY
    #                    key = f"{plot_name}__{proc_name}"
    #                else:
    #                    ## Variation: Jet_pt_jesUp__DY
    #                    key = f"{plot_name}_{suffix}__{proc_name}"

    #                h = bh.Histogram(
    #                    bh.axis.Variable(edges),
    #                    storage=bh.storage.Weight()
    #                )
    #                h.view().value    = counts
    #                h.view().variance = sumw2
    #                f[key] = h

    #                integral = counts.sum()
    #                print(f"  [ROOT] {key:60s}  integral={integral:.2f}")

    #print(f"\nSaved: {outfile}")
    import ROOT
    ROOT.gROOT.SetBatch(True)

    f = ROOT.TFile(outfile, "RECREATE")

    for plot_name, processes in histogram_dict.items():
        for proc_name, variations in processes.items():
            for var_name, (counts, sumw2, edges) in variations.items():

                var_config = VARIATIONS[var_name]
                suffix = var_config["suffix"]
                key = f"{plot_name}__{proc_name}" if suffix is None \
                      else f"{plot_name}_{suffix}__{proc_name}"

                nbins = len(counts)
                h = ROOT.TH1D(key, key, nbins, edges)
                h.Sumw2()  # <-- explicitly activate Sumw2 storage

                for i, (c, s) in enumerate(zip(counts, sumw2)):
                    h.SetBinContent(i + 1, c)
                    h.SetBinError(i + 1, s ** 0.5)  # error = sqrt(sumw2)

                h.Write()
                print(f"  [ROOT] {key:60s}  integral={counts.sum():.2f}  "
                      f"err={sumw2.sum()**0.5:.2f}")

    f.Close()
    ## Write data_obs (Asimov = sum of background nominals)
    write_data_obs(histogram_dict, outfile)

    print(f"\nmake_datacard.py will look for histograms like:")
    print(f"  Nominal:   {list(PLOT_SETTINGS.keys())[0]}__%s" % "<process>")
    print(f"  JES up:    {list(PLOT_SETTINGS.keys())[0]}_jesUp__%s" % "<process>")
    print(f"  JES down:  {list(PLOT_SETTINGS.keys())[0]}_jesDown__%s" % "<process>")
    print(f"  JER up:    {list(PLOT_SETTINGS.keys())[0]}_jerUp__%s" % "<process>")
    print(f"  JER down:  {list(PLOT_SETTINGS.keys())[0]}_jerDown__%s" % "<process>")
    print(f"  PU up:     {list(PLOT_SETTINGS.keys())[0]}_puUp__%s" % "<process>")
    print(f"  PU down:   {list(PLOT_SETTINGS.keys())[0]}_puDown__%s" % "<process>")


def write_data_obs(histogram_dict, outfile):
    """
    Combine always requires a data_obs histogram even when running blinded.
    Write it as the sum of all background nominal histograms (Asimov dataset).
    """
    import boost_histogram as bh

    #bkg_groups = list(ALL_SAMPLES_DICT.keys())

    #with uproot.update(outfile) as f:
    #    for plot_name, processes in histogram_dict.items():
    #        total_counts = None
    #        total_edges  = None

    #        for proc_name, variations in processes.items():
    #            if proc_name not in bkg_groups:
    #                continue
    #            if "nominal" not in variations:
    #                continue
    #            counts, sumw2, edges = variations["nominal"]
    #            if total_counts is None:
    #                total_counts = counts.copy()
    #                total_edges  = edges
    #            else:
    #                total_counts += counts

    #        if total_counts is not None:
    #            h = bh.Histogram(
    #                bh.axis.Variable(total_edges),
    #                storage=bh.storage.Weight()
    #            )
    #            h.view().value    = total_counts
    #            h.view().variance = total_counts   # Poisson: variance = counts
    #            key = f"{plot_name}__data_obs"
    #            f[key] = h
    #            print(f"  [ROOT] {key:60s}  (Asimov sum of backgrounds)")
    
    import ROOT
    f = ROOT.TFile(outfile, "UPDATE")

    bkg_groups = list(ALL_SAMPLES_DICT.keys())

    for plot_name, processes in histogram_dict.items():
        total_counts = None
        total_edges  = None

        for proc_name, variations in processes.items():
            if proc_name not in bkg_groups:
                continue
            if "nominal" not in variations:
                continue
            counts, sumw2, edges = variations["nominal"]
            if total_counts is None:
                total_counts = counts.copy()
                total_edges  = edges
            else:
                total_counts += counts

        if total_counts is not None:
            key = f"{plot_name}__data_obs"
            nbins = len(total_counts)
            h = ROOT.TH1D(key, key, nbins, total_edges)
            h.Sumw2()
            for i, c in enumerate(total_counts):
                h.SetBinContent(i + 1, c)
                h.SetBinError(i + 1, c ** 0.5)  # Poisson for data
            h.Write()
            print(f"  [ROOT] {key:60s}  (Asimov sum of backgrounds)")

    f.Close()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Signal + background histograms with JES/JER/PU variations for Combine"
    )
    p.add_argument("--basedir",     default=BASE_DIR,
                   help="Base directory for signal samples")
    p.add_argument("--bkg-basedir", default=BKG_BASE_DIR,
                   help="Base directory for background subsamples")
    p.add_argument("--masses",      nargs="+", type=int, default=MASSES)
    p.add_argument("--ctaus",       nargs="+", type=int, default=CTAUS)
    p.add_argument("--outroot",     default="histograms.root")
    p.add_argument("--lumi",        type=float, default=TARGET_LUMI)
    p.add_argument("--xsec-file",   default="./plots_config/xsec_and_br.json")
    p.add_argument("--sumw-file",   default="./plots_config/Summer22_CHS_v19/mutau/v8/all_total_sumw.json")
    p.add_argument("--schema",      default="PFNanoAODSchema",
                   help="NanoAODSchema or PFNanoAODSchema")
    p.add_argument("--skip-bkg",    action="store_true",
                   help="Skip background processing (signal only)")
    p.add_argument("--skip-syst",   action="store_true",
                   help="Skip systematic variations (nominal only)")
    return p.parse_args()


def main():
    args = parse_args()

    # Schema
    if args.schema == "PFNanoAODSchema":
        from coffea.nanoevents import PFNanoAODSchema
        PFNanoAODSchema.warn_missing_crossrefs = False
        PFNanoAODSchema.mixins["DisMuon"] = "Muon"
        PFNanoAODSchema.mixins["CorrectedJet"] = "Jet"
        schema = PFNanoAODSchema
    else:
        schema = NanoAODSchema

    # Load xsec/br and sumw JSON files
    with open(args.xsec_file, "r") as f:
        xsec_and_br = json.load(f)
    xsec = xsec_and_br["xsec"]
    br   = xsec_and_br["br"]
    with open(args.sumw_file, "r") as f:
        sum_gen_w = json.load(f)

    # histogram_dict:
    #   { plot_name: { process_name: { "nominal": (c, s, e), "jesUp": (c, s, e), ... } } }
    histogram_dict = {}

    # Determine which variations to run
    if args.skip_syst:
        active_variations = {"nominal": VARIATIONS["nominal"]}
    else:
        active_variations = VARIATIONS

    # -------------------------------------------------------------------
    # Signal samples — one set of histograms per (mass, ctau) grid point
    # -------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Processing signal samples ...")
    print("=" * 60)

    for mass in args.masses:
        for ctau in args.ctaus:
            sname = signal_name(mass, ctau)
            sdir  = os.path.join(args.basedir, "faster_trial_" + sname)
            print(f"\nINFO: Loading {sname} ...")

            try:
                files = find_files(sdir)
            except FileNotFoundError as e:
                print(f"  [WARNING] {e} — skipping.")
                continue

            # Check xsec/sumw available
            if sname not in xsec:
                print(f"  [WARNING] {sname} not in xsec JSON — skipping.")
                continue
            if sname not in sum_gen_w:
                print(f"  [WARNING] {sname} not in sumw JSON — skipping.")
                continue
            
            # Load events with coffea (for nominal jets)
            events = NanoEventsFactory.from_root(
                {f: "Events" for f in files}, schemaclass=schema
            ).events()
            if ctau == 500:
                tree = uproot.open(f"{sdir}/faster_trial_{sname}.root:Events")
                weight500 = tree["weight_ct500"].array()
                events = ak.with_field(events, weight500, "weight500")
            
            score_mask = ak.flatten(events.CorrectedJet.disTauTag_score1 > 0.99)
            events = events[score_mask]

            # Load variation branches with uproot (bypasses coffea name mangling)
            varied_branches = load_varied_branches(files)
            varied_branches = {k: v[score_mask] for k, v in varied_branches.items()}

            # Compute lumi weight
            lumi_weight = (
                args.lumi * xsec[sname] * br[sname] * 1000 / sum_gen_w[sname]
            )
            if ctau == 500:
                lumi_weight = lumi_weight * events["weight500"]
            nominal_weights = events.weight * lumi_weight

            # Fill all variations
            for plot_name, settings in PLOT_SETTINGS.items():
                var_results = fill_all_variations(
                    events, varied_branches, nominal_weights, lumi_weight,
                    settings, is_MC=True
                )

                proc_vars = (
                    histogram_dict
                    .setdefault(plot_name, {})
                    .setdefault(sname, {})
                )
                for var_name, (counts, sumw2, edges) in var_results.items():
                    proc_vars[var_name] = (
                        np.array(counts, dtype=float),
                        np.array(sumw2, dtype=float),
                        np.array(edges),
                    )

                nom_integral = proc_vars.get("nominal", (np.array([0]),))[0].sum()
                print(f"  [{sname}] nominal integral = {nom_integral:.2f}")

    # -------------------------------------------------------------------
    # Background samples — subsamples summed per group, with variations
    # -------------------------------------------------------------------
    if not args.skip_bkg:
        print("\n" + "=" * 60)
        print("Processing background samples ...")
        print("=" * 60)

        for group, subsamples in ALL_SAMPLES_DICT.items():
            print(f"\nINFO: Group [{group}]")

            for subsample in subsamples:
                sdir = os.path.join(args.bkg_basedir, "faster_trial_" + subsample)
                print(f"  Loading subsample: {subsample} ...")

                try:
                    files = find_files(sdir)
                except FileNotFoundError as e:
                    print(f"    [WARNING] {e} — skipping subsample.")
                    continue

                if subsample not in xsec:
                    print(f"    [WARNING] {subsample} not in xsec JSON — skipping.")
                    continue
                if subsample not in sum_gen_w:
                    print(f"    [WARNING] {subsample} not in sumw JSON — skipping.")
                    continue

                # Load events with coffea
                events = NanoEventsFactory.from_root(
                    {f: "Events" for f in files}, schemaclass=schema
                ).events()
                score_mask = ak.flatten(events.CorrectedJet.disTauTag_score1 > 0.99)
                events = events[score_mask]
                # Load variation branches with uproot
                varied_branches = load_varied_branches(files)
                varied_branches = {k: v[score_mask] for k, v in varied_branches.items()}

                # Lumi weight
                lumi_weight = (
                    args.lumi
                    * xsec[subsample]
                    * br[subsample]
                    * 1000
                    / sum_gen_w[subsample]
                )
                nominal_weights = events.weight * lumi_weight

                # Fill all variations for this subsample
                for plot_name, settings in PLOT_SETTINGS.items():
                    var_results = fill_all_variations(
                        events, varied_branches, nominal_weights, lumi_weight,
                        settings, is_MC=True
                    )

                    # Accumulate into group histograms
                    group_vars = (
                        histogram_dict
                        .setdefault(plot_name, {})
                        .setdefault(group, {})
                    )

                    for var_name, (counts, sumw2, edges) in var_results.items():
                        counts = np.array(counts, dtype=float)
                        sumw2  = np.array(sumw2, dtype=float)
                        edges  = np.array(edges)

                        if var_name not in group_vars:
                            group_vars[var_name] = (
                                counts.copy(), sumw2.copy(), edges.copy()
                            )
                        else:
                            existing_counts, existing_sumw2, existing_edges = group_vars[var_name]
                            group_vars[var_name] = (
                                existing_counts + counts,
                                existing_sumw2  + sumw2,
                                existing_edges,
                            )

                    nom_integral = group_vars.get("nominal", (np.array([0]),))[0].sum()
                    print(f"    [{group}] running nominal integral = {nom_integral:.2f}")

    # -------------------------------------------------------------------
    # Print κ summary before saving
    # -------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("κ SUMMARY (yield_varied / yield_nominal)")
    print("=" * 60)

    for plot_name, processes in histogram_dict.items():
        for proc_name, variations in processes.items():
            if "nominal" not in variations:
                continue
            nom_yield = variations["nominal"][0].sum()
            if nom_yield == 0:
                continue

            kappas = []
            for var_name in ["jesUp", "jesDown", "jerUp", "jerDown", "puUp", "puDown"]:
                if var_name in variations:
                    var_yield = variations[var_name][0].sum()
                    kappa = var_yield / nom_yield
                    kappas.append(f"{var_name}={kappa:.4f}")

            if kappas:
                print(f"  {proc_name:30s}  nominal={nom_yield:10.2f}  {', '.join(kappas)}")
            nom_counts, nom_sumw2, _ = variations["nominal"]
            nom_yield = nom_counts.sum()
            nom_unc   = np.sqrt(nom_sumw2.sum())
            # then in the print:
            print(f"  {proc_name:30s}  nominal={nom_yield:10.2f} ± {nom_unc:.2f}  {', '.join(kappas)}")

    # -------------------------------------------------------------------
    # Save to ROOT
    # -------------------------------------------------------------------
    if not histogram_dict:
        print("ERROR: No histograms filled — check your paths and sample lists.")
        return

    print("\n" + "=" * 60)
    print("Writing ROOT file ...")
    print("=" * 60)
    save_to_root(histogram_dict, args.outroot)

    print("\nDone.")


if __name__ == "__main__":
    main()

