import os
import json
import hist
from hist import Hist
import numpy as np
import awkward as ak
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
import matplotlib.ticker as ticker
import mplhep as hep
import argparse
import uproot
from coffea.lumi_tools import LumiMask
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema, PFNanoAODSchema, BaseSchema
import pdb, glob


PFNanoAODSchema.warn_missing_crossrefs = False
PFNanoAODSchema.mixins["DisMuon"] = "Muon"
PFNanoAODSchema.mixins["CorrectedJet"] = "Jet"

hep.style.use("CMS")

nanov = 'Summer22_CHS_v19/'
# nanov = ''
sample_folder = "/eos/uscms/store/user/lpcdisptau/dally/displacedTaus/selected/Summer22_CHS_v19/mutau/v8/QCD_CR_score_wUncertainties/faster_trial/"
## if a sample is not ready yet, comment it out
all_samples_dict = {
    "DY" : [
        "DYJetsToLL_M-50",
        "DYto2Tau-2Jets_MLL-50_0J",
        "DYto2Tau-2Jets_MLL-50_1J",
        "DYto2Tau-2Jets_MLL-50_2J",
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
    "Wto2Q" : [
        "Wto2Q-2Jets_PTQQ-100to200_1J",
        "Wto2Q-2Jets_PTQQ-100to200_2J",
        "Wto2Q-2Jets_PTQQ-200to400_1J",
        "Wto2Q-2Jets_PTQQ-200to400_2J",
        "Wto2Q-2Jets_PTQQ-400to600_1J",
        "Wto2Q-2Jets_PTQQ-400to600_2J",
        "Wto2Q-2Jets_PTQQ-600_1J",
        "Wto2Q-2Jets_PTQQ-600_2J",
     ],
    "WtoLNu" : [
        "WtoLNu-4Jets",
      ],
    "TT" : [
      "TTto2L2Nu", 
      "TTtoLNu2Q", 
      "TTto4Q"
      ],
    "singleT": [
      "TbarWplustoLNu2Q",
      "TbarWplusto2L2Nu",
      "TWminusto2L2Nu",
      "TBbarQ_t-channel_4FS",
      "TWminustoLNu2Q",
      "TbarBQ_t-channel_4FS",
      ],  
    "VBF" : [
      "VBFto2L_MLL-50", 
      ],
    "EWK" : [
      "WW", 
      "WZ", 
      "ZZ", 
      ],
    "JetMET": [
      "JetMET_Run2022E",
      "JetMET_Run2022F",
      "JetMET_Run2022G",
      ], 
}

## build reverse lookup dict to be used when retrieving sum gen events etc
reverse_samples_lookup = {
    subsample: key
    for key, subs in all_samples_dict.items()
    for subsample in subs
}

available_processes = [sub for subs in all_samples_dict.values() for sub in subs]

## load xsec and br settings from JSON file
with open("./plots_config/xsec_and_br.json", "r") as file:
    xsec_and_br = json.load(file)
xsec = xsec_and_br['xsec']
br = xsec_and_br['br']

## load sum_gen_w from another JSON file
with open(f"./plots_config/{nanov}/mutau/v8/all_total_sumw.json", "r") as file:
    sum_gen_w = json.load(file)
    
## target lumi, for now fixed value
target_lumi = 26.7

BRANCHES = {
    "jet_pt":         "CorrectedJet_pt",
    "jet_pt_jesUp":   "CorrectedJet_pt_jesUp",
    "jet_pt_jesDown": "CorrectedJet_pt_jesDown",
    "jet_pt_jerUp":   "CorrectedJet_pt_jerUp",
    "jet_pt_jerDown": "CorrectedJet_pt_jerDown",
    "weight_puUp":    "weightUp",
    "weight_puDown":  "weightDown",
}

BACKGROUNDS = ["QCD", "DY", "Top", "WJets", "EWK", "VBF", "Data"]
binning = [0, 500, 101]

# Directory creation for plots (directory_path = "plots/" + args.process)
directory_path = "plots/"
if not os.path.exists(directory_path):
    os.makedirs(directory_path)
    print(f"Directory '{directory_path}' created successfully.")

# Dictionary for histograms and binnings
histogram_dict = {}
binning_dict = {}
event_dict = {}

for process in available_processes:
    if process not in os.listdir(sample_folder + '../'): continue
    if "QCD" not in process: continue
    print(process)
    is_MC  = False if 'jetmet' in process.lower() else True
    tmp_string = f"faster_trial_{process}/faster_trial_{process}.root"
    tmp_file = sample_folder +  tmp_string
    events = NanoEventsFactory.from_root({tmp_file:"Events"}, schemaclass= PFNanoAODSchema).events()
    #############################################################################################
    ### I foolishly named the up and down PU uncertainty variations as weight_up and weight_down
    ### Because of this, coffea doesn't read them correctly, so use uproot to access them and add
    ### them back into events with names weightUp and weightDown
    if is_MC:
        tree = uproot.open(f"{tmp_file}:Events")
        weightUp = tree["weight_down"].array()
        weightDown = tree["weight_up"].array()
        events = ak.with_field(events, weightUp, "weightUp")
        events = ak.with_field(events, weightDown, "weightDown")
    #############################################################################################
    events = events[ak.flatten(abs(events.CorrectedJet.dxy) < 999)]
    events = events[ak.flatten(abs(events.CorrectedJet.dxyErr) < 999)]
    events = events[events.CorrectedPFMET.pt > 105]
    
    if 'W' in sample_folder or 'TT' in sample_folder:
        events = events[ak.flatten(abs(events.DisMuon.mT) > 30)]

    if 'SR' in sample_folder:
        events = events[ak.flatten(events.CorrectedJet.disTauTag_score1 > 0.9975)]
    #if is_MC:
        ### Raw PU weights (divide out genWeight)
        #pu_nom  = events.weight / events.genWeight
        #pu_up   = events.weightUp / events.genWeight
        #pu_down = events.weightDown / events.genWeight
        #
        ### Check distributions
        #print(f"PU nominal — mean: {ak.mean(pu_nom):.4f}, std: {ak.std(pu_nom):.4f}")
        #print(f"PU up      — mean: {ak.mean(pu_up):.4f}, std: {ak.std(pu_up):.4f}")
        #print(f"PU down    — mean: {ak.mean(pu_down):.4f}, std: {ak.std(pu_down):.4f}")
        #
        ### Per-event ratios
        #print(f"\nup/nom   — mean: {ak.mean(pu_up/pu_nom):.4f}")
        #print(f"down/nom — mean: {ak.mean(pu_down/pu_nom):.4f}")
        #
        ### Check a few events
        #print(f"\nFirst 10 events:")
        #print(f"  nominal: {pu_nom[:10]}")
        #print(f"  up:      {pu_up[:10]}")
        #print(f"  down:    {pu_down[:10]}")

    weights = events.run / events.run
    if "jetmet" in process.lower() or "muon" in process.lower():
        weights = weights
    else:
        lumi_weight = target_lumi * xsec[process] * br[process] * 1000 / sum_gen_w[process]
        weights = events.weight * lumi_weight #* events.Muon.eventWeight ## am i missing the sumGenW here?

    for name, branch  in BRANCHES.items():
        if "CorrectedJet" in branch:
            if not is_MC and ('up' in name.lower() or 'down' in name.lower()): continue
            events_corrected = events[ak.flatten(events["CorrectedJet"]['_'.join(branch.split('_')[1:])] > 32)]
            weights_corrected = weights[ak.flatten(events["CorrectedJet"]['_'.join(branch.split('_')[1:])] > 32)]
            var_values = getattr(getattr(events_corrected, branch.split('_')[0]), '_'.join(branch.split('_')[1:]))
            vals_flat = ak.flatten(var_values)
            weights_broadcast = ak.broadcast_arrays(var_values, weights_corrected)[1]
            weights_flat = ak.flatten(weights_broadcast)
            histo = Hist(hist.axis.Regular(binning[-1] - 1, binning[0], binning[1], name = name, underflow = True, overflow = True), storage=hist.storage.Weight())
            histo.fill(vals_flat, weight=weights_flat)

        else:
            if not is_MC: continue
            events_corrected = events[ak.flatten(events["CorrectedJet"]["pt"] > 32)]
            weights_corrected = weights[ak.flatten(events["CorrectedJet"]["pt"] > 32)]
            var_values = getattr(events_corrected, 'CorrectedJet')
            var_values = getattr(var_values, 'pt')
            vals_flat = ak.flatten(var_values)
            weights_corr = weights_corrected / events_corrected.weight * events_corrected[branch]
            weights_broadcast = ak.broadcast_arrays(var_values, weights_corr)[1]
            weights_flat = ak.flatten(weights_broadcast)
            histo = Hist(hist.axis.Regular(binning[-1] - 1, binning[0], binning[1], name = name, underflow = True, overflow = True), storage=hist.storage.Weight())
            histo.fill(vals_flat, weight=weights_flat)

        if name not in histogram_dict:
            histogram_dict[name] = {}

        if process not in histogram_dict[name]:
            histogram_dict[name][process] = []

        histogram_dict[name][process].append(histo)

        if name not in binning_dict:
            binning_dict[name] = {}

        if process not in binning_dict[name]:
            binning_dict[name][process] = np.linspace(*binning)
# Plotting part
total_hist_dict = {}
sum_hist_dict = {}
for plot_name, histograms in histogram_dict.items():
    print('INFO: Now making plot for', plot_name, '...')
    
    if plot_name not in event_dict:
        event_dict[plot_name] = {}

    for bg in BACKGROUNDS:
        total_hist_dict[bg] = Hist(hist.axis.Regular(binning[-1] - 1, binning[0], binning[1], name = plot_name, underflow = True, overflow = True), storage=hist.storage.Weight())
    
    for process, histogram in histograms.items():
        histogram = sum(histogram)
        if process in all_samples_dict['JetMET']:
           total_hist_dict["Data"] += histogram
        elif process in all_samples_dict['TT']:
            total_hist_dict["Top"] += histogram
        elif process in all_samples_dict['singleT']:
            total_hist_dict["Top"] += histogram
        elif process in all_samples_dict['DY']:
            total_hist_dict["DY"] += histogram
        elif process in all_samples_dict['WtoLNu']:
            total_hist_dict["WJets"] += histogram
        elif process in all_samples_dict['Wto2Q']:
            total_hist_dict["WJets"] += histogram
        elif process in all_samples_dict['EWK']:
            total_hist_dict["EWK"] += histogram
        elif process in all_samples_dict['VBF']:
            total_hist_dict["VBF"] += histogram
        elif process in all_samples_dict['QCD']:
            total_hist_dict["QCD"] += histogram

    for bg in BACKGROUNDS:
        sum_hist_dict[bg] = np.sum(total_hist_dict[bg].values(flow=True))

        if bg not in event_dict[plot_name]:
            event_dict[plot_name][bg] = 0
        event_dict[plot_name][bg] += sum_hist_dict[bg] 

## ============================================================
##  APPEND THIS TO THE END OF YOUR EXISTING SCRIPT
## ============================================================

## ── Build κ values from event_dict ──────────────────────────
## event_dict structure:
##   event_dict["jet_pt"][bg]          = nominal yield
##   event_dict["jet_pt_jesUp"][bg]    = JES up yield
##   event_dict["jet_pt_jesDown"][bg]  = JES down yield
##   event_dict["jet_pt_jerUp"][bg]    = JER up yield
##   event_dict["jet_pt_jerDown"][bg]  = JER down yield
##   event_dict["weight_puUp"][bg]     = PU up yield
##   event_dict["weight_puDown"][bg]   = PU down yield

kappa_dict = {}
datacard_dict = {}
print(np.sqrt(np.sum(total_hist_dict["QCD"].view().variance)))
systematic_pairs = {
    "CMS_scale_j": ("jet_pt_jesUp", "jet_pt_jesDown"),
    "CMS_res_j":   ("jet_pt_jerUp", "jet_pt_jerDown"),
    "CMS_pileup":  ("weight_puUp",  "weight_puDown"),
}

nominal_key = "jet_pt"

for bg in BACKGROUNDS:
    if bg == "Data":
        continue

    nominal_yield = event_dict.get(nominal_key, {}).get(bg, 0.0)
    if nominal_yield == 0.0:
        continue

    kappa_dict[bg] = {"nominal_yield": round(nominal_yield, 4)}
    datacard_dict[bg] = {"nominal_yield": round(nominal_yield, 4)}

    for syst_name, (up_key, down_key) in systematic_pairs.items():
        yield_up   = event_dict.get(up_key, {}).get(bg, 0.0)
        yield_down = event_dict.get(down_key, {}).get(bg, 0.0)

        kappa_up   = yield_up   / nominal_yield if nominal_yield != 0 else 1.0
        kappa_down = yield_down / nominal_yield if nominal_yield != 0 else 1.0

        kappa_dict[bg][syst_name] = {
            "yield_nominal": round(nominal_yield, 4),
            "yield_up":      round(yield_up, 4),
            "yield_down":    round(yield_down, 4),
            "kappa_up":      round(kappa_up, 4),
            "kappa_down":    round(kappa_down, 4),
        }

        ## Format for the datacard lnN line
        ## Asymmetric if |kappa_up - 1| and |kappa_down - 1| differ by more than 20%
        delta_up   = abs(kappa_up - 1.0)
        delta_down = abs(kappa_down - 1.0)
        if delta_up > 0 and delta_down > 0 and max(delta_up, delta_down) / min(delta_up, delta_down) > 1.2:
            datacard_dict[bg][syst_name] = f"{kappa_down:.4f}/{kappa_up:.4f}"
        else:
            ## Symmetric: average the deviations
            avg_kappa = 1.0 + (delta_up + delta_down) / 2.0
            datacard_dict[bg][syst_name] = f"{avg_kappa:.4f}"


## ── Print summary ───────────────────────────────────────────
print("\n" + "="*70)
print("  κ VALUES FOR DATACARD")
print("="*70)

for bg, systs in kappa_dict.items():
    print(f"\n{bg}:")
    print(f"  Nominal yield: {systs['nominal_yield']:.1f}")
    for syst_name in systematic_pairs:
        if syst_name in systs:
            s = systs[syst_name]
            print(f"  {syst_name:20s}  {s['kappa_down']:.4f}/{s['kappa_up']:.4f}  "
                  f"(yields: {s['yield_down']:.1f} / {s['yield_nominal']:.1f} / {s['yield_up']:.1f})")


## ── Print datacard-ready lines ──────────────────────────────
print("\n" + "="*70)
print("  DATACARD lnN LINES (copy-paste ready)")
print("="*70)

## Get ordered list of backgrounds that have entries
bg_order = [bg for bg in BACKGROUNDS if bg in datacard_dict and bg != "Data"]

header = f"{'':25s}" + "".join(f"{bg:>15s}" for bg in bg_order)
print(header)

for syst_name in systematic_pairs:
    line = f"{syst_name:20s} lnN"
    for bg in bg_order:
        if syst_name in datacard_dict.get(bg, {}):
            line += f"{datacard_dict[bg][syst_name]:>15s}"
        else:
            line += f"{'-':>15s}"
    print(line)


## ── Export to JSON (append, don't overwrite) ────────────────

## Use ABSOLUTE path so it's the same regardless of working directory
#output_json = "kappa_values.json"
output_json = "nonsense.json"

## Determine the channel name from the sample_folder path
if "SR" in sample_folder:
    channel_name = sample_folder.rstrip("/").split("/")[-2].split("_")[0]
else:
    channel_name = '_'.join(sample_folder.rstrip("/").split("/")[-2].split("_")[0:2])

## Get the observed data count from event_dict
data_obs = round(event_dict.get("jet_pt", {}).get("Data", 0.0))

## Build the new entry for this channel
new_entry = {
    "channel": channel_name,
    "luminosity_pb": target_lumi * 1000,
    "observation": data_obs,
    "backgrounds": kappa_dict,
    "datacard_lnN": datacard_dict,
    "event_counts": {k: {bg: round(v, 4) for bg, v in counts.items()} for k, counts in event_dict.items()},
}

## Wrap it under the channel name key
json_output_nested = {channel_name: new_entry}

## Load existing JSON if it exists, otherwise start fresh
print(f"\n→ Looking for existing JSON at: {output_json}")
try:
    with open(output_json, "r") as f:
        json_input = json.load(f)
    print(f"  Found existing file with channels: {list(json_input.keys())}")
    json_input.update(json_output_nested)
except (FileNotFoundError, json.JSONDecodeError) as e:
    json_input = json_output_nested
    print(f"  No existing file found ({e}), starting fresh")

## Write back
with open(output_json, "w") as f:
    json.dump(json_input, f, indent=2)

print(f"✓ κ values for '{channel_name}' exported to {output_json}")
#print(f"  Observation (Data): {data_obs}")
print(f"  Channels now in file: {list(json_input.keys())}")

