import os
import json
import numpy as np
import awkward as ak
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
import mplhep as hep
import argparse
from coffea.lumi_tools import LumiMask
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema, PFNanoAODSchema
import pdb, glob

PFNanoAODSchema.warn_missing_crossrefs = False
PFNanoAODSchema.mixins["DisMuon"] = "Muon"

hep.style.use("CMS")

nanov = 'Summer22_CHS_v10/'
# nanov = ''
#sample_folder = f"/eos/uscms/store/user/dally/skim/{nanov}prompt_mutau/v8/selected/faster_trial/"
#sample_folder = f"/eos/uscms/store/user/dally/skim/{nanov}mutau/v3/selected/W_CR/faster_trial/"
sample_folder = f"/eos/uscms/store/group/lpcdisptau/dally/displacedTaus/selected/Summer22_CHS_v10/mutau/v3/PR/faster_trial/"
## if a sample is not ready yet, comment it out
all_samples_dict = {
    "WtoLNu" : [
        "WtoLNu-4Jets",
##         "WtoLNu-2Jets_0J",
##        "WtoLNu-2Jets_1J",
##         "WtoLNu-2Jets_2J",
##        "WtoLNu-4Jets_3J",
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
with open(f"./plots_config/{nanov}/v8/all_total_sumw.json", "r") as file:
    sum_gen_w = json.load(file)
    
## target lumi, for now fixed value
target_lumi = 26.7

# int_lumi = ak.from_parquet("../sample_processing/my_skimData2018C").intLumi[0]
#print(int_lumi)
#print(ak.from_parquet("../sample_processing/my_skimData2018C").intLumi)


## Parser setup
# parser = argparse.ArgumentParser(description="Make control plots for Z to mu mu studies.")
# parser.add_argument("--dataset", choices=["all"] + available_processes, default = "all", type=str, help="specifying which plot you want") # changeable in future... 
# parser.add_argument("--groupProcesses", action="store_true", default = "false", help="saying which processes do you want to group")
# args = parser.parse_args()

dataset_name = 'all'

# Load plot settings from JSON file
with open("./plots_config/dR_plot_settings.json", "r") as file:
    plot_settings = json.load(file)

# Directory creation for plots (directory_path = "plots/" + args.process)
directory_path = "plots/"
if not os.path.exists(directory_path):
    os.makedirs(directory_path)
    print(f"Directory '{directory_path}' created successfully.")

# Dictionary for histograms and binnings
histogram_dict = {}
binning_dict = {}

for process in available_processes:
    # print(process)
    tmp_string = f"faster_trial_{process}/faster_trial_{process}.root"
    tmp_file = sample_folder +  tmp_string
    PFNanoAODSchema.mixins["DisMuon"] = "Muon"
    PFNanoAODSchema.mixins["CorrectedJet"] = "Jet"
    events = NanoEventsFactory.from_root({tmp_file:"Events"}, schemaclass= PFNanoAODSchema).events()

    jet_id = (
       (events.CorrectedJet.neHEF < 0.99) &
       (events.CorrectedJet.neEmEF < 0.9) &
       (events.CorrectedJet.chMultiplicity + events.CorrectedJet.neMultiplicity > 1) &
       (events.CorrectedJet.chMultiplicity > 0) &
       (events.CorrectedJet.muEF < 0.5) &
       (events.CorrectedJet.chEmEF < 0.8)
    )

    events = events[ak.flatten(jet_id, axis = None)]

    mutau_dr = events.DisMuon.metric_table(events.CorrectedJet)
    events["DisMuon"] = ak.with_field(events.DisMuon, mutau_dr, "dR")

    weights = events.run / events.run
    if "jetmet" in process.lower():
        weights = weights
#    if "muon" in process.lower():
#        weights = weights
    else:
        lumi_weight = target_lumi * xsec[process] * br[process] * 1000 / sum_gen_w[process]
#         lumi_weight = target_lumi * xsec[process] * br[process] * 1000 / sum_gen_w[reverse_samples_lookup[process]][process]
        weights = events.weight * lumi_weight ## am i missing the sumGenW here?

    for plot_name, settings in plot_settings.items():
        ## test, not clear why
#         print("vals type:", type(var_values), "vals layout:", var_values.layout.form)
        if not settings["per_event"]:
            var_values = getattr(getattr(events, settings["field"]), settings["variable"])
            vals_flat = ak.flatten(var_values)
            weights_broadcast = ak.broadcast_arrays(var_values, weights)[1]
            weights_flat = ak.flatten(weights_broadcast)
            hist, _ = np.histogram(vals_flat, weights=weights_flat, bins=np.linspace(*settings["binning_linspace"]))

        else:
            if settings["variable"] == "":
                hist, _ = np.histogram(getattr(events, settings["field"]), weights=ak.broadcast_arrays(getattr(events, settings["field"]), weights)[1], bins=np.linspace(*settings["binning_linspace"]))
            else:
                hist, _ = np.histogram(getattr(getattr(events, settings["field"]), settings["variable"]), weights=ak.broadcast_arrays(getattr(getattr(events, settings["field"]), settings["variable"]), weights)[1], bins=np.linspace(*settings["binning_linspace"]))

        if plot_name not in histogram_dict:
            histogram_dict[plot_name] = {}

        if process not in histogram_dict[plot_name]:
            histogram_dict[plot_name][process] = []

        histogram_dict[plot_name][process].append(hist)

        if plot_name not in binning_dict:
            binning_dict[plot_name] = {}

        if process not in binning_dict[plot_name]:
            binning_dict[plot_name][process] = np.linspace(*settings["binning_linspace"])



# Plotting part
groupProcesses = True

fig, ax_main = plt.subplots()
print(np.asarray(histogram_dict['DisMuon_dR']['WtoLNu-4Jets']))
hep.hist2dplot([np.asarray(histogram_dict['DisMuon_dR']['WtoLNu-4Jets']).flatten(), np.asarray(histogram_dict['Jet_score']['WtoLNu-4Jets']).flatten()],
                bins = [binning_dict['DisMuon_dR']['WtoLNu-4Jets'], binning_dict['Jet_score']['WtoLNu-4Jets']], 
                ax = ax_main
)

plt.savefig(f"./plots/preselection_W_dR/all_DisMuon_dR_Jet_score_2DHist.pdf")
plt.savefig("./plots/preselection_W_dR/all_DisMuon_dR_Jet_score_2DHist.png")

for plot_name, histograms in histogram_dict.items():
    print('INFO: Now making plot for', plot_name, '...')
    binning = np.linspace(*plot_settings[plot_name].get("binning_linspace"))
    do_stack = not plot_settings[plot_name].get("density")
    # if args.groupProcesses:
    if groupProcesses:
        hist_WtoLNu    = np.zeros(len(binning)-1)

    hists_to_plot = []
    labels = []

    for process, histogram in histograms.items():
#         print('INFO: Now looking at process', process, '...')
        histogram = np.asarray(histogram)
        histogram = histogram.flatten()
        # Fix later, hist should not be a list in the first place and should be 60 1d and not (60,1)
        # if args.groupProcesses:
        if groupProcesses:
            if process in all_samples_dict['WtoLNu']:
                hist_WtoLNu += histogram

    if groupProcesses:
        hists_to_plot.append(hist_WtoLNu)
        labels.append('WtoLNu')

    fig, ax_main = plt.subplots(1, 1, sharex=True)
    hep.histplot(hists_to_plot, bins=binning,
                 histtype = 'fill', color = "#964a8b",
                 label=labels, #color=colours, #sort='label_r', 
                 density=plot_settings[plot_name].get("density"), ax=ax_main)

    ax_main.set_ylabel(plot_settings[plot_name].get("ylabel"))
    ax_main.set_xlabel(plot_settings[plot_name].get("xlabel"), usetex=False)
    ax_main.legend()
    ax_main.xaxis.set_major_locator(MultipleLocator(plot_settings[plot_name].get("x_major_ticks")))
    ax_main.xaxis.set_minor_locator(MultipleLocator(plot_settings[plot_name].get("x_minor_ticks")))

    # Decorating with CMS label
    hep.cms.label(data=True, loc=0, label="Private Work", com=13.6, lumi=round(target_lumi, 1), ax=ax_main)
    #hep.cms.add_text(r'$|\eta|$ > 1.2', loc = 'upper left')
    
    # Saving with special name
    filedir = "preselection_W_dR"
    if filedir not in os.listdir('plots/'):
        os.mkdir(f'plots/{filedir}')
    filename = f"./plots/{filedir}/{dataset_name}_{plot_name}_jetID"
    if plot_settings[plot_name].get("density"):
        filename += "_normalized"
    else: 
        filename += "_stacked"
    filename += ".pdf"
    print(filename)
    plt.savefig(filename)
    plt.savefig(filename.replace('.pdf', '.png'))
    plt.savefig(filename.replace('.pdf', '.eps'))
    ax_main.set_yscale('log')
    plt.savefig(filename.replace('.pdf', '_log.pdf'))
    plt.savefig(filename.replace('.pdf', '_log.png'))
    plt.savefig(filename.replace('.pdf', '_log.eps'))
    plt.clf()
