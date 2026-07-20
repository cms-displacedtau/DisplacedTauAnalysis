
import uproot, os
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
target_lumi = 26.7
nanov = 'Summer22_CHS_v10/'
# nanov = ''
sample_folder = f"/eos/uscms/store/user/dally/skim/Summer22_CHS_v10/mutau/v3/material_veto/faster_trial/"

all_samples_dict = {
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

with uproot.open("../skimming/Material_Map_HIST.root") as f:
    h2 = f["material_map"]
values = h2.values()
xedges = h2.axes[0].edges()
yedges = h2.axes[1].edges()

dataset_name = 'all'

# Load plot settings from JSON file
with open("./plots_config/DiMuon_plot_settings.json", "r") as file:
    plot_settings = json.load(file)

# Directory creation for plots (directory_path = "plots/" + args.process)
directory_path = "plots/"
if not os.path.exists(directory_path):
    os.makedirs(directory_path)
    print(f"Directory '{directory_path}' created successfully.")

# Dictionary for histograms and binnings
data_histogram_dict = {}
material_histogram_dict = {}
binning_dict = {}

vtx_x_hist = []
vtx_y_hist = []

for process in available_processes:
    # print(process)
    tmp_string = f"faster_trial_{process}/faster_trial_{process}.root"
    tmp_file = sample_folder +  tmp_string
    events = NanoEventsFactory.from_root({tmp_file:"Events"}, schemaclass= PFNanoAODSchema).events()
    
    DiMuon = ak.concatenate([events.L1DiMuon, events.L2DiMuon], axis  = 1)
    events = ak.with_field(events, DiMuon, "DiMuon")
    vtx_x_hist.append(events.DiMuon.vtx_x)
    vtx_y_hist.append(events.DiMuon.vtx_y)
    r = (events.DiMuon.vtx_x ** 2 + events.DiMuon.vtx_y ** 2) ** (1/2)
    events["DiMuon"] = ak.with_field(events.DiMuon, r, 'vtx_r')

    for plot_name, settings in plot_settings.items():
        var_values = getattr(getattr(events, settings["field"]), settings["variable"])
        vals_flat = ak.flatten(var_values, axis = None)
        hist, _ = np.histogram(vals_flat, bins=np.linspace(*settings["binning_linspace"]))

        if plot_name not in data_histogram_dict:
            data_histogram_dict[plot_name] = {}

        if process not in data_histogram_dict[plot_name]:
            data_histogram_dict[plot_name][process] = []

        data_histogram_dict[plot_name][process].append(hist)

        if plot_name not in binning_dict:
            binning_dict[plot_name] = {}

        if process not in binning_dict[plot_name]:
            binning_dict[plot_name][process] = np.linspace(*settings["binning_linspace"])
    
        
groupProcesses = True

vtx_x_hist = ak.flatten(vtx_x_hist, axis = None)
vtx_y_hist = ak.flatten(vtx_y_hist, axis = None)

fig, ax = plt.subplots()
hep.hist2dplot(values, xedges[0:600], yedges[0:600], cbar=True)
plt.scatter(vtx_x_hist, vtx_y_hist, s = 2)

filedir = "Material_Interaction"
if filedir not in os.listdir('plots/'):
    os.mkdir(f'plots/{filedir}')
filename = f"./plots/{filedir}/pos_map"
# #if args.groupProcesses:
#filename += "_etaleq1p2.pdf"
#filename += "_etag1p2.pdf"
filename += ".pdf"
print(filename)
plt.savefig(filename)
plt.savefig(filename.replace('.pdf', '.png'))
plt.savefig(filename.replace('.pdf', '.eps'))
plt.clf()


for plot_name in data_histogram_dict.keys():
    print('INFO: Now making plot for', plot_name, '...')
    binning = np.linspace(*plot_settings[plot_name].get("binning_linspace"))
    do_stack = not plot_settings[plot_name].get("density")
    # if args.groupProcesses:
    if groupProcesses:
        hist_Data      = np.zeros(len(binning)-1)
        hist_mat_Data      = np.zeros(len(binning)-1)
    
    data_hists = []
    mat_data_hists = []

    for process, histogram in data_histogram_dict[plot_name].items():
        histogram = np.asarray(histogram)
        histogram = histogram.flatten()
        if groupProcesses:
            if process in all_samples_dict['JetMET']:
                hist_Data += histogram

    if groupProcesses:
        data_hists.append(hist_Data)

    
    fig, (ax_main, ax_ratio) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    fig.subplots_adjust(hspace=0.0)
    hep.histplot(data_hists, bins=binning, stack=do_stack, histtype='fill', 
                 label = 'Muon from muon vtx',
                 density=plot_settings[plot_name].get("density"), ax=ax_main)
    ax_main.set_ylabel(plot_settings[plot_name].get("ylabel"))
    ax_main.legend()


    
    ## this part is still to be done 
    #if groupProcesses:
    # # # if args.groupProcesses:
        #sum_histogram = np.sum(np.asarray(hists_to_plot), axis=0)
        #sum_data_histogram = np.sum(np.asarray(data_hists), axis=0)
        #ratio_hist = sum_data_histogram / (sum_histogram + np.finfo(float).eps)
    # #     # Adding relative sqrtN Poisson uncertainty for now, should be improved when using the hist package
        #rel_unc = np.sqrt(sum_data_histogram) / sum_data_histogram
        #rel_unc *= ratio_hist
        #rel_unc[rel_unc < 0] = 0 # Not exactly sure why we have negative values, but this solves it for the moment
        #hep.histplot(ratio_hist, bins=binning, histtype='errorbar', yerr=rel_unc, color='black', label='Ratio', ax=ax_ratio)
        #ax_ratio.axhline(1, color='gray', linestyle='--')
    #ax_ratio.set_xlabel(plot_settings[plot_name].get("xlabel"), usetex=False)
    #ax_main.xaxis.set_major_locator(MultipleLocator(plot_settings[plot_name].get("x_major_ticks")))
    #ax_main.xaxis.set_minor_locator(MultipleLocator(plot_settings[plot_name].get("x_minor_ticks")))
    #ax_ratio.xaxis.set_major_locator(MultipleLocator(plot_settings[plot_name].get("x_major_ticks")))
    #ax_ratio.xaxis.set_minor_locator(MultipleLocator(plot_settings[plot_name].get("x_minor_ticks")))
    #ax_ratio.set_ylabel('Data / MC')
    #ax_ratio.set_xlim(binning[0], binning[-1])
    #ax_ratio.set_ylim(0.6, 1.4)
    
    # Decorating with CMS label
    hep.cms.label(data=True, loc=0, label="Private Work", com=13.6, lumi=round(target_lumi, 1), ax=ax_main)
    #hep.cms.add_text(r'$|\eta|$ > 1.2', loc = 'upper left')
    
    # Saving with special name
    #filename = f"/eos/uscms/store/user/dally/DisplacedTauAnalysis/plots/{dataset_name}_{plot_name}"
    filedir = "Material_Interaction"
    if filedir not in os.listdir('plots/'):
        os.mkdir(f'plots/{filedir}')
    filename = f"./plots/{filedir}/{dataset_name}_{plot_name}"
    # #if args.groupProcesses:
    if plot_settings[plot_name].get("density"):
        filename += "_normalized"
    else: 
        filename += "_stacked"
    #filename += "_etaleq1p2.pdf"
    #filename += "_etag1p2.pdf"
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


