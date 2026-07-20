import os
import json
import hist
from hist import Hist
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

print(hep.__version__)

nanov = 'Summer22_CHS_v10/'
PFNanoAODSchema.warn_missing_crossrefs = False
PFNanoAODSchema.mixins["DisMuon"] = "Muon"
PFNanoAODSchema.mixins["CorrectedJet"] = "Jet"

hep.style.use("CMS")

sample_folder = f"/eos/uscms/store/group/lpcdisptau/dally/displacedTaus/selected/Summer22_CHS_v10/mutau/v4/PR/faster_trial/"
## if a sample is not ready yet, comment it out
all_samples_dict = {
    "QCD" : [
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

dataset_name = 'all'

# Load plot settings from JSON file
with open("./plots_config/displaced_corrected_plot_settings.json", "r") as file:
    plot_settings = json.load(file)

# Directory creation for plots (directory_path = "plots/" + args.process)
directory_path = "plots/"
if not os.path.exists(directory_path):
    os.makedirs(directory_path)
    print(f"Directory '{directory_path}' created successfully.")

# Dictionary for histograms and binnings
histogram_l0p2_dict = {}
histogram_geq0p2_dict = {}
binning_dict = {}

for process in available_processes:
    print(process)
    tmp_string = f"faster_trial_{process}/faster_trial_{process}.root"
    tmp_file = sample_folder +  tmp_string
    events = NanoEventsFactory.from_root({tmp_file:"Events"}, schemaclass= PFNanoAODSchema).events()
    events = events[events.CorrectedPFMET.pt > 105]
    
    dphi = abs(events.CorrectedJet.phi - events.CorrectedPFMET.phi)
    dphi = np.where(dphi > np.pi, 2*np.pi - dphi, dphi)
    events["CorrectedJet"] = ak.with_field(events.CorrectedJet, dphi, "dphi")
    mu_dphi = abs(events.CorrectedJet.phi - events.DisMuon.phi)
    mu_dphi = np.where(dphi > np.pi, 2*np.pi - dphi, dphi)
    events["DisMuon"] = ak.with_field(events.DisMuon, mu_dphi, "dphi")
    deta = abs(events.CorrectedJet.eta - events.DisMuon.eta)
    events["DisMuon"] = ak.with_field(events.DisMuon, deta, "deta")
    dR = events.DisMuon.metric_table(events.CorrectedJet)
    dR = ak.flatten(dR)
    events["DisMuon"] = ak.with_field(events.DisMuon, dR, "dR")

    pt_raw = (1 - events.Jet.rawFactor) * events.Jet.pt
    events["Jet"] = ak.with_field(events.Jet, pt_raw, "ptRaw")

    weights = events.run / events.run
    if "jetmet" in process.lower():
        weights = weights
    else:
        lumi_weight = target_lumi * xsec[process] * br[process] * 1000 / sum_gen_w[process]
        weights = events.weight * lumi_weight ## am i missing the sumGenW here?
    print(len(events))

    #events_l0p2 = events[ak.ravel(events.DisMuon.pfRelIso03_all < 0.2)]
    #events_geq0p2 = events[ak.ravel(events.DisMuon.pfRelIso03_all >= 0.2)]

    #weights_l0p2 = weights[ak.ravel(events.DisMuon.pfRelIso03_all < 0.2)]
    #weights_geq0p2 = weights[ak.ravel(events.DisMuon.pfRelIso03_all >= 0.2)]

    events_l0p2 = events[ak.ravel(events.DisMuon.mT < 50)]
    events_geq0p2 = events[ak.ravel(events.DisMuon.mT >= 50)]

    weights_l0p2 = weights[ak.ravel(events.DisMuon.mT < 50)]
    weights_geq0p2 = weights[ak.ravel(events.DisMuon.mT >= 50)]

    for plot_name, settings in plot_settings.items():
        if "weight" in plot_name: continue
        if "correction" in plot_name: continue
        if "resolution" in plot_name: continue
        if "Sig" in plot_name: continue
        if not settings["per_event"]:
            var_values_l0p2 = getattr(getattr(events_l0p2, settings["field"]), settings["variable"])
            vals_flat_l0p2 = ak.flatten(var_values_l0p2)
            weights_broadcast_l0p2 = ak.broadcast_arrays(var_values_l0p2, weights_l0p2)[1]
            weights_flat_l0p2 = ak.flatten(weights_broadcast_l0p2)
            histo_l0p2 = Hist(hist.axis.Regular(settings["binning_linspace"][-1] - 1, settings["binning_linspace"][0], settings["binning_linspace"][1], name = plot_name, underflow = True, overflow = True))
            histo_l0p2.fill(vals_flat_l0p2, weight=weights_flat_l0p2)

            var_values_geq0p2 = getattr(getattr(events_geq0p2, settings["field"]), settings["variable"])
            vals_flat_geq0p2 = ak.flatten(var_values_geq0p2)
            weights_broadcast_geq0p2 = ak.broadcast_arrays(var_values_geq0p2, weights_geq0p2)[1]
            weights_flat_geq0p2 = ak.flatten(weights_broadcast_geq0p2)
            histo_geq0p2 = Hist(hist.axis.Regular(settings["binning_linspace"][-1] - 1, settings["binning_linspace"][0], settings["binning_linspace"][1], name = plot_name, underflow = True, overflow = True))
            histo_geq0p2.fill(vals_flat_geq0p2, weight=weights_flat_geq0p2)
        else:
            if settings["variable"] == "":
                var_values_l0p2 = getattr(events_l0p2, settings["field"])
                weights_broadcast_l0p2 = ak.broadcast_arrays(var_values_l0p2, weights_l0p2)[1]
                histo_l0p2 = Hist(hist.axis.Regular(settings["binning_linspace"][-1] - 1, settings["binning_linspace"][0], settings["binning_linspace"][1], name = plot_name, underflow = True, overflow = True))
                histo_l0p2.fill(var_values_l0p2, weight=weights_broadcast_l0p2)

                var_values_geq0p2 = getattr(events_geq0p2, settings["field"])
                weights_broadcast_geq0p2 = ak.broadcast_arrays(var_values_geq0p2, weights_geq0p2)[1]
                histo_geq0p2 = Hist(hist.axis.Regular(settings["binning_linspace"][-1] - 1, settings["binning_linspace"][0], settings["binning_linspace"][1], name = plot_name, underflow = True, overflow = True))
                histo_geq0p2.fill(var_values_geq0p2, weight=weights_broadcast_geq0p2)

            else:
                var_values_l0p2 = getattr(getattr(events_l0p2, settings["field"]), settings["variable"])
                weights_broadcast_l0p2 = ak.broadcast_arrays(var_values_l0p2, weights_l0p2)[1]
                histo_l0p2 = Hist(hist.axis.Regular(settings["binning_linspace"][-1] - 1, settings["binning_linspace"][0], settings["binning_linspace"][1], name = plot_name, underflow = True, overflow = True))
                histo_l0p2.fill(var_values_l0p2, weight=weights_broadcast_l0p2)

                var_values_geq0p2 = getattr(getattr(events_geq0p2, settings["field"]), settings["variable"])
                weights_broadcast_geq0p2 = ak.broadcast_arrays(var_values_geq0p2, weights_geq0p2)[1]
                histo_geq0p2 = Hist(hist.axis.Regular(settings["binning_linspace"][-1] - 1, settings["binning_linspace"][0], settings["binning_linspace"][1], name = plot_name, underflow = True, overflow = True))
                histo_geq0p2.fill(var_values_geq0p2, weight=weights_broadcast_geq0p2)

        if plot_name not in histogram_l0p2_dict:
            histogram_l0p2_dict[plot_name] = {}
        if plot_name not in histogram_geq0p2_dict:
            histogram_geq0p2_dict[plot_name] = {}

        if process not in histogram_l0p2_dict[plot_name]:
            histogram_l0p2_dict[plot_name][process] = []
        if process not in histogram_geq0p2_dict[plot_name]:
            histogram_geq0p2_dict[plot_name][process] = []

        histogram_l0p2_dict[plot_name][process].append(histo_l0p2)
        histogram_geq0p2_dict[plot_name][process].append(histo_geq0p2)

        if plot_name not in binning_dict:
            binning_dict[plot_name] = {}

        if process not in binning_dict[plot_name]:
            binning_dict[plot_name][process] = np.linspace(*settings["binning_linspace"])

# Plotting part
groupProcesses = True

for plot_name, histograms in histogram_l0p2_dict.items():
    print('INFO: Now making plot for', plot_name, '...')
    binning = np.linspace(*plot_settings[plot_name].get("binning_linspace"))
    do_stack = not plot_settings[plot_name].get("density")
    # if args.groupProcesses:
    if groupProcesses:
        hist_l0p2_QCD       = Hist(hist.axis.Regular(plot_settings[plot_name]["binning_linspace"][-1] - 1, plot_settings[plot_name]["binning_linspace"][0], plot_settings[plot_name]["binning_linspace"][1], name = plot_name, underflow = True, overflow = True))
        hist_geq0p2_QCD       = Hist(hist.axis.Regular(plot_settings[plot_name]["binning_linspace"][-1] - 1, plot_settings[plot_name]["binning_linspace"][0], plot_settings[plot_name]["binning_linspace"][1], name = plot_name, underflow = True, overflow = True))
    
    hists_l0p2_to_plot = []
    hists_geq0p2_to_plot = []
    labels = []

    for process in  histograms.keys():
        histogram_l0p2 = histogram_l0p2_dict[plot_name][process]
        histogram_geq0p2 = histogram_geq0p2_dict[plot_name][process]
        histogram_l0p2 = sum(histogram_l0p2)
        histogram_geq0p2 = sum(histogram_geq0p2)
        if groupProcesses:
            if process in all_samples_dict['QCD']:
                hist_l0p2_QCD += histogram_l0p2
                hist_geq0p2_QCD += histogram_geq0p2
            #elif process in all_samples_dict['Stau']:
            #    hist_Sig += histogram

    
    sum_hist_l0p2_QCD   = np.sum(hist_l0p2_QCD.values())
    sum_hist_geq0p2_QCD = np.sum(hist_geq0p2_QCD.values())

    if groupProcesses:
        hists_l0p2_to_plot.append(hist_l0p2_QCD/sum_hist_l0p2_QCD)
        labels.append('pfRelIso03_all < 0.2')
        hists_geq0p2_to_plot.append(hist_geq0p2_QCD/sum_hist_geq0p2_QCD)
        labels.append('pfRelIso03_all >= 0.2')

    colours = ["#5790fc", "#f89c20", "#e42536", "#964a8b", "#9c9ca1", "#7a21dd", "#FF99C9", "#C8E9A0", "#6DD3CE",] #"#127475", "#FF99C9"]
    
    fig, ax_main = plt.subplots(1, 1, sharex=True)
    fig.subplots_adjust(hspace=0.0)
    hep.histplot(hists_l0p2_to_plot, 
                 histtype = 'step',
                 label='DisMuon-PFMET mT < 50', color="#5790fc", #sort='label_r', 
                 flow = 'sum',
                 density=plot_settings[plot_name].get("density"), ax=ax_main)
    hep.histplot(hists_geq0p2_to_plot, histtype='step', 
                  color="#f89c20", label='DisMuon-PFMET mT >= 50', density=plot_settings[plot_name].get("density"), ax=ax_main,
                 flow = 'sum',)
    ax_main.set_ylabel(plot_settings[plot_name].get("ylabel"))
    ax_main.set_xlabel(plot_settings[plot_name].get("xlabel"), usetex=False)
    ax_main.legend()

    ax_main.xaxis.set_major_locator(MultipleLocator(plot_settings[plot_name].get("x_major_ticks")))
    ax_main.xaxis.set_minor_locator(MultipleLocator(plot_settings[plot_name].get("x_minor_ticks")))

    # Decorating with CMS label
    hep.cms.label(data=True, loc=0, label="Private Work", com=13.6, lumi=round(target_lumi, 1), ax=ax_main)
    #hep.cms.add_text(r'$|\eta|$ > 1.2', loc = 'upper left')
    
    # Saving with special name
    #filename = f"/eos/uscms/store/user/dally/DisplacedTauAnalysis/plots/{dataset_name}_{plot_name}"
    filedir = "QCD_shapes_mT"
    if filedir not in os.listdir('plots/'):
        os.mkdir(f'plots/{filedir}')
    filename = f"./plots/{filedir}/{dataset_name}_{plot_name}"
    # #if args.groupProcesses:
    if plot_settings[plot_name].get("density"):
        filename += "_normalized"
    else: 
        filename += "_stacked"
    #filename += "_etaleq1p4.pdf"
    #filename += "_etag1p4.pdf"
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
