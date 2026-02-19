import os
import json
import numpy as np
import awkward as ak
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
import mplhep as hep
import hist
from hist import Hist
import argparse
from coffea.lumi_tools import LumiMask
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema, PFNanoAODSchema
import gzip, correctionlib, importlib, pickle
import pdb, glob

PFNanoAODSchema.warn_missing_crossrefs = False
PFNanoAODSchema.mixins["DisMuon"] = "Muon"

hep.style.use("CMS")

nanov = 'Summer22_CHS_v10/'
# nanov = ''
#sample_folder = f"/eos/uscms/store/user/dally/skim/{nanov}prompt_mutau/v8_nobJetVeto/selected/faster_trial/"
#sample_folder = f"/eos/uscms/store/user/dally/skim/{nanov}mutau/v3_CorrectedJet/selected/faster_trial/"
#sample_folder = f"/eos/uscms/store/group/lpcdisptau/dally/displacedTaus/selected/Summer22_CHS_v10/mutau/v3/QCD_CR/faster_trial/"
sample_folder = f"/eos/uscms/store/group/lpcdisptau/dally/displacedTaus/selected/Summer22_CHS_v10/mutau/v4/W_CR/faster_trial/"
normalize = False

#if "PR" in sample_folder:
#    normalize = True

## if a sample is not ready yet, comment it out
all_samples_dict = {
    "DY" : [
        "DYJetsToLL_M-50",
        "DYto2Tau-2Jets_MLL-50_0J",
        "DYto2Tau-2Jets_MLL-50_1J",
        "DYto2Tau-2Jets_MLL-50_2J",
      ],
    "QCD" : [
        #"QCD_PT-50to80",
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
        "QCD_PT-2400to3200",
        "QCD_PT-3200",
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
    "Wto2Q" : [
       "Wto2Q-2Jets_PTQQ-100to200_1J",
        #"Wto2Q-2Jets_PTQQ-100to200_2J",
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
    "Stau_500" : [
        "Stau_500_0p01mm",
        "Stau_500_0p1mm",
        "Stau_500_1mm",
        "Stau_500_10mm",
        "Stau_500_100mm",
        "Stau_500_1000mm",
    ],
    "Stau_300" : [
        "Stau_300_0p01mm",
        "Stau_300_0p1mm",
        "Stau_300_1mm",
        "Stau_300_10mm",
        "Stau_300_100mm",
        "Stau_300_1000mm",
    ],
    "Stau_200" : [
        "Stau_200_0p01mm",
        "Stau_200_0p1mm",
        "Stau_200_1mm",
        "Stau_200_10mm",
        "Stau_200_100mm",
        "Stau_200_1000mm",
    ],
    "Stau_100" : [
        "Stau_100_0p01mm",
        "Stau_100_0p1mm",
        "Stau_100_1mm",
        "Stau_100_10mm",
        "Stau_100_100mm",
        "Stau_100_1000mm",
    ],
}

signal_mass = ['100', '200', '300', '500']
signal_lifetime = ['0p01mm', '0p1mm', '1mm', '10mm', '100mm', '1000mm']


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
with open("./plots_config/displaced_corrected_plot_settings.json", "r") as file:
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
    print(f"Starting {process}")
    tmp_string = f"faster_trial_{process}/faster_trial_{process}.root"
    tmp_file = sample_folder +  tmp_string
    events = NanoEventsFactory.from_root({tmp_file:"Events"}, schemaclass= PFNanoAODSchema).events()
    events = events[events.CorrectedPFMET.pt > 105]
    #events = events[ak.flatten(events.DisMuon.pfRelIso03_all <  0.1)]
    #events = events[ak.flatten(events.CorrectedJet.btagPNetB >  0.6915)]
    #events = events[ak.ravel(abs(events.Tau.eta) > 1.2)]
    #events = events[ak.ravel(abs(events.Tau.eta) > 1.2)]
    
    #mutau_dr = events.Muon.metric_table(events.Tau)
    #mutau_pt = events.Muon.pt + events.Tau.pt

    #met = events.PuppiMET.pt
    #met_phi = events.PuppiMET.phi
    #if "Muon" not in process:
    #    met = events.CorrectedPuppiMET.pt
    #    met_phi = events.CorrectedPuppiMET.phi
    #dphi = abs(events.Tau.phi - met_phi)
    #dphi = np.where(dphi > np.pi, 2*np.pi - dphi, dphi)
    #mT = np.sqrt(2 * events.Muon.pt * met * (1 - np.cos(dphi)))
    #events = ak.with_field(events, mT, "Puppi_mT")

    #events["mutau"] = ak.with_field(events.mutau, mutau_dr, where = 'dR')
    #events["mutau"] = ak.with_field(events.mutau, mutau_pt, where = 'pt') 

    #if "muon" not in process.lower():
    #    events["PuppiMET"] = events.CorrectedPuppiMET
    ## disable for now
#     print ('need to put back weights')

    pt_raw = (1 - events.Jet.rawFactor) * events.Jet.pt
    events["Jet"] = ak.with_field(events.Jet, pt_raw, "ptRaw")

    #dphi = abs(events.DisMuon.phi - events.CorrectedPuppiMET.phi)
    #dphi = np.where(dphi > np.pi, 2*np.pi - dphi, dphi)
    #mT = np.sqrt(2 * events.DisMuon.pt * events.CorrectedPuppiMET.pt * (1 - np.cos(dphi)))
    #events["DisMuon"] = ak.with_field(events.DisMuon, mT, "mT")

    weights = events.run / events.run
    if "jetmet" in process.lower():
        weights = weights
#    if "muon" in process.lower():
#        weights = weights
    else:
        if normalize == False:
            lumi_weight = target_lumi * xsec[process] * br[process] * 1000 / sum_gen_w[process]
#         lumi_weight = target_lumi * xsec[process] * br[process] * 1000 / sum_gen_w[reverse_samples_lookup[process]][process]
        else:
            lumi_weight = 1
        weights = events.weight * lumi_weight ## am i missing the sumGenW here?
    #events["weight"] = abs(weights)
    for plot_name, settings in plot_settings.items():
        if "weight" in plot_name: continue
        if "correction" in plot_name: continue
        if "resolution" in plot_name: continue
        #if "btag" not in plot_name: continue
        #if 'dxy' not in plot_name: continue
        ## test, not clear why
#         print("vals type:", type(var_values), "vals layout:", var_values.layout.form)
        if not settings["per_event"]:
            var_values = getattr(getattr(events, settings["field"]), settings["variable"])
            vals_flat = ak.flatten(var_values)
            weights_broadcast = ak.broadcast_arrays(var_values, weights)[1]
            weights_flat = ak.flatten(weights_broadcast)
            #hist, _ = np.histogram(vals_flat, weights=weights_flat, bins=np.linspace(*settings["binning_linspace"]))
            histo = Hist(hist.axis.Regular(settings["binning_linspace"][-1] - 1, settings["binning_linspace"][0], settings["binning_linspace"][1], name = plot_name, underflow = True, overflow = True))
            histo.fill(vals_flat, weight=weights_flat)

        else:
            if settings["variable"] == "":
                #hist, _ = np.histogram(getattr(events, settings["field"]), weights=ak.broadcast_arrays(getattr(events, settings["field"]), weights)[1], bins=np.linspace(*settings["binning_linspace"]))
                var_values = getattr(events, settings["field"])
                weights_broadcast = ak.broadcast_arrays(var_values, weights)[1]
                histo = Hist(hist.axis.Regular(settings["binning_linspace"][-1] - 1, settings["binning_linspace"][0], settings["binning_linspace"][1], name = plot_name, underflow = True, overflow = True))
                histo.fill(var_values, weight=weights_broadcast)
            else:
                #if settings["variable"] == "mass" and settings["field"] == "mutau":
                    #mutau = events.Muon + events.Tau
                    #events = events[ak.ravel(mutau.charge == 0)]
                    #weights = weights[ak.ravel(mutau.charge == 0)]
                #hist, _ = np.histogram(getattr(getattr(events, settings["field"]), settings["variable"]), weights=ak.broadcast_arrays(getattr(getattr(events, settings["field"]), settings["variable"]), weights)[1], bins=np.linspace(*settings["binning_linspace"]))
                var_values = getattr(getattr(events, settings["field"]), settings["variable"])
                weights_broadcast = ak.broadcast_arrays(var_values, weights)[1]
                histo = Hist(hist.axis.Regular(settings["binning_linspace"][-1] - 1, settings["binning_linspace"][0], settings["binning_linspace"][1], name = plot_name, underflow = True, overflow = True))
                histo.fill(var_values, weight=weights_broadcast)

        if plot_name not in histogram_dict:
            histogram_dict[plot_name] = {}

        if process not in histogram_dict[plot_name]:
            histogram_dict[plot_name][process] = []

        histogram_dict[plot_name][process].append(histo)

        if plot_name not in binning_dict:
            binning_dict[plot_name] = {}

        if process not in binning_dict[plot_name]:
            binning_dict[plot_name][process] = np.linspace(*settings["binning_linspace"])



# Plotting part
groupProcesses = True

for plot_name, histograms in histogram_dict.items():
    print('INFO: Now making plot for', plot_name, '...')
    binning = np.linspace(*plot_settings[plot_name].get("binning_linspace"))
    do_stack = not plot_settings[plot_name].get("density")
    # if args.groupProcesses:
    if groupProcesses:
        hist_Stau_500       = Hist(hist.axis.Regular(plot_settings[plot_name]["binning_linspace"][-1] - 1, plot_settings[plot_name]["binning_linspace"][0], plot_settings[plot_name]["binning_linspace"][1], name = plot_name, underflow = True, overflow = True))
        hist_Stau_300       = Hist(hist.axis.Regular(plot_settings[plot_name]["binning_linspace"][-1] - 1, plot_settings[plot_name]["binning_linspace"][0], plot_settings[plot_name]["binning_linspace"][1], name = plot_name, underflow = True, overflow = True))
        hist_Stau_200       = Hist(hist.axis.Regular(plot_settings[plot_name]["binning_linspace"][-1] - 1, plot_settings[plot_name]["binning_linspace"][0], plot_settings[plot_name]["binning_linspace"][1], name = plot_name, underflow = True, overflow = True))
        hist_Stau_100       = Hist(hist.axis.Regular(plot_settings[plot_name]["binning_linspace"][-1] - 1, plot_settings[plot_name]["binning_linspace"][0], plot_settings[plot_name]["binning_linspace"][1], name = plot_name, underflow = True, overflow = True))
        hist_Wto2Q     = Hist(hist.axis.Regular(plot_settings[plot_name]["binning_linspace"][-1] - 1, plot_settings[plot_name]["binning_linspace"][0], plot_settings[plot_name]["binning_linspace"][1], name = plot_name, underflow = True, overflow = True))
        hist_WtoLNu    = Hist(hist.axis.Regular(plot_settings[plot_name]["binning_linspace"][-1] - 1, plot_settings[plot_name]["binning_linspace"][0], plot_settings[plot_name]["binning_linspace"][1], name = plot_name, underflow = True, overflow = True))
        hist_EWK       = Hist(hist.axis.Regular(plot_settings[plot_name]["binning_linspace"][-1] - 1, plot_settings[plot_name]["binning_linspace"][0], plot_settings[plot_name]["binning_linspace"][1], name = plot_name, underflow = True, overflow = True))
        hist_TT        = Hist(hist.axis.Regular(plot_settings[plot_name]["binning_linspace"][-1] - 1, plot_settings[plot_name]["binning_linspace"][0], plot_settings[plot_name]["binning_linspace"][1], name = plot_name, underflow = True, overflow = True))
        hist_singleT   = Hist(hist.axis.Regular(plot_settings[plot_name]["binning_linspace"][-1] - 1, plot_settings[plot_name]["binning_linspace"][0], plot_settings[plot_name]["binning_linspace"][1], name = plot_name, underflow = True, overflow = True))
        hist_Top   = Hist(hist.axis.Regular(plot_settings[plot_name]["binning_linspace"][-1] - 1, plot_settings[plot_name]["binning_linspace"][0], plot_settings[plot_name]["binning_linspace"][1], name = plot_name, underflow = True, overflow = True))
        hist_WJets   = Hist(hist.axis.Regular(plot_settings[plot_name]["binning_linspace"][-1] - 1, plot_settings[plot_name]["binning_linspace"][0], plot_settings[plot_name]["binning_linspace"][1], name = plot_name, underflow = True, overflow = True))
        hist_QCD       = Hist(hist.axis.Regular(plot_settings[plot_name]["binning_linspace"][-1] - 1, plot_settings[plot_name]["binning_linspace"][0], plot_settings[plot_name]["binning_linspace"][1], name = plot_name, underflow = True, overflow = True))
        #hist_Data_Muon      = np.zeros(len(binning)-1)
        #hist_Data_None      = np.zeros(len(binning)-1)
        #hist_Data_E      = np.zeros(len(binning)-1)
        #hist_Data_F      = np.zeros(len(binning)-1)
        #hist_Data_G      = np.zeros(len(binning)-1)
    
    hists_to_plot = []
    data_hists = []
    data_muon_hists = []
    data_none_hists = []
    labels = []
    data_labels = []

    signal_hists = {}
    for mass in signal_mass:
        signal_hists[mass] = {}
        for lifetime in signal_lifetime:
            signal_hists[mass][lifetime] = Hist(hist.axis.Regular(plot_settings[plot_name]["binning_linspace"][-1] - 1, plot_settings[plot_name]["binning_linspace"][0], plot_settings[plot_name]["binning_linspace"][1], name = plot_name, underflow= True, overflow = True))

    for process, histogram in histograms.items():
#         print('INFO: Now looking at process', process, '...')
        #histogram = np.asarray(histogram)
        #histogram = histogram.flatten()
        histogram = sum(histogram)
        # Fix later, hist should not be a list in the first place and should be 60 1d and not (60,1)
        # if args.groupProcesses:
        if groupProcesses:
            if process in all_samples_dict['TT']:
                hist_TT += histogram
                hist_Top += histogram
            elif process in all_samples_dict['singleT']:
                hist_singleT += histogram
                hist_Top += histogram
            elif process in all_samples_dict['DY']:
                hist_EWK += histogram
            #elif process in all_samples_dict['DYTau_0']:
            #    hist_DYTau_0 += histogram
            #elif process in all_samples_dict['DYTau_1']:
            #    hist_DYTau_1 += histogram
            #elif process in all_samples_dict['DYTau_2']:
            #    hist_DYTau_2 += histogram
            elif process in all_samples_dict['WtoLNu']:
                hist_WtoLNu += histogram
                hist_WJets += histogram
            elif process in all_samples_dict['Wto2Q']:
                hist_Wto2Q += histogram
                hist_WJets += histogram
            elif process in all_samples_dict['QCD']:
                hist_QCD += histogram
            #elif process in all_samples_dict['Stau']:
            #    hist_Sig += histogram
            elif process in all_samples_dict['Stau_500']:
                signal_hists['500'][process.split('_')[-1]] = histogram
            elif process in all_samples_dict['Stau_300']:
                signal_hists['300'][process.split('_')[-1]] = histogram
            elif process in all_samples_dict['Stau_200']:
                signal_hists['200'][process.split('_')[-1]] = histogram
            elif process in all_samples_dict['Stau_100']:
                signal_hists['100'][process.split('_')[-1]] = histogram
        else:
            if process == "Data2018C":
                data_hist = histogram
            elif process == "DYJetsToLL":
                hists_to_plot.append(histogram)
                labels.append(process)
            else:
                hists_to_plot.append(histogram)
                labels.append(process)

    #sum_hist_Wto2Q   = np.sum(hist_Wto2Q, axis = 0) 
    #sum_hist_WtoLNu  = np.sum(hist_WtoLNu, axis = 0)
    #sum_hist_EWK     = np.sum(hist_EWK, axis = 0)
    #sum_hist_TT      = np.sum(hist_TT, axis = 0)
    #sum_hist_singleT = np.sum(hist_singleT, axis = 0)
    #sum_hist_QCD     = np.sum(hist_QCD, axis = 0)

    # print (hist_EWK)
    if groupProcesses:
     #if args.groupProcesses:
        if normalize == True:
            hists_to_plot.append(hist_Wto2Q/sum_hist_Wto2Q)
            labels.append('Wto2Q')
            hists_to_plot.append(hist_WtoLNu/sum_hist_WtoLNu)
            labels.append('WtoLNu')
            hists_to_plot.append(hist_singleT/sum_hist_singleT)
            labels.append('singleT')
            hists_to_plot.append(hist_TT/sum_hist_TT)
            labels.append('TT')
            hists_to_plot.append(hist_EWK/sum_hist_EWK)
            labels.append('DYJetsToLL')
            hists_to_plot.append(hist_QCD/sum_hist_QCD)
            labels.append('QCD')
        else:
            hists_to_plot.append(hist_Wto2Q)
            labels.append('Wto2Q')
            hists_to_plot.append(hist_WtoLNu)
            labels.append('WtoLNu')
            hists_to_plot.append(hist_singleT)
            labels.append('singleT')
            hists_to_plot.append(hist_TT)
            labels.append('TT')
            #hists_to_plot.append(hist_Top)
            #labels.append('singleT + TT')
            hists_to_plot.append(hist_EWK)
            labels.append('DYJetsToLL')
            hists_to_plot.append(hist_QCD)
            labels.append('QCD')
            

    colours = [ "#6DD3CE", "#127475", "#FF99C9", "#FECDAA", "#06D6A0", "#B30089"]
    for mass in signal_mass:
        colour = 0
        fig, ax_main = plt.subplots(1, 1, sharex=True)
        hep.histplot(hists_to_plot,  histtype='fill', 
                 stack=do_stack,
                 flow = 'sum',
                 label=labels, #color=colours, #sort='label_r', 
                 density=plot_settings[plot_name].get("density"), ax=ax_main)
        for lifetime in signal_lifetime:
            if normalize == True:
                hep.histplot(signal_hists[mass][lifetime]/np.sum(signal_hists[mass][lifetime], axis = 0),  histtype='step', flow = 'sum', label=f'Stau_{mass}_{lifetime}', color=colours[colour], ax=ax_main)
            else:
                hep.histplot(signal_hists[mass][lifetime],  histtype='step', flow = 'sum', label=f'Stau_{mass}_{lifetime}', color=colours[colour], ax=ax_main)
            colour += 1
        ax_main.set_ylabel(plot_settings[plot_name].get("ylabel"))
        ax_main.set_xlabel(plot_settings[plot_name].get("xlabel"), usetex=False)
        ax_main.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop = {"size":16})
        
        # this part is still to be done 
        #if groupProcesses:
        ## # # if args.groupProcesses:
        #    sum_histogram = np.sum(np.asarray(hists_to_plot), axis=0)
        #    sum_data_histogram = np.sum(np.asarray(data_hists), axis=0)
        #    ratio_hist = sum_data_histogram / (sum_histogram + np.finfo(float).eps)
        ## #     # Adding relative sqrtN Poisson uncertainty for now, should be improved when using the hist package
        #    rel_unc = np.sqrt(sum_data_histogram) / sum_data_histogram
        #    rel_unc *= ratio_hist
        #    rel_unc[rel_unc < 0] = 0 # Not exactly sure why we have negative values, but this solves it for the moment
        #    hep.histplot(ratio_hist, bins=binning, histtype='errorbar', yerr=rel_unc, color='black', label='Ratio', ax=ax_ratio)
        #    ax_ratio.axhline(1, color='gray', linestyle='--')
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
        plt.subplots_adjust(right=0.8)
        filedir = "W_CR_signal"
        if filedir not in os.listdir('plots/'):
            os.mkdir(f'plots/{filedir}')
        filename = f"./plots/{filedir}/{dataset_name}_{plot_name}_{mass}"
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
        #plt.clf()
