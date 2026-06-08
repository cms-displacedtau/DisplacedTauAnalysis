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
from coffea.lumi_tools import LumiMask
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema, PFNanoAODSchema
import pdb, glob

print(hep.__version__)

PFNanoAODSchema.warn_missing_crossrefs = False
PFNanoAODSchema.mixins["DisMuon"] = "Muon"
PFNanoAODSchema.mixins["CorrectedJet"] = "Jet"

hep.style.use("CMS")

nanov = 'Summer22_CHS_v19/'
# nanov = ''
#sample_folder = f"/eos/uscms/store/user/dally/skim/{nanov}prompt_mutau/v8/selected/faster_trial/"
#sample_folder = f"/eos/uscms/store/user/dally/skim/{nanov}mutau/v3/selected/W_CR/faster_trial/"
#sample_folder = f"/eos/uscms/store/group/lpcdisptau/dally/displacedTaus/selected/Summer22_CHS_v10/mutau/v4/SR/faster_trial/"
#sample_folder = "/eos/uscms/store/group/lpcdisptau/dally/displacedTaus/selected/Summer22_CHS_v10/mutau/v5/TTMinusB_CR_Mu50_noScore/faster_trial/"
#sample_folder = "/eos/uscms/store/group/lpcdisptau/dally/displacedTaus/selected/Summer22_CHS_v10/mutau/v5/TTMinusB_CR_IsoMu50_noScore_PromptMuon/faster_trial/"
#sample_folder = "/eos/uscms/store/group/lpcdisptau/dally/displacedTaus/selected/Summer22_CHS_v10/mutau/v4/TTMinusB_CR_MET120_IsoTrk50_noScore/faster_trial/"
#sample_folder = "/eos/uscms/store/group/lpcdisptau/dally/displacedTaus/selected/Summer22_CHS_v10/mutau/v4/TTMinusB_CR_PFMET120_PFMHT120_IDTight_PFHT60_noScore/faster_trial/"
#sample_folder = "/eos/uscms/store/group/lpcdisptau/dally/displacedTaus/selected/Summer22_CHS_v10/mutau/v4/TTMinusB_CR_MET120_IsoTrk50_noScore_withJetMapVeto/faster_trial/"
#sample_folder = "/eos/uscms/store/group/lpcdisptau/dally/displacedTaus/selected/Summer22_CHS_v10/mutau/v6/TTMinusB_CR_Mu50_noScore_JetDxy0p1_MuDxy0p045/faster_trial/"
#sample_folder = "/eos/uscms/store/group/lpcdisptau/dally/displacedTaus/selected/Summer22_CHS_v10/pmutau/v7/PR_Mu50_LeadingPtJet_PackedSelection_PromptMuon/faster_trial/"
#sample_folder = "/eos/uscms/store/group/lpcdisptau/dally/displacedTaus/selected/Summer22_CHS_v10/mutau/v6/PR_Mu50_LeadingPtJet_PackedSelection/faster_trial/"
#sample_folder = f"/eos/uscms/store/group/lpcdisptau/dally/displacedTaus/selected/Summer22_CHS_v10/mutau/v4/PR/faster_trial/"
#sample_folder = f"/eos/uscms/store/group/lpcdisptau/dally/displacedTaus/selected/Summer22_CHS_v10/mutau/v4/TTMinusB_CR_MET120_IsoTrk50_noScore_JetDxy0p1/faster_trial/"
sample_folder = "/eos/uscms/store/user/lpcdisptau/dally/displacedTaus/selected/Summer22_CHS_v19/mutau/v8/PR_score/faster_trial/"
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
      #"JetMET_Run2022E",
      "JetMET_Run2022F",
      #"JetMET_Run2022G",
      ], 
    "Muon": [
        #"Muon_Run2022E",
        #"Muon_Run2022F",
        #"Muon_Run2022G",
    ],
    "Stau" : [],
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
#target_lumi = 26.7
## For era 2022 F
target_lumi = 17.78

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
    print(process)
    tmp_string = f"faster_trial_{process}/faster_trial_{process}.root"
    tmp_file = sample_folder +  tmp_string
    events = NanoEventsFactory.from_root({tmp_file:"Events"}, schemaclass= PFNanoAODSchema).events()
    events = events[ak.flatten(abs(events.CorrectedJet.dxy) < 999)]
    events = events[ak.flatten(abs(events.CorrectedJet.dxyErr) < 999)]
    events = events[ak.flatten(events.CorrectedJet.pt >= 200)]


    ##target = "MET120_IsoTrk50"
    #target = "PFMET120_PFMHT120_IDTight_PFHT60"
    #pass_target = events.HLT[target]
    #fail_others = ak.ones_like(pass_target, dtype=bool)

    ##print("Number of events that pass MET105_IsoTrk50", len(events[events["HLT"]["MET105_IsoTrk50"]])) 
    #for trig in events.HLT.fields:
    #    if trig == target:
    #        continue
    #    #if "MET105_IsoTrk50" in trig:
    #    #    continue
    #    fail_others = fail_others & (~events.HLT[trig])
    #    print(f"For {trig}, the number of events that fail is {sum(fail_others)}")

    #events = events[fail_others]

    

    #events = events[ak.ravel(abs(events.CorrectedJet.eta) > 1.4)]
    #events = events[ak.ravel(abs(events.DisMuon.eta) <= 1.4)]
    #events = events[ak.ravel(events.DisMuon.pt > 50)]
    #events = events[events.CorrectedPFMET.pt > 105]
    #events = events[ak.ravel(abs(events.DisMuon.dxy) <  0.005)]
    #events = events[ak.ravel(abs(events.CorrectedJet.dxy) <  0.02)]
    
    
    #events["Jet"] = events["Jet"][events["CorrectedJet"]["jetId"] == True]    
    #events["Jet"] = events["Jet"][abs(events["CorrectedJet"]["dxy"]) < 0.02]
    #events["Jet"] = events["Jet"][abs(events["Jet"]["pt"]) > ]

    #loose_bjets  = events["Jet"][events["CorrectedJet"]["btagPNetB"] > 0.0499]
    #medium_bjets = events["Jet"][events["CorrectedJet"]["btagPNetB"] > 0.2605]
    #tight_bjets  = events["Jet"][events["CorrectedJet"]["btagPNetB"] > 0.6915]
    #num_corrected_jets = ak.count_nonzero(events["Jet"]["pt"], axis = 1)
    #num_loose_bjets = ak.count_nonzero(loose_bjets["pt"], axis = 1)
    #num_medium_bjets = ak.count_nonzero(medium_bjets["pt"], axis = 1)
    #num_tight_bjets = ak.count_nonzero(tight_bjets["pt"], axis = 1)

    #events["Jet"] = ak.with_field(events.Jet, num_corrected_jets, "numJets")
    #events["Jet"] = ak.with_field(events.Jet, num_loose_bjets, "numLooseBJets")
    #events["Jet"] = ak.with_field(events.Jet, num_medium_bjets, "numMediumBJets")
    #events["Jet"] = ak.with_field(events.Jet, num_tight_bjets, "numTightBJets")
    #mutau_dr = events.Muon.metric_table(events.Tau)
    #mutau_pt = events.Muon.pt + events.Tau.pt

    #met = events.CorrectedPuppiMET.pt
    #met_phi = events.CorrectedPuppiMET.phi
    #dphi = abs(events.Muon.phi - met_phi)
    #dphi = np.where(dphi > np.pi, 2*np.pi - dphi, dphi)
    #mT = np.sqrt(2 * events.Muon.pt * met * (1 - np.cos(dphi)))
    #events["Muon"] = ak.with_field(events.Muon, mT, "mT")

    dphi = abs(events.CorrectedJet.phi - events.CorrectedPFMET.phi)
    dphi = np.where(dphi > np.pi, 2*np.pi - dphi, dphi)
    events["CorrectedJet"] = ak.with_field(events.CorrectedJet, dphi, "dphi")
    #events["mutau"] = ak.with_field(events.mutau, mutau_dr, where = 'dR')
    #events["mutau"] = ak.with_field(events.mutau, mutau_pt, where = 'pt') 

    #mu_dphi = abs(events.CorrectedJet.phi - events.DisMuon.phi)
    #mu_dphi = np.where(dphi > np.pi, 2*np.pi - dphi, dphi)
    #events["DisMuon"] = ak.with_field(events.DisMuon, mu_dphi, "dphi")
    #deta = abs(events.CorrectedJet.eta - events.DisMuon.eta)
    #events["DisMuon"] = ak.with_field(events.DisMuon, deta, "deta")
    #dR = events.DisMuon.metric_table(events.CorrectedJet)
    #dR = ak.flatten(dR)
    #events["DisMuon"] = ak.with_field(events.DisMuon, dR, "dR")
   # mu_dphi = abs(events.CorrectedJet.phi - events.DisMuon.phi)
   # mu_dphi = np.where(dphi > np.pi, 2*np.pi - dphi, dphi)
   # events["DisMuon"] = ak.with_field(events.DisMuon, mu_dphi, "dphi")
   # deta = abs(events.CorrectedJet.eta - events.DisMuon.eta)
   # events["DisMuon"] = ak.with_field(events.DisMuon, deta, "deta")
   # dR = events.DisMuon.metric_table(events.CorrectedJet)
   # dR = ak.flatten(dR)
   # events["DisMuon"] = ak.with_field(events.DisMuon, dR, "dR")
   # dxySig = events.DisMuon.dxy/events.DisMuon.dxyErr
   # dzSig  = events.DisMuon.dz/events.DisMuon.dzErr
   # events["DisMuon"] = ak.with_field(events.DisMuon, dxySig, "dxySig")
   # events["DisMuon"] = ak.with_field(events.DisMuon, dzSig, "dzSig")

    #if "muon" not in process.lower():
    #    events["PuppiMET"] = events.CorrectedPuppiMET
    ## disable for now
#     print ('need to put back weights')

    #pt_raw = (1 - events.Jet.rawFactor) * events.Jet.pt
    #events["Jet"] = ak.with_field(events.Jet, pt_raw, "ptRaw")
    dxySig = events.CorrectedJet.dxyErr/events.CorrectedJet.dxy
    events["CorrectedJet"] = ak.with_field(events.CorrectedJet, dxySig, "dxySig")

    dxySigInv = 1/events.CorrectedJet.dxySig
    events["CorrectedJet"] = ak.with_field(events["CorrectedJet"], dxySigInv, "dxySigInv") 

    weights = events.run / events.run
    if "jetmet" in process.lower() or "muon" in process.lower():
        weights = weights
    else:
        lumi_weight = target_lumi * xsec[process] * br[process] * 1000 / sum_gen_w[process]
#         lumi_weight = target_lumi * xsec[process] * br[process] * 1000 / sum_gen_w[reverse_samples_lookup[process]][process]
        weights = events.weight * lumi_weight #* events.Muon.eventWeight ## am i missing the sumGenW here?

    for plot_name, settings in plot_settings.items():
        if "weight" in plot_name: continue
        if "Jet_eta" not in plot_name: continue
        if "eventWeight" in plot_name: continue
        if "correction" in plot_name: continue
        if "resolution" in plot_name: continue
        if "PFCands" in plot_name:continue
        if "cut" in plot_name: continue
        if not settings["per_event"]:
            var_values = getattr(getattr(events, settings["field"]), settings["variable"])
            vals_flat = ak.flatten(var_values)
            weights_broadcast = ak.broadcast_arrays(var_values, weights)[1]
            weights_flat = ak.flatten(weights_broadcast)
            histo = Hist(hist.axis.Regular(settings["binning_linspace"][-1] - 1, settings["binning_linspace"][0], settings["binning_linspace"][1], name = plot_name, underflow = True, overflow = True), storage=hist.storage.Weight())
            histo.fill(vals_flat, weight=weights_flat)

        else:
            if settings["variable"] == "":
                var_values = getattr(events, settings["field"])
                weights_broadcast = ak.broadcast_arrays(var_values, weights)[1]
                histo = Hist(hist.axis.Regular(settings["binning_linspace"][-1] - 1, settings["binning_linspace"][0], settings["binning_linspace"][1], name = plot_name, underflow = True, overflow = True), storage=hist.storage.Weight())
                histo.fill(var_values, weight=weights_broadcast)
            else:
                var_values = getattr(getattr(events, settings["field"]), settings["variable"])
                weights_broadcast = ak.broadcast_arrays(var_values, weights)[1]
                histo = Hist(hist.axis.Regular(settings["binning_linspace"][-1] - 1, settings["binning_linspace"][0], settings["binning_linspace"][1], name = plot_name, underflow = True, overflow = True), storage=hist.storage.Weight())
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
        hist_Sig       = Hist(hist.axis.Regular(plot_settings[plot_name]["binning_linspace"][-1] - 1, plot_settings[plot_name]["binning_linspace"][0], plot_settings[plot_name]["binning_linspace"][1], name = plot_name, underflow = True, overflow = True), storage=hist.storage.Weight())
        hist_Wto2Q     = Hist(hist.axis.Regular(plot_settings[plot_name]["binning_linspace"][-1] - 1, plot_settings[plot_name]["binning_linspace"][0], plot_settings[plot_name]["binning_linspace"][1], name = plot_name, underflow = True, overflow = True), storage=hist.storage.Weight())
        hist_WtoLNu    = Hist(hist.axis.Regular(plot_settings[plot_name]["binning_linspace"][-1] - 1, plot_settings[plot_name]["binning_linspace"][0], plot_settings[plot_name]["binning_linspace"][1], name = plot_name, underflow = True, overflow = True), storage=hist.storage.Weight())
        hist_DY        = Hist(hist.axis.Regular(plot_settings[plot_name]["binning_linspace"][-1] - 1, plot_settings[plot_name]["binning_linspace"][0], plot_settings[plot_name]["binning_linspace"][1], name = plot_name, underflow = True, overflow = True), storage=hist.storage.Weight())
        hist_TT        = Hist(hist.axis.Regular(plot_settings[plot_name]["binning_linspace"][-1] - 1, plot_settings[plot_name]["binning_linspace"][0], plot_settings[plot_name]["binning_linspace"][1], name = plot_name, underflow = True, overflow = True), storage=hist.storage.Weight())
        hist_singleT   = Hist(hist.axis.Regular(plot_settings[plot_name]["binning_linspace"][-1] - 1, plot_settings[plot_name]["binning_linspace"][0], plot_settings[plot_name]["binning_linspace"][1], name = plot_name, underflow = True, overflow = True), storage=hist.storage.Weight())
        hist_Top   = Hist(hist.axis.Regular(plot_settings[plot_name]["binning_linspace"][-1] - 1, plot_settings[plot_name]["binning_linspace"][0], plot_settings[plot_name]["binning_linspace"][1], name = plot_name, underflow = True, overflow = True), storage=hist.storage.Weight())
        hist_WJets   = Hist(hist.axis.Regular(plot_settings[plot_name]["binning_linspace"][-1] - 1, plot_settings[plot_name]["binning_linspace"][0], plot_settings[plot_name]["binning_linspace"][1], name = plot_name, underflow = True, overflow = True), storage=hist.storage.Weight())
        hist_QCD       = Hist(hist.axis.Regular(plot_settings[plot_name]["binning_linspace"][-1] - 1, plot_settings[plot_name]["binning_linspace"][0], plot_settings[plot_name]["binning_linspace"][1], name = plot_name, underflow = True, overflow = True), storage=hist.storage.Weight())
        hist_EWK       = Hist(hist.axis.Regular(plot_settings[plot_name]["binning_linspace"][-1] - 1, plot_settings[plot_name]["binning_linspace"][0], plot_settings[plot_name]["binning_linspace"][1], name = plot_name, underflow = True, overflow = True), storage=hist.storage.Weight())
        hist_VBF       = Hist(hist.axis.Regular(plot_settings[plot_name]["binning_linspace"][-1] - 1, plot_settings[plot_name]["binning_linspace"][0], plot_settings[plot_name]["binning_linspace"][1], name = plot_name, underflow = True, overflow = True), storage=hist.storage.Weight())
        hist_Data      = Hist(hist.axis.Regular(plot_settings[plot_name]["binning_linspace"][-1] - 1, plot_settings[plot_name]["binning_linspace"][0], plot_settings[plot_name]["binning_linspace"][1], name = plot_name, underflow = True, overflow = True), storage=hist.storage.Weight())
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

    for process, histogram in histograms.items():
#         print('INFO: Now looking at process', process, '...')
        #histogram = np.asarray(histogram)
        #histogram = histogram.flatten()
        histogram = sum(histogram)
        # Fix later, hist should not be a list in the first place and should be 60 1d and not (60,1)
        # if args.groupProcesses:
        if groupProcesses:
            #if process in all_samples_dict["JetMET_Run2022E"]:
            #    hist_Data_E += histogram
            #elif process in all_samples_dict["JetMET_Run2022F"]:
            #    hist_Data_F += histogram
            #eliif process in all_samples_dict["JetMET_Run2022G"]:
            #    hist_Data_G += histogram
            if process in all_samples_dict['JetMET']:
               hist_Data += histogram
            if process in all_samples_dict['Muon']:
               hist_Data += histogram
            #if process in all_samples_dict['JetMET_Muon']:
            #    hist_Data_Muon += histogram
            elif process in all_samples_dict['TT']:
                hist_TT += histogram
                hist_Top += histogram
            elif process in all_samples_dict['singleT']:
                hist_singleT += histogram
                hist_Top += histogram
            elif process in all_samples_dict['DY']:
                hist_DY += histogram
            elif process in all_samples_dict['WtoLNu']:
                hist_WtoLNu += histogram
                hist_WJets += histogram
            elif process in all_samples_dict['Wto2Q']:
                hist_Wto2Q += histogram
                hist_WJets += histogram
            elif process in all_samples_dict['EWK']:
                hist_EWK += histogram
            elif process in all_samples_dict['VBF']:
                hist_VBF += histogram
            elif process in all_samples_dict['QCD']:
                hist_QCD += histogram
            #elif process in all_samples_dict['Stau']:
            #    hist_Sig += histogram
        else:
            if process == "Data2018C":
                data_hist = histogram
            elif process == "DYJetsToLL":
                hists_to_plot.append(histogram)
                labels.append(process)
            else:
                hists_to_plot.append(histogram)
                labels.append(process)

    sum_hist_Wto2Q   = np.sum(hist_Wto2Q.values(flow=True)) 
    sum_hist_WtoLNu  = np.sum(hist_WtoLNu.values(flow=True))
    sum_hist_DY      = np.sum(hist_DY.values(flow=True))
    sum_hist_TT      = np.sum(hist_TT.values(flow=True))
    sum_hist_singleT = np.sum(hist_singleT.values(flow=True))
    sum_hist_VBF     = np.sum(hist_VBF.values(flow=True))
    sum_hist_EWK     = np.sum(hist_EWK.values(flow=True))
    sum_hist_QCD     = np.sum(hist_QCD.values(flow=True))
    sum_hist_Data    = np.sum(hist_Data.values(flow=True))
    sum_hist_bg    = np.sum([sum_hist_Wto2Q, sum_hist_WtoLNu, sum_hist_DY, sum_hist_TT, sum_hist_singleT, sum_hist_VBF, sum_hist_EWK, sum_hist_QCD])
    #sum_hist_bg    = np.sum([sum_hist_Wto2Q, sum_hist_WtoLNu, sum_hist_DY, sum_hist_TT, sum_hist_singleT, sum_hist_QCD])

    # print (hist_EWK)
    if groupProcesses:
     #if args.groupProcesses:
        hists_to_plot.append(hist_Wto2Q)
        labels.append(f'Wto2Q: %.2f'%sum_hist_Wto2Q)
        hists_to_plot.append(hist_WtoLNu)
        labels.append(f'WtoLNu: %.2f' %sum_hist_WtoLNu)
        hists_to_plot.append(hist_singleT)
        labels.append(f'singleT: %.2f' %sum_hist_singleT)
        hists_to_plot.append(hist_TT)
        labels.append(f'TT: %.2f' %sum_hist_TT)
        hists_to_plot.append(hist_DY)
        labels.append(f'DY: %.2f' %sum_hist_DY)
        hists_to_plot.append(hist_QCD)
        labels.append(f'QCD: %.2f' %sum_hist_QCD)
        hists_to_plot.append(hist_VBF)
        labels.append(f'VBF: %.2f' %sum_hist_VBF)
        hists_to_plot.append(hist_EWK)
        labels.append(f'EWK: %.2f' %sum_hist_EWK)
        #hists_to_plot.append(hist_DYTau_0)
        #labels.append('DYtoTauTau_0Jets')
        #hists_to_plot.append(hist_DYTau_1)
        #labels.append('DYtoTauTau_1Jets')
        #hists_to_plot.append(hist_DYTau_2)
        #labels.append('DYtoTauTau_2Jets')
        #hists_to_plot.append(hist_Top)
        #labels.append('tt + singlet')
        #hists_to_plot.append(hist_WJets)
        #labels.append('W to jets')
        #hists_to_plot.append(hist_Sig)
        #labels.append('signal')
        data_hists.append(hist_Data)
        #data_muon_hists.append(hist_Data_Muon)
        #data_hists.append(hist_Data_E)
        #data_labels.append('E')
        #data_hists.append(hist_Data_F)
        #data_labels.append('F')
        #data_hists.append(hist_Data_G)
        #data_labels.append('G')

    colours = ["#5790fc", "#f89c20", "#e42536", "#964a8b", "#9c9ca1", "#7a21dd", "#FF99C9", "#C8E9A0",]# "#6DD3CE",] #"#127475", "#FF99C9"]

    fig, (ax_main, ax_ratio) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]})
    #fig, ax_main = plt.subplots(1, 1, sharex=True)
    fig.subplots_adjust(hspace=0.0)
    
    hep.comp.data_model(data_hist = hist_Data, stacked_components = hists_to_plot, stacked_labels = labels, stacked_colors = colours, xlabel = plot_settings[plot_name].get("xlabel"), ylabel = plot_settings[plot_name].get("ylabel"), data_label='data: %.2f'%sum_hist_Data, flow = 'sum', fig = fig, ax_main = ax_main, ax_comparison = ax_ratio, comparison='split_ratio')

    ax_ratio.axhline(1, color='gray', linestyle='--')
    ax_main.xaxis.set_major_locator(MultipleLocator(plot_settings[plot_name].get("x_major_ticks")))
    ax_main.xaxis.set_minor_locator(MultipleLocator(plot_settings[plot_name].get("x_minor_ticks")))
    ax_ratio.xaxis.set_major_locator(MultipleLocator(plot_settings[plot_name].get("x_major_ticks")))
    ax_ratio.xaxis.set_minor_locator(MultipleLocator(plot_settings[plot_name].get("x_minor_ticks")))
    ax_ratio.xaxis.set_major_formatter(ticker.ScalarFormatter())

    ax_ratio.set_ylabel('Data / MC')
    #ax_ratio.set_xlim(binning[0], binning[-1])
    ax_ratio.set_ylim(0.5, 2.)
    #ax_ratio.set_xlabel(plot_settings[plot_name].get("xlabel"), usetex=False)
   
    ax_ratio.xaxis.set_tick_params(labelbottom=True)
    plt.setp(ax_ratio.get_xticklabels(), visible=True)    

    # Decorating with CMS label
    txt = hep.cms.label(f'Total Integral: %.2f Private Work'%sum_hist_bg, data=True, loc=0, label="Private Work", com=13.6, lumi=round(target_lumi, 1), ax=ax_main, fontsize = "x-small")
    #hep.append_text(f'Total Integral: %.2f'%sum_hist_bg, txt, loc = "below")
    #hep.cms.add_text(r'$|\eta|$ > 1.2', loc = 'upper left')
    #print("x tick locations:", ax_ratio.get_xticks())
    #print("x tick labels:", [t.get_text() for t in ax_ratio.get_xticklabels()])
    #print("x tick label visible:", [t.get_visible() for t in ax_ratio.get_xticklabels()])
    #print("labelbottom:", ax_ratio.xaxis.get_tick_params()['labelbottom'])
    #print("sharex partner:", ax_ratio.get_shared_x_axes().get_siblings(ax_ratio)) 
    # Saving with special name
    #filename = f"/eos/uscms/store/user/dally/DisplacedTauAnalysis/plots/{dataset_name}_{plot_name}"
    filedir = "PR_score_JetPt200"
    if filedir not in os.listdir('plots/'):
        os.mkdir(f'plots/{filedir}')
    #if filedir in os.listdir('plots/'):
    #    if 'endcap' not in os.listdir(f'plots/{filedir}'):
    #        os.mkdir(f'plots/{filedir}/endcap')
    filename = f"./plots/{filedir}/{dataset_name}_{plot_name}"
    # #if args.groupProcesses:
    if plot_settings[plot_name].get("density"):
        filename += "_normalized"
    else: 
        filename += "_stacked"
    #filename += "_etaleq1p4.pdf"
    #filename += "_etag1p4.pdf"
    #filename += "_PFMET105.pdf"
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
