import os, sys
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


NanoAODSchema.warn_missing_crossrefs = False
NanoAODSchema.mixins["DisMuon"] = "Muon"
hep.style.use("CMS")

nanov = 'Summer22_CHS_v19/'
# nanov = ''
#sample_folder = f"/eos/uscms/store/user/dally/skim/{nanov}prompt_mutau/v8_nobJetVeto/selected/faster_trial/"
#sample_folder = f"/eos/uscms/store/user/dally/skim/{nanov}mutau/v3_CorrectedJet/selected/faster_trial/"
#sample_folder = f"/eos/uscms/store/group/lpcdisptau/dally/displacedTaus/selected/Summer22_CHS_v10/mutau/v3/QCD_CR/faster_trial/"
sample_folder = f"/eos/uscms/store/user/lpcdisptau/dally/displacedTaus/selected/Summer22_CHS_v19/mutau/v8/PR_score/faster_trial/"
normalize = False

#if "PR" in sample_folder:
#    normalize = True
import awkward as ak
import numpy as np

import awkward as ak
import numpy as np

## load xsec and br settings from JSON file
with open("./plots_config/xsec_and_br.json", "r") as file:
    xsec_and_br = json.load(file)
xsec = xsec_and_br['xsec']
br = xsec_and_br['br']

## load sum_gen_w from another JSON file
with open(f"./plots_config/{nanov}/mutau/v8/all_total_sumw.json", "r") as file:
    sum_gen_w = json.load(file)
target_lumi = 27.6

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
       "TWminustoLNu2Q",
      "TBbarQ_t-channel_4FS",
      "TbarBQ_t-channel_4FS",
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
}
## build reverse lookup dict to be used when retrieving sum gen events etc
reverse_samples_lookup = {
    subsample: key
    for key, subs in all_samples_dict.items()
    for subsample in subs
}

available_processes = [sub for subs in all_samples_dict.values() for sub in subs]

ISOLATION = ["Rel. Tracker Iso < 0.075", "Rel. PF Iso < 0.15", "Rel. PF Iso < 0.15 & Trk Iso < 0.075"]

binning = [0, 10, 101] 


# Directory creation for plots (directory_path = "plots/" + args.process)
directory_path = "plots/"
if not os.path.exists(directory_path):
    os.makedirs(directory_path)
    print(f"Directory '{directory_path}' created successfully.")


d0_hists = {}
d0_hists["nominal"] = hist.Hist(hist.axis.Regular(100, 0, 10, name = "d0", underflow = True, overflow = True), storage = hist.storage.Weight())
for iso in ISOLATION:
    d0_hists[iso] = hist.Hist(hist.axis.Regular(100, 0, 10, name = "d0", underflow = True, overflow = True), storage = hist.storage.Weight())

d0_eff_hists = {}
for process in available_processes:
    print(f"Starting {process}")
    tmp_string = f"faster_trial_{process}/faster_trial_{process}.root"
    tmp_file = sample_folder +  tmp_string
    events = NanoEventsFactory.from_root({tmp_file:"Events"}, schemaclass= NanoAODSchema).events()

    d0_hists["nominal"].fill(ak.flatten(abs(events.DisMuon.dxy), axis = None))

    

    for iso in ISOLATION:
        if "Tracker" in iso:
            events_tracker = events[ak.flatten(abs(events.DisMuon.tkRelIso) < 0.075, axis = None)]
            d0_hists[iso].fill(ak.flatten(abs(events_tracker.DisMuon.dxy), axis = None))
        if "PF Iso" in iso and "Trk" not in iso:
            events_pf = events[ak.flatten(abs(events.DisMuon.pfRelIso04_all) < 0.15, axis = None)]
            d0_hists[iso].fill(ak.flatten(abs(events_pf.DisMuon.dxy), axis = None))
        if "Trk" in iso:
            events_both = events[ak.flatten(abs(events.DisMuon.pfRelIso04_all) < 0.15, axis = None) & ak.flatten(abs(events.DisMuon.tkRelIso) < 0.075, axis = None)]
            d0_hists[iso].fill(ak.flatten(abs(events_both.DisMuon.dxy), axis = None))

    

for iso in ISOLATION:
    d0_eff_hists[iso] = hep.comp.get_comparison(d0_hists[iso], d0_hists["nominal"], comparison = 'efficiency')


fig, ax = plt.subplots(1, 1, sharex=True)
    
colours = ["#6DD3CE", "#127475", "#FF99C9", "#FECDAA", "#06D6A0",
          "#B30089", "#8338EC", "#FFD166", "#EF476F", "#2D2D2D"]


color_idx = 0
for iso in ISOLATION:
    ax.errorbar(d0_hists["nominal"].axes[0].centers, d0_eff_hists[iso][0], yerr=[d0_eff_hists[iso][1], d0_eff_hists[iso][-1]], color = colours[color_idx], label = iso, fmt='o', linestyle='none')
    color_idx += 1
#ax.set_yscale('log')
ax.legend(fontsize="x-small")
#ax.set_ylim(1E-4, 1E3)
ax.set_ylabel("Efficiency")
ax.set_xlabel(r"Muon |$d_0$| [cm]") 
hep.cms.label("Private Work", data=True, loc=0, com=13.6, ax=ax, fontsize = "x-small")
fig.savefig("muonIsoEff.pdf")
fig.savefig("muonIsoEff.png")

