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

def get_distinct_parent(events, gen_particles, max_depth=20):
    """
    Replicates distinctParent behavior without the memory blow-up.
    Walks up the GenPart chain until the pdgId changes, operating
    only on the selected particles rather than the full GenPart table.

    All arrays are materialized to plain numpy/awkward upfront to
    avoid lazy references back to the full GenPart table.

    Parameters
    ----------
    events : NanoEvents
        The full events object (needed to access GenPart table).
    gen_particles : awkward array
        The selected GenPart subset (e.g., events.GenPart[events.DisMuon.genPartIdx]).
    max_depth : int
        Maximum number of steps to walk up the chain (default 20, safety limit).

    Returns
    -------
    parent_pdgId : awkward array
        The absolute pdgId of the distinct parent for each particle.
        Returns -1 if no distinct parent is found.
    """
    # Materialize the full GenPart lookup tables as plain awkward arrays
    # This strips away lazy references that cause memory blow-ups
    all_genpart_pdgId = ak.to_packed(ak.without_parameters(events.GenPart.pdgId))
    all_genpart_mother = ak.to_packed(ak.without_parameters(events.GenPart.genPartIdxMother))

    # Materialize selected particle info
    current_idx = ak.to_packed(ak.without_parameters(gen_particles.genPartIdxMother))
    my_pdgId = ak.to_packed(ak.without_parameters(gen_particles.pdgId))

    for _ in range(max_depth):
        has_mother = current_idx >= 0

        mother_pdgId = ak.where(
            has_mother,
            all_genpart_pdgId[current_idx],
            0
        )

        is_distinct = (abs(mother_pdgId) != abs(my_pdgId)) & has_mother

        if ak.all(is_distinct | ~has_mother):
            break

        next_idx = ak.where(
            has_mother,
            all_genpart_mother[current_idx],
            -1
        )

        current_idx = ak.where(is_distinct, current_idx, next_idx)

    has_parent = current_idx >= 0
    parent_pdgId = ak.where(
        has_parent,
        abs(all_genpart_pdgId[current_idx]),
        -1
    )

    return parent_pdgId


def get_immediate_parent(events, gen_particles):
    """
    Simple version: just gets the immediate mother's pdgId.
    No chain walking, minimal memory usage.

    Parameters
    ----------
    events : NanoEvents
        The full events object.
    gen_particles : awkward array
        The selected GenPart subset.

    Returns
    -------
    parent_pdgId : awkward array
        The absolute pdgId of the immediate mother.
        Returns -1 if no mother exists.
    """
    all_genpart_pdgId = ak.to_packed(ak.without_parameters(events.GenPart.pdgId))
    mother_idx = ak.to_packed(ak.without_parameters(gen_particles.genPartIdxMother))
    has_mother = mother_idx >= 0

    parent_pdgId = ak.where(
        has_mother,
        abs(all_genpart_pdgId[mother_idx]),
        -1
    )

    return parent_pdgId


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


binning = [0, 10, 101] 
var_binning = [0, 50, 100, 150, 200, 250, 350, 450, 550, 750, 950, 1400]

MOTHERPARTICLES = ["q/g", "EWK bosons", "tau", "mesons", "baryons"] 

# Directory creation for plots (directory_path = "plots/" + args.process)
directory_path = "plots/"
if not os.path.exists(directory_path):
    os.makedirs(directory_path)
    print(f"Directory '{directory_path}' created successfully.")

DisMuon_mother = {}

d0_hists = {}
d0_hists["nominal"] = Hist(hist.axis.Variable(var_binning, name = "d0", underflow = True, overflow = True))
d0_hists["unknown"] = Hist(hist.axis.Variable(var_binning, name = "d0", underflow = True, overflow = True))
for part in MOTHERPARTICLES:
    d0_hists[part] = Hist(hist.axis.Variable(var_binning, name = "d0", underflow = True, overflow = True))

d0_eff_hists = {}
for process in available_processes:
    print(f"Starting {process}")
    tmp_string = f"faster_trial_{process}/faster_trial_{process}.root"
    tmp_file = sample_folder +  tmp_string
    events = NanoEventsFactory.from_root({tmp_file:"Events"}, schemaclass= NanoAODSchema).events()
    #events = events[ak.flatten(events.DisMuon.inTime)]
    events["DisMuon"]["dxy"] = events.DisMuon.dxy * 1E6
    d0_hists["nominal"].fill(ak.flatten(abs(events.DisMuon.dxy), axis = None))
    DisMuon_unknown = events.DisMuon[events.DisMuon.genPartIdx == -1]
    d0_hists["unknown"].fill(ak.flatten(abs(DisMuon_unknown.dxy), axis = None))
    events["DisMuon"] = events.DisMuon[events.DisMuon.genPartIdx >=0]

    GenMuon = events.GenPart[events.DisMuon.genPartIdx] 
    GenMuon_parent = abs(get_distinct_parent(events, GenMuon))
    #GenMuon_parent = abs(GenMuon.distinctParent.pdgId)

    for part in MOTHERPARTICLES:
        if "q/g" in part:
            DisMuon_mother[part] = events.DisMuon[(GenMuon_parent < 10) | (GenMuon_parent == 21)]
            d0_hists[part].fill(ak.flatten(abs(DisMuon_mother[part].dxy), axis = None))
        if "EWK bosons" in part:
            DisMuon_mother[part] = events.DisMuon[(GenMuon_parent == 22) | (GenMuon_parent == 23) | (GenMuon_parent == 24) | (GenMuon_parent == 25)]
            d0_hists[part].fill(ak.flatten(abs(DisMuon_mother[part].dxy), axis = None))
        if "tau" in part:
            DisMuon_mother[part] = events.DisMuon[GenMuon_parent == 15]
            d0_hists[part].fill(ak.flatten(abs(DisMuon_mother[part].dxy), axis = None))
        if "mesons" in part:
            DisMuon_mother[part] = events.DisMuon[((GenMuon_parent > 100) & (GenMuon_parent < 1000)) | (GenMuon_parent > 10000)]
            d0_hists[part].fill(ak.flatten(abs(DisMuon_mother[part].dxy), axis = None))
        if "baryons" in part:
            DisMuon_mother[part] = events.DisMuon[(GenMuon_parent > 1000) & (GenMuon_parent < 10000)]
            d0_hists[part].fill(ak.flatten(abs(DisMuon_mother[part].dxy), axis = None))

    
d0_eff_hists["unknown"] = hep.comp.get_comparison(d0_hists["unknown"], d0_hists["nominal"], comparison = 'efficiency')    

for part in MOTHERPARTICLES:
    d0_eff_hists[part] = hep.comp.get_comparison(d0_hists[part], d0_hists["nominal"], comparison = 'efficiency')


fig, ax = plt.subplots(1, 1, sharex=True)
    
colours = ["#6DD3CE", "#127475", "#FF99C9", "#FECDAA", "#06D6A0",
          "#B30089", "#8338EC", "#FFD166", "#EF476F", "#2D2D2D"]


ax.errorbar(d0_hists["nominal"].axes[0].centers, d0_eff_hists["unknown"][0], yerr=[d0_eff_hists["unknown"][1], d0_eff_hists["unknown"][-1]], color = colours[0], label = "unknown")

color_idx = 1
for part in MOTHERPARTICLES:
    ax.errorbar(d0_hists["nominal"].axes[0].centers, d0_eff_hists[part][0], yerr=[d0_eff_hists[part][1], d0_eff_hists[part][-1]], color = colours[color_idx], label = part)
    color_idx += 1
ax.set_yscale('log')
ax.legend(fontsize="small")
ax.set_ylim(1E-4, 1E3)
ax.set_ylabel("Muons from parent/total muons")
ax.set_xlabel(r"Muon |$d_0$| [cm]") 
hep.cms.label("Private Work", data=True, loc=0, com=13.6, ax=ax, fontsize = "x-small")
fig.savefig("muonParent_1400.pdf")
fig.savefig("muonParent_1400.png")





