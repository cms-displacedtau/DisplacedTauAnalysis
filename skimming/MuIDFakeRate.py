import awkward as ak
import numpy as np
import uproot, sys, os
from coffea import processor
from coffea.nanoevents.methods import candidate
from coffea.nanoevents import NanoEventsFactory, PFNanoAODSchema
from dask.distributed import Client, wait, progress, LocalCluster
from lpcjobqueue import LPCCondorCluster, schedd
#from dask_lxplus import CernCluster
from dask import config as cfg
from dask_jobqueue import HTCondorCluster
cfg.set({'distributed.scheduler.worker-ttl': None}) # Check if this solves some dask issues
cfg.set({'distributed.scheduler.allowed-failures': 20}) # Check if this solves some dask issues
import fsspec_xrootd
from  fsspec_xrootd import XRootDFileSystem
import hist
from matplotlib import pyplot as plt
import mplhep as hep

import os, argparse, importlib, pdb, socket
import time
from datetime import datetime

from collections import defaultdict
import json

from utils import process_n_files, is_rootcompat, is_good_hlt, is_included, uproot_writeable
# import sys
# sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath('../selections/lumi_selections.py') ) ) )
from selections.lumi_selections import select_lumis


PFNanoAODSchema.warn_missing_crossrefs = False
PFNanoAODSchema.mixins["DisMuon"] = "Muon"

parser = argparse.ArgumentParser(description="")
parser.add_argument(
	"--sample",
	choices=['QCD','DY', 'signal', 'WtoLNu', 'Wto2Q', 'TT', 'singleT', 'JetMET_2022', 'Muon', 'DYEMu', 'DYTau', 'VBF', 'EWK'],
	required=True,
	help='Specify the sample you want to process')
parser.add_argument(
	"--subsample",
	nargs='*',
	default='all',
	required=False,
	help='Specify the exact sample you want to process')
parser.add_argument(
	"--nfiles",
	default='-1',
	required=False,
	help='Specify the number of input files to process')
parser.add_argument(
	"--usePkl",
	default=True,
	required=False,
	help='Turn it to false to use the non-preprocessed samples')
parser.add_argument(
	"--nanov",
	choices=['Summer22_CHS_v10', 'Summer22_CHS_v7', 'Summer22_CHS_v19'],
	default='Summer22_CHS_v19',
	required=False,
	help='Specify the custom nanoaod version to process')
parser.add_argument(
	"--testjob",
        action="store_true",
        help="Run a test job locally")
args = parser.parse_args()


out_folder = f'root://cmseos.fnal.gov//store/user/lpcdisptau/dally/displacedTaus/skim/{args.nanov}/mutau/v8/'
out_folder_json = out_folder.replace('root://cmseos.fnal.gov/','/eos/uscms')
custom_nano_v = args.nanov + '/'
custom_nano_v_p = args.nanov + '.'

samples = {
    "Wto2Q": f"samples.{custom_nano_v_p}fileset_Wto2Q",
    "WtoLNu": f"samples.{custom_nano_v_p}fileset_WtoLNu",
    "QCD": f"samples.{custom_nano_v_p}fileset_QCD",
    "DY": f"samples.{custom_nano_v_p}fileset_DY",
    "DYEMu": f"samples.{custom_nano_v_p}fileset_DYEMu",
    "DYTau": f"samples.{custom_nano_v_p}fileset_DYTau",
    "signal": f"samples.{custom_nano_v_p}fileset_signal",
    "TT": f"samples.{custom_nano_v_p}fileset_TT",
    "singleT": f"samples.{custom_nano_v_p}fileset_singleT",
    "JetMET": f"samples.{custom_nano_v_p}fileset_JetMET_2022",
    "Muon": f"samples.{custom_nano_v_p}fileset_Muon_2022",
    "VBF": f"samples.{custom_nano_v_p}fileset_VBF",
    "EWK": f"samples.{custom_nano_v_p}fileset_EWK",
}

all_fileset = {}
if args.usePkl == True:
    import pickle 
    with open(f"samples/{custom_nano_v}{args.sample}_preprocessed.pkl", "rb") as  f:
        input_dataset = pickle.load(f)
else:
    samples = {
        "Wto2Q": f"samples.{custom_nano_v_p}fileset_Wto2Q",
        "WtoLNu": f"samples.{custom_nano_v_p}fileset_WtoLNu",
        "QCD": f"samples.{custom_nano_v_p}fileset_QCD",
        "DY": f"samples.{custom_nano_v_p}fileset_DY",
        "DYEMu": f"samples.{custom_nano_v_p}fileset_DYEMu",
        "DYTau": f"samples.{custom_nano_v_p}fileset_DYTau",
        "signal": f"samples.{custom_nano_v_p}fileset_signal",
        "TT": f"samples.{custom_nano_v_p}fileset_TT",
        "singleT": f"samples.{custom_nano_v_p}fileset_singleT",
        "JetMET": f"samples.{custom_nano_v_p}fileset_JetMET_2022",
        "Muon": f"samples.{custom_nano_v_p}fileset_Muon_2022",
        "VBF": f"samples.{custom_nano_v_p}fileset_VBF",
        "EWK": f"samples.{custom_nano_v_p}fileset_EWK",
    }
  
    module = importlib.import_module(samples[args.sample])
    input_dataset = module.fileset   

## restrict to specific sub-samples
if args.subsample == 'all':
    fileset = input_dataset
else:  
    with open(f"samples/{args.nanov}/{skim_folder}/{args.skimversion}/{args.subsample[0]}_preprocessed.pkl", "rb") as  f:
        fileset = pickle.load(f)


## restrict to n files
process_n_files(int(args.nfiles), fileset)

GEN_STAU_PDGID   = 1000015   # stau PDG ID
GEN_TAU_PDGID    = 15
GEN_MU_PDGID     = 13
MATCH_DR         = 0.3        # deltaR cone for reco<->gen matching

PT_MIN = 20
ETA_MAX = 2.4

# Muon ID working points available in NanoAOD
MUON_IDS = {
    "Loose":  lambda mu: mu.looseId,
    "Medium": lambda mu: mu.mediumId,
    "Tight":  lambda mu: mu.tightId,
}


variables_with_bins = {
    "reco_mu_pt": [(16, 20, 100), "GeV"],
    "is_fake_pt": [(16, 20, 100), "GeV"],
    "reco_mu_eta" : [(12, -2.4, 2.4), ""],
    "is_fake_eta" : [(12, -2.4, 2.4), ""],
    "reco_mu_dxy" : [(30, 0, 15), "cm"],
    "is_fake_dxy" : [(30, 0, 15), "cm"]
    }

SAMPLE_GROUPS = {
    "QCD":  ["QCD"],
    "Top":  ["TT", "singleT"],
    "W":    ["WtoLNu", "Wto2Q"],
    "DY":   ["DY", "DYEMu", "DYTau"],
}

class MCProcessor(processor.ProcessorABC):
    def __init__(self, vars_with_bins, sample):
        self.vars_with_bins = vars_with_bins
        self.sample = sample
        print("Initializing ExampleProcessor")

    def initialize_histograms(self):
        histograms = {}
        # Initialize histograms for each variable based on provided binning
        for var, bin_info in self.vars_with_bins.items():
            print(f"Creating histogram for {var} with bin_info {bin_info}")

            histograms[var] = hist.Hist(hist.axis.Variable(np.linspace(bin_info[0][1],bin_info[0][2],bin_info[0][0] + 1), name=var, label = var + ' ' + bin_info[1], underflow = True, overflow = True),
                                        storage = hist.storage.Weight())

            print(f"Successfully created histogram for {var}")
        return histograms
    
    def gen_muons_not_from_stau(self, events, reco_mu):
        """
        Return gen-level muons whose ancestry is stau->tau->mu.
        Works with the GenPart collection (NanoAOD).
        """
        dis_mu = reco_mu
        gp = events.GenPart
    
        gen_idx  = dis_mu.genPartIdx  # (event, part) -> int
        safe_gen = ak.where(gen_idx >= 0, gen_idx, 0)
        rgp = gp[safe_gen]
    
        # Absolute PDG IDs
        abs_pdg = abs(rgp.pdgId)
    
        # Build parent/grandparent masks using GenPart_genPartIdxMother
        # NanoAOD stores the mother index; -1 means no mother.
        mother_idx  = rgp.genPartIdxMother  # (event, part) -> int
        # Clip negative indices so ak indexing doesn't wrap
        safe_mother = ak.where(mother_idx >= 0, mother_idx, 0)
    
        # PDG ID of the direct mother of each particle
        mother_pdg  = abs(gp[safe_mother].pdgId)
        # PDG ID of the grandparent
        gmother_idx = gp[safe_mother].genPartIdxMother
        safe_gmother = ak.where(gmother_idx >= 0, gmother_idx, 0)
        gmother_pdg = abs(gp[safe_gmother].pdgId)
    
        is_mu        = (abs_pdg == GEN_MU_PDGID)
        from_tau     = (mother_pdg == GEN_TAU_PDGID)
        from_stau    = (gmother_pdg == GEN_STAU_PDGID)
        is_final     = rgp.hasFlags("isLastCopy")
    
        signal_mask  = ~(is_mu & from_tau & from_stau & is_final)
        return events.DisMuon[signal_mask]

    def process(self, events):

        ## NB: to be double checked if the string identifying the buggy DY dataset is correct
        if self.sample == 'DYEMu': 
            lhe_part = events.LHEPart
            outcoming = lhe_part[lhe_part.status > 0]
            lhe_z = outcoming[(outcoming.status == 2) & (outcoming.pdgId==23)]
            out_tau_tau = outcoming[(abs(outcoming.pdgId)==15)]
            counts_tautau = ak.num(out_tau_tau, axis=1)  
            mask_ztautau = (counts_tautau == 2)
            mask_zll = ~mask_ztautau
            events = events[mask_zll]
            print (f" Removed non zll events from {self.sample}")
                    
        no_id = ak.full_like(events.DisMuon.pt, True, dtype=bool)
        print(no_id)
        events["DisMuon"] = ak.with_field(events["DisMuon"], no_id, "noId")
        mu_id = {"None": lambda mu: mu.noId}
        ALL_MUON_IDS = mu_id | MUON_IDS
        # ---- reco muons ------------------------------------------------
        events["DisMuon"] = events.DisMuon[(events.DisMuon.pt > PT_MIN) & (abs(events.DisMuon.eta) < ETA_MAX)]
        reco_mu = events.DisMuon
        print("Cuts applied to reco muon")

        id_hists = {}        

        for id_name, id_sel in ALL_MUON_IDS.items():

            reco_id = reco_mu[id_sel(reco_mu)]
            is_fake = reco_mu

            histograms = self.initialize_histograms()
            # Loop over variables and fill histograms
            histograms["reco_mu_pt"].fill(
                **{"reco_mu_pt": ak.flatten(reco_mu.pt, axis=None)}
            )
            histograms["reco_mu_eta"].fill(
                **{"reco_mu_eta": ak.flatten(reco_mu.eta, axis=None)}
            )
            histograms["reco_mu_dxy"].fill(
                **{"reco_mu_dxy": ak.flatten(abs(reco_mu.dxy), axis=None)}
            )

            # Fill numerator histograms (fakes only)
            histograms["is_fake_pt"].fill(
                **{"is_fake_pt": ak.flatten(reco_id.pt, axis=None)}
            )
            histograms["is_fake_eta"].fill(
                **{"is_fake_eta": ak.flatten(reco_id.eta, axis=None)}
            )
            histograms["is_fake_dxy"].fill(
                **{"is_fake_dxy": ak.flatten(abs(reco_id.dxy), axis=None)}
            )
            #for var in histograms:
            #    var_name = '_'.join(var.split('_')[-1])
            #    histograms[var].fill(
            #        **{var: ak.flatten(var.split('_')[0:2][var_name], axis = None)},
            #    )
            id_hists[id_name] = histograms             
            
        output = {"histograms": id_hists, "sample": self.sample}
        print(output)
        return output

    def postprocess(self, accumulator):
        return accumulator

# Colors for each muon ID
ID_COLORS = {
    "None":   "#E23F59",
    "Loose":  "#EAB508",
    "Medium": "#7FB00C",
    "Tight":  "#8A15B1",
}

ID_MARKERS = {
    "None":   "o",
    "Loose":  "s",
    "Medium": "D",
    "Tight":  "^",
}

if __name__ == "__main__":

    import pickle

    output_dir = f"outputs/{args.nanov}"
    os.makedirs(output_dir, exist_ok=True)
    output_pkl = os.path.join(output_dir, f"fake_rate_{args.sample}.pkl")

    proc = MCProcessor(vars_with_bins=variables_with_bins, sample=args.sample)

    if args.testjob:
        cluster = LocalCluster(n_workers=10, threads_per_worker=1)
    else:
        cluster = LPCCondorCluster(
                cores=4,
                memory='10GB',
                log_directory = f"/uscmst1b_scratch/lpc1/3DayLifetime/condor/log/selected/v8",
                transfer_input_files = ["utils.py",],
                job_extra_directives={
                    "should_transfer_files": "YES",
                    '+JobFlavour': '"longlunch"',
                    },
                job_script_prologue=[
                    "export XRD_RUNFORKHANDLER=1",  ### enables fork-safety in the XRootD client, to avoid deadlock when accessing EOS files
                    f"export X509_USER_PROXY=$HOME/x509up_u57864",
                    "export PYTHONPATH=$PYTHONPATH:$_CONDOR_SCRATCH_DIR:$HOME",
                ],
                )
         #minimum > 0: https://github.com/CoffeaTeam/coffea/issues/465
        cluster.adapt(minimum=0, maximum=200)
        print(cluster.job_script())

    client = Client(cluster)
    runner = processor.Runner(
        executor=processor.DaskExecutor(client=client, compression=None),
        schema=PFNanoAODSchema,
        savemetrics=True,
        xrootdtimeout=600,
    )

    output, report = runner(
        fileset,
        treename="Events",
        processor_instance=proc,
        uproot_options={"allow_read_errors_with_report": (OSError, KeyError)},
    )

    # Save this sample's output
    with open(output_pkl, "wb") as f:
        pickle.dump(output["histograms"], f)
    print(f"Saved output to {output_pkl}")
    def compute_fake_rate(h_num, h_den):
        """
        Compute fake rate = num / den with binomial uncertainties.
        Returns (centres, rate, err_lo, err_hi).
        """
        rate, err = hep.comp.get_efficiency(h_num, h_den)
        centers = (h_num.axes[0].edges[:-1] + h_num.axes[0].edges[1:]) / 2    
        return centers, rate, err
    
    
    def plot_fake_rate_for_group(group_name, group_hists, kin_var, out_dir, nanov):
        """
        For a given sample group, plot fake rate vs *kin_var* (pt, eta, dxy)
        with one curve per muon ID on the same canvas.
        """
        id_colors  = {"None": "#E23F59", "Loose": "#EAB508", "Medium": "#7FB00C", "Tight": "#8A15B1"}
    
        fig, ax = plt.subplots(figsize=(10, 7))
    
        for id_name, histograms in group_hists.items():
            den_key = f"reco_mu_{kin_var}"
            num_key = f"is_fake_{kin_var}"
    
            h_den = histograms[den_key]
            h_num = histograms[num_key]
    
            centers, rate, err = compute_fake_rate(h_num, h_den)
    
            ax.errorbar(
                centers, rate, yerr=err,
                fmt="o",
                color=id_colors.get(id_name, "gray"),
                label=f"{id_name}",
                
            )
    
        ax.set_xlabel(h_den.axes[0].label)
        ax.set_ylabel("Fake rate")
        ax.set_ylim(0, 1.1)
        ax.legend()
        hep.cms.label(f"{group_name} Private Work", data=False, ax=ax)
    
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"fake_rate_{kin_var}_{group_name}.pdf")
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_path}")
    
    
    def plot_all_groups(all_outputs, nanov):
        """
        Merge per-sample outputs into groups and produce fake-rate plots.
        """
        out_dir = f"plots/{nanov}"

        for group_name, sample_list in SAMPLE_GROUPS.items():
            # Collect available samples for this group
            available = [s for s in sample_list if s in all_outputs]
            if not available:
                print(f"Skipping group {group_name}: no outputs found")
                continue

            # Merge histograms across samples within the group
            merged_id_hists = {}
            for sample_key in available:
                sample_hists = all_outputs[sample_key]
                for id_name, hists in sample_hists.items():
                    if id_name not in merged_id_hists:
                        # Deep copy the first sample's histograms
                        merged_id_hists[id_name] = {k: v.copy() for k, v in hists.items()}
                    else:
                        for k in hists:
                            merged_id_hists[id_name][k] = merged_id_hists[id_name][k] + hists[k]

            # Plot each kinematic variable
            for kin_var in ["pt", "eta", "dxy"]:
                plot_fake_rate_for_group(group_name, merged_id_hists, kin_var, out_dir, nanov)

    # ---- Collect all available sample outputs and make group plots ----------
    all_outputs = {}
    for sample_name in ["QCD", "DY", "DYEMu", "DYTau", "TT", "singleT",
                        "WtoLNu", "Wto2Q", "signal", "VBF", "EWK"]:
        pkl_path = os.path.join(output_dir, f"fake_rate_{sample_name}.pkl")
        if os.path.exists(pkl_path):
            with open(pkl_path, "rb") as f:
                all_outputs[sample_name] = pickle.load(f)
            print(f"Loaded: {pkl_path}")

    if all_outputs:
        plot_all_groups(all_outputs, args.nanov)
    else:
        print("No sample outputs found yet — run more samples to produce group plots.")
