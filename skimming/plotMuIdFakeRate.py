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
import hist.dask as hda

import os, argparse, importlib, pdb, socket
import time
from datetime import datetime

from collections import defaultdict
import json
import mplhep as hep

from utils import process_n_files, is_rootcompat, is_good_hlt, is_included, uproot_writeable
# import sys
# sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath('../selections/lumi_selections.py') ) ) )
from selections.lumi_selections import select_lumis


PFNanoAODSchema.warn_missing_crossrefs = False
PFNanoAODSchema.mixins["DisMuon"] = "Muon"

SAMPLE_GROUPS = {
    "QCD":  ["QCD"],
    "Top":  ["TT", "singleT"],
    "W":    ["WtoLNu", "Wto2Q"],
    "DY":   ["DY", "DYEMu", "DYTau"],
}
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

        centees, rate, err = compute_fake_rate(h_num, h_den)

        ax.errorbar(
            centres, rate, yerr=err,
            fmt="o",
            color=id_colors.get(id_name, "gray"),
            label=f"{id_name}",
            
        )

    ax.set_xlabel(h_den.axes[0].label)
    ax.set_ylabel("Fake rate")
    ax.set_ylim(0, None)
    ax.legend()
    hep.cms.label("{group_name} Private Work", data=False, ax=ax)

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
            sample_hists = all_outputs[sample_key]["histograms"]
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
