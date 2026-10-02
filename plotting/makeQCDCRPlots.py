#!/usr/bin/env python3
"""QCD control region plots matching Figure 7.7 of Daniel's thesis, backgrounds only:
muon pT, jet pT, muon dxy and jet dxy, 26.7 fb^-1, log y axis,
underflow/overflow added into the first/last bin.
QCD CR (thesis Table 7.3): muon |d0| < 0.045 cm, muon iso > 0.1, jet |d0| < 0.1 cm,
jet score < 0.2, b-tag > 0, plus the preselection (muon pT > 30, jet pT > 32, MET > 105).
Run from the plotting/ folder:
    python3 makeQCDCRPlots.py
"""
import json, os
import numpy as np
import awkward as ak
import uproot
import hist
from hist import Hist
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import mplhep as hep

hep.style.use("CMS")

SELECTION     = "QCD_CR_thesis"  # folder written by selection_cuts.py --selection QCD_CR_thesis
SAMPLE_FOLDER = ("/eos/uscms/store/user/lpcdisptau/bskipwor/displacedTaus/selected/"
                 f"Summer22_CHS_v19/mutau/v19/{SELECTION}/faster_trial/")
TARGET_LUMI   = 26.7  # fb^-1, same as the thesis figures

# (name, branch, n bins, low, high, x label, y label, major tick, minor tick)
PLOTS = [
    ("DisMuon_pt", "DisMuon_pt", 50, 0, 500, r"$\mu$ $p_T$ [GeV]", "Events / 10 GeV", 50, 10),
    ("Jet_pt", "CorrectedJet_pt", 50, 0, 500, r"jet $p_T$ [GeV]", "Events / 10 GeV", 50, 10),
    ("DisMuon_dxy", "DisMuon_dxy", 10, -0.05, 0.05, r"$\mu$ $d_{xy}$ [cm]", "Events / 0.01 cm", 0.05, 0.01),
    ("Jet_dxy", "CorrectedJet_dxy", 20, -0.1, 0.1, r"jet $d_{xy}$ [cm]", "Events / 0.01 cm", 0.1, 0.02),
]

# (legend label, subsamples, colour), stacked bottom -> top as in the thesis
GROUPS = [
    ("Wto2Q", ["Wto2Q-2Jets_PTQQ-100to200_1J", "Wto2Q-2Jets_PTQQ-100to200_2J",
               "Wto2Q-2Jets_PTQQ-200to400_1J", "Wto2Q-2Jets_PTQQ-200to400_2J",
               "Wto2Q-2Jets_PTQQ-400to600_1J", "Wto2Q-2Jets_PTQQ-400to600_2J",
               "Wto2Q-2Jets_PTQQ-600_1J", "Wto2Q-2Jets_PTQQ-600_2J"], "#5790FC"),
    ("WtoLNu", ["WtoLNu-4Jets"], "#F89C20"),
    ("singleT", ["TbarWplustoLNu2Q", "TbarWplusto2L2Nu", "TWminusto2L2Nu", "TWminustoLNu2Q",
                 "TBbarQ_t-channel_4FS", "TbarBQ_t-channel_4FS"], "#E42536"),
    ("TT", ["TTto2L2Nu", "TTtoLNu2Q", "TTto4Q"], "#964A8B"),
    ("DY", ["DYJetsToLL_M-50", "DYto2Tau-2Jets_MLL-50_0J", "DYto2Tau-2Jets_MLL-50_1J",
            "DYto2Tau-2Jets_MLL-50_2J"], "#9C9CA1"),
    ("QCD", ["QCD_PT-50to80", "QCD_PT-80to120", "QCD_PT-120to170", "QCD_PT-170to300",
             "QCD_PT-300to470", "QCD_PT-470to600", "QCD_PT-600to800", "QCD_PT-800to1000",
             "QCD_PT-1000to1400", "QCD_PT-1400to1800", "QCD_PT-1800to2400",
             "QCD_PT-2400to3200", "QCD_PT-3200"], "#7A21DD"),
    ("EWK", ["WW", "WZ", "ZZ"], "#FF99C9"),
]

BRANCHES = ["weight", "CorrectedPFMET_pt",
            "DisMuon_pt", "DisMuon_dxy", "DisMuon_pfRelIso03_all",
            "CorrectedJet_pt", "CorrectedJet_dxy", "CorrectedJet_dxyErr",
            "CorrectedJet_disTauTag_score1", "CorrectedJet_btagPNetB"]

with open("./plots_config/xsec_and_br.json") as f:
    xsec_and_br = json.load(f)
xsec, br = xsec_and_br["xsec"], xsec_and_br["br"]
with open("./plots_config/Summer22_CHS_v19/mutau/v19/all_total_sumw.json") as f:
    sum_gen_w = json.load(f)

# hists[plot name] = list of one histogram per group, in GROUPS order
hists = {p[0]: [] for p in PLOTS}
print(f"QCD CR, {TARGET_LUMI} fb^-1")
print(f"{'group':8s} {'sim events':>10s} {'total':>12s}")
for label, subsamples, colour in GROUPS:
    group_hists = {name: Hist(hist.axis.Regular(nb, lo, hi, underflow=True, overflow=True),
                              storage=hist.storage.Weight())
                   for name, _, nb, lo, hi, *_ in PLOTS}
    n_sim, total = 0, 0.0
    for sub in subsamples:
        path = f"{SAMPLE_FOLDER}faster_trial_{sub}/faster_trial_{sub}.root"
        if not os.path.exists(path):
            print(f"  WARNING: no merged file for {sub}, leaving it out")
            continue
        if sub not in sum_gen_w:
            print(f"  WARNING: {sub} is not in all_total_sumw.json, leaving it out")
            continue
        ev = uproot.open(path)["Events"].arrays(BRANCHES)

        def first(branch):
            # the one muon / one jet kept per event, as a plain numpy array
            return ak.to_numpy(ak.fill_none(ak.firsts(ev[branch]), np.nan))

        values = {b: first(b) for b in BRANCHES if b.startswith(("DisMuon", "CorrectedJet"))}
        score = values["CorrectedJet_disTauTag_score1"]
        # thesis QCD CR + preselection (repeating cuts the selection already made is harmless)
        keep = ((values["DisMuon_pt"] > 30)
                & (np.abs(values["DisMuon_dxy"]) < 0.045)
                & (values["DisMuon_pfRelIso03_all"] > 0.1)
                & (values["CorrectedJet_pt"] > 32)
                & (np.abs(values["CorrectedJet_dxy"]) < 0.1)
                & (np.abs(values["CorrectedJet_dxyErr"]) < 999)
                & (score > 0) & (score < 0.2)
                & (values["CorrectedJet_btagPNetB"] > 0)
                & (ak.to_numpy(ev["CorrectedPFMET_pt"]) > 105))

        w = ak.to_numpy(ev["weight"])[keep] * TARGET_LUMI * xsec[sub] * br[sub] * 1000 / sum_gen_w[sub]
        for name, branch, *_ in PLOTS:
            group_hists[name].fill(values[branch][keep], weight=w)
        n_sim += int(keep.sum())
        total += w.sum()

    print(f"{label:8s} {n_sim:>10d} {total:>12.2f}")
    for name in hists:
        hists[name].append(group_hists[name])

labels = [g[0] for g in GROUPS]
colours = [g[2] for g in GROUPS]
os.makedirs("plots/qcd_cr_thesis", exist_ok=True)
for name, _, nb, lo, hi, xlabel, ylabel, major, minor in PLOTS:
    fig, ax = plt.subplots()
    hep.histplot(hists[name], histtype="fill", stack=True, flow="sum", label=labels, color=colours, ax=ax)
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.xaxis.set_major_locator(MultipleLocator(major))
    ax.xaxis.set_minor_locator(MultipleLocator(minor))
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), prop={"size": 16})
    hep.cms.label("Work in progress", data=False, loc=0, com=13.6, lumi=TARGET_LUMI, ax=ax, fontsize="small")
    plt.subplots_adjust(right=0.75)
    out = f"plots/qcd_cr_thesis/{name}_{SELECTION}"
    plt.savefig(out + ".png")
    plt.savefig(out + ".pdf")
    plt.close(fig)
    print("saved", out + ".png")
