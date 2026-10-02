#!/usr/bin/env python3
"""Muon dxy N-1 plot (every SR cut except the muon dxy cut), backgrounds only,
made to match the muon d0 figure in Daniel's thesis:
  - signed muon dxy, 60 bins of 0.01 cm from -0.3 to 0.3 cm
  - underflow/overflow added into the first/last bin
  - 26.7 fb^-1, log y axis, same stacking order, colours and axis range
Run from the plotting/ folder, like the other plotting scripts:
    python3 makeMuonDxyPlot.py
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

SELECTION     = "N1_muonDxy"  # folder written by selection_cuts.py --selection N1_muonDxy
SAMPLE_FOLDER = ("/eos/uscms/store/user/lpcdisptau/bskipwor/displacedTaus/selected/"
                 f"Summer22_CHS_v19/mutau/v19/{SELECTION}/faster_trial/")
TARGET_LUMI   = 26.7   # fb^-1, same as the thesis figure
JET_SCORE_MIN = 0.9   # thesis Table 7.1; selection_function.py's SR uses 0.9

# thesis binning: 0.01 cm bins from -0.3 to 0.3 cm
N_BINS, X_MIN, X_MAX = 60, -0.3, 0.3

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

hists, labels, colours = [], [], []
print(f"jet score > {JET_SCORE_MIN}, {TARGET_LUMI} fb^-1")
print(f"{'group':8s} {'sim events':>10s} {'total':>10s} {'-0.01 to 0':>11s} {'0 to 0.01':>10s}")
for label, subsamples, colour in GROUPS:
    h = Hist(hist.axis.Regular(N_BINS, X_MIN, X_MAX, underflow=True, overflow=True),
             storage=hist.storage.Weight())
    n_sim = 0
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

        mu_dxy = first("DisMuon_dxy")
        jet_dxy = first("CorrectedJet_dxy")
        # every SR cut except the muon dxy cut (the selection already applied most of
        # these; repeating them is harmless and lets this script also run on PR files)
        keep = ((first("DisMuon_pt") > 30)
                & (first("DisMuon_pfRelIso03_all") < 0.18)
                & (first("CorrectedJet_pt") > 32)
                & (np.abs(jet_dxy) > 0.1) & (np.abs(jet_dxy) < 999)
                & (np.abs(first("CorrectedJet_dxyErr")) < 999)
                & (first("CorrectedJet_disTauTag_score1") > JET_SCORE_MIN)
                & (first("CorrectedJet_btagPNetB") > 0)
                & (ak.to_numpy(ev["CorrectedPFMET_pt"]) > 105))

        lumi_weight = TARGET_LUMI * xsec[sub] * br[sub] * 1000 / sum_gen_w[sub]
        h.fill(mu_dxy[keep], weight=ak.to_numpy(ev["weight"])[keep] * lumi_weight)
        n_sim += int(keep.sum())

    v = h.values()
    print(f"{label:8s} {n_sim:>10d} {h.values(flow=True).sum():>10.2f} {v[29]:>11.2f} {v[30]:>10.2f}")
    hists.append(h)
    labels.append(label)
    colours.append(colour)

fig, ax = plt.subplots()
hep.histplot(hists, histtype="fill", stack=True, flow="sum", label=labels, color=colours, ax=ax)
ax.set_yscale("log")
stack_max = sum(h.values() for h in hists).max()
ax.set_ylim(2.2e-5, max(3.6e3, 3 * stack_max))  # thesis figure's y range
ax.set_xlabel(r"$\mu$ dxy [cm]")
ax.set_ylabel("Events / 0.01 cm")
ax.xaxis.set_major_locator(MultipleLocator(0.1))
ax.xaxis.set_minor_locator(MultipleLocator(0.025))
ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), prop={"size": 16})
hep.cms.label("Work in progress", data=False, loc=0, com=13.6, lumi=TARGET_LUMI, ax=ax)
plt.subplots_adjust(right=0.75)

os.makedirs("plots/muon_dxy_thesis", exist_ok=True)
out = f"plots/muon_dxy_thesis/DisMuon_dxy_{SELECTION}_score{JET_SCORE_MIN}"
plt.savefig(out + ".png")
plt.savefig(out + ".pdf")
print("saved", out + ".png")
