import os, sys
import numpy as np
import awkward as ak
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from coffea.nanoevents import NanoEventsFactory, PFNanoAODSchema

sys.path.insert(0, "../skimming")
from selection_function import manual_blinding, selections_dict

PFNanoAODSchema.warn_missing_crossrefs = False
PFNanoAODSchema.mixins["DisMuon"] = "Muon"
PFNanoAODSchema.mixins["CorrectedJet"] = "Jet"
PFNanoAODSchema.mixins["JetPFCands"] = "Jet"
PFNanoAODSchema.mixins["PFCands"] = "Jet"

SAMPLE_FOLDER = "/eos/uscms/store/user/lpcdisptau/bskipwor/displacedTaus/selected/Summer22_CHS_v19/mutau/v19/PR/faster_trial/"

## signal MC: mix of short and long lifetimes so both sides of the dxy cut are populated
PROCESSES = [
    "Stau_100_0p1mm",
    "Stau_100_1mm",
    "Stau_100_100mm",
    "Stau_500_0p1mm",
    "Stau_500_1mm",
    "Stau_500_100mm",
    "Stau_500_1000mm",
]
OUTDIR = "plots/blinding_check"

DATA_NAMES = ("jetmet", "muon_2022", "egamma", "data")
SR = selections_dict["SR"]
os.makedirs(OUTDIR, exist_ok=True)

## dxy spans microns (prompt) to tens of cm (ctau 1000mm), so use log bins
DXY_BINS = np.logspace(-4, 2, 61)     # 1e-4 to 100 cm
ISO_BINS = np.linspace(0, 1, 61)
SCORE_BINS = np.linspace(0, 1, 61)

def load(process):
    path = f"{SAMPLE_FOLDER}faster_trial_{process}/faster_trial_{process}.root"
    if any(d in process.lower() for d in DATA_NAMES):
        sys.exit(f"REFUSING: '{process}' looks like DATA. This test is MC-only.")
    ev = NanoEventsFactory.from_root({path: "Events"}, schemaclass=PFNanoAODSchema).events()
    if "genWeight" not in ev.fields:
        sys.exit(f"REFUSING: {path} has no genWeight branch, so it is not MC.")
    print(f"  {process}: {len(ev)} events (MC confirmed: genWeight present)")
    return ev

def muon_in_box(ev):
    return ak.flatten((abs(ev.DisMuon.dxy) > SR["muon_dxy_min"]) &
                      (abs(ev.DisMuon.dxy) < SR["muon_dxy_max"]) &
                      (ev.DisMuon.pfRelIso03_all > SR["muon_iso_min"]) &
                      (ev.DisMuon.pfRelIso03_all < SR["muon_iso_max"]))

def jet_in_box(ev):
    return ak.flatten((ev.CorrectedJet.disTauTag_score1 > SR["jet_score_min"]) &
                      (ev.CorrectedJet.disTauTag_score1 < SR["jet_score_max"]) &
                      (abs(ev.CorrectedJet.dxy) > SR["jet_dxy_min"]) &
                      (abs(ev.CorrectedJet.dxy) < SR["jet_dxy_max"]))

def arr(x, lo, hi):
    """flatten to numpy and clip into range so nothing falls off the plot"""
    v = ak.to_numpy(ak.flatten(x)).astype(float)
    return np.clip(v, lo, hi)

def plot2d(xb, yb, xa, ya, xlabel, ylabel, title, vline, hline, xbins, ybins, fname):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True, sharey=True)
    for ax, (x, y, tag) in zip(axes, [(xb, yb, "before blinding"),
                                      (xa, ya, "after blinding")]):
        if len(x):
            h = ax.hist2d(x, y, bins=[xbins, ybins], norm=matplotlib.colors.LogNorm())
            fig.colorbar(h[3], ax=ax)
        ax.set_xscale("log")
        ax.axvline(vline, color="red", lw=1.5)
        ax.axhline(hline, color="red", lw=1.5)
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        ax.set_title(f"{tag}  (N={len(x)})")
    fig.suptitle(title)
    fig.tight_layout()
    out = f"{OUTDIR}/{fname}.png"
    fig.savefig(out); plt.close(fig)
    print("  wrote", out)

print("Loading signal MC...")
evs = [load(p) for p in PROCESSES]
before = ak.concatenate(evs)
print(f"total before blinding: {len(before)}")

after = manual_blinding(before)
print(f"total after  blinding: {len(after)}")
print(f"removed: {len(before) - len(after)}")

mb, jb = muon_in_box(before), jet_in_box(before)
print(f"expected removed (muon box AND jet box): {int(ak.sum(mb & jb))}")

ma, ja = muon_in_box(after), jet_in_box(after)
print(f"events still in BOTH boxes after blinding: {int(ak.sum(ma & ja))}   <-- must be 0")

lo, hi = DXY_BINS[0], DXY_BINS[-1]
print("\nPlots:")
## muon plane, split on whether the jet is in its box
for tag, sel_b, sel_a in [("jetInBox", jb, ja), ("jetNotInBox", ~jb, ~ja)]:
    b, a = before[sel_b], after[sel_a]
    plot2d(arr(abs(b.DisMuon.dxy), lo, hi), arr(b.DisMuon.pfRelIso03_all, 0, 1),
           arr(abs(a.DisMuon.dxy), lo, hi), arr(a.DisMuon.pfRelIso03_all, 0, 1),
           "|muon dxy| [cm]", "muon pfRelIso03_all",
           f"muon plane, {tag}", SR["muon_dxy_min"], SR["muon_iso_max"],
           DXY_BINS, ISO_BINS, f"muon_iso_vs_dxy_{tag}")

## jet plane, split on whether the muon is in its box
for tag, sel_b, sel_a in [("muonInBox", mb, ma), ("muonNotInBox", ~mb, ~ma)]:
    b, a = before[sel_b], after[sel_a]
    plot2d(arr(abs(b.CorrectedJet.dxy), lo, hi), arr(b.CorrectedJet.disTauTag_score1, 0, 1),
           arr(abs(a.CorrectedJet.dxy), lo, hi), arr(a.CorrectedJet.disTauTag_score1, 0, 1),
           "|jet dxy| [cm]", "disTauTag_score1",
           f"jet plane, {tag}", SR["jet_dxy_min"], SR["jet_score_min"],
           DXY_BINS, SCORE_BINS, f"jet_score_vs_dxy_{tag}")
