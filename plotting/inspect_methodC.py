#!/usr/bin/env python3
"""
Follow-up to compare_methodC.py for one dataset:
  1. events whose candidate (DisMuon index 0) fails one of the selection's muon cuts: which cut, and the values
  2. events Method C gained (not in the old ntuple): how many muons sit ahead of the candidate in pT, and what they are

Run inside the coffea container, from plotting/:
  python inspect_methodC.py --dataset Stau_300_1000mm
"""
import os, sys, argparse
import numpy as np
import awkward as ak
import uproot

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../skimming"))
from selection_function import selections_dict

base = "/eos/uscms/store/user/lpcdisptau/bskipwor/displacedTaus/selected/Summer22_CHS_v19/mutau"
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True)
parser.add_argument("--old", default=f"{base}/v19/SR/faster_trial")
parser.add_argument("--new", default=f"{base}/v20/SR_methodC2/faster_trial")
parser.add_argument("--selection", default="SR")
parser.add_argument("--nprint", type=int, default=10)
args = parser.parse_args()
sel = selections_dict[args.selection]
ds = args.dataset


def tree(folder):
    return uproot.open(f"{folder}/faster_trial_{ds}/faster_trial_{ds}.root")["Events"]


t_new = tree(args.new)
mu_fields = [k for k in t_new.keys() if k.startswith("DisMuon_")]
new = t_new.arrays(["run", "luminosityBlock", "event", "CorrectedPFMET_pt"] + mu_fields)
old = tree(args.old).arrays(["run", "luminosityBlock", "event"])
old_keys = set(zip(ak.to_numpy(old.run).tolist(), ak.to_numpy(old.luminosityBlock).tolist(), ak.to_numpy(old.event).tolist()))
new_keys = list(zip(ak.to_numpy(new.run).tolist(), ak.to_numpy(new.luminosityBlock).tolist(), ak.to_numpy(new.event).tolist()))
in_old = np.array([k in old_keys for k in new_keys])
print(f"{ds}: {len(new)} new events, {int(in_old.sum())} also in old, {int((~in_old).sum())} gained")
print(f"DisMuon branches in the new ntuple: {mu_fields}\n")

## 1. candidate cut failures, cut by cut
cand = {f.replace("DisMuon_", ""): ak.to_numpy(new[f][:, 0]) for f in mu_fields}
cuts = {
    "pt":    cand["pt"] > sel["muon_pt_min"],
    "eta":   np.abs(cand["eta"]) < 2.4,
    "dxy":   (np.abs(cand["dxy"]) > sel["muon_dxy_min"]) & (np.abs(cand["dxy"]) < sel["muon_dxy_max"]),
    "iso03": (cand["pfRelIso03_all"] > sel["muon_iso_min"]) & (cand["pfRelIso03_all"] < sel["muon_iso_max"]),
    "iso04": (cand["pfRelIso04_all"] > sel["muon_iso04_min"]) & (cand["pfRelIso04_all"] < sel["muon_iso04_max"]),
    "mT":    (cand["mT"] > sel["muon_mt_min"]) & (cand["mT"] < sel["muon_mt_max"]),
}
if "mediumId" in cand:
    cuts["mediumId"] = cand["mediumId"] == 1
fail = ~np.logical_and.reduce(list(cuts.values()))
print(f"--- 1. candidate fails a muon cut in {int(fail.sum())} events")
for name, ok in cuts.items():
    print(f"    fails {name:8s}: {int((~ok).sum())}")
show = ["pt", "eta", "dxy", "pfRelIso03_all", "pfRelIso04_all", "mT"] + (["mediumId"] if "mediumId" in cand else [])
for i in np.flatnonzero(fail)[:args.nprint]:
    vals = "  ".join(f"{k}={cand[k][i]:.6g}" for k in show)
    print(f"    event {new_keys[i]} in_old={bool(in_old[i])} nMu={len(new.DisMuon_pt[i])}  {vals}")

## 2. gained events: what was ahead of the candidate?
print(f"\n--- 2. gained events")
gained = np.flatnonzero(~in_old)
if len(gained):
    g = new[gained]
    n_ahead = ak.to_numpy(ak.sum(g.DisMuon_pt[:, 1:] > g.DisMuon_pt[:, 0], axis=1))
    print(f"    muons ahead of the candidate: " + ", ".join(f"{n}: {int((n_ahead == n).sum())}" for n in np.unique(n_ahead)))
    print(f"    (0 ahead means the candidate IS the leading muon, so the old code should have kept the event too)")
    for j in range(min(args.nprint, len(g))):
        pts = ak.to_list(g.DisMuon_pt[j])
        extra = ""
        if "DisMuon_mediumId" in g.fields:
            extra = f" mediumId={ak.to_list(g.DisMuon_mediumId[j])}"
        dxys = [round(x, 4) for x in ak.to_list(g.DisMuon_dxy[j])]
        isos = [round(x, 3) for x in ak.to_list(g.DisMuon_pfRelIso03_all[j])]
        print(f"    event {new_keys[gained[j]]}: pt={[round(p, 1) for p in pts]} dxy={dxys} iso03={isos}{extra}  MET={float(g.CorrectedPFMET_pt[j]):.1f}")
