#!/usr/bin/env python3
"""
Compare two merged selections event by event (e.g. leading-muon vs Method C) and make slide-ready plots.

What a correct Method C looks like, compared with the leading-muon selection on the same input:
  * every reference event is also in the test ntuple (refOnly = 0), with the SAME candidate muon
    (if the leading mediumId muon passes, it is also the highest-pT muon passing all cuts)
  * gained events (testOnly) always have >= 1 higher-pT muon ahead of the candidate
  * the candidate (DisMuon index 0) passes every muon cut of the selection
  * partner muons (index 1+) are kept in some events

Run inside the coffea container, from plotting/, e.g.:
  python compare_methodC.py            (defaults: v19/SR leading muon vs v20/SR_methodC2)
Output: plots/methodC_validation/<name>/  (summary.txt, summary.csv, PDFs + PNGs; unweighted event counts)
"""
import os, sys, glob, argparse
import numpy as np
import awkward as ak
import uproot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import mplhep as hep
    hep.style.use("CMS")
except ImportError:
    hep = None

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../skimming"))
from selection_function import selections_dict

base = "/eos/uscms/store/user/lpcdisptau/bskipwor/displacedTaus/selected/Summer22_CHS_v19/mutau"
parser = argparse.ArgumentParser()
parser.add_argument("--ref", default=f"{base}/v19/SR/faster_trial", help="merged reference ntuples")
parser.add_argument("--test", default=f"{base}/v20/SR_methodC2/faster_trial", help="merged ntuples being validated")
parser.add_argument("--ref-label", default="Leading muon (v19)")
parser.add_argument("--test-label", default="Method C (v20)")
parser.add_argument("--name", default="leading_vs_methodC2", help="output subfolder")
parser.add_argument("--selection", default="SR", help="selection both were made with (for the candidate-cut check)")
args = parser.parse_args()
outdir = f"plots/methodC_validation/{args.name}"
os.makedirs(outdir, exist_ok=True)
sel = selections_dict[args.selection]

branches_wanted = ["run", "luminosityBlock", "event",
                   "DisMuon_pt", "DisMuon_eta", "DisMuon_phi", "DisMuon_dxy",
                   "DisMuon_pfRelIso03_all", "DisMuon_pfRelIso04_all", "DisMuon_mT"]


def load(folder, dataset):
    path = f"{folder}/faster_trial_{dataset}/faster_trial_{dataset}.root"
    if not os.path.exists(path):
        return None
    tree = uproot.open(path)["Events"]
    missing = [b for b in branches_wanted if b not in tree.keys()]
    if missing:
        sys.exit(f"{path} is missing branches {missing}")
    return tree.arrays(branches_wanted)


def event_keys(arr):
    return list(zip(ak.to_numpy(arr.run).tolist(), ak.to_numpy(arr.luminosityBlock).tolist(), ak.to_numpy(arr.event).tolist()))


def cos_alpha(eta1, phi1, eta2, phi2):
    """3D opening angle between two directions: -1 = back-to-back (cosmic legs), +1 = same direction (duplicates)"""
    return np.cos(phi1 - phi2) / (np.cosh(eta1) * np.cosh(eta2)) + np.tanh(eta1) * np.tanh(eta2)


def delta_r(eta1, phi1, eta2, phi2):
    dphi = (phi1 - phi2 + np.pi) % (2 * np.pi) - np.pi
    return np.sqrt((eta1 - eta2) ** 2 + dphi ** 2)


def label(ax):
    if hep is not None:
        hep.cms.label(label="Private Work", data=False, com=13.6, ax=ax, loc=0, fontsize=16)


def save(fig, fname):
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{outdir}/{fname}.{ext}", dpi=150)
    plt.close(fig)


datasets = sorted(os.path.basename(d).replace("faster_trial_", "", 1)
                  for d in glob.glob(f"{args.test}/faster_trial_*") if os.path.isdir(d))
if not datasets:
    sys.exit(f"No merged datasets found in {args.test}")

groups = ("signal", "background")
keys = ["nmu_ref", "nmu_test", "pt_ref", "pt_test", "dxy_ref", "dxy_test",
        "partner_pt", "partner_dr", "partner_cosa", "n_ahead"]
h = {k: {g: [] for g in groups} for k in keys}

rows = []
for ds in datasets:
    ref, test = load(args.ref, ds), load(args.test, ds)
    if test is None or len(test) == 0:
        continue
    grp = "signal" if ds.startswith("Stau") else "background"
    n_ref = 0 if ref is None else len(ref)

    ## 1. event overlap
    test_keys = event_keys(test)
    test_index = {k: i for i, k in enumerate(test_keys)}
    ref_keys = [] if ref is None else event_keys(ref)
    common = [(i_ref, test_index[k]) for i_ref, k in enumerate(ref_keys) if k in test_index]
    ref_only, test_only = n_ref - len(common), len(test) - len(common)

    ## 2. same candidate muon in shared events?
    n_same = 0
    if common:
        ir, it = (np.array(x) for x in zip(*common))
        r0, t0 = ref[ir], test[it]
        same = ((ak.to_numpy(r0.DisMuon_pt[:, 0]) == ak.to_numpy(t0.DisMuon_pt[:, 0]))
                & (ak.to_numpy(r0.DisMuon_dxy[:, 0]) == ak.to_numpy(t0.DisMuon_dxy[:, 0]))
                & (ak.to_numpy(r0.DisMuon_eta[:, 0]) == ak.to_numpy(t0.DisMuon_eta[:, 0])))
        n_same = int(same.sum())

    ## 3. candidate passes every muon cut of the selection
    c = {f: ak.to_numpy(test[f"DisMuon_{f}"][:, 0]) for f in ["pt", "eta", "dxy", "pfRelIso03_all", "pfRelIso04_all", "mT"]}
    passes = ((c["pt"] > sel["muon_pt_min"]) & (np.abs(c["eta"]) < 2.4)
              & (np.abs(c["dxy"]) > sel["muon_dxy_min"]) & (np.abs(c["dxy"]) < sel["muon_dxy_max"])
              & (c["pfRelIso03_all"] > sel["muon_iso_min"]) & (c["pfRelIso03_all"] < sel["muon_iso_max"])
              & (c["pfRelIso04_all"] > sel["muon_iso04_min"]) & (c["pfRelIso04_all"] < sel["muon_iso04_max"])
              & (c["mT"] > sel["muon_mt_min"]) & (c["mT"] < sel["muon_mt_max"]))
    nmu = ak.to_numpy(ak.num(test.DisMuon_pt))

    ## 4. gained events: muons ahead of the candidate
    n_ahead0 = 0
    if test_only:
        shared = set(i for _, i in common)
        g = test[np.array([i for i in range(len(test)) if i not in shared])]
        n_ahead = ak.to_numpy(ak.sum(g.DisMuon_pt[:, 1:] > g.DisMuon_pt[:, 0], axis=1))
        n_ahead0 = int((n_ahead == 0).sum())
        h["n_ahead"][grp].append(n_ahead)

    rows.append(dict(dataset=ds, group=grp, n_ref=n_ref, n_test=len(test), common=len(common), ref_only=ref_only,
                     test_only=test_only, same_cand=n_same, gained_none_ahead=n_ahead0,
                     cand_fail=int((~passes).sum()), frac_partner=float(np.mean(nmu > 1))))

    ## histograms
    h["nmu_test"][grp].append(nmu)
    h["pt_test"][grp].append(c["pt"])
    h["dxy_test"][grp].append(np.abs(c["dxy"]))
    if n_ref:
        h["nmu_ref"][grp].append(ak.to_numpy(ak.num(ref.DisMuon_pt)))
        h["pt_ref"][grp].append(ak.to_numpy(ref.DisMuon_pt[:, 0]))
        h["dxy_ref"][grp].append(np.abs(ak.to_numpy(ref.DisMuon_dxy[:, 0])))
    p_eta, p_phi = test.DisMuon_eta[:, 1:], test.DisMuon_phi[:, 1:]
    c_eta, c_phi = test.DisMuon_eta[:, 0], test.DisMuon_phi[:, 0]
    h["partner_pt"][grp].append(ak.to_numpy(ak.flatten(test.DisMuon_pt[:, 1:])))
    h["partner_dr"][grp].append(ak.to_numpy(ak.flatten(delta_r(p_eta, p_phi, c_eta, c_phi))))
    h["partner_cosa"][grp].append(ak.to_numpy(ak.flatten(cos_alpha(p_eta, p_phi, c_eta, c_phi))))

## ---------------- table ----------------
cols = ["n_ref", "n_test", "common", "ref_only", "test_only", "same_cand", "gained_none_ahead", "cand_fail", "frac_partner"]
w = {c: max(len(c), 7) for c in cols}
hdr = f"{'dataset':32s} " + " ".join(f"{c:>{w[c]}s}" for c in cols)
lines = [f"ref : {args.ref_label}  ({args.ref})", f"test: {args.test_label}  ({args.test})", "", hdr, "-" * len(hdr)]
for r in rows:
    lines.append(f"{r['dataset']:32s} " + " ".join(f"{r[c]:{w[c]}.3f}" if c == "frac_partner" else f"{r[c]:{w[c]}d}" for c in cols))
tot = {c: sum(r[c] for r in rows) for c in cols if c != "frac_partner"}
lines.append(f"{'TOTAL':32s} " + " ".join(f"{tot[c]:{w[c]}d}" if c in tot else f"{'':{w[c]}s}" for c in cols))
lines += ["",
          "Expect for leading-muon -> Method C on the same input: ref_only = 0, same_cand = common,",
          "gained_none_ahead = 0, cand_fail = 0. For a rerun of the same method: test_only = ref_only = 0 too."]
summary = "\n".join(lines)
print(summary)
with open(f"{outdir}/summary.txt", "w") as f:
    f.write(summary + "\n")
with open(f"{outdir}/summary.csv", "w") as f:
    f.write("dataset," + ",".join(cols) + "\n")
    for r in rows:
        f.write(r["dataset"] + "," + ",".join(str(r[c]) for c in cols) + "\n")

## ---------------- per-dataset closure plot ----------------
for grp in groups:
    rs = [r for r in rows if r["group"] == grp and r["n_ref"] > 0]
    if not rs:
        continue
    x = np.arange(len(rs))
    kept = np.array([r["same_cand"] / r["n_ref"] for r in rs])
    gain = np.array([r["test_only"] / r["n_ref"] for r in rs])
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(max(10, 0.38 * len(rs)), 8), sharex=True)
    a1.plot(x, 100 * kept, "o", color="#5790fc")
    a1.axhline(100, color="gray", ls="--", lw=1)
    a1.set_ylabel("Reference events kept,\nsame candidate [%]", fontsize=14)
    a1.set_ylim(min(99, 100 * kept.min() - 0.5), 100.5)
    label(a1)
    a2.bar(x, 100 * gain, color="#f89c20")
    a2.set_ylabel("Events gained [%]", fontsize=14)
    a2.set_xticks(x)
    a2.set_xticklabels([r["dataset"] for r in rs], rotation=90, fontsize=10)
    a1.text(0.98, 0.08, f"{args.test_label} vs {args.ref_label}", transform=a1.transAxes, ha="right", fontsize=13)
    save(fig, f"closure_per_dataset_{grp}")

## ---------------- overlays with ratio panel ----------------
def overlay(kref, ktest, bins, xlabel, fname, logx=False):
    for grp in groups:
        have_ref, have_test = bool(h[kref][grp]), bool(h[ktest][grp])
        if not have_test:
            continue
        test_v = np.concatenate(h[ktest][grp])
        ct, edges = np.histogram(test_v, bins=bins)
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
        if have_ref:
            ref_v = np.concatenate(h[kref][grp])
            cr, _ = np.histogram(ref_v, bins=bins)
            a1.stairs(cr, edges, label=f"{args.ref_label} (N={ref_v.size})", lw=2, color="#5790fc")
        a1.stairs(ct, edges, label=f"{args.test_label} (N={test_v.size})", lw=2, ls="--", color="#e42536")
        a1.set_yscale("log")
        a1.set_ylabel("Events (unweighted)")
        a1.legend(fontsize=13, title=grp, title_fontsize=13)
        label(a1)
        if have_ref:
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.where(cr > 0, ct / cr, np.nan)
            centers = 0.5 * (edges[1:] + edges[:-1])
            a2.plot(centers, ratio, "o", color="black", ms=4)
            a2.axhline(1, color="gray", ls="--", lw=1)
            finite = ratio[np.isfinite(ratio)]
            lo, hi = (finite.min(), finite.max()) if finite.size else (1, 1)
            a2.set_ylim(min(0.95, lo - 0.05), max(1.05, hi * 1.08 + 0.02))
            a2.set_ylabel("test / ref", fontsize=14)
        a2.set_xlabel(xlabel)
        if logx:
            a2.set_xscale("log")
        save(fig, f"{fname}_{grp}")


overlay("nmu_ref", "nmu_test", np.arange(0.5, 7.5, 1), "Number of DisMuons per event", "n_dismuons")
overlay("pt_ref", "pt_test", np.linspace(0, 500, 51), r"Candidate muon $p_T$ [GeV]", "candidate_pt")
overlay("dxy_ref", "dxy_test", np.logspace(-1, 1, 41), r"Candidate muon $|d_{xy}|$ [cm]", "candidate_dxy", logx=True)

## ---------------- single distributions ----------------
def single(key, bins, xlabel, fname, note=None):
    for grp in groups:
        if not h[key][grp]:
            continue
        v = np.concatenate(h[key][grp])
        if v.size == 0:
            continue
        cnt, edges = np.histogram(v, bins=bins)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.stairs(cnt, edges, lw=2, color="#964a8b", label=f"{args.test_label}, {grp} (N={v.size})")
        ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Entries (unweighted)")
        ax.legend(fontsize=13)
        if note:
            ax.text(0.97, 0.75, note(v), transform=ax.transAxes, ha="right", fontsize=13)
        label(ax)
        save(fig, f"{fname}_{grp}")


single("n_ahead", np.arange(-0.5, 5.5, 1), r"Higher-$p_T$ muons ahead of the candidate (gained events)", "gained_n_ahead",
       note=lambda v: f"0 ahead: {int((v == 0).sum())} events")
single("partner_pt", np.linspace(0, 300, 61), r"Partner muon $p_T$ [GeV]", "partner_pt")
single("partner_dr", np.linspace(0, 6, 61), r"$\Delta R$(partner, candidate)", "partner_dR")
single("partner_cosa", np.linspace(-1, 1, 41), r"$\cos\alpha$(partner, candidate)", "partner_cosalpha")
print(f"\nPlots written to {outdir}/")
