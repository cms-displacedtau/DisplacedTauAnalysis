#!/usr/bin/env python3
"""
Basic data validation, to run every time a new data skim is made.

1. Luminosity input for brilcalc: the run/lumisection list of the data actually skimmed
   (golden-JSON format), plus a coverage check against the golden JSON.
2. Yield per run / luminosity per run at the loosest selection (the skim: MET triggers, filters,
   >=1 DisMuon, >=1 jet). It should be flat across runs; steps or outliers point to data problems.
   Needs the per-run luminosity from brilcalc (--brilcalc); without it, raw yields per run are plotted.
3. 2D eta-phi maps of DisMuons and jets, per era and combined (holes or hot spots = detector issues).

Blinding: no number printed or saved by this script isolates events in a blinded region.
 - Yields per run are counted at the loosest selection (the whole skim). Their total is the known skim
   total, which says nothing about the signal region.
 - The eta-phi maps leave out every event that could be in the SR box: any DisMuon with |dxy| > 0.1 cm and
   iso03 < 0.18 together with any jet with |dxy| > 0.1 cm and disTauTag score > 0.8 (looser than the real
   SR box, so it removes more than needed, never less). How many events this removes is never counted,
   because that number would itself be a count in a blinded region.

Run inside the coffea container, from plotting/:
  python data_validation.py                                  # lumi JSONs, raw yields, maps
  python data_validation.py --brilcalc lumi_by_run_EG.csv    # + yield / lumi per run
Output: plots/data_validation/<tag>/
"""
import os, glob, json, argparse
from collections import defaultdict
import numpy as np
import awkward as ak
import uproot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

try:
    import mplhep as hep
    hep.style.use("CMS")
except ImportError:
    hep = None

parser = argparse.ArgumentParser()
parser.add_argument("--skim", default="/eos/uscms/store/user/lpcdisptau/bskipwor/displacedTaus/skim/Summer22_CHS_v19/mutau/v20")
parser.add_argument("--datasets", nargs="*", default=["JetMET_Run2022E", "JetMET_Run2022G"])
parser.add_argument("--golden", default="/eos/uscms/store/user/lpcdisptau/json/Cert_Collisions2022_355100_362760_Golden.json")
parser.add_argument("--brilcalc", default=None, help="brilcalc per-run CSV (brilcalc lumi ... -u /pb -o file.csv)")
parser.add_argument("--lumi", type=float, default=None, help="total luminosity in fb^-1 for the plot labels")
parser.add_argument("--tag", default="JetMET_2022EG")
args = parser.parse_args()
outdir = f"plots/data_validation/{args.tag}"
os.makedirs(outdir, exist_ok=True)

BLIND_MU_DXY, BLIND_MU_ISO, BLIND_JET_DXY, BLIND_JET_SCORE = 0.1, 0.18, 0.1, 0.8
branches = ["run", "luminosityBlock",
            "DisMuon_pt", "DisMuon_eta", "DisMuon_phi", "DisMuon_dxy", "DisMuon_pfRelIso03_all",
            "Jet_pt", "Jet_eta", "Jet_phi", "Jet_dxy", "Jet_disTauTag_score1"]
eta_bins, phi_bins = np.linspace(-2.5, 2.5, 51), np.linspace(-np.pi, np.pi, 65)

yields = defaultdict(int)                      # run -> events after the blinding veto
run_era = {}
lumisections = set()                           # (run, ls) seen in the skim (all events, before the veto)
maps = {"DisMuon": defaultdict(lambda: np.zeros((50, 64))), "Jet": defaultdict(lambda: np.zeros((50, 64)))}
n_total = defaultdict(int)

if not any(glob.glob(f"{args.skim}/{ds}/*.root") for ds in args.datasets):
    raise SystemExit(f"No skim files found in {args.skim} for {args.datasets}")
for ds in args.datasets:
    files = sorted(glob.glob(f"{args.skim}/{ds}/*.root"))
    print(f"{ds}: {len(files)} skim files")
    for f in files:
        t = uproot.open(f)["Events"]
        if t.num_entries == 0:
            continue
        for a in t.iterate(branches, step_size="200 MB"):
            run = ak.to_numpy(a.run).astype(np.int64)
            ls = ak.to_numpy(a.luminosityBlock).astype(np.int64)
            lumisections.update(zip(*np.unique(np.stack([run, ls]), axis=1)))

            ## yields per run: the loosest selection, every skimmed event
            n_total[ds] += len(run)
            for r, n in zip(*np.unique(run, return_counts=True)):
                yields[int(r)] += int(n); run_era[int(r)] = ds

            ## maps: leave out every event that could be in the SR box (never counted)
            mu_box = (abs(a.DisMuon_dxy) > BLIND_MU_DXY) & (a.DisMuon_pfRelIso03_all < BLIND_MU_ISO)
            jet_box = (abs(a.Jet_dxy) > BLIND_JET_DXY) & (a.Jet_disTauTag_score1 > BLIND_JET_SCORE)
            a = a[~ak.to_numpy(ak.any(mu_box, axis=1) & ak.any(jet_box, axis=1))]
            for obj, ptmin in (("DisMuon", 20.), ("Jet", 20.)):
                sel = a[f"{obj}_pt"] > ptmin
                h, _, _ = np.histogram2d(ak.to_numpy(ak.flatten(a[f"{obj}_eta"][sel])),
                                         ak.to_numpy(ak.flatten(a[f"{obj}_phi"][sel])), bins=[eta_bins, phi_bins])
                maps[obj][ds] += h

for ds in args.datasets:
    print(f"{ds}: {n_total[ds]} skimmed events")


def label(ax, lumi=args.lumi):
    if hep is None:
        return
    kw = dict(label="Private Work", data=True, com=13.6, ax=ax, loc=0, fontsize=16)
    if lumi is not None:
        kw["lumi"] = lumi
    hep.cms.label(**kw)


def save(fig, name):
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{outdir}/{name}.{ext}", dpi=150)
    plt.close(fig)


def to_ranges(lss):
    out = []
    for l in sorted(lss):
        if out and l == out[-1][1] + 1:
            out[-1][1] = l
        else:
            out.append([l, l])
    return out


## ---------- 1. lumisection lists for brilcalc ----------
by_run = defaultdict(set)
for r, l in lumisections:
    by_run[int(r)].add(int(l))
skim_json = {str(r): to_ranges(by_run[r]) for r in sorted(by_run)}
golden = json.load(open(args.golden))
golden_runs_in_skim = {r: v for r, v in golden.items() if int(r) in by_run}
n_gold_ls = sum(b - a + 1 for v in golden_runs_in_skim.values() for a, b in v)
n_skim_ls = sum(len(v) for v in by_run.values())
not_golden = sum(1 for r, ls in by_run.items() for l in ls
                 if not any(a <= l <= b for a, b in golden.get(str(r), [])))
for name, content in (("lumis_from_skim", skim_json), ("lumis_golden_runs_in_skim", golden_runs_in_skim)):
    with open(f"{outdir}/{name}_{args.tag}.json", "w") as fp:
        json.dump(content, fp)
print(f"\nruns in skim: {len(by_run)}, lumisections in skim: {n_skim_ls}, golden lumisections in those runs: {n_gold_ls} "
      f"({100. * n_skim_ls / max(n_gold_ls, 1):.2f}% have >= 1 skimmed event); skim lumisections NOT in golden JSON: {not_golden} (must be 0)")
print(f"wrote {outdir}/lumis_from_skim_{args.tag}.json and {outdir}/lumis_golden_runs_in_skim_{args.tag}.json for brilcalc")

## ---------- 2. yield per run (/ luminosity) ----------
runs = np.array(sorted(yields))
n = np.array([yields[r] for r in runs], dtype=float)
eras = sorted(set(run_era.values()))
colors = dict(zip(eras, ["#5790fc", "#f89c20", "#e42536", "#964a8b"]))
with open(f"{outdir}/yields_per_run_{args.tag}.csv", "w") as fp:
    fp.write("run,era,events\n")
    for r, y in zip(runs, n):
        fp.write(f"{r},{run_era[r]},{int(y)}\n")

lumi_pb = {}
if args.brilcalc:
    for line in open(args.brilcalc):
        if line.startswith("#") or not line.strip():
            continue
        fields = line.strip().split(",")
        lumi_pb[int(fields[0].split(":")[0])] = float(fields[-1])     # recorded, in the unit given to -u (use /pb)
    missing = [r for r in runs if r not in lumi_pb]
    if missing:
        print(f"WARNING: {len(missing)} runs have events but no brilcalc luminosity, e.g. {missing[:5]}")
    print(f"total recorded luminosity from brilcalc: {sum(lumi_pb.values()) / 1000.:.3f} fb^-1 (if -u /pb was used)")

fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(runs))
if lumi_pb:
    ok = np.array([r in lumi_pb and lumi_pb[r] > 0 for r in runs])
    L = np.array([lumi_pb.get(r, np.nan) for r in runs])
    y, err = n / L, np.sqrt(n) / L
    mean = np.nansum(n[ok]) / np.nansum(L[ok])
    for era in eras:
        m = ok & np.array([run_era[r] == era for r in runs])
        ax.errorbar(x[m], y[m], yerr=err[m], fmt="o", ms=3, color=colors[era], label=era)
    ax.axhline(mean, color="gray", ls="--", label=f"average {mean:.1f} events/pb")
    ax.set_ylabel("Events / pb$^{-1}$ (loosest selection)")
    pull = (y - mean) / err
    out = [(int(r), round(float(p), 1)) for r, p, o in zip(runs, pull, ok) if o and abs(p) > 5]
    print(f"runs more than 5 sigma from the average: {len(out)} {out[:20]}")
else:
    for era in eras:
        m = np.array([run_era[r] == era for r in runs])
        ax.plot(x[m], n[m], "o", ms=3, color=colors[era], label=era)
    ax.set_ylabel("Events (loosest selection)")
    ax.text(0.5, 0.9, "no luminosity given (--brilcalc): raw yields, not expected to be flat",
            transform=ax.transAxes, ha="center", fontsize=12)
step = max(1, len(runs) // 15)
ax.set_xticks(x[::step]); ax.set_xticklabels(runs[::step], rotation=60, fontsize=10)
ax.set_xlabel("Run")
ax.legend(fontsize=12)
label(ax)
save(fig, "yield_per_lumi_vs_run" if lumi_pb else "yield_vs_run")

## ---------- 3. eta-phi maps ----------
for obj, title in (("DisMuon", "DisMuon (pT > 20 GeV)"), ("Jet", "Jet (pT > 20 GeV)")):
    for ds in eras + ["all"]:
        h = sum(maps[obj].values()) if ds == "all" else maps[obj][ds]
        if not np.any(h):
            continue
        fig, ax = plt.subplots(figsize=(10, 7))
        mesh = ax.pcolormesh(eta_bins, phi_bins, h.T, norm=LogNorm(vmin=max(h[h > 0].min(), 1)), cmap="viridis")
        fig.colorbar(mesh, ax=ax, label="Objects")
        ax.set_xlabel(r"$\eta$"); ax.set_ylabel(r"$\phi$")
        ax.text(0.02, 0.97, f"{title}, {ds}", transform=ax.transAxes, va="top", fontsize=13,
                bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"))
        label(ax)
        save(fig, f"etaphi_{obj}_{ds}")
print(f"\nPlots written to {outdir}/")
