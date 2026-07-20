import uproot
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------
# Input files
# -------------------------------------------------
file_nominal = "higgsCombine.scanFromSnapshot_wo.MultiDimFit.mH120.root"
file_mcstat  = "higgsCombine.scanFromSnapshot_with.MultiDimFit.mH120.root"

f = uproot.open("higgsCombine.scanFromSnapshot_with.MultiDimFit.mH120.root")
t = f["limit"]

print(t.arrays(["r", "deltaNLL"], library="np"))
# -------------------------------------------------
# Read trees
# -------------------------------------------------
with uproot.open(file_nominal) as f:
    tree = f["limit"]
    r_nom = tree["r"].array(library="np")
    delta_nom = tree["deltaNLL"].array(library="np")

with uproot.open(file_mcstat) as f:
    tree = f["limit"]
    r_mc = tree["r"].array(library="np")
    delta_mc = tree["deltaNLL"].array(library="np")

# -------------------------------------------------
# Remove duplicate best-fit entry (deltaNLL < 0)
# -------------------------------------------------
mask_nom = delta_nom >= 0
mask_mc = delta_mc >= 0

r_nom = r_nom[mask_nom]
delta_nom = delta_nom[mask_nom]

r_mc = r_mc[mask_mc]
delta_mc = delta_mc[mask_mc]

# -------------------------------------------------
# Sort by r
# -------------------------------------------------
idx = np.argsort(r_nom)
r_nom = r_nom[idx]
delta_nom = delta_nom[idx]

idx = np.argsort(r_mc)
r_mc = r_mc[idx]
delta_mc = delta_mc[idx]

# -------------------------------------------------
# Plot
# -------------------------------------------------
plt.figure(figsize=(7,5))

plt.plot(r_nom, 2*delta_nom,
         lw=2,
         label="Without QCD MC-stat")

plt.plot(r_mc, 2*delta_mc,
         lw=2,
         label="With QCD MC-stat")

# 95% CL threshold for one POI
plt.axhline(3.84,
            color="black",
            linestyle="--",
            label="95% CL")

plt.xlabel(r"Signal strength $r$")
plt.ylabel(r"$2\Delta \mathrm{NLL}$")

plt.xlim(left=0)
plt.ylim(bottom=0)

plt.grid(alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig("profile_likelihood_comparison.pdf")
plt.savefig("profile_likelihood_comparison.png", dpi=300)

