"""
Reweight a cτ₀ = 1000 mm sample to produce a cτ₀ = 500 mm sample.

The per-event weight is:
    w = (cτ_src / cτ_tgt) * exp( ct * (1/cτ_src - 1/cτ_tgt) )

where ct is the generator-level proper decay length of each stau.
For pair-produced staus where both must be reconstructed,
the event weight is the product of the two individual weights.

This script computes ct from GenPart_vx/vy/vz using the stau → τ decay topology.
"""

import numpy as np
import awkward as ak
import uproot

# ─── Configuration ───────────────────────────────────────────────────────────
INPUT_FILE   = "/eos/uscms/store/group/lpcdisptau/dally/displacedTaus/selected/Summer22_CHS_v19/mutau/v8/SR_score_wUncertainties/faster_trial/faster_trial_Stau_500_1000mm/faster_trial_Stau_500_1000mm.root"
TREE_NAME    = "Events"
OUTPUT_FILE  = "/eos/uscms/store/group/lpcdisptau/dally/displacedTaus/selected/Summer22_CHS_v19/mutau/v8/SR_score_wUncertainties/faster_trial/faster_trial_Stau_500_500mm/faster_trial_Stau_500_500mm.root"

CTAU_SOURCE  = 1000.0  # mm — the sample you have
CTAU_TARGET  =  500.0  # mm — the sample you want

STAU_PDGID   = 1000015
TAU_PDGID    = 15
STAU_MASS    = 100.0   # GeV — adjust to your mass point


# ─── Core reweighting function ───────────────────────────────────────────────
def lifetime_weight(ct: np.ndarray, ctau_src: float, ctau_tgt: float) -> np.ndarray:
    """
    Compute per-particle lifetime reweighting factor (shape only).

    Parameters
    ----------
    ct        : proper decay length (mm) for each LLP
    ctau_src  : proper decay length of the source MC sample (mm)
    ctau_tgt  : proper decay length of the target hypothesis (mm)

    Returns
    -------
    weight array
    """
    exponent = ct * (1.0 / ctau_src - 1.0 / ctau_tgt)
    return np.exp(exponent)


# ─── Compute ct from GenPart branches ────────────────────────────────────────
def compute_stau_ct(tree):
    """
    Compute the proper decay length ct (mm) for each stau in each event.

    Strategy:
      1. Find gen-level staus (|pdgId| == 1000015)
      2. Find their daughter taus (|pdgId| == 15 with mother index pointing to a stau)
      3. Stau production vertex = GenPart_v{x,y,z} of the stau
         Stau decay vertex     = GenPart_v{x,y,z} of the daughter tau
      4. L_3D = 3D distance between the two vertices (mm, since GenPart_v{x,y,z} is in cm → convert)
      5. ct = L_3D * m_stau / p_stau

    Returns
    -------
    ct1, ct2 : numpy arrays of shape (n_events,) — proper decay lengths of the two staus
    """
    # Read needed branches
    pdgId      = tree["GenPart_pdgId"].array()
    mother_idx = tree["GenPart_genPartIdxMother"].array()
    pt         = tree["GenPart_pt"].array()
    eta        = tree["GenPart_eta"].array()
    phi        = tree["GenPart_phi"].array()
    mass       = tree["GenPart_mass"].array()
    vx         = tree["GenPart_vx"].array()
    vy         = tree["GenPart_vy"].array()
    vz         = tree["GenPart_vz"].array()

    n_events = len(pdgId)
    ct1 = np.zeros(n_events)
    ct2 = np.zeros(n_events)

    for ievt in range(n_events):
        evt_pdg    = pdgId[ievt]
        evt_mother = mother_idx[ievt]
        evt_pt     = pt[ievt]
        evt_eta    = eta[ievt]
        evt_vx     = vx[ievt]
        evt_vy     = vy[ievt]
        evt_vz     = vz[ievt]

        # Find stau indices
        stau_indices = [i for i, pid in enumerate(evt_pdg) if abs(pid) == STAU_PDGID]

        stau_ct_list = []

        for stau_idx in stau_indices:
            # Find daughter tau: particle with |pdgId|==15 whose mother is this stau
            daughter_idx = None
            for j, pid in enumerate(evt_pdg):
                if abs(pid) == TAU_PDGID and evt_mother[j] == stau_idx:
                    daughter_idx = j
                    break

            if daughter_idx is None:
                # Try: stau may radiate and appear multiple times in the chain.
                # Look for any daughter whose mother chain traces back to this stau.
                # For simplicity, also check if the stau decays to itself (status copies)
                # and follow the chain.
                last_copy = stau_idx
                while True:
                    found_copy = None
                    for j, pid in enumerate(evt_pdg):
                        if abs(pid) == STAU_PDGID and evt_mother[j] == last_copy and j != last_copy:
                            found_copy = j
                            break
                    if found_copy is not None:
                        last_copy = found_copy
                    else:
                        break
                # Now search for tau daughter of the last copy
                for j, pid in enumerate(evt_pdg):
                    if abs(pid) == TAU_PDGID and evt_mother[j] == last_copy:
                        daughter_idx = j
                        stau_idx = last_copy  # use the last copy for vertex
                        break

            if daughter_idx is None:
                continue

            # Production vertex of stau (cm) and decay vertex = daughter's production vertex (cm)
            # GenPart_vx/vy/vz are in cm in NanoAOD → convert to mm
            dx = (evt_vx[daughter_idx] - evt_vx[stau_idx]) * 10.0  # cm → mm
            dy = (evt_vy[daughter_idx] - evt_vy[stau_idx]) * 10.0
            dz = (evt_vz[daughter_idx] - evt_vz[stau_idx]) * 10.0

            L3D = np.sqrt(dx**2 + dy**2 + dz**2)

            # Stau momentum
            stau_pt  = evt_pt[stau_idx]
            stau_eta = evt_eta[stau_idx]
            p_stau   = stau_pt * np.cosh(stau_eta)  # |p| = pt * cosh(eta)

            # Proper decay length: ct = L * m / p
            if p_stau > 0:
                ct_val = L3D * STAU_MASS / p_stau
            else:
                ct_val = 0.0

            stau_ct_list.append(ct_val)

        # Store the two stau ct values
        if len(stau_ct_list) >= 2:
            ct1[ievt] = stau_ct_list[0]
            ct2[ievt] = stau_ct_list[1]
        elif len(stau_ct_list) == 1:
            ct1[ievt] = stau_ct_list[0]
            ct2[ievt] = 0.0
        # else: both remain 0 (shouldn't happen in signal)

    return ct1, ct2


# ─── Main workflow ───────────────────────────────────────────────────────────
def reweight():
    """Read the tree, compute stau ct, calculate weights, write output."""

    print(f"Reading {INPUT_FILE}...")
    with uproot.open(f"{INPUT_FILE}:{TREE_NAME}") as tree:
        ct1, ct2 = compute_stau_ct(tree)

    # Sanity checks
    print(f"\nStau 1 ct (mm): mean={ct1.mean():.2f}, median={np.median(ct1):.2f}, "
          f"max={ct1.max():.2f}")
    print(f"Stau 2 ct (mm): mean={ct2.mean():.2f}, median={np.median(ct2):.2f}, "
          f"max={ct2.max():.2f}")
    print(f"Expected mean ct for {CTAU_SOURCE} mm sample: {CTAU_SOURCE:.1f} mm")

    # Compute per-stau weights (shape only — no prefactor)
    w1 = lifetime_weight(ct1, CTAU_SOURCE, CTAU_TARGET)
    w2 = lifetime_weight(ct2, CTAU_SOURCE, CTAU_TARGET)

    # ── Choose your event weight ──────────────────────────────────────────
    # Option A: Both staus must be reconstructed (your semi-leptonic analysis
    #           requires both a displaced muon AND a displaced tau_h)
    weight_pair = w1 * w2

    # Option B: If only one stau matters for your selection, use just that one.
    # weight_single = w1  # or w2, depending on which you reconstruct

    weights = weight_pair  # ← change to weight_single if appropriate

    # Diagnostics
    n = len(weights)
    print(f"\n{'='*50}")
    print(f"Events:            {n}")
    print(f"Mean weight:       {weights.mean():.4f}")
    print(f"Max weight:        {weights.max():.4f}")
    print(f"Min weight:        {weights.min():.6f}")
    eff_n = weights.sum()**2 / (weights**2).sum()
    print(f"Eff. events:       {eff_n:.0f}  ({100*eff_n/n:.1f}%)")
    print(f"{'='*50}")

    # Write output
    print(f"\nWriting {OUTPUT_FILE}...")
    with uproot.recreate(OUTPUT_FILE) as fout:
        with uproot.open(f"{INPUT_FILE}:{TREE_NAME}") as tree:
            branches = tree.arrays(library="np")
        branches["weight_ct500"] = weights
        branches["GenStau1_ct"]  = ct1   # store for debugging/validation
        branches["GenStau2_ct"]  = ct2
        branches["weight_ct500_stau1"] = w1
        branches["weight_ct500_stau2"] = w2
        fout[TREE_NAME] = branches

    print("Done. New branches: weight_ct500, GenStau1_ct, GenStau2_ct, "
          "weight_ct500_stau1, weight_ct500_stau2")

    # ── Validation ────────────────────────────────────────────────────────
    print(f"\n─── Validation ───")
    print(f"[Stau 1] Mean ct = {ct1.mean():.2f} mm")
    print(f"[Stau 2] Mean ct = {ct2.mean():.2f} mm")
    if ct1.mean() < CTAU_SOURCE * 0.5:
        print(f"  ℹ Note: Mean ct is well below {CTAU_SOURCE} mm. This is expected")
        print(f"    for skimmed samples — your SR selections preferentially keep")
        print(f"    events with moderate ct (inside the tracker acceptance).")
        print(f"    The reweighting is still valid.")
    else:
        ratio = ct1.mean() / CTAU_SOURCE
        if abs(ratio - 1.0) > 0.1:
            print(f"  ⚠ WARNING: mean ct differs from expected by {100*(ratio-1):.1f}%.")
            print(f"    Check GenPart vertex units or stau chain logic.")
        else:
            print(f"  ✓ Mean ct consistent with {CTAU_SOURCE} mm sample (pre-selection).")


if __name__ == "__main__":
    reweight()
