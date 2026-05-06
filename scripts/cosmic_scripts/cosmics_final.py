import os
import pickle
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from hist import Hist, axis
from coffea import processor
from coffea.nanoevents import PFNanoAODSchema
import coffea.nanoevents.methods.vector as vector
from coffea.dataset_tools import preprocess

PFNanoAODSchema.warn_missing_crossrefs = False
PFNanoAODSchema.mixins["DisMuon"] = "Muon"

def save_comparison_overlay(h, var_name, PREFIX, OUTPUT_DIR, title_suffix="", filename_suffix="", log_y=False, normalize=True):
    import numpy as np
    
    if np.sum(h.values()) == 0:
        print(f"Skipping {var_name} (Empty)")
        return

    fig, ax = plt.subplots(figsize=(8, 7))
    for label in h.axes["cat"]:
        h_slice = h[{"cat": label}]
        total_events = np.sum(h_slice.values())
        
        if total_events == 0:
            continue
            
        if normalize:
            # Scale the histogram so the area under the curve is 1.0
            h_scaled = h_slice * (1.0 / total_events)
            h_scaled.plot1d(ax=ax, label=label)
        else:
            # Plot raw event counts
            h_slice.plot1d(ax=ax, label=label)
    
    ax.legend(title="Category")
    
    if normalize:
        ax.set_ylabel("Fraction of Events")
    else:
        ax.set_ylabel("Events")
        
    ax.set_title(f"Comparison: {var_name} {title_suffix}")
    
    if log_y:
        ax.set_yscale("log")
    
    outpath = os.path.join(OUTPUT_DIR, f"{PREFIX}{var_name}_{filename_suffix}.pdf")
    fig.savefig(outpath)
    plt.close(fig)
    print(f"    Saved comparison plot to: {outpath}")

def save_2d_profile_overlay(h, var_name, PREFIX, OUTPUT_DIR, title_suffix="", filename_suffix=""):
    import numpy as np
    if np.sum(h.values()) == 0:
        return

    # Iterate through each category
    for label in h.axes["cat"]:
        h_slice = h[{"cat": label}]
        if np.sum(h_slice.values()) == 0:
            continue

        fig, ax = plt.subplots(figsize=(8, 7))

        # Plot the 2D colormap
        h_slice.plot2d(ax=ax, cmap="viridis")

        # Calculate the profile (Mean Y per X bin)
        x_centers = h_slice.axes[0].centers
        y_centers = h_slice.axes[1].centers
        counts = h_slice.values() 

        profile_x = []
        profile_y = []
        profile_yerr = []

        for i in range(len(x_centers)):
            bin_counts = counts[i, :]
            total_in_bin = np.sum(bin_counts)
            if total_in_bin > 0:
                mean_y = np.average(y_centers, weights=bin_counts)
                # Calculate standard error of the mean for accurate error bars
                variance = np.average((y_centers - mean_y)**2, weights=bin_counts)
                std_dev = np.sqrt(variance)
                std_err = std_dev / np.sqrt(total_in_bin) 

                profile_x.append(x_centers[i])
                profile_y.append(mean_y)
                profile_yerr.append(std_err)

        # Overlay the profile
        if profile_x:
            ax.errorbar(
                profile_x, profile_y, yerr=profile_yerr, 
                fmt='o', color='red', markersize=5, ecolor='red', 
                capsize=3, label="Profile (Mean ± StdErr)"
            )
            ax.legend()

        ax.set_title(f"{label}: {var_name} {title_suffix}")

        safe_label = label.replace(" ", "_").replace("(", "").replace(")", "")
        outpath = os.path.join(OUTPUT_DIR, f"{PREFIX}{var_name}_{safe_label}_{filename_suffix}.pdf")
        fig.savefig(outpath)
        plt.close(fig)
        print(f"    Saved 2D profile plot to: {outpath}")

def save_simple_2d_plot(h, var_name, PREFIX, OUTPUT_DIR, title_suffix="", filename_suffix="", log_z=False):
    import numpy as np
    import matplotlib.colors as mcolors
    if np.sum(h.values()) == 0:
        return

    for label in h.axes["cat"]:
        h_slice = h[{"cat": label}]
        if np.sum(h_slice.values()) == 0:
            continue

        fig, ax = plt.subplots(figsize=(8, 7))
        
        if log_z:
            # Apply logarithmic color scale and set vmin to 1 to avoid log(0) errors
            h_slice.plot2d(ax=ax, cmap="viridis", norm=mcolors.LogNorm(vmin=1))
        else:
            h_slice.plot2d(ax=ax, cmap="viridis")
        
        ax.set_title(f"{label}: {var_name} {title_suffix}")
        
        safe_label = label.replace(" ", "_").replace("(", "").replace(")", "")
        outpath = os.path.join(OUTPUT_DIR, f"{PREFIX}{var_name}_{safe_label}_{filename_suffix}_2D.pdf")
        fig.savefig(outpath)
        plt.close(fig)
        print(f"    Saved 2D plot to: {outpath}")


def delta_r_mb2_prop(reco_obj, gen_obj):
    """
    Calculates dR using the propagated fields
    """
    dphi = np.abs(reco_obj.phi_at_mb2 - gen_obj.prop_phi_at_mb2)
    dphi = ak.where(dphi > np.pi, 2*np.pi - dphi, dphi)
    deta = reco_obj.eta_at_mb2 - gen_obj.prop_eta_at_mb2
    return np.sqrt(deta**2 + dphi**2)

def delta_r_mb2_std(reco_obj, gen_obj):
    """
    Calculates dR using the standard unpropagated fields
    """
    dphi = np.abs(reco_obj.phi_at_mb2 - gen_obj.phi_at_mb2)
    dphi = ak.where(dphi > np.pi, 2*np.pi - dphi, dphi)
    deta = reco_obj.eta_at_mb2 - gen_obj.eta_at_mb2
    return np.sqrt(deta**2 + dphi**2)

class SingleMuonProcessor(processor.ProcessorABC):
    def __init__(self):
        self.output = {
            "n_events_initial": 0,
            "n_duplicates_removed": 0,
            "n_events_3plus_muons_post_cleaning": 0,
            
            "n_cosmic_events_total": 0,
            "n_cosmic_events_1_muon": 0,
            "n_cosmic_events_2plus_muons": 0,
            
            "n_nobptx_events_total": 0,
            "n_nobptx_events_1_muon": 0,
            "n_nobptx_events_2plus_muons": 0,

            "n_nobptx_raw_events_1": 0,
            "n_nobptx_raw_events_2": 0,
            "n_nobptx_raw_events_3plus": 0,
            "n_nobptx_raw_muons_3plus": 0,

            "n_cosmic_2mu_pre_cosA": 0,
            "n_cosmic_2mu_post_cosA": 0,
            "n_nobptx_2mu_pre_cosA": 0,
            "n_nobptx_2mu_post_cosA": 0,
            
            "delta_time_upper_lower": Hist(
                axis.StrCategory([], name="cat", label="Dataset", growth=True), 
                axis.Regular(100, -60, 60, name="val", label=r"$\Delta t$ (Upper - Lower) [ns]")
            ),
            
            "single_muon_pt": Hist(axis.StrCategory([], name="cat", label="Muon Category", growth=True), axis.Regular(100, 0, 100, name="val", label=r"Single Muon $p_T$ [GeV]")),
            "single_muon_eta": Hist(axis.StrCategory([], name="cat", label="Muon Category", growth=True), axis.Regular(100, -2.5, 2.5, name="val", label=r"Single Muon $\eta$")),
            "single_muon_phi": Hist(axis.StrCategory([], name="cat", label="Muon Category", growth=True), axis.Regular(100, -np.pi, np.pi, name="val", label=r"Single Muon $\phi$")),
            "single_muon_dxy": Hist(axis.StrCategory([], name="cat", label="Muon Category", growth=True), axis.Regular(100, -100, 100, name="val", label=r"Single Muon $d_{xy}$")),
            "single_muon_dz_overlay": Hist(axis.StrCategory([], name="cat", label="Muon Category", growth=True), axis.Regular(50, -300, 300, name="val", label=r"Single Muon $d_z$ [cm]")),
            "single_muon_validDTHits": Hist(axis.StrCategory([], name="cat", label="Muon Category", growth=True), axis.Regular(60, 0, 60, name="val", label="Valid DT Hits")),
            "single_muon_validCSCHits": Hist(axis.StrCategory([], name="cat", label="Muon Category", growth=True), axis.Regular(60, 0, 60, name="val", label="Valid CSC Hits")),
            "single_muon_validHits": Hist(axis.StrCategory([], name="cat", label="Muon Category", growth=True), axis.Regular(80, 0, 80, name="val", label="Total Valid Muon Hits")),
            "single_muon_dtStations": Hist(axis.StrCategory([], name="cat", label="Muon Category", growth=True), axis.Regular(50, 0, 50, name="val", label="DT Stations with Valid Hits")),
            "single_muon_timeAtIpInOut": Hist(axis.StrCategory([], name="cat", label="Muon Category", growth=True), axis.Regular(25, -60, 60, name="val", label="Time at IP InOut [ns]")),
            "single_muon_timeAtIpInOutErr": Hist(axis.StrCategory([], name="cat", label="Muon Category", growth=True), axis.Regular(75, 0, 5, name="val", label="Time at IP InOut Error [ns]")),
            "single_muon_timeNDof": Hist(axis.StrCategory([], name="cat", label="Muon Category", growth=True), axis.Regular(50, 0, 50, name="val", label="timeNDof")),
            
            "debug_dr3_eta_mb2": Hist(axis.StrCategory([], name="cat", label="Source", growth=True), axis.Regular(110, -105, 5, name="val", label=r"$\eta$ at MB2 ($\Delta R > 3.0$)")),
            "debug_dr3_phi_mb2": Hist(axis.StrCategory([], name="cat", label="Source", growth=True), axis.Regular(110, -105, 5, name="val", label=r"$\phi$ at MB2 ($\Delta R > 3.0$)")),

            "two_muons_cos_alpha": Hist(axis.StrCategory([], name="cat", label="Muon Category", growth=True), axis.Regular(100, -1.01, -0.9, name="val", label=r"$\cos\alpha$ (Upper vs Lower)")),
            
            "single_muon_timing_err_ndof7": Hist(axis.StrCategory([], name="cat", growth=True), axis.Regular(60, 0, 3, name="val", label="Time Error [ns] (Single, nDof > 7)")),
            "single_muon_timeErr_DT_only": Hist(axis.StrCategory([], name="cat", growth=True), axis.Regular(60, 0, 3, name="val", label="Time Error [ns] (DT > 0, CSC == 0)")),
            "single_muon_timeErr_CSC_only": Hist(axis.StrCategory([], name="cat", growth=True), axis.Regular(60, 0, 3, name="val", label="Time Error [ns] (DT == 0, CSC > 0)")),
            "single_muon_timeErr_DT_CSC_both": Hist(axis.StrCategory([], name="cat", growth=True), axis.Regular(60, 0, 3, name="val", label="Time Error [ns] (DT > 0, CSC > 0)")),

            "single_muon_timeErr_vs_DTHits": Hist(
                axis.StrCategory([], name="cat", growth=True), 
                axis.Regular(60, 0, 60, name="hits", label="Valid DT Hits"), 
                axis.Regular(45, 0, 3, name="err", label="Time Error [ns]")
            ),
            "single_muon_timeErr_vs_CSCHits": Hist(
                axis.StrCategory([], name="cat", growth=True), 
                axis.Regular(60, 0, 60, name="hits", label="Valid CSC Hits"), 
                axis.Regular(45, 0, 3, name="err", label="Time Error [ns]")
            ),
            "single_muon_timeErr_vs_TotalHits": Hist(
                axis.StrCategory([], name="cat", growth=True), 
                axis.Regular(60, 0, 60, name="hits", label="Total DT + CSC Hits"), 
                axis.Regular(45, 0, 3, name="err", label="Time Error [ns]")
            ),

            "uncut_upper_vs_lower_phi": Hist(
                axis.StrCategory([], name="cat", growth=True), 
                axis.Regular(50, 0, np.pi, name="upper", label=r"Upper Muon $\phi$"), 
                axis.Regular(50, -np.pi, 0, name="lower", label=r"Lower Muon $\phi$")
            ),
            "uncut_upper_vs_lower_eta": Hist(
                axis.StrCategory([], name="cat", growth=True), 
                axis.Regular(100, -2.5, 2.5, name="upper", label=r"Upper Muon $\eta$"), 
                axis.Regular(100, -2.5, 2.5, name="lower", label=r"Lower Muon $\eta$")
            ),
            "event_muon_multiplicity": Hist(
                axis.StrCategory([], name="cat", label="Dataset", growth=True), 
                axis.Regular(10, 0, 10, name="val", label="Total DisMuons per Event")
            ),
            "event_total_valid_hits": Hist(
                axis.StrCategory([], name="cat", label="Dataset", growth=True), 
                axis.Regular(20, 0, 200, name="val", label="Sum of Valid Muon Hits in Event")
            ),
            "event_standalone_fraction": Hist(
                axis.StrCategory([], name="cat", label="Dataset", growth=True), 
                axis.Regular(20, 0, 1.05, name="val", label="Fraction of Standalone Muons in Event")
            ),
        }

    def process(self, events):
        dataset = events.metadata.get("dataset", "Unknown")
        self.output["n_events_initial"] += len(events)
        
        events["DisMuon"] = ak.zip(
            {
                "pt": events.DisMuon.pt,
                "eta": events.DisMuon.eta,
                "phi": events.DisMuon.phi,
                "mass": events.DisMuon.mass,
                "charge": events.DisMuon.charge,          
                "timeNDof": events.DisMuon.timeNDof,
                "isStandalone": events.DisMuon.isStandalone,
                "isGlobal": events.DisMuon.isGlobal,
                "dxy": events.DisMuon.dxy,
                "dz": events.DisMuon.dz,
                "numberOfValidMuonDTHits": events.DisMuon.numberOfValidMuonDTHits,
                "numberOfValidMuonCSCHits": events.DisMuon.numberOfValidMuonCSCHits,
                "numberOfValidMuonHits": events.DisMuon.numberOfValidMuonHits,
                "dtStationsWithValidHits": events.DisMuon.dtStationsWithValidHits,
                "timeAtIpInOut": events.DisMuon.timeAtIpInOut,
                "timeAtIpInOutErr": events.DisMuon.timeAtIpInOutErr,
                "mediumId": events.DisMuon.mediumId,
                "pfRelIso03_all": events.DisMuon.pfRelIso03_all,
                "eta_at_mb2": events.DisMuon.eta_at_mb2,
                "phi_at_mb2": events.DisMuon.phi_at_mb2,
            },
            with_name="PtEtaPhiMLorentzVector",
            behavior=vector.behavior,
        )

        # Ensure we do not crash when searching for GenPart in data
        has_gen = "GenPart" in events.fields
        
        if has_gen:
            genpart_dict = {
                "pt": events.GenPart.pt,
                "eta": events.GenPart.eta,
                "phi": events.GenPart.phi,
                "mass": events.GenPart.mass,
                "pdgId": events.GenPart.pdgId,
                "status": events.GenPart.status,
                "eta_at_mb2": events.GenPart.eta_at_mb2, 
                "phi_at_mb2": events.GenPart.phi_at_mb2,
            }

            if "prop_eta_at_mb2" in events.GenPart.fields:
                genpart_dict["prop_eta_at_mb2"] = events.GenPart.prop_eta_at_mb2
                genpart_dict["prop_phi_at_mb2"] = events.GenPart.prop_phi_at_mb2
            elif "GenPart_prop_eta_at_mb2" in events.fields:
                genpart_dict["prop_eta_at_mb2"] = events["GenPart_prop_eta_at_mb2"]
                genpart_dict["prop_phi_at_mb2"] = events["GenPart_prop_phi_at_mb2"]

            events["GenPart"] = ak.zip(
                genpart_dict,
                with_name="PtEtaPhiMLorentzVector",
                behavior=vector.behavior,
            )

            gen_muons = events.GenPart[
                (abs(events.GenPart.pdgId) == 13) & 
                (events.GenPart.status == 1)
            ]
            
            gen_muons = gen_muons[(gen_muons.pt > 20) & 
                (abs(gen_muons.eta) < 2.4)
            ]
            
        else:
            gen_muons = None
        
        dismuon_mask = (events.DisMuon.pt > 20) & (abs(events.DisMuon.eta) < 2.4)
        events["DisMuon"] = events.DisMuon[dismuon_mask]

        dis_muons = events.DisMuon
        n_dismuons = ak.num(dis_muons)

        is_cosmic = "Cosmic" in dataset or dataset.startswith("LooseMu") or dataset == "test_cosmics_calib"
        is_nobptx = "NoBPTX" in dataset
        is_signal = not (is_cosmic or is_nobptx)
        ds_label = "Cosmic" if is_cosmic else ("NoBPTX" if is_nobptx else "Signal")

        # Track the total raw events explicitly before filtering
        if is_cosmic:
            self.output["n_cosmic_events_total"] += len(events)
        elif is_nobptx:
            self.output["n_nobptx_events_total"] += len(events)
        
        quality_mask = (events.DisMuon.mediumId == True) & (events.DisMuon.pfRelIso03_all < 0.18)
        quality_muons = events.DisMuon[quality_mask]
        
        mask_has_quality_muons = (ak.num(quality_muons) > 0)
        
        if ak.sum(mask_has_quality_muons) > 0:
            study_muons = quality_muons[mask_has_quality_muons]
            n_muons_in_event = ak.num(study_muons)
            sum_valid_hits = ak.sum(study_muons.numberOfValidMuonHits, axis=1)
            n_standalone = ak.sum((study_muons.isStandalone == True) & (study_muons.isGlobal == False), axis=1)
            standalone_fraction = n_standalone / n_muons_in_event
            
            self.output["event_muon_multiplicity"].fill(cat=ds_label, val=n_muons_in_event)
            self.output["event_total_valid_hits"].fill(cat=ds_label, val=sum_valid_hits)
            self.output["event_standalone_fraction"].fill(cat=ds_label, val=standalone_fraction)

            if is_nobptx:
                mask_1 = (n_dismuons == 1)
                mask_2 = (n_dismuons == 2)
                mask_3plus = (n_dismuons > 2)
                
                self.output["n_nobptx_raw_events_1"] += ak.sum(mask_1)
                self.output["n_nobptx_raw_events_2"] += ak.sum(mask_2)
                self.output["n_nobptx_raw_events_3plus"] += ak.sum(mask_3plus)
                self.output["n_nobptx_raw_muons_3plus"] += ak.sum(n_dismuons[mask_3plus])

        # ==========================================
        # SIGNAL-ONLY EVENT FILTERING
        # ==========================================
        if is_signal and has_gen:
            jets = events.Jet[
                (abs(events.Jet.eta) < 2.4) & 
                (events.Jet.pt > 20) & 
                (events.Jet.neHEF < 0.99) & 
                (events.Jet.neEmEF < 0.9) & 
                ((events.Jet.chMultiplicity + events.Jet.neMultiplicity) > 1) & 
                (events.Jet.chMultiplicity > 0) & 
                (events.Jet.muEF < 0.1) & 
                (events.Jet.chEmEF < 0.8)
            ]

            # Require exactly one GenMuon and exactly one Reco Jet (post-kinematic cuts)
            signal_mask = (ak.num(gen_muons) == 1) & (ak.num(jets) == 1)

            # Apply mask strictly to the signal events
            events = events[signal_mask]
            gen_muons = gen_muons[signal_mask]
            
            # Redefine DisMuons based on the newly filtered events array
            dis_muons = events.DisMuon
            n_dismuons = ak.num(dis_muons)

        # ==========================================
        # EXACTLY TWO MUONS (UNCUT) LOGIC
        # ==========================================
        mask_exactly_two_uncut = (n_dismuons == 2)
        if ak.sum(mask_exactly_two_uncut) > 0:
            two_muons_uncut = dis_muons[mask_exactly_two_uncut]
            
            # Sort by phi so [0] is the highest and [1] is the lowest
            sorted_by_phi = two_muons_uncut[ak.argsort(two_muons_uncut.phi, axis=1, ascending=False)]
            upper_candidates = sorted_by_phi[:, 0]
            lower_candidates = sorted_by_phi[:, 1]
            
            # Enforce that the upper candidate is strictly positive and the lower is strictly negative
            mask_opposite_hemispheres = (upper_candidates.phi > 0) & (lower_candidates.phi < 0)
            
            upper_final = upper_candidates[mask_opposite_hemispheres]
            lower_final = lower_candidates[mask_opposite_hemispheres]
            
            self.output["uncut_upper_vs_lower_phi"].fill(
                cat=ds_label, 
                upper=upper_final.phi, 
                lower=lower_final.phi
            )
            self.output["uncut_upper_vs_lower_eta"].fill(
                cat=ds_label, 
                upper=upper_final.eta, 
                lower=lower_final.eta
            )

        # ==========================================
        # SINGLE MUON LOGIC
        # ==========================================
        if has_gen:
            mask_single = (n_dismuons == 1) & (ak.num(gen_muons) >= 1)
        else:
            mask_single = (n_dismuons == 1)
        
        if ak.sum(mask_single) > 0:
            single_muons = events.DisMuon[mask_single][:, 0]
            if has_gen:
                valid_gen_muons = gen_muons[mask_single]

            mask_medium = (single_muons.mediumId == True)
            mask_iso = (single_muons.pfRelIso03_all < 0.18)
            
            single_muons = single_muons[mask_medium & mask_iso]
            if has_gen:
                valid_gen_muons = valid_gen_muons[mask_medium & mask_iso]
            
            if is_cosmic or is_nobptx:
                if is_cosmic:
                    self.output["n_cosmic_events_1_muon"] += len(single_muons)
                elif is_nobptx:
                    self.output["n_nobptx_events_1_muon"] += len(single_muons)
                    '''
                    # Create a mask for pT > 20
                    high_pt_mask = single_muons.pt > 20
                    high_pt_muons = single_muons[high_pt_mask]
                    
                    # If this chunk has ANY NoBPTX muons over 20 GeV, print them
                    if len(high_pt_muons) > 0:
                        print(f"\n--- Found {len(high_pt_muons)} NoBPTX muons with pT > 20 GeV! ---")
                        # We convert to a list or use print(ak.to_list()) to make the console output cleaner
                        print(f"pT values:  {high_pt_muons.pt.to_list()}")
                        print(f"eta values: {high_pt_muons.eta.to_list()}")
                    '''
                # Set prefix dynamically so we do not have to duplicate the block
                prefix = "Cosmic" if is_cosmic else "NoBPTX"

                mask_upper_single = single_muons.phi > 0
                mask_lower_single = single_muons.phi < 0
                upper_singles = single_muons[mask_upper_single]
                lower_singles = single_muons[mask_lower_single]
                
                if len(upper_singles) > 0:
                    self.output["single_muon_dz_overlay"].fill(cat=f"Upper {prefix}", val=upper_singles.dz)
                    self.output["single_muon_pt"].fill(cat=f"Upper {prefix}", val=upper_singles.pt)
                    self.output["single_muon_eta"].fill(cat=f"Upper {prefix}", val=upper_singles.eta)
                    self.output["single_muon_phi"].fill(cat=f"Upper {prefix}", val=upper_singles.phi)
                    self.output["single_muon_dxy"].fill(cat=f"Upper {prefix}", val=upper_singles.dxy)
                    self.output["single_muon_validDTHits"].fill(cat=f"Upper {prefix}", val=upper_singles.numberOfValidMuonDTHits)
                    self.output["single_muon_validCSCHits"].fill(cat=f"Upper {prefix}", val=upper_singles.numberOfValidMuonCSCHits)
                    self.output["single_muon_validHits"].fill(cat=f"Upper {prefix}", val=upper_singles.numberOfValidMuonHits)
                    self.output["single_muon_dtStations"].fill(cat=f"Upper {prefix}", val=upper_singles.dtStationsWithValidHits)
                    self.output["single_muon_timeAtIpInOut"].fill(cat=f"Upper {prefix}", val=upper_singles.timeAtIpInOut)
                    self.output["single_muon_timeAtIpInOutErr"].fill(cat=f"Upper {prefix}", val=upper_singles.timeAtIpInOutErr)
                    self.output["single_muon_timeNDof"].fill(cat=f"Upper {prefix}", val=upper_singles.timeNDof)
                    self.output["single_muon_timing_err_ndof7"].fill(cat=f"Upper {prefix}", val=upper_singles[upper_singles.timeNDof > 7].timeAtIpInOutErr)
                    
                    up_dt_only = (upper_singles.numberOfValidMuonDTHits > 0) & (upper_singles.numberOfValidMuonCSCHits == 0)
                    up_csc_only = (upper_singles.numberOfValidMuonDTHits == 0) & (upper_singles.numberOfValidMuonCSCHits > 0)
                    up_both = (upper_singles.numberOfValidMuonDTHits > 0) & (upper_singles.numberOfValidMuonCSCHits > 0)
                    
                    self.output["single_muon_timeErr_DT_only"].fill(cat=f"Upper {prefix}", val=upper_singles[up_dt_only].timeAtIpInOutErr)
                    self.output["single_muon_timeErr_CSC_only"].fill(cat=f"Upper {prefix}", val=upper_singles[up_csc_only].timeAtIpInOutErr)
                    self.output["single_muon_timeErr_DT_CSC_both"].fill(cat=f"Upper {prefix}", val=upper_singles[up_both].timeAtIpInOutErr)

                    self.output["single_muon_timeErr_vs_DTHits"].fill(cat=f"Upper {prefix}", hits=upper_singles.numberOfValidMuonDTHits[up_dt_only], err=upper_singles.timeAtIpInOutErr[up_dt_only])
                    self.output["single_muon_timeErr_vs_CSCHits"].fill(cat=f"Upper {prefix}", hits=upper_singles.numberOfValidMuonCSCHits[up_csc_only], err=upper_singles.timeAtIpInOutErr[up_csc_only])
                    self.output["single_muon_timeErr_vs_TotalHits"].fill(cat=f"Upper {prefix}", hits=(upper_singles.numberOfValidMuonDTHits[up_both] + upper_singles.numberOfValidMuonCSCHits[up_both]), err=upper_singles.timeAtIpInOutErr[up_both])
                
                if len(lower_singles) > 0:
                    self.output["single_muon_dz_overlay"].fill(cat=f"Lower {prefix}", val=lower_singles.dz)
                    self.output["single_muon_pt"].fill(cat=f"Lower {prefix}", val=lower_singles.pt)
                    self.output["single_muon_eta"].fill(cat=f"Lower {prefix}", val=lower_singles.eta)
                    self.output["single_muon_phi"].fill(cat=f"Lower {prefix}", val=lower_singles.phi)
                    self.output["single_muon_dxy"].fill(cat=f"Lower {prefix}", val=lower_singles.dxy)
                    self.output["single_muon_validDTHits"].fill(cat=f"Lower {prefix}", val=lower_singles.numberOfValidMuonDTHits)
                    self.output["single_muon_validCSCHits"].fill(cat=f"Lower {prefix}", val=lower_singles.numberOfValidMuonCSCHits)
                    self.output["single_muon_validHits"].fill(cat=f"Lower {prefix}", val=lower_singles.numberOfValidMuonHits)
                    self.output["single_muon_dtStations"].fill(cat=f"Lower {prefix}", val=lower_singles.dtStationsWithValidHits)
                    self.output["single_muon_timeAtIpInOut"].fill(cat=f"Lower {prefix}", val=lower_singles.timeAtIpInOut)
                    self.output["single_muon_timeAtIpInOutErr"].fill(cat=f"Lower {prefix}", val=lower_singles.timeAtIpInOutErr)
                    self.output["single_muon_timeNDof"].fill(cat=f"Lower {prefix}", val=lower_singles.timeNDof)
                    self.output["single_muon_timing_err_ndof7"].fill(cat=f"Lower {prefix}", val=lower_singles[lower_singles.timeNDof > 7].timeAtIpInOutErr)
                    
                    dn_dt_only = (lower_singles.numberOfValidMuonDTHits > 0) & (lower_singles.numberOfValidMuonCSCHits == 0)
                    dn_csc_only = (lower_singles.numberOfValidMuonDTHits == 0) & (lower_singles.numberOfValidMuonCSCHits > 0)
                    dn_both = (lower_singles.numberOfValidMuonDTHits > 0) & (lower_singles.numberOfValidMuonCSCHits > 0)
                    
                    self.output["single_muon_timeErr_DT_only"].fill(cat=f"Lower {prefix}", val=lower_singles[dn_dt_only].timeAtIpInOutErr)
                    self.output["single_muon_timeErr_CSC_only"].fill(cat=f"Lower {prefix}", val=lower_singles[dn_csc_only].timeAtIpInOutErr)
                    self.output["single_muon_timeErr_DT_CSC_both"].fill(cat=f"Lower {prefix}", val=lower_singles[dn_both].timeAtIpInOutErr)

                    self.output["single_muon_timeErr_vs_DTHits"].fill(cat=f"Lower {prefix}", hits=lower_singles.numberOfValidMuonDTHits[dn_dt_only], err=lower_singles.timeAtIpInOutErr[dn_dt_only])
                    self.output["single_muon_timeErr_vs_CSCHits"].fill(cat=f"Lower {prefix}", hits=lower_singles.numberOfValidMuonCSCHits[dn_csc_only], err=lower_singles.timeAtIpInOutErr[dn_csc_only])
                    self.output["single_muon_timeErr_vs_TotalHits"].fill(cat=f"Lower {prefix}", hits=(lower_singles.numberOfValidMuonDTHits[dn_both] + lower_singles.numberOfValidMuonCSCHits[dn_both]), err=lower_singles.timeAtIpInOutErr[dn_both])
            
            elif has_gen: # Signal logic
                dr_mb2_prop_array = delta_r_mb2_prop(single_muons, valid_gen_muons)
                min_dr_mb2_prop = ak.min(dr_mb2_prop_array, axis=1)
                mask_dr_prop = ak.fill_none(min_dr_mb2_prop < 0.4, False)
                matched_muons_prop = single_muons[mask_dr_prop]
                
                if len(matched_muons_prop) > 0:
                    self.output["single_muon_dz_overlay"].fill(cat="Signal (Propagated)", val=matched_muons_prop.dz)
                    self.output["single_muon_pt"].fill(cat="Signal (Propagated)", val=matched_muons_prop.pt)
                    self.output["single_muon_eta"].fill(cat="Signal (Propagated)", val=matched_muons_prop.eta)
                    self.output["single_muon_phi"].fill(cat="Signal (Propagated)", val=matched_muons_prop.phi)
                    self.output["single_muon_dxy"].fill(cat="Signal (Propagated)", val=matched_muons_prop.dxy)
                    self.output["single_muon_validDTHits"].fill(cat="Signal (Propagated)", val=matched_muons_prop.numberOfValidMuonDTHits)
                    self.output["single_muon_validCSCHits"].fill(cat="Signal (Propagated)", val=matched_muons_prop.numberOfValidMuonCSCHits)
                    self.output["single_muon_validHits"].fill(cat="Signal (Propagated)", val=matched_muons_prop.numberOfValidMuonHits)
                    self.output["single_muon_dtStations"].fill(cat="Signal (Propagated)", val=matched_muons_prop.dtStationsWithValidHits)
                    self.output["single_muon_timeAtIpInOut"].fill(cat="Signal (Propagated)", val=matched_muons_prop.timeAtIpInOut)
                    self.output["single_muon_timeAtIpInOutErr"].fill(cat="Signal (Propagated)", val=matched_muons_prop.timeAtIpInOutErr)
                    self.output["single_muon_timeNDof"].fill(cat="Signal (Propagated)", val=matched_muons_prop.timeNDof)
                    self.output["single_muon_timing_err_ndof7"].fill(cat="Signal (Propagated)", val=matched_muons_prop[matched_muons_prop.timeNDof > 7].timeAtIpInOutErr)
                    
                    sig_dt_only = (matched_muons_prop.numberOfValidMuonDTHits > 0) & (matched_muons_prop.numberOfValidMuonCSCHits == 0)
                    sig_csc_only = (matched_muons_prop.numberOfValidMuonDTHits == 0) & (matched_muons_prop.numberOfValidMuonCSCHits > 0)
                    sig_both = (matched_muons_prop.numberOfValidMuonDTHits > 0) & (matched_muons_prop.numberOfValidMuonCSCHits > 0)
                    
                    self.output["single_muon_timeErr_DT_only"].fill(cat="Signal (Propagated)", val=matched_muons_prop[sig_dt_only].timeAtIpInOutErr)
                    self.output["single_muon_timeErr_CSC_only"].fill(cat="Signal (Propagated)", val=matched_muons_prop[sig_csc_only].timeAtIpInOutErr)
                    self.output["single_muon_timeErr_DT_CSC_both"].fill(cat="Signal (Propagated)", val=matched_muons_prop[sig_both].timeAtIpInOutErr)

                    self.output["single_muon_timeErr_vs_DTHits"].fill(cat="Signal (Propagated)", hits=matched_muons_prop.numberOfValidMuonDTHits[sig_dt_only], err=matched_muons_prop.timeAtIpInOutErr[sig_dt_only])
                    self.output["single_muon_timeErr_vs_CSCHits"].fill(cat="Signal (Propagated)", hits=matched_muons_prop.numberOfValidMuonCSCHits[sig_csc_only], err=matched_muons_prop.timeAtIpInOutErr[sig_csc_only])
                    self.output["single_muon_timeErr_vs_TotalHits"].fill(cat="Signal (Propagated)", hits=(matched_muons_prop.numberOfValidMuonDTHits[sig_both] + matched_muons_prop.numberOfValidMuonCSCHits[sig_both]), err=matched_muons_prop.timeAtIpInOutErr[sig_both])
        
        # ==========================================
        # MULTIPLE MUON LOGIC
        # ==========================================
        if has_gen:
            mask_multiple_disMuon_event = (ak.num(events.DisMuon) >= 2) & (ak.num(gen_muons) >= 1)
        else:
            mask_multiple_disMuon_event = (ak.num(events.DisMuon) >= 2)
        
        if ak.sum(mask_multiple_disMuon_event) > 0:
            events = events[mask_multiple_disMuon_event]
            if has_gen:
                gen_muons = gen_muons[mask_multiple_disMuon_event]
            dis_muons = events.DisMuon

            # --- Apply cuts to Leading pT Muon First ---
            sorted_muons_temp = dis_muons[ak.argsort(dis_muons.pt, axis=1, ascending=False)]
            lead_muon_eval = sorted_muons_temp[:, 0]
            
            lead_quality_mask = (lead_muon_eval.mediumId == True) & (lead_muon_eval.pfRelIso03_all < 0.18)
            
            # Remove the event entirely if the leading muon fails the cuts
            events = events[lead_quality_mask]
            if has_gen:
                gen_muons = gen_muons[lead_quality_mask]
            sorted_muons = sorted_muons_temp[lead_quality_mask]
            
            # --- COUNTER: >= 2 Muons Post-Cut ---
            if is_cosmic:
                self.output["n_cosmic_events_2plus_muons"] += len(events)
            elif is_nobptx:
                self.output["n_nobptx_events_2plus_muons"] += len(events)

            # Duplicate track removal
            #######################################################################################################
            if len(events) > 0:
                lead_muon = sorted_muons[:, 0]
                
                deta = sorted_muons.eta - lead_muon.eta
                dphi = sorted_muons.delta_phi(lead_muon)
                dpt = sorted_muons.pt - lead_muon.pt
                mask_same_charge = (sorted_muons.charge * lead_muon.charge) > 0

                is_duplicate_track = mask_same_charge & (abs(deta) < 0.01) & (abs(dphi) < 0.001) & (abs(dpt) < 0.5)
                is_duplicate_track = is_duplicate_track & (ak.local_index(sorted_muons, axis=1) > 0)
                
                self.output["n_duplicates_removed"] += ak.sum(is_duplicate_track)

                cleaned_muons = sorted_muons[~is_duplicate_track]

                has_3_muons_still = (ak.num(cleaned_muons) >= 3)
                self.output["n_events_3plus_muons_post_cleaning"] += ak.sum(has_3_muons_still)
                #######################################################################################################
                
                mask_exactly_two = (ak.num(cleaned_muons) == 2)
                
                events = events[mask_exactly_two]
                if has_gen:
                    gen_muons = gen_muons[mask_exactly_two]
                final_muons = cleaned_muons[mask_exactly_two]
                
                if ak.sum(mask_exactly_two) > 0:
                    upper_candidates = final_muons[final_muons.phi > 0]
                    lower_candidates = final_muons[final_muons.phi < 0]

                    n_upper = ak.num(upper_candidates)
                    n_lower = ak.num(lower_candidates)

                    has_both_legs = (n_upper == 1) & (n_lower == 1)
                    
                    events = events[has_both_legs]
                    if has_gen:
                        gen_muons = gen_muons[has_both_legs]
                    upper_candidates = upper_candidates[has_both_legs]
                    lower_candidates = lower_candidates[has_both_legs]

                    if ak.sum(has_both_legs) > 0:
                        upper = upper_candidates[:, 0]
                        lower = lower_candidates[:, 0]

                        # Calculate Opening Angle
                        dot_product = upper.px * lower.px + upper.py * lower.py + upper.pz * lower.pz
                        denominator = upper.p * lower.p
                        cosA = ak.where(denominator != 0, dot_product / denominator, -1000.0)

                        self.output["two_muons_cos_alpha"].fill(
                                cat=ds_label,
                                val=cosA
                            )
                        
                        # Calculate Delta Time (Upper - Lower)
                        dt = upper.timeAtIpInOut - lower.timeAtIpInOut
                        
                        # Only fill Delta Time for Cosmics and Data
                        if is_cosmic or is_nobptx:
                            self.output["delta_time_upper_lower"].fill(
                                cat=ds_label,
                                val=dt
                            )
                            
                        # Track how many events survive a cut of cos(alpha) >= -0.99
                        mask_pass_cosA = (cosA >= -0.99)
                        
                        if is_cosmic:
                            self.output["n_cosmic_2mu_pre_cosA"] += len(cosA)
                            self.output["n_cosmic_2mu_post_cosA"] += ak.sum(mask_pass_cosA)
                        elif is_nobptx:
                            self.output["n_nobptx_2mu_pre_cosA"] += len(cosA)
                            self.output["n_nobptx_2mu_post_cosA"] += ak.sum(mask_pass_cosA)

        return self.output

    def postprocess(self, accumulator):
        return accumulator

if __name__ == '__main__':
    
    # Load the preprocessed Cosmic pickle file
    cosmic_pkl = "samples/Summer22_CHS_v19_Cosmic/Cosmic_preprocessed.pkl"
    print(f"Loading preprocessed Cosmics from {cosmic_pkl}...")
    with open(cosmic_pkl, "rb") as f:
        combined_runnable = pickle.load(f)
    
    # --- COMMENTED OUT NOBPTX LOAD ---
    nobptx_pkl = "samples/Summer22_CHS_v19_Cosmic/NoBPTX_preprocessed.pkl"
    print(f"Loading preprocessed NoBPTX from {nobptx_pkl}...")
    with open(nobptx_pkl, "rb") as f:
        nobptx_runnable = pickle.load(f)

    # Merge the dictionaries
    combined_runnable.update(nobptx_runnable)

    # --- ADDED BACK SIGNAL LOAD ---
    signal_pkl = "samples/Signal/Stau_300_100mm_preprocessed.pkl"
    print(f"Loading preprocessed Signal from {signal_pkl}...")
    with open(signal_pkl, "rb") as f:
        signal_runnable = pickle.load(f)

    # Merge the dictionaries
    combined_runnable.update(signal_runnable)
    
    # Run the Processor
    print("Starting Processor...")
    executor = processor.FuturesExecutor(workers=8)
    runner = processor.Runner(
        executor=executor,
        schema=PFNanoAODSchema,
        chunksize=50_000,
        skipbadfiles=True,
    )
    
    out = runner(
        combined_runnable,
        treename="Events",
        processor_instance=SingleMuonProcessor(),
    )

    print("\n" + "="*50)
    print(f"Total events processed: {out['n_events_initial']}")
    print(f"Total duplicate tracks removed: {out['n_duplicates_removed']}")
    print(f"WARNING: Total events with >=3 muons AFTER cleaning: {out['n_events_3plus_muons_post_cleaning']}")
    
    tot_cosmic = out['n_cosmic_events_total']
    if tot_cosmic > 0:
        pct_1 = (out['n_cosmic_events_1_muon'] / tot_cosmic) * 100
        pct_2 = (out['n_cosmic_events_2plus_muons'] / tot_cosmic) * 100
        print("\nCOSMIC MUON MULTIPLICITY (After Lead/Single Quality Cuts):")
        print(f"  Total Cosmic Events Processed: {tot_cosmic}")
        print(f"  Single DisMuon (Passes Cuts):  {out['n_cosmic_events_1_muon']} ({pct_1:.2f}%)")
        print(f"  >= 2 DisMuons (Lead Pass Cuts): {out['n_cosmic_events_2plus_muons']} ({pct_2:.2f}%)")

    tot_nobptx = out['n_nobptx_events_total']
    if tot_nobptx > 0:
        pct_1_n = (out['n_nobptx_events_1_muon'] / tot_nobptx) * 100
        pct_2_n = (out['n_nobptx_events_2plus_muons'] / tot_nobptx) * 100

        print("\nNoBPTX RAW RECONSTRUCTED COUNTS (Before Cuts):")
        print(f"  Events with Exactly 1 DisMuon:  {out['n_nobptx_raw_events_1']} (Total Muons: {out['n_nobptx_raw_events_1'] * 1})")
        print(f"  Events with Exactly 2 DisMuons: {out['n_nobptx_raw_events_2']} (Total Muons: {out['n_nobptx_raw_events_2'] * 2})")
        print(f"  Events with > 2 DisMuons:       {out['n_nobptx_raw_events_3plus']} (Total Muons: {out['n_nobptx_raw_muons_3plus']})")
        
        print("\nNoBPTX MUON MULTIPLICITY (After Lead/Single Quality Cuts):")
        print(f"  Total NoBPTX Events Processed: {tot_nobptx}")
        print(f"  Single DisMuon (Passes Cuts):  {out['n_nobptx_events_1_muon']} ({pct_1_n:.2f}%)")
        print(f"  >= 2 DisMuons (Lead Pass Cuts): {out['n_nobptx_events_2plus_muons']} ({pct_2_n:.2f}%)")
    print("="*50 + "\n")
    
    print("\ncos(alpha) CUT EFFICIENCY (Events with 1 Upper + 1 Lower Muon):")

    pre_c = out['n_cosmic_2mu_pre_cosA']
    post_c = out['n_cosmic_2mu_post_cosA']
    if pre_c > 0:
        print(f"  Cosmics: {pre_c} events pre-cut -> {post_c} survive cosA >= -0.99 ({(post_c/pre_c)*100:.2f}%)")
        
    pre_n = out['n_nobptx_2mu_pre_cosA']
    post_n = out['n_nobptx_2mu_post_cosA']
    if pre_n > 0:
        print(f"  NoBPTX:  {pre_n} events pre-cut -> {post_n} survive cosA >= -0.99 ({(post_n/pre_n)*100:.2f}%)")
    
    print("="*50 + "\n")

    OUTPUT_DIR = "single_muon_signal_vs_cosmic_plots"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Apply formatting to the titles and filenames
    PREFIX = "single_muon_"
    TITLE_MODIFIER = "(300 GeV 100 mm vs Cosmics)"
    FILE_MODIFIER = "Stau_300_100mm_overlay"

    overlay_plots = [
        "delta_time_upper_lower",
        "event_muon_multiplicity",
        "event_total_valid_hits",
        "event_standalone_fraction",
        #"single_muon_dz_overlay",
        #"single_muon_validDTHits",
        #"single_muon_validCSCHits",
        #"single_muon_validHits",
        #"single_muon_dtStations",
        #"single_muon_pt",
        #"single_muon_eta",
        #"single_muon_phi",
        #"single_muon_dxy",
        #"single_muon_timeNDof",
        #"single_muon_timeAtIpInOut",
        #"single_muon_timeAtIpInOutErr",
        #"single_muon_timing_err_ndof7",
        #"single_muon_timeErr_DT_only",
        #"single_muon_timeErr_CSC_only",
        #"single_muon_timeErr_DT_CSC_both",
        #"two_muons_cos_alpha",
    ]

    profile_plots = [
        #"single_muon_timeErr_vs_DTHits",
        #"single_muon_timeErr_vs_CSCHits",
        #"single_muon_timeErr_vs_TotalHits"
    ]

    simple_2d_plots = [
        #"uncut_upper_vs_lower_phi",
        #"uncut_upper_vs_lower_eta"
    ]

    for key, hist_obj in out.items():
        if isinstance(hist_obj, (int, float)): 
            continue
  
        if key in overlay_plots:
            save_comparison_overlay(hist_obj, key, PREFIX, OUTPUT_DIR, title_suffix=TITLE_MODIFIER, filename_suffix=FILE_MODIFIER, normalize=True)
            
        elif key in profile_plots:
            save_2d_profile_overlay(hist_obj, key, PREFIX, OUTPUT_DIR, title_suffix=TITLE_MODIFIER, filename_suffix=FILE_MODIFIER)

        elif key in simple_2d_plots:
            save_simple_2d_plot(hist_obj, key, PREFIX, OUTPUT_DIR, title_suffix=TITLE_MODIFIER, filename_suffix=FILE_MODIFIER, log_z=True)
            
    print("Done!")