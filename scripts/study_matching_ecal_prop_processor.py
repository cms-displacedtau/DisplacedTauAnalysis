# tau_processor.py
import numpy as np
import awkward as ak
from coffea import processor
from coffea.nanoevents import NanoAODSchema, PFNanoAODSchema
PFNanoAODSchema.warn_missing_crossrefs = False

from hist import Hist
import hist

max_eta = 2.4
maxLxy = 100

dxy_bins_low = np.arange(0, 8, 1)
dxy_bins_med = np.arange(8, 20, 4)
dxy_bins_high = np.arange(20, 110, 10)
dxy_bins_eff = np.concatenate([dxy_bins_low, dxy_bins_med, dxy_bins_high])

lxy_bins_eff = np.arange(0, 130, 1)

new_dxy_bins_eff = np.logspace(-2, 1.7, 10)
pt_bins_eff = np.logspace(.001, 2.7, 15) ## np.logspace(1.5, 2.7, 12)

dxy_bins = np.arange(0, 100, 1)

class TauProcessor(processor.ProcessorABC):

    def __init__(self, min_reco_pf_pt=2, min_gen_tau_pt=20, decayM=0, min_gen_pion_pt = 2, min_jet_pt = 20, delta_r_matching = 0.4):
        self.min_reco_pf_pt = min_reco_pf_pt
        self.min_gen_tau_pt = min_gen_tau_pt
        self.min_gen_pion_pt = min_gen_pion_pt
        self.min_jet_pt_ = min_jet_pt
        self.decayM_ = decayM
        self.delta_r_matching_threshold_ = delta_r_matching

    def get_gen_hadrons_from_taus(self, event_):
        hadrons_from_taus_ = event_.GenPart[(abs(event_.GenPart.pdgId) == 211) | (abs(event_.GenPart.pdgId) == 321) ]
        hadrons_from_taus_ = hadrons_from_taus_[abs(hadrons_from_taus_.distinctParent.pdgId) == 15]
        hadrons_from_taus_ = hadrons_from_taus_[abs(hadrons_from_taus_.distinctParent.distinctParent.pdgId) == 1000015]
        hadrons_from_taus_ = hadrons_from_taus_[hadrons_from_taus_.pt > self.min_gen_pion_pt]
        ## sara: should add eta requirement??
        return hadrons_from_taus_

    def get_gen_dau_from_taus(self, event_):
        taus = event_.GenPart[(abs(event_.GenPart.pdgId) == 15) & (event_.GenPart.hasFlags("isLastCopy"))]
        taus = taus[(abs(taus.distinctParent.pdgId) == 1000015)]
        return taus.distinctChildren


    
    def get_gen_muons_from_taus(self, event_):
        muons_from_tau_ = event_.GenPart[(abs(event_.GenPart.pdgId) == 13) & (event_.GenPart.hasFlags("isLastCopy"))]
        muons_from_tau_ = muons_from_tau_[
            (muons_from_tau_.pt > 20) &
            (abs(muons_from_tau_.eta) < max_eta) &
            (abs(muons_from_tau_.distinctParent.distinctParent.pdgId) == 1000015)
        ]
        return muons_from_tau_

    def get_gen_vis_taus(self, event_):
        gvt_ = event_.GenVisTau[
            (abs(event_.GenVisTau.parent.pdgId) == 15) &
            (abs(event_.GenVisTau.parent.distinctParent.pdgId) == 1000015) &
            (event_.GenVisTau.parent.distinctParent.hasFlags("isLastCopy")) &
            (event_.GenVisTau.parent.hasFlags("fromHardProcess")) &
            (event_.GenVisTau.status == self.decayM_) &
            (event_.GenVisTau.pt > self.min_gen_tau_pt) &
            (abs(event_.GenVisTau.eta) < max_eta)
        ]
        tau_vx = gvt_.parent.distinctParent.vx - ak.firsts(gvt_.parent.distinctChildren.vx, axis =2)
        tau_vy = gvt_.parent.distinctParent.vy - ak.firsts(gvt_.parent.distinctChildren.vy, axis =2)
        Lxy = np.sqrt(tau_vx**2 + tau_vy**2)
        gvt_ = ak.with_field(gvt_, Lxy, "pion_lxy")
        return gvt_
    

    def calc_and_add_dxy_gen_pion(self, pions_from_taus_, events_):
        counts_pions_ = ak.num(pions_from_taus_, axis=1)
        pv_y_ = np.asarray(events_.PVBS.y)
        pv_x_ = np.asarray(events_.PVBS.x)
        pv_y_expanded_pions_ = ak.unflatten(np.repeat(pv_y_, np.asarray(counts_pions_)), counts_pions_)
        pv_x_expanded_pions_ = ak.unflatten(np.repeat(pv_x_, np.asarray(counts_pions_)), counts_pions_)
        pions_from_taus_["dxy"] = (pions_from_taus_.vy - pv_y_expanded_pions_) * np.cos(pions_from_taus_.phi) - \
                                  (pions_from_taus_.vx - pv_x_expanded_pions_) * np.sin(pions_from_taus_.phi)

    def get_leading_gen_pion(self, event_):
        gen_pions_from_taus_ = self.get_gen_hadrons_from_taus(event_)
        self.calc_and_add_dxy_gen_pion(gen_pions_from_taus_, event_)
        sorted_gen_pions_ = gen_pions_from_taus_[ak.argsort(gen_pions_from_taus_.pt, ascending=False)]
        leading_gen_pion_ = sorted_gen_pions_[:, :1]
        return leading_gen_pion_
    
    def calc_and_add_dxy_gen_muon(self, muons_from_taus_, events_):
        counts_muons_ = ak.num(muons_from_taus_, axis=1)
        pv_y_ = np.asarray(events_.PVBS.y)
        pv_x_ = np.asarray(events_.PVBS.x)
        pv_y_expanded = ak.unflatten(np.repeat(pv_y_, np.asarray(counts_muons_)), counts_muons_)
        pv_x_expanded = ak.unflatten(np.repeat(pv_x_, np.asarray(counts_muons_)), counts_muons_)
        muons_from_taus_["dxy"] = (muons_from_taus_.vy - pv_y_expanded) * np.cos(muons_from_taus_.phi) - \
                                  (muons_from_taus_.vx - pv_x_expanded) * np.sin(muons_from_taus_.phi)
    
    def select_jets(self, event_):
        jets_ = event_.Jet[(abs(event_.Jet.eta) < max_eta) & (event_.Jet.pt > self.min_jet_pt_)]
        jets_ = jets_[
           (jets_.neHEF < 0.99) &
           (jets_.neEmEF < 0.9) &
           (jets_.chMultiplicity + jets_.neMultiplicity > 1) &
    #        (jets_.chHEF > 0.01) &
           (jets_.chMultiplicity > 0) &
           (jets_.muEF < 0.8) &
           (jets_.chEmEF < 0.8)
        ]
        jets_ = jets_[(jets_.muEF < 0.5)]
        return jets_
    
    def get_leading_jet(self, jets_):
        sorted_jets_ = jets_[ak.argsort(jets_.disTauTag_score1, ascending=False)]
        leading_jet_ = sorted_jets_[:, :1]
        leading_jet_ = leading_jet_[leading_jet_.disTauTag_score1 > 0.0]
        return leading_jet_
    
    def get_charged_pf(self, leading_jet_, min_reco_pf_pt_):
        reco_pf_ = leading_jet_.constituents.pf
        reco_pf_ = reco_pf_[(reco_pf_.charge != 0)]
        reco_pf_ = reco_pf_[(reco_pf_.pt > min_reco_pf_pt_)]
        return reco_pf_
    
    def delta_r_ecal(self, reco_obj, gen_obj):
        ### compute deltaR using coords at the ecal surface.
        ### this is specific to pion as it uses at_ecal_pion
    
        dphi = np.abs(reco_obj.phi_at_ecal - gen_obj.phi_at_ecal_pion)
        dphi = ak.where(dphi > np.pi, 2*np.pi - dphi, dphi)
        deta = reco_obj.eta_at_ecal - gen_obj.eta_at_ecal_pion
        return np.sqrt(deta**2 + dphi**2)


    def process(self, events):
        dataset = events.metadata["dataset"]
        cutflow = {}
        
        cutflow['initial'] = len(events)
        print(f"[{dataset}] Events before any filter: {cutflow['initial']}")

        edges_x = [-101, -98, -58]
        edges_x += list(np.linspace(-3, 3, 61))  
        h_allgenpions_eta_ecal_vs_eta = ( Hist.new
            .Var(edges_x, name="gen_eta_at_ecal", label="gen_eta_at_ecal")
            .Reg(50,  -2.5, 2.5, name="gen_eta", label="gen_eta")
            .Double()
        )

        hpt = ( Hist.new
            .Reg(200, 0, 100, name="recoPT", label="reco pT (GeV)")
            .Reg(200, 0, 100, name="genPT" , label="gen pT (GeV)")
            .Double()
        )
        hdelta = ( Hist.new
            .Reg(100, 0, 4, name="deltaR", label="deltaR")
            .Reg(100, 0, 4, name="deltaR_custom", label="deltaR at ECAL")
            .Double()
        )

        hdeltaR_vs_dxy= ( Hist.new
            .Reg(1500, 0, 150, name="deltaR_custom", label="deltaR at ECAL")
            .Reg(200,  0, 100, name="dxy", label="dxy")
            .Double()
        )
        heta_ecal_vs_dxy= ( Hist.new
            .Reg(250, -102, 2.5, name="eta_at_ecal", label="eta_at_ecal")
            .Reg(120,  0, 60, name="gen_dxy", label="gen_dxy")
            .Double()
        )
        heta_ecal_vs_pt = ( Hist.new
            .Reg(250, -102, 2.5, name="eta_at_ecal", label="eta_at_ecal")
            .Reg(200,  0, 200, name="gen_pt", label="gen pion pt")
            .Double()
        )
        heta_ecal_vs_eta = ( Hist.new
            .Var(edges_x, name="eta_at_ecal", label="eta_at_ecal")
            .Reg(50,  -2.5, 2.5, name="gen_eta", label="gen_eta")
#             .Reg(125,  -2.5, 2.5, name="gen_eta", label="gen_eta")
            .Double()
        )
        heta_ecal_vs_lxy= ( Hist.new
            .Var(edges_x, name="eta_at_ecal", label="eta_at_ecal")
            .Reg(200,  0, 200, name="gen_lxy", label="gen_lxy")
            .Double()
        )
        heta_ecal_vs_tau_eta = ( Hist.new
            .Var(edges_x, name="eta_at_ecal", label="eta_at_ecal")
            .Reg(125,  -2.5, 2.5, name="gen_tau_eta", label="gen_tau_eta")
            .Double()
        )
        hdxy_vs_lxy = ( Hist.new
            .Reg(200,  0, 100, name="gen_dxy", label="gen_dxy")
            .Reg(200,  0, 200, name="gen_lxy", label="gen_lxy")
            .Double()
        )
        hfail_pt_vs_eta = ( Hist.new
            .Reg(125, -2.5, 2.5, name="eta", label="eta")
            .Reg(120,  40, 300, name="pt", label="pt")
            .Double()
        )

        hdeltaR = ( Hist.new
            .Reg(100, 0, 6, name="deltaR", label="deltaR")
            .Double()
        )
        hdeltaR_custom = ( Hist.new
            .Reg(100, 0, 6, name="deltaR_custom", label="deltaR_custom")
            .Double()
        )
        hdeltaR_custom_zoomout = ( Hist.new
            .Reg(400, 110, 150, name="deltaR_custom", label="deltaR_custom")
            .Double()
        )
        hphi_ecal_vs_phi_bump = ( Hist.new
            .Reg(80,  -3.2, 3.2, name="phi_at_ecal", label="phi_at_ecal")
            .Reg(80,  -3.2, 3.2, name="phi", label="phi")
            .Double()
        )
        hgenphi_ecal_vs_genphi_bump = ( Hist.new
            .Reg(80,  -3.2, 3.2, name="genphi_at_ecal", label="genphi_at_ecal")
            .Reg(80,  -3.2, 3.2, name="genphi", label="genphi")
            .Double()
        )

        h_deta_bump = Hist.new.Reg(100, -5, 5, name="deta", label="deltaEta (origin)").Double()
        h_dphi_bump = Hist.new.Reg(100, 0, 3.5, name="dphi", label="deltaPhi (origin)").Double()
        h_deta_ecal_bump = Hist.new.Reg(100, -5, 5, name="deta_ecal", label="deltaEta (ECAL)").Double()
        h_dphi_ecal_bump = Hist.new.Reg(100, 0, 3.5, name="dphi_ecal", label="deltaPhi (ECAL)").Double()
        h_dpt_bump = Hist.new.Reg(200, -60, 60, name="dpt", label="deltaPt").Double()
        h_dpt_all = Hist.new.Reg(200, -60, 60, name="dpt", label="deltaPt").Double()
        h_dcharge_bump = Hist.new.Reg(3, 0, 3, name="dcharge", label="deltaCharge").Double()
        h_dcharge_all = Hist.new.Reg(3, 0, 3, name="dcharge", label="deltaCharge").Double()
        h_lxy_bump = Hist.new.Reg(120, 0, 120, name="lxy", label="lxy").Double()
        h_lxy_all = Hist.new.Reg(120, 0, 120, name="lxy", label="lxy").Double()
        h_pdgid_bump = Hist.new.Reg(8, -400, 400, name="pdgid", label="pdgid").Double()
        h_geneta_bump = Hist.new.Reg(50, -2.5, 2.5, name="geneta", label="eta").Double()

        hgen_tau_pt = Hist.new.Reg(300,0,300, name="gen_tau_pt", label="gen tau pt").Double()
        
        hreco_dxy = ( Hist.new
            .Var(dxy_bins_eff, name="reco_dxy", label="reco dxy")
            .Double()
        )
        hreco_ecal_dxy = ( Hist.new
            .Var(dxy_bins_eff, name="reco_ecal_dxy", label="ecal reco dxy")
            .Double()
        )
        hgen_dxy = ( Hist.new
            .Var(dxy_bins_eff, name="gen_dxy", label="gen dxy")
            .Double()
        )
        hall_dxy = ( Hist.new
            .Var(dxy_bins, name="gen_dxy", label="gen dxy")
            .Double()
        )
        hfail_dxy = ( Hist.new
            .Var(dxy_bins, name="gen_dxy", label="gen dxy")
            .Double()
        )
        hpass_dxy = ( Hist.new
            .Var(dxy_bins, name="gen_dxy", label="gen dxy")
            .Double()
        )

        hreco_dxy_zoom = ( Hist.new
                     .Var(new_dxy_bins_eff, name="reco_dxy", label="reco dxy").Double())
        hreco_ecal_dxy_zoom = ( Hist.new
                     .Var(new_dxy_bins_eff, name="reco_ecal_dxy", label="ecal reco dxy").Double())
        hgen_dxy_zoom = ( Hist.new
                     .Var(new_dxy_bins_eff, name="gen_dxy", label="gen dxy").Double())

        hreco_lxy = ( Hist.new
                     .Var(lxy_bins_eff, name="reco_lxy", label="reco lxy").Double())
        hreco_ecal_lxy = ( Hist.new
                     .Var(lxy_bins_eff, name="reco_ecal_lxy", label="ecal reco lxy").Double())
        hgen_lxy = ( Hist.new
                     .Var(lxy_bins_eff, name="gen_lxy", label="gen lxy").Double())
        hreco_pt = ( Hist.new
                    .Var(pt_bins_eff, name="reco_pt", label="reco pt").Double())
        hreco_ecal_pt = ( Hist.new
                    .Var(pt_bins_eff, name="reco_ecal_pt", label="ecal reco pt").Double())
        hgen_pt = ( Hist.new
                    .Var(pt_bins_eff, name="gen_pt", label="gen pt").Double())
        hmatched_jet_dxy = ( Hist.new
                    .Var(new_dxy_bins_eff, name="jet_dxy", label="matched Jet dxy (cm)").Double())
        hmatched_jet_pt = ( Hist.new
                    .Var(pt_bins_eff, name="jet_pt", label="matched Jet pT (GeV)").Double())
        hmatched_jet_pion_pt = ( Hist.new
                    .Var(pt_bins_eff, name="jet_pion_pt", label="matched-jet leading pion pT (GeV)").Double())

        hmatched_alljet_pt = ( Hist.new
                    .Var(pt_bins_eff, name="jet_pt", label="matched Jet pT (GeV)").Double())
        hgen_pt_for_alljet = ( Hist.new
                    .Var(pt_bins_eff, name="gen_pt", label="gen pt").Double())

        hreco_pt_plot = ( Hist.new
                    .Reg(300, 0, 300, name="reco_pt", label="reco pt").Double())
        hreco_pt_zoom_plot = ( Hist.new
                    .Reg(300, 0, 30, name="reco_pt", label="reco pt").Double())
        hgen_pt_plot = ( Hist.new
                    .Reg(300, 0, 300, name="gen_pt", label="gen pt").Double())
        hgen_pt_zoom_plot = ( Hist.new
                    .Reg(300, 0, 30, name="gen_pt", label="gen pt").Double())
 
        hreco_gvtpt = ( Hist.new
                    .Var(pt_bins_eff, name="reco_gvtpt", label="reco gvt pt").Double())
        hreco_ecal_gvtpt = ( Hist.new
                    .Var(pt_bins_eff, name="reco_ecal_gvtpt", label="ecal reco gvt pt").Double())
        hgen_gvtpt = ( Hist.new
                    .Var(pt_bins_eff, name="gen_gvtpt", label="gen gvt pt").Double())
      
        
        hphi_ecal_vs_phi_worse = ( Hist.new
            .Reg(80,  -3.2, 3.2, name="phi_at_ecal", label="phi_at_ecal")
            .Reg(80,  -3.2, 3.2, name="phi", label="phi")
            .Double()
        )
        hgenphi_ecal_vs_genphi_worse = ( Hist.new
            .Reg(80,  -3.2, 3.2, name="genphi_at_ecal", label="genphi_at_ecal")
            .Reg(80,  -3.2, 3.2, name="genphi", label="genphi")
            .Double()
        )
        h_deta_worse = Hist.new.Reg(100, -5, 5, name="deta", label="deltaEta (origin)").Double()
        h_dphi_worse = Hist.new.Reg(100, 0, 3.5, name="dphi", label="deltaPhi (origin)").Double()
        h_deta_ecal_worse = Hist.new.Reg(100, -5, 5, name="deta_ecal", label="deltaEta (ECAL)").Double()
        h_dphi_ecal_worse = Hist.new.Reg(100, 0, 3.5, name="dphi_ecal", label="deltaPhi (ECAL)").Double()
        h_dpt_worse = Hist.new.Reg(200, -60, 60, name="dpt", label="deltaPt").Double()
        h_dcharge_worse = Hist.new.Reg(3, 0, 3, name="dcharge", label="deltaCharge").Double()
        h_lxy_worse = Hist.new.Reg(120, 0, 120, name="lxy", label="lxy").Double()
        hpt_vs_genpt_worse = ( Hist.new
            .Reg(100,  0, 100, name="genpt", label="genpt")
            .Reg(100, 0, 100, name="recopt", label="recopt")
            .Double()
        )
        hdxy_vs_gendxy_worse = ( Hist.new
            .Reg(100, 0, 2, name="gendxy", label="gendxy")
            .Reg(100, 0, 2, name="recodxy", label="recodxy")
            .Double()
        )


        # Base return dictionary for early exits
        def get_return_dict(events_arr):
            return {dataset: {
                "entries": ak.num(events_arr, axis=0), "cutflow": cutflow,
                "hdelta": hdelta, "hdeltaR": hdeltaR, "hdeltaR_custom": hdeltaR_custom,
                "hdeltaR_custom_zoomout": hdeltaR_custom_zoomout, "hreco_dxy": hreco_dxy,
                "hreco_ecal_dxy": hreco_ecal_dxy, "hgen_lxy": hgen_lxy, "hgen_dxy": hgen_dxy,
                "hall_dxy": hall_dxy, "hpass_dxy": hpass_dxy, "hfail_dxy": hfail_dxy,
                "hreco_dxy_zoom": hreco_dxy_zoom, "hreco_ecal_dxy_zoom": hreco_ecal_dxy_zoom,
                "hgen_dxy_zoom": hgen_dxy_zoom, "hreco_pt": hreco_pt, "hreco_ecal_pt": hreco_ecal_pt,
                "hgen_pt": hgen_pt, "hgen_pt_plot": hgen_pt_plot, "hgen_pt_zoom_plot": hgen_pt_zoom_plot,
                "hreco_pt_plot": hreco_pt_plot, "hreco_pt_zoom_plot": hreco_pt_zoom_plot,
                "hmatched_jet_pt": hmatched_jet_pt, "hmatched_jet_dxy": hmatched_jet_dxy,
                "hmatched_jet_pion_pt": hmatched_jet_pion_pt, 
                "hdeltaR_vs_dxy": hdeltaR_vs_dxy, "heta_ecal_vs_dxy": heta_ecal_vs_dxy,
                "heta_ecal_vs_pt": heta_ecal_vs_pt, "heta_ecal_vs_eta": heta_ecal_vs_eta,
                "hdxy_vs_lxy": hdxy_vs_lxy, "hfail_pt_vs_eta": hfail_pt_vs_eta,
                "h_deta_bump": h_deta_bump, "h_dphi_bump": h_dphi_bump,
                "h_deta_ecal_bump": h_deta_ecal_bump, "h_dphi_ecal_bump": h_dphi_ecal_bump,
                "h_dpt_bump": h_dpt_bump, "h_dpt_all": h_dpt_all, 
                "h_dcharge_all": h_dcharge_all, "h_dcharge_bump": h_dcharge_bump, 
                "h_lxy_all": h_lxy_all, "h_lxy_bump": h_lxy_bump, 
                "heta_ecal_vs_lxy": heta_ecal_vs_lxy, "heta_ecal_vs_tau_eta": heta_ecal_vs_tau_eta,
                "h_allgenpions_eta_ecal_vs_eta": h_allgenpions_eta_ecal_vs_eta,
                "hphi_ecal_vs_phi_bump": hphi_ecal_vs_phi_bump,
                "hgenphi_ecal_vs_genphi_bump": hgenphi_ecal_vs_genphi_bump,
                "hphi_ecal_vs_phi_worse": hphi_ecal_vs_phi_worse,
                "hgenphi_ecal_vs_genphi_worse": hgenphi_ecal_vs_genphi_worse,
                "h_pdgid_bump": h_pdgid_bump, "h_geneta_bump": h_geneta_bump,
                "h_deta_worse": h_deta_worse, "h_dphi_worse": h_dphi_worse,
                "h_deta_ecal_worse": h_deta_ecal_worse, "h_dphi_ecal_worse": h_dphi_ecal_worse,
                "h_dpt_worse": h_dpt_worse, "h_dcharge_worse": h_dcharge_worse, 
                "h_lxy_worse": h_lxy_worse, 
                "hgen_tau_pt": hgen_tau_pt,
                "hpt_vs_genpt_worse": hpt_vs_genpt_worse,
                "hdxy_vs_gendxy_worse": hdxy_vs_gendxy_worse,
                "hmatched_alljet_pt":hmatched_alljet_pt,
                "hgen_pt_for_alljet":hgen_pt_for_alljet,
                "hreco_gvtpt": hreco_gvtpt,
                "hreco_ecal_gvtpt": hreco_ecal_gvtpt,
                "hgen_gvtpt": hgen_gvtpt,
                
            }}

        # select gen visible taus from the desired long-lived particle chain
        gen_vis_taus = self.get_gen_vis_taus(events)
        gen_vis_taus = gen_vis_taus[abs(gen_vis_taus.pion_lxy) < maxLxy]

        muons_from_taus = self.get_gen_muons_from_taus(events)
        mask = (ak.num(gen_vis_taus) == 1) & (ak.num(muons_from_taus) == 1)
        ## ************************** ##
        filter_events = events[mask]
        cutflow['after_1tau_1muon'] = len(filter_events)
        print(f"[{dataset}] Events after 1 GenVisTau and 1 GenMuon filter: {cutflow['after_1tau_1muon']}")
        ## ************************** ##

        pions_from_taus = self.get_gen_hadrons_from_taus(filter_events)
        counts_pions = ak.num(pions_from_taus, axis=1)
        ## ************************** ##
        ## check for gen decay mode
#         zero_filter_events = filter_events[counts_pions == 0]
#         cutflow['after_gen_pions_equal_0'] = len(zero_filter_events)
#         print(f"[{dataset}] Events after Gen Pions == 0 filter: {cutflow['after_gen_pions_equal_0']}")
#         dau_from_taus = self.get_gen_dau_from_taus(zero_filter_events)
#         print(f"[{dataset}] Tau children pdgIds for 0-pion events:\n{dau_from_taus.pdgId}")
        ## ************************** ##
        filter_events = filter_events[counts_pions > 0]
        cutflow['after_gen_pions_gt_0'] = len(filter_events)
        print(f"[{dataset}] Events after Gen Pions > 0 filter: {cutflow['after_gen_pions_gt_0']}")
        ## ************************** ##
        if len(filter_events) == 0:
            return get_return_dict(events)
    
        
        ### fill plots for GEN pions, no requests on RECO yet
        leading_all_gen_pion = self.get_leading_gen_pion(filter_events)
        flat_allgenpions_eta_ecal = ak.flatten(leading_all_gen_pion.eta_at_ecal_pion)
        flat_allgenpions_eta = ak.flatten(leading_all_gen_pion.eta)
        h_allgenpions_eta_ecal_vs_eta.fill(gen_eta_at_ecal = flat_allgenpions_eta_ecal, gen_eta = ak.to_numpy(flat_allgenpions_eta))

#                               
# ##      Aug 7, not used
# #         pions_from_taus = self.get_gen_hadrons_from_taus(filter_events)
# #         muons_from_taus = self.get_gen_muons_from_taus(filter_events)
# #         self.calc_and_add_dxy_gen_muon(muons_from_taus, filter_events)

        ## RECO side:
        ## want a jet passing ID and with at least one charged candidate
        jets = self.select_jets(filter_events)
        counts_jets = ak.num(jets, axis=1)
        ## ************************** ##
        filter_events = filter_events[counts_jets > 0]
        cutflow['after_tight_jet'] = len(filter_events)
        print(f"[{dataset}] Events after tight jet requirement: {cutflow['after_tight_jet']}")
        ## ************************** ##
        if len(filter_events) == 0:
            return get_return_dict(events)

        ### Plots for the efficiency with standard DR and no requirement on PF charged candidate in the jet
        ##  Match jets to GenVisTaus and plot matched jet pT
        ##  using firsts is safe only because running on mu-tau events
        current_gen_vis_taus = self.get_gen_vis_taus(filter_events)
        first_gen_tau = ak.firsts(current_gen_vis_taus)

        jets = self.select_jets(filter_events)
        leading_jet = self.get_leading_jet(jets)

        # Find if the gen vis tau has a match
        dr_jets_taus = leading_jet.delta_r(first_gen_tau)
        has_matched_jet = ak.fill_none(ak.any(dr_jets_taus < self.delta_r_matching_threshold_, axis=1), False)
        # Filter the taus using the mask and fill the histogram with their pt
        matched_gen_taus = first_gen_tau[has_matched_jet]
#         hmatched_alljet_pt.fill(jet_pt = ak.to_numpy(ak.drop_none(matched_gen_taus.pt)))
#         hgen_pt_for_alljet.fill(gen_pt = ak.to_numpy(first_gen_tau.pt))
        hmatched_alljet_pt.fill(jet_pt = ak.to_numpy(ak.flatten(matched_gen_taus.pt, axis=None)))
        hgen_pt_for_alljet.fill(gen_pt = ak.to_numpy(ak.flatten(first_gen_tau.pt, axis=None)))
        ### end of efficiency with standard DR and no requirement on PF charged candidate in the jet
         
        
        ## require a charged pf candidate in the jet, and select the leading one 
        charged_reco_pf = self.get_charged_pf(leading_jet, self.min_reco_pf_pt)
        # Count how many valid PF candidates exist per jet (axis=2), then sum per event (axis=1)
        counts_charged_pf_per_event = ak.sum(ak.num(charged_reco_pf, axis=2), axis=1)
        has_charged_in_leading_event = counts_charged_pf_per_event > 0

        ## ************************** ##
        pf_charged_events = filter_events[has_charged_in_leading_event]
        cutflow['after_charged_pf_in_leading_jet'] = len(pf_charged_events)
        print(f"[{dataset}] Events after charged PF in leading jet filter: {cutflow['after_charged_pf_in_leading_jet']}")
        ## ************************** ##
        if len(pf_charged_events) == 0:
            return get_return_dict(events)



        # recompute with selected events
        leading_gen_pion = self.get_leading_gen_pion(pf_charged_events)
#         gen_pions_from_taus = self.get_gen_hadrons_from_taus(pf_charged_events)
#         sorted_gen_pions = gen_pions_from_taus[ak.argsort(gen_pions_from_taus.pt, ascending=False)]
#         leading_gen_pion = sorted_gen_pions[:, :1]
#         print(f"[{dataset}] GEN leading pions pts :\n{ak.to_list(sorted_gen_pions.pt[:100])}", flush=True)

        leading_gen_pion = leading_gen_pion[leading_gen_pion.eta_at_ecal_pion > -10]
        counts_gen_leading = ak.num(leading_gen_pion, axis=1)  # number of pions from reco tau per event

        
        ## RECO 
        jets = self.select_jets(pf_charged_events)
        leading_jet = self.get_leading_jet(jets)
        charged_reco_pf = self.get_charged_pf(leading_jet, self.min_reco_pf_pt)
        sorted_reco_pf = charged_reco_pf[ak.argsort(charged_reco_pf.pt, ascending=False)]
        leading_reco_pf = sorted_reco_pf[ak.argmax(sorted_reco_pf.pt, axis=2, keepdims=True)]
        leading_reco_pf = leading_reco_pf[leading_reco_pf.eta_at_ecal > -10]

        counts_pf_leading = ak.num(leading_reco_pf, axis=1)  # number of pions from reco tau per event
        ## ************************** ##
        ## select only events with at least one RECO charged PFCand inside the leading (score) jet
        leading_events = pf_charged_events[(counts_pf_leading > 0) ]
        cutflow['after_counts_pf_leading_gt_0'] = len(leading_events)
        print(f"[{dataset}] Events after counts_pf_leading > 0 filter: {cutflow['after_counts_pf_leading_gt_0']}")
        
        leading_events = leading_events[(counts_gen_leading > 0) ]
        cutflow['after_counts_gen_leading_gt_0'] = len(leading_events)
        print(f"[{dataset}] Events after counts_gen_leading > 0 filter: {cutflow['after_counts_gen_leading_gt_0']}")
        ## ************************** ## 

        if len(leading_events) == 0:
            return get_return_dict(events)

        ## recompute everything again
        gen_vis_taus = self.get_gen_vis_taus(leading_events)
        gen_vis_taus = gen_vis_taus[abs(gen_vis_taus.pion_lxy) < maxLxy]
        leading_gen_pion = self.get_leading_gen_pion(leading_events)
        leading_gen_pion = leading_gen_pion[leading_gen_pion.eta_at_ecal_pion > -10]

#         return get_return_dict(leading_events)

#         leading_all_gen_pion = self.get_leading_gen_pion(leading_events)
#         pions_from_taus = self.get_gen_hadrons_from_taus(leading_events)
        ## calculate gen level dxy for pions
#         self.calc_and_add_dxy_gen_pion(pions_from_taus, leading_events)
#         sorted_gen_pions = pions_from_taus[ak.argsort(pions_from_taus.pt, ascending=False)]
#         leading_gen_pion = sorted_gen_pions[:, :1]

        gen_charge = ak.where(leading_gen_pion.pdgId > 0, 1, -1)
        leading_gen_pion = ak.with_field(leading_gen_pion, gen_charge, "charge")
        # Broadcast the tau pT to match the 0-or-1 shape of the pion
        tau_pt_matched_shape = ak.broadcast_arrays(gen_vis_taus[:, :1].pt, leading_gen_pion.pt)[0]
        # Now insert it safely without corrupting the other data types
        leading_gen_pion = ak.with_field(leading_gen_pion, tau_pt_matched_shape, "parent_tau_pt")

        gen_vis_taus = ak.with_field(gen_vis_taus, leading_gen_pion.dxy, "pion_dxy")
        gen_vis_taus = ak.with_field(gen_vis_taus, leading_gen_pion.pt, "pion_pt")

        jets = self.select_jets(leading_events)
        leading_jet = self.get_leading_jet(jets)
        reco_pf =  self.get_charged_pf(leading_jet, self.min_reco_pf_pt)
        sorted_reco_pf = reco_pf[ak.argsort(reco_pf.pt, ascending=False)]
        leading_reco_pf = sorted_reco_pf[ak.argmax(sorted_reco_pf.pt, axis=2, keepdims=True)]
        leading_reco_pf = leading_reco_pf[leading_reco_pf.eta_at_ecal > -10]

        ## try to get the matched pions
        matched_pf_to_pions = ak.flatten(leading_reco_pf).nearest(leading_gen_pion, threshold = self.delta_r_matching_threshold_)
        
        ## try to replicate with custom deltaR (at ECAL surface)
        # Compute deltaR matrix between each reco and gen candidate in an event
#         dr_matrix = self.delta_r_ecal(
#             ak.cartesian({"pf": ak.flatten(leading_reco_pf), "pion": leading_gen_pion}, nested=True).pf,
#             ak.cartesian({"pf": ak.flatten(leading_reco_pf), "pion": leading_gen_pion}, nested=True).pion
#         )        
#         ## find the index of the gen_pion closest to each reco_pf
#         ecal_nearest_idx = ak.argmin(dr_matrix, axis=2)  # per reco_pf        
#         ## get the corresponding gen_pion
#         ecal_matched_pf_to_pions = leading_gen_pion[ecal_nearest_idx]
#         dr_min = ak.min(dr_matrix, axis=2)
#         ecal_matched_pf_to_pions_masked = ak.mask(ecal_matched_pf_to_pions, dr_min < self.delta_r_matching_threshold_)


        flat_leading_reco_pf = ak.flatten(leading_reco_pf, axis=1)
        # compute dR values
        delta_r_origin = flat_leading_reco_pf[:, :, None].delta_r(leading_gen_pion[:, None, :])
        deta_origin = flat_leading_reco_pf.eta[:, :, None] - leading_gen_pion.eta[:, None, :]
        dphi_origin = abs(flat_leading_reco_pf.phi[:, :, None] - leading_gen_pion.phi[:, None, :])
        dphi_origin = ak.where(dphi_origin > np.pi, 2*np.pi - dphi_origin, dphi_origin)

        
        # Mask: For each gen_pion, is there ANY reco_pf within threshold?
        # is_matched shape: [events, gen_pion]
        is_matched = ak.any(delta_r_origin < self.delta_r_matching_threshold_, axis=1)
        matched_gen_pions = leading_gen_pion[is_matched]

#         delta_r_origin = leading_reco_pf[:, :, None].delta_r(leading_gen_pion[:, None, :])
#         # Find if the gen pion has a match
#         min_dr = ak.min(delta_r_origin, axis=2)
# #         is_matched = min_dr < self.delta_r_matching_threshold_
#         is_matched = ak.any(delta_r_origin < self.delta_r_matching_threshold_, axis=1)
#         matched_gen_pions = leading_gen_pion[is_matched]

        # Compute dR values (ECAL)
        deta_ecal = flat_leading_reco_pf.eta_at_ecal[:, :, None] - leading_gen_pion.eta_at_ecal_pion[:, None, :]
        dphi_ecal = abs(flat_leading_reco_pf.phi_at_ecal[:, :, None] - leading_gen_pion.phi_at_ecal_pion[:, None, :])
        dphi_ecal = ak.where(dphi_ecal > np.pi, 2*np.pi - dphi_ecal, dphi_ecal)
        delta_r_custom = np.sqrt(deta_ecal**2 + dphi_ecal**2)

        dpt = flat_leading_reco_pf.pt[:, :, None] - leading_gen_pion.pt[:, None, :]
        dcharge = abs(flat_leading_reco_pf.charge[:, :, None] - leading_gen_pion.charge[:, None, :])

        leading_pion_lxy = np.sqrt(leading_gen_pion.vx**2 + leading_gen_pion.vy **2)
        leading_gen_pion = ak.with_field(leading_gen_pion, leading_pion_lxy, where="lxy")    
        
        ### broadcast lxy to [events, reco_pf, gen_pion]
        gen_lxy_broadcasted, _ = ak.broadcast_arrays(leading_gen_pion.lxy[:, None, :], delta_r_custom)
        gen_dxy_broadcasted, _ = ak.broadcast_arrays(leading_gen_pion.dxy[:, None, :], delta_r_custom)
        gen_phi_broadcasted, _ = ak.broadcast_arrays(leading_gen_pion.phi[:, None, :], delta_r_custom)
        gen_phi_ecal_broadcasted, _ = ak.broadcast_arrays(leading_gen_pion.phi_at_ecal_pion[:, None, :], delta_r_custom)
        gen_pt_broadcasted, _ = ak.broadcast_arrays(leading_gen_pion.pt[:, None, :], delta_r_custom)
        gen_pdgid_broadcasted, _ = ak.broadcast_arrays(leading_gen_pion.pdgId[:, None, :], delta_r_custom)
        gen_eta_broadcasted, _ = ak.broadcast_arrays(leading_gen_pion.eta[:, None, :], delta_r_custom)
        phi_broadcasted, _ = ak.broadcast_arrays(flat_leading_reco_pf.phi[:, None, :], delta_r_custom)
        phi_ecal_broadcasted, _ = ak.broadcast_arrays(flat_leading_reco_pf.phi_at_ecal[:, None, :], delta_r_custom)
        pt_broadcasted, _ = ak.broadcast_arrays(flat_leading_reco_pf.pt[:, None, :], delta_r_custom)
        dxy_broadcasted, _ = ak.broadcast_arrays(flat_leading_reco_pf.d0[:, None, :], delta_r_custom)

        bump_mask = (delta_r_custom > 3.0) & (delta_r_custom < 4.0)
        worse_mask = (delta_r_custom > delta_r_origin)


        # Mask for ECAL
        ecal_is_matched = ak.any(delta_r_custom < self.delta_r_matching_threshold_, axis=1)
        ecal_matched_gen_pions = leading_gen_pion[ecal_is_matched]

        flat_delta_r = ak.flatten(ak.flatten(delta_r_origin))
        flat_delta_r = ak.fill_none(flat_delta_r, 999)
        flat_delta_r_custom = ak.flatten(ak.flatten(delta_r_custom))
        flat_delta_r_custom = ak.fill_none(flat_delta_r_custom, 999)

#         flat_gen_dxy = ak.flatten(leading_gen_pion.dxy)
#         flat_reco_dxy = ak.flatten(matched_gen_pions.dxy)
#         flat_ecal_reco_dxy = ak.flatten(ecal_matched_gen_pions.dxy)
# 
#         flat_gen_pt = ak.flatten(leading_gen_pion.pt)
#         flat_gen_eta = ak.flatten(leading_gen_pion.eta)
#         flat_gen_eta_ecal = ak.flatten(leading_gen_pion.eta_at_ecal_pion)
#         flat_reco_pt = ak.flatten(matched_gen_pions.pt)
#         flat_ecal_reco_pt = ak.flatten(ecal_matched_gen_pions.pt)
# 
#         flat_fail_gen_pt  = ak.flatten(leading_gen_pion[leading_gen_pion.eta_at_ecal_pion < -10].pt)
#         flat_fail_gen_eta = ak.flatten(leading_gen_pion[leading_gen_pion.eta_at_ecal_pion < -10].eta)
#         flat_fail_gen_dxy = ak.flatten(leading_gen_pion[leading_gen_pion.eta_at_ecal_pion < -10].dxy)
#         flat_pass_gen_dxy = ak.flatten(leading_gen_pion[leading_gen_pion.eta_at_ecal_pion > -10].dxy)

        flat_gen_dxy = ak.fill_none(ak.flatten(leading_gen_pion.dxy), -99)
        flat_reco_dxy = ak.fill_none(ak.flatten(matched_gen_pions.dxy), -99)
        flat_ecal_reco_dxy = ak.fill_none(ak.flatten(ecal_matched_gen_pions.dxy), -99)

        flat_gen_pt = ak.fill_none(ak.flatten(leading_gen_pion.pt), -99)
        flat_gen_eta = ak.fill_none(ak.flatten(leading_gen_pion.eta), -99)
        flat_gen_eta_ecal = ak.fill_none(ak.flatten(leading_gen_pion.eta_at_ecal_pion), -99)
        flat_reco_pt = ak.fill_none(ak.flatten(matched_gen_pions.pt), -99)
        flat_ecal_reco_pt = ak.fill_none(ak.flatten(ecal_matched_gen_pions.pt), -99)

        flat_fail_gen_pt  = ak.fill_none(ak.flatten(leading_gen_pion[leading_gen_pion.eta_at_ecal_pion < -10].pt), -99)
        flat_fail_gen_eta = ak.fill_none(ak.flatten(leading_gen_pion[leading_gen_pion.eta_at_ecal_pion < -10].eta), -99)
        flat_fail_gen_dxy = ak.fill_none(ak.flatten(leading_gen_pion[leading_gen_pion.eta_at_ecal_pion < -10].dxy), -99)
        flat_pass_gen_dxy = ak.fill_none(ak.flatten(leading_gen_pion[leading_gen_pion.eta_at_ecal_pion > -10].dxy), -99)

        
        # fill histograms
        hdelta.fill(deltaR = ak.to_numpy(flat_delta_r), deltaR_custom = ak.to_numpy(flat_delta_r_custom))
        hdeltaR.fill(deltaR = ak.to_numpy(flat_delta_r))
        hdeltaR_custom.fill(deltaR_custom = ak.to_numpy(flat_delta_r_custom))
        hdeltaR_custom_zoomout.fill(deltaR_custom = ak.to_numpy(flat_delta_r_custom))

        hgen_dxy.fill(gen_dxy = ak.to_numpy(flat_gen_dxy))
        hreco_dxy.fill(reco_dxy = ak.to_numpy(flat_reco_dxy))
        hreco_ecal_dxy.fill(reco_ecal_dxy = ak.to_numpy(flat_ecal_reco_dxy))

        hall_dxy.fill(gen_dxy = ak.to_numpy(flat_gen_dxy))
        hpass_dxy.fill(gen_dxy = ak.to_numpy(flat_pass_gen_dxy))
        hfail_dxy.fill(gen_dxy = ak.to_numpy(flat_fail_gen_dxy))

        hgen_dxy_zoom.fill(gen_dxy = ak.to_numpy(flat_gen_dxy))
        hreco_dxy_zoom.fill(reco_dxy = ak.to_numpy(flat_reco_dxy))
        hreco_ecal_dxy_zoom.fill(reco_ecal_dxy = ak.to_numpy(flat_ecal_reco_dxy))

        hgen_pt.fill(gen_pt = ak.to_numpy(flat_gen_pt))
        hreco_pt.fill(reco_pt = ak.to_numpy(flat_reco_pt))
        hreco_ecal_pt.fill(reco_ecal_pt = ak.to_numpy(flat_ecal_reco_pt))
        
        hreco_gvtpt.fill(reco_gvtpt = ak.to_numpy(ak.flatten(matched_gen_pions.parent_tau_pt)))
        hreco_ecal_gvtpt.fill(reco_ecal_gvtpt = ak.to_numpy(ak.flatten(ecal_matched_gen_pions.parent_tau_pt)))
        hgen_gvtpt.fill(gen_gvtpt = ak.to_numpy(flat_gen_pt))

        ## plot all gen and reco pT (no matching) 
        hgen_pt_plot.fill(gen_pt = ak.to_numpy(flat_gen_pt))
        hgen_pt_zoom_plot.fill(gen_pt = ak.to_numpy(flat_gen_pt))
        hreco_pt_plot.fill(reco_pt = ak.to_numpy(ak.flatten(ak.flatten(leading_reco_pf.pt))))
        hreco_pt_zoom_plot.fill(reco_pt = ak.to_numpy(ak.flatten(ak.flatten(leading_reco_pf.pt))))
        
        heta_ecal_vs_dxy.fill(eta_at_ecal = ak.to_numpy(flat_gen_eta_ecal), gen_dxy = ak.to_numpy(flat_gen_dxy))
        heta_ecal_vs_pt.fill (eta_at_ecal = ak.to_numpy(flat_gen_eta_ecal), gen_pt  = ak.to_numpy(flat_gen_pt))
        heta_ecal_vs_eta.fill(eta_at_ecal = ak.to_numpy(flat_gen_eta_ecal), gen_eta = ak.to_numpy(flat_gen_eta))
        hfail_pt_vs_eta.fill (pt = ak.to_numpy(flat_fail_gen_pt), eta = ak.to_numpy(flat_fail_gen_eta))

  
        h_deta_bump.fill(deta = ak.to_numpy(ak.flatten(deta_origin[bump_mask], axis=None)))
        h_dphi_bump.fill(dphi = ak.to_numpy(ak.flatten(dphi_origin[bump_mask], axis=None)))
        h_deta_ecal_bump.fill(deta_ecal = ak.to_numpy(ak.flatten(deta_ecal[bump_mask], axis=None)))
        h_dphi_ecal_bump.fill(dphi_ecal = ak.to_numpy(ak.flatten(dphi_ecal[bump_mask], axis=None)))
        h_dpt_bump.fill(dpt = ak.to_numpy(ak.flatten(dpt[bump_mask], axis=None)))
        h_dpt_all.fill(dpt = ak.to_numpy(ak.flatten(dpt, axis=None)))
        h_dcharge_bump.fill(dcharge = ak.to_numpy(ak.flatten(dcharge[bump_mask], axis=None)))
        h_dcharge_all.fill(dcharge = ak.to_numpy(ak.flatten(dcharge, axis=None)))
        h_lxy_bump.fill(lxy = ak.to_numpy(ak.flatten(gen_lxy_broadcasted[bump_mask], axis=None)))
        h_lxy_all.fill(lxy = ak.to_numpy(ak.flatten(gen_lxy_broadcasted, axis=None)))
      
        hphi_ecal_vs_phi_bump.fill(
            phi_at_ecal = ak.to_numpy(ak.flatten(phi_ecal_broadcasted[bump_mask], axis=None)), 
            phi         = ak.to_numpy(ak.flatten(phi_broadcasted[bump_mask], axis=None))
        )
        
        hgenphi_ecal_vs_genphi_bump.fill(
            genphi_at_ecal = ak.to_numpy(ak.flatten(gen_phi_ecal_broadcasted[bump_mask], axis=None)), 
            genphi         = ak.to_numpy(ak.flatten(gen_phi_broadcasted[bump_mask], axis=None))
        )
        h_pdgid_bump.fill(pdgid = ak.to_numpy(ak.flatten(gen_pdgid_broadcasted[bump_mask], axis=None))) 
        h_geneta_bump.fill(geneta = ak.to_numpy(ak.flatten(gen_eta_broadcasted[bump_mask], axis=None))) 


        ## understand why some events have worse deltaR ecal than deltaR
        hphi_ecal_vs_phi_worse.fill(
            phi_at_ecal = ak.to_numpy(ak.flatten(phi_ecal_broadcasted[worse_mask], axis=None)), 
            phi         = ak.to_numpy(ak.flatten(phi_broadcasted[worse_mask], axis=None))
        )
        
        hgenphi_ecal_vs_genphi_worse.fill(
            genphi_at_ecal = ak.to_numpy(ak.flatten(gen_phi_ecal_broadcasted[worse_mask], axis=None)), 
            genphi         = ak.to_numpy(ak.flatten(gen_phi_broadcasted[worse_mask], axis=None))
        )
        h_dpt_worse.fill(dpt = ak.to_numpy(ak.flatten(dpt[worse_mask], axis=None)))
        h_deta_worse.fill(deta = ak.to_numpy(ak.flatten(deta_origin[worse_mask], axis=None)))
        h_dphi_worse.fill(dphi = ak.to_numpy(ak.flatten(dphi_origin[worse_mask], axis=None)))
        h_deta_ecal_worse.fill(deta_ecal = ak.to_numpy(ak.flatten(deta_ecal[worse_mask], axis=None)))
        h_dphi_ecal_worse.fill(dphi_ecal = ak.to_numpy(ak.flatten(dphi_ecal[worse_mask], axis=None)))
        h_dcharge_worse.fill(dcharge = ak.to_numpy(ak.flatten(dcharge[worse_mask], axis=None)))
        h_lxy_worse.fill(lxy = ak.to_numpy(ak.flatten(gen_lxy_broadcasted[worse_mask], axis=None)))
        hpt_vs_genpt_worse.fill(
            genpt = ak.to_numpy(ak.flatten(gen_pt_broadcasted[worse_mask], axis=None)), 
            recopt         = ak.to_numpy(ak.flatten(pt_broadcasted[worse_mask], axis=None))
        )
        hdxy_vs_gendxy_worse.fill(
            gendxy = ak.to_numpy(ak.flatten(gen_dxy_broadcasted[worse_mask], axis=None)), 
            recodxy         = ak.to_numpy(ak.flatten(dxy_broadcasted[worse_mask], axis=None))
        )
        
      

        ## plot GEN lxy 
        flat_lxy = ak.flatten(gen_vis_taus.pion_lxy)
        flat_gentau_eta = ak.flatten(gen_vis_taus.eta)
        hgen_lxy.fill(gen_lxy = ak.to_numpy(flat_lxy))
        hgen_tau_pt.fill(gen_tau_pt = ak.to_numpy(ak.flatten(gen_vis_taus.pt)))
        
        heta_ecal_vs_lxy    .fill(eta_at_ecal = flat_gen_eta_ecal, gen_lxy = ak.to_numpy(flat_lxy))
        heta_ecal_vs_tau_eta.fill(eta_at_ecal = flat_gen_eta_ecal, gen_tau_eta = ak.to_numpy(flat_gentau_eta))
        hdxy_vs_lxy.fill(gen_dxy = ak.to_numpy(flat_gen_dxy), gen_lxy = ak.to_numpy(flat_lxy))

        # Match jets to GenVisTaus and plot matched jet pT
        ##  using firsts is safe only because running on mu-tau events
        first_gen_tau = ak.firsts(gen_vis_taus)

        # retrieve subgroup of all jets such that
        # they are leading jets and they have a leading_reco_pf
        ## since leading_reco_pf has shape: [events, jets, pf_candidates]
        has_valid_pf = ak.num(leading_reco_pf, axis=2) > 0
        valid_leading_jet = leading_jet[has_valid_pf]
        ## before it was jets.delta_r()
        dr_jets_taus = valid_leading_jet.delta_r(first_gen_tau)
#         matched_jets = jets[ak.fill_none(dr_jets_taus < 0.4, False)]
#         hmatched_jet_pt.fill(jet_pt = ak.to_numpy(ak.flatten(matched_jets.pt, axis=None)))

        # Find if the gen vis tau has a match
        has_matched_jet = ak.fill_none(ak.any(dr_jets_taus < self.delta_r_matching_threshold_, axis=1), False)
        # Filter the taus using the mask and fill the histogram with their pt
        matched_gen_taus = first_gen_tau[has_matched_jet]
        hmatched_jet_pt.fill(jet_pt = ak.to_numpy(ak.flatten(matched_gen_taus.pt, axis=None)))
        hmatched_jet_pion_pt.fill(jet_pion_pt = ak.to_numpy(ak.flatten(matched_gen_taus.pion_pt, axis=None)))
        hmatched_jet_dxy.fill(jet_dxy = ak.to_numpy(ak.flatten(matched_gen_taus.pion_dxy, axis=None)))        

        ## otherwise
#         # This compares EVERY jet to EVERY tau in the event
#         delta_r_matrix = jets[:, :, None].delta_r(gen_vis_taus[:, None, :])
#         # Check if a jet is matched to ANY tau within dR < 0.4
#         is_matched = ak.any(delta_r_matrix < 0.4, axis=2)
#         matched_jets = jets[is_matched]

#        dataset = events.metadata["dataset"]    
        return get_return_dict(leading_events)



    def postprocess(self, accumulator):
        return accumulator





import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pdb
from coffea.util import save
import itertools

if __name__ == "__main__":

    iterative_run = processor.Runner(
#         executor = processor.IterativeExecutor(compression=None),
        executor=processor.FuturesExecutor(compression=None, workers = 4),        
        schema=PFNanoAODSchema,
        maxchunks=4,
    )

    stau_fileset = {
#       "Stau_300_100mm": {
#         "files": {
#           "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v13/SMS-TStauStau_MStau-300_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_0_0.root" : "Events",
#           "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v13/SMS-TStauStau_MStau-300_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_10_0.root" : "Events",
#           "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v13/SMS-TStauStau_MStau-300_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_11_0.root" : "Events",
#           "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v13/SMS-TStauStau_MStau-300_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_12_0.root" : "Events",
#           "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v13/SMS-TStauStau_MStau-300_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_13_0.root" : "Events",
#           "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v13/SMS-TStauStau_MStau-300_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_14_0.root" : "Events",
#           "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v13/SMS-TStauStau_MStau-300_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_15_0.root" : "Events",
#           "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v13/SMS-TStauStau_MStau-300_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_16_0.root" : "Events",
#           "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v13/SMS-TStauStau_MStau-300_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_17_0.root" : "Events",
#           "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v13/SMS-TStauStau_MStau-300_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_18_0.root" : "Events",
#           "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v13/SMS-TStauStau_MStau-300_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_19_0.root" : "Events",
#           "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v13/SMS-TStauStau_MStau-300_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_1_0.root" : "Events",
#           "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v13/SMS-TStauStau_MStau-300_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_20_0.root" : "Events",
#           "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v13/SMS-TStauStau_MStau-300_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_21_0.root" : "Events",
#           "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v13/SMS-TStauStau_MStau-300_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_22_0.root" : "Events",
#           "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v13/SMS-TStauStau_MStau-300_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_23_0.root" : "Events",
#           "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v13/SMS-TStauStau_MStau-300_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_24_0.root" : "Events",
#           "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v13/SMS-TStauStau_MStau-300_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_25_0.root" : "Events",
#           "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v13/SMS-TStauStau_MStau-300_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_26_0.root" : "Events",
#           "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v13/SMS-TStauStau_MStau-300_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_27_0.root" : "Events",
#           "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v13/SMS-TStauStau_MStau-300_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_28_0.root" : "Events",
#           "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v13/SMS-TStauStau_MStau-300_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_29_0.root" : "Events",
#           "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v13/SMS-TStauStau_MStau-300_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_2_0.root" : "Events",
#           "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v13/SMS-TStauStau_MStau-300_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_30_0.root" : "Events",
#           "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v13/SMS-TStauStau_MStau-300_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_31_0.root" : "Events",
#           "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v13/SMS-TStauStau_MStau-300_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_32_0.root" : "Events",
#           "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v13/SMS-TStauStau_MStau-300_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_33_0.root" : "Events",
#           "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v13/SMS-TStauStau_MStau-300_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_34_0.root" : "Events",
#           "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v13/SMS-TStauStau_MStau-300_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_35_0.root" : "Events",
#           "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v13/SMS-TStauStau_MStau-300_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_36_0.root" : "Events",
#           "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v13/SMS-TStauStau_MStau-300_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_37_0.root" : "Events",
#           "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v13/SMS-TStauStau_MStau-300_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_38_0.root" : "Events",
#           "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v13/SMS-TStauStau_MStau-300_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_39_0.root" : "Events",
#           "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v13/SMS-TStauStau_MStau-300_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_3_0.root" : "Events",
#           "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v13/SMS-TStauStau_MStau-300_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_40_0.root" : "Events",
#           "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v13/SMS-TStauStau_MStau-300_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_4_0.root" : "Events",
#           "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v13/SMS-TStauStau_MStau-300_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_5_0.root" : "Events",
#           "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v13/SMS-TStauStau_MStau-300_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_6_0.root" : "Events",
#           "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v13/SMS-TStauStau_MStau-300_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_7_0.root" : "Events",
#           "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v13/SMS-TStauStau_MStau-300_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_8_0.root" : "Events",
#           "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v13/SMS-TStauStau_MStau-300_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_9_0.root" : "Events",
#          }
#       },
      ## adding requirement on vertex in tracker
#       "Stau_300_1000mm": {
#         "files": {
#           "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v13/SMS-TStauStau_MStau-300_ctau-1000mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_0_0.root" : "Events",
#           "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v13/SMS-TStauStau_MStau-300_ctau-1000mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_10_0.root" : "Events",
#           "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v13/SMS-TStauStau_MStau-300_ctau-1000mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_11_0.root" : "Events",
#           "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v13/SMS-TStauStau_MStau-300_ctau-1000mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_12_0.root" : "Events",
#           "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v13/SMS-TStauStau_MStau-300_ctau-1000mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_13_0.root" : "Events",
#           "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v13/SMS-TStauStau_MStau-300_ctau-1000mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_14_0.root" : "Events",
#           "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v13/SMS-TStauStau_MStau-300_ctau-1000mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_15_0.root" : "Events",
#           "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v13/SMS-TStauStau_MStau-300_ctau-1000mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_16_0.root" : "Events",
#           "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v13/SMS-TStauStau_MStau-300_ctau-1000mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_17_0.root" : "Events",
#           "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v13/SMS-TStauStau_MStau-300_ctau-1000mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_18_0.root" : "Events",
#           "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v13/SMS-TStauStau_MStau-300_ctau-1000mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_19_0.root" : "Events",
#           "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v13/SMS-TStauStau_MStau-300_ctau-1000mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_1_0.root" : "Events",
#           "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v13/SMS-TStauStau_MStau-300_ctau-1000mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_20_0.root" : "Events",
#           "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v13/SMS-TStauStau_MStau-300_ctau-1000mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_21_0.root" : "Events",
#           "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v13/SMS-TStauStau_MStau-300_ctau-1000mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_22_0.root" : "Events",
#           "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v13/SMS-TStauStau_MStau-300_ctau-1000mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_23_0.root" : "Events",
#           "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v13/SMS-TStauStau_MStau-300_ctau-1000mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_24_0.root" : "Events",
#           "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v13/SMS-TStauStau_MStau-300_ctau-1000mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_25_0.root" : "Events",
#           "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v13/SMS-TStauStau_MStau-300_ctau-1000mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_26_0.root" : "Events",
#           "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v13/SMS-TStauStau_MStau-300_ctau-1000mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_27_0.root" : "Events",
#           "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v13/SMS-TStauStau_MStau-300_ctau-1000mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_28_0.root" : "Events",
#           "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v13/SMS-TStauStau_MStau-300_ctau-1000mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_29_0.root" : "Events",
#           "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v13/SMS-TStauStau_MStau-300_ctau-1000mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_2_0.root" : "Events",
#           "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v13/SMS-TStauStau_MStau-300_ctau-1000mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_30_0.root" : "Events",
#           "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v13/SMS-TStauStau_MStau-300_ctau-1000mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_31_0.root" : "Events",
#           "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v13/SMS-TStauStau_MStau-300_ctau-1000mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_32_0.root" : "Events",
#           "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v13/SMS-TStauStau_MStau-300_ctau-1000mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_33_0.root" : "Events",
#           "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v13/SMS-TStauStau_MStau-300_ctau-1000mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_34_0.root" : "Events",
#           "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v13/SMS-TStauStau_MStau-300_ctau-1000mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_35_0.root" : "Events",
#           "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v13/SMS-TStauStau_MStau-300_ctau-1000mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_36_0.root" : "Events",
#           "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v13/SMS-TStauStau_MStau-300_ctau-1000mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_38_0.root" : "Events",
#           "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v13/SMS-TStauStau_MStau-300_ctau-1000mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_3_0.root" : "Events",
#           "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v13/SMS-TStauStau_MStau-300_ctau-1000mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_4_0.root" : "Events",
#           "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v13/SMS-TStauStau_MStau-300_ctau-1000mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_5_0.root" : "Events",
#           "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v13/SMS-TStauStau_MStau-300_ctau-1000mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_6_0.root" : "Events",
#           "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v13/SMS-TStauStau_MStau-300_ctau-1000mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_7_0.root" : "Events",
#           "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v13/SMS-TStauStau_MStau-300_ctau-1000mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_8_0.root" : "Events",
#           "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/Run3_Summer22_chs_AK4PFCands_v13/SMS-TStauStau_MStau-300_ctau-1000mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_9_0.root" : "Events",
#          }
#       },
#       "Stau_300_1mm": {
#         "files": {
#           "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/summary/Run3_Summer22_chs_AK4PFCands_v12/SMS-TStauStau_MStau-300_ctau-1mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_0.root" : "Events",
#           "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/summary/Run3_Summer22_chs_AK4PFCands_v12/SMS-TStauStau_MStau-300_ctau-1mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_1.root" : "Events",
#           "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/summary/Run3_Summer22_chs_AK4PFCands_v12/SMS-TStauStau_MStau-300_ctau-1mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_2.root" : "Events",
#           "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/summary/Run3_Summer22_chs_AK4PFCands_v12/SMS-TStauStau_MStau-300_ctau-1mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_3.root" : "Events",
#          }
#       },
      "Stau_300_100mm": {
        "files": {
          "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/summary/Run3_Summer22_chs_AK4PFCands_v21/SMS-TStauStau_MStau-300_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_0.root" : "Events",
#           "root://cmseos.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/summary/Run3_Summer22_chs_AK4PFCands_v21/SMS-TStauStau_MStau-300_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_1.root" : "Events",
#           "root://cmseos.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/summary/Run3_Summer22_chs_AK4PFCands_v21/SMS-TStauStau_MStau-300_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_2.root" : "Events",
#           "root://cmseos.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/summary/Run3_Summer22_chs_AK4PFCands_v21/SMS-TStauStau_MStau-300_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_3.root" : "Events",
#           "/eos/cms/store/user/fiorendi/displacedTaus/inputFiles_fortesting/nano_0.root" : "Events",
         }
      }
#       "Stau_300_1000mm": {
#         "files": {
#           "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/summary/Run3_Summer22_chs_AK4PFCands_v12/SMS-TStauStau_MStau-300_ctau-1000mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_0.root" : "Events",
#           "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/summary/Run3_Summer22_chs_AK4PFCands_v12/SMS-TStauStau_MStau-300_ctau-1000mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_1.root" : "Events",
#           "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/summary/Run3_Summer22_chs_AK4PFCands_v12/SMS-TStauStau_MStau-300_ctau-1000mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_2.root" : "Events",
#           "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/summary/Run3_Summer22_chs_AK4PFCands_v12/SMS-TStauStau_MStau-300_ctau-1000mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_3.root" : "Events",
#          }
#       },
#       "Stau_100_100mm": {
#         "files": {
#           "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/summary/Run3_Summer22_chs_AK4PFCands_v12/SMS-TStauStau_MStau-100_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_0.root" : "Events",
#           "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/summary/Run3_Summer22_chs_AK4PFCands_v12/SMS-TStauStau_MStau-100_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_1.root" : "Events",
#          }
#       },
#       "Stau_100_1000mm": {
#         "files": {
#           "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/summary/Run3_Summer22_chs_AK4PFCands_v12/SMS-TStauStau_MStau-100_ctau-1000mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_0.root" : "Events",
#           "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/summary/Run3_Summer22_chs_AK4PFCands_v12/SMS-TStauStau_MStau-100_ctau-1000mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_1.root" : "Events",
#           "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/summary/Run3_Summer22_chs_AK4PFCands_v12/SMS-TStauStau_MStau-100_ctau-1000mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_2.root" : "Events",
#          }
#       },
#       "Stau_500_1000mm": {
#         "files": {
#           "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/summary/Run3_Summer22_chs_AK4PFCands_v12/SMS-TStauStau_MStau-500_ctau-1000mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_0.root" : "Events",
#           "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/summary/Run3_Summer22_chs_AK4PFCands_v12/SMS-TStauStau_MStau-500_ctau-1000mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_1.root" : "Events",
#           "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/summary/Run3_Summer22_chs_AK4PFCands_v12/SMS-TStauStau_MStau-500_ctau-1000mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_2.root" : "Events",
#           "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/summary/Run3_Summer22_chs_AK4PFCands_v12/SMS-TStauStau_MStau-500_ctau-1000mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_3.root" : "Events",
#          }
#       }
    }
    
    plot_dir = 'eff_vs_dxy'

    decayM = 0
    min_gen_tau_pt = 20
#     delta_r_matching_threshold = 0.3
    min_jet_pt = 20
    dr_thresholds = [0.3]
#     dr_thresholds = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    pion_pt_thresholds = [0]
#     pion_pt_thresholds = [0, 1, 2, 5, 10]
#     pion_pt_thresholds = [1, 2, 5, 10]
    
    for min_reco_pion_pt,delta_r_matching_threshold in itertools.product(pion_pt_thresholds, dr_thresholds):
        print(f"\n=================================================")
        print(f" Running processor with deltaR threshold = {delta_r_matching_threshold}")
        print(f"=================================================")    
        out = iterative_run(
            stau_fileset,
            treename="Events",
            processor_instance=TauProcessor(
                min_reco_pf_pt = min_reco_pion_pt, 
                min_gen_tau_pt = min_gen_tau_pt, 
                decayM=  decayM, 
                min_gen_pion_pt = 0, 
                min_jet_pt = min_jet_pt, 
                delta_r_matching = delta_r_matching_threshold
                ),
        )
    
        for sample_name in  [
    #                          'Stau_300_100mm',
    #                          'Stau_300_1mm',
                             'Stau_300_100mm',
    #                          'Stau_300_1000mm',
    #                          'Stau_100_100mm',
    #                          'Stau_100_1000mm',
    #                          'Stau_500_1000mm'
                            ]:
    
            sample_label = sample_name 
            print(f"\n--- Cutflow for {sample_name} (dR = {delta_r_matching_threshold}) ---")
            for cut, count in out[sample_name]['cutflow'].items():
                print(f"{cut}: {count}")
    
            output = {
            "h_allgenpions_eta_ecal_vs_eta": out[sample_name]['h_allgenpions_eta_ecal_vs_eta'],
            "hgen_dxy": out[sample_name]['hgen_dxy'],
            "hall_dxy": out[sample_name]['hall_dxy'],
            "hpass_dxy": out[sample_name]['hpass_dxy'],
            "hfail_dxy": out[sample_name]['hfail_dxy'],
            "hreco_dxy": out[sample_name]['hreco_dxy'],
            "hreco_ecal_dxy": out[sample_name]['hreco_ecal_dxy'],
            "hgen_dxy_zoom": out[sample_name]['hgen_dxy_zoom'],
            "hreco_dxy_zoom": out[sample_name]['hreco_dxy_zoom'],
            "hreco_ecal_dxy_zoom": out[sample_name]['hreco_ecal_dxy_zoom'],
            "hgen_lxy": out[sample_name]['hgen_lxy'],
#             "hreco_lxy": out[sample_name]['hreco_lxy'],
#             "hreco_ecal_lxy": out[sample_name]['hreco_ecal_lxy'],
            "hgen_pt": out[sample_name]['hgen_pt'],
            "hreco_pt": out[sample_name]['hreco_pt'],
            "hreco_ecal_pt": out[sample_name]['hreco_ecal_pt'],
            "hdelta": out[sample_name]['hdelta'],
            "hdeltaR": out[sample_name]['hdeltaR'],
            "hdeltaR_custom": out[sample_name]['hdeltaR_custom'],
            "hdeltaR_custom_zoomout": out[sample_name]['hdeltaR_custom_zoomout'],
            "heta_ecal_vs_dxy": out[sample_name]['heta_ecal_vs_dxy'],
            "heta_ecal_vs_pt": out[sample_name]['heta_ecal_vs_pt'],
            "heta_ecal_vs_eta": out[sample_name]['heta_ecal_vs_eta'],
            "heta_ecal_vs_lxy": out[sample_name]['heta_ecal_vs_lxy'],
            "heta_ecal_vs_tau_eta": out[sample_name]['heta_ecal_vs_tau_eta'],
            "hdxy_vs_lxy": out[sample_name]['hdxy_vs_lxy'],
            "hfail_pt_vs_eta": out[sample_name]['hfail_pt_vs_eta'],
            "hgen_pt_plot": out[sample_name]['hgen_pt_plot'],
            "hgen_pt_zoom_plot": out[sample_name]['hgen_pt_zoom_plot'],
            "hreco_pt_plot": out[sample_name]['hreco_pt_plot'],
            "hreco_pt_zoom_plot": out[sample_name]['hreco_pt_zoom_plot'],
            "hmatched_jet_pt": out[sample_name]['hmatched_jet_pt'],
            "hmatched_jet_pion_pt": out[sample_name]['hmatched_jet_pion_pt'],
            "hmatched_jet_dxy": out[sample_name]['hmatched_jet_dxy'],
            "h_deta_bump": out[sample_name]['h_deta_bump'],
            "h_dphi_bump": out[sample_name]['h_dphi_bump'],
            "h_deta_ecal_bump": out[sample_name]['h_deta_ecal_bump'],
            "h_dphi_ecal_bump": out[sample_name]['h_dphi_ecal_bump'],
            "h_dpt_bump": out[sample_name]['h_dpt_bump'],
            "h_dpt_all": out[sample_name]['h_dpt_all'],
            "h_dcharge_bump": out[sample_name]['h_dcharge_bump'],
            "h_dcharge_all": out[sample_name]['h_dcharge_all'],
            "h_lxy_all": out[sample_name]['h_lxy_all'],
            "h_lxy_bump": out[sample_name]['h_lxy_bump'], 
            "hphi_ecal_vs_phi_bump": out[sample_name]['hphi_ecal_vs_phi_bump'], 
            "hgenphi_ecal_vs_genphi_bump": out[sample_name]['hgenphi_ecal_vs_genphi_bump'], 
            "h_pdgid_bump": out[sample_name]['h_pdgid_bump'], 
            "h_geneta_bump": out[sample_name]['h_geneta_bump'], 
            "hgen_tau_pt": out[sample_name]['hgen_tau_pt'], 
            "hphi_ecal_vs_phi_worse": out[sample_name]['hphi_ecal_vs_phi_worse'], 
            "hgenphi_ecal_vs_genphi_worse": out[sample_name]['hgenphi_ecal_vs_genphi_worse'], 
            "h_deta_worse": out[sample_name]['h_deta_worse'],
            "h_dphi_worse": out[sample_name]['h_dphi_worse'],
            "h_deta_ecal_worse": out[sample_name]['h_deta_ecal_worse'],
            "h_dphi_ecal_worse": out[sample_name]['h_dphi_ecal_worse'],
            "h_dpt_worse": out[sample_name]['h_dpt_worse'],
            "h_dcharge_worse": out[sample_name]['h_dcharge_worse'],
            "h_lxy_worse": out[sample_name]['h_lxy_worse'], 
            "hpt_vs_genpt_worse": out[sample_name]['hpt_vs_genpt_worse'], 
            "hdxy_vs_gendxy_worse": out[sample_name]['hdxy_vs_gendxy_worse'], 
            "hmatched_alljet_pt": out[sample_name]['hmatched_alljet_pt'], 
            "hgen_pt_for_alljet": out[sample_name]['hgen_pt_for_alljet'], 
            "hreco_gvtpt": out[sample_name]['hreco_gvtpt'], 
            "hreco_ecal_gvtpt": out[sample_name]['hreco_ecal_gvtpt'], 
            "hgen_gvtpt": out[sample_name]['hgen_gvtpt'], 
        }
            delta_r_matching_threshold_string = str(delta_r_matching_threshold).replace('.','p')
            out_filename = f"inputs_study_propagation/histograms_{sample_name}_dR{delta_r_matching_threshold_string}_MaxLxy{maxLxy}_decayM{decayM}_pfPt{min_reco_pion_pt}_genPionPt0_genTauPt{min_gen_tau_pt}_noTaggerScore_goodRecoProp.coffea"
            save(output, out_filename)
            print(f"Saved output to {out_filename}")
