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
# keep the helper functions (pure functions that operate on event-like objects)
def get_gen_pions_from_taus(event_):
    pions_from_taus_ = event_.GenPart[abs(event_.GenPart.pdgId) == 211]
    pions_from_taus_ = pions_from_taus_[abs(pions_from_taus_.distinctParent.pdgId) == 15]
    pions_from_taus_ = pions_from_taus_[abs(pions_from_taus_.distinctParent.distinctParent.pdgId) == 1000015]
    return pions_from_taus_

def get_gen_muons_from_taus(event_):
    muons_from_tau_ = event_.GenPart[(abs(event_.GenPart.pdgId) == 13) & (event_.GenPart.hasFlags("isLastCopy"))]
    muons_from_tau_ = muons_from_tau_[
        (muons_from_tau_.pt > 20) &
        (abs(muons_from_tau_.eta) < max_eta) &
        (abs(muons_from_tau_.distinctParent.distinctParent.pdgId) == 1000015)
    ]
    return muons_from_tau_

def calc_and_add_dxy_gen_pion(pions_from_taus_, events_):
    counts_pions_ = ak.num(pions_from_taus_, axis=1)
    pv_y_ = np.asarray(events_.PVBS.y)
    pv_x_ = np.asarray(events_.PVBS.x)
    pv_y_expanded_pions_ = ak.unflatten(np.repeat(pv_y_, np.asarray(counts_pions_)), counts_pions_)
    pv_x_expanded_pions_ = ak.unflatten(np.repeat(pv_x_, np.asarray(counts_pions_)), counts_pions_)
    pions_from_taus_["dxy"] = (pions_from_taus_.vy - pv_y_expanded_pions_) * np.cos(pions_from_taus_.phi) - \
                              (pions_from_taus_.vx - pv_x_expanded_pions_) * np.sin(pions_from_taus_.phi)

def calc_and_add_dxy_gen_muon(muons_from_taus_, events_):
    counts_muons_ = ak.num(muons_from_taus_, axis=1)
    pv_y_ = np.asarray(events_.PVBS.y)
    pv_x_ = np.asarray(events_.PVBS.x)
    pv_y_expanded = ak.unflatten(np.repeat(pv_y_, np.asarray(counts_muons_)), counts_muons_)
    pv_x_expanded = ak.unflatten(np.repeat(pv_x_, np.asarray(counts_muons_)), counts_muons_)
    muons_from_taus_["dxy"] = (muons_from_taus_.vy - pv_y_expanded) * np.cos(muons_from_taus_.phi) - \
                              (muons_from_taus_.vx - pv_x_expanded) * np.sin(muons_from_taus_.phi)

def select_jets(event_):
    jets_ = event_.Jet[(abs(event_.Jet.eta) < max_eta) & (event_.Jet.pt > 30)]
    jets_ = jets_[
       (jets_.neHEF < 0.99) &
       (jets_.neEmEF < 0.9) &
       (jets_.chMultiplicity + jets_.neMultiplicity > 1) &
       (jets_.chHEF > 0.01) &
       (jets_.chMultiplicity > 0) &
       (jets_.muEF < 0.8) &
       (jets_.chEmEF < 0.8)
    ]
    jets_ = jets_[(jets_.muEF < 0.5)]
    return jets_

def get_leading_jet(jets_):
    sorted_jets_ = jets_[ak.argsort(jets_.disTauTag_score1, ascending=False)]
    leading_jet_ = sorted_jets_[:, :1]
    leading_jet_ = leading_jet_[leading_jet_.disTauTag_score1 > 0.9]
    return leading_jet_

def get_charged_pf(leading_jet_, min_reco_pf_pt_):
    reco_pf_ = leading_jet_.constituents.pf
    reco_pf_ = reco_pf_[(reco_pf_.charge != 0)]
    reco_pf_ = reco_pf_[(reco_pf_.pt > min_reco_pf_pt_)]
    return reco_pf_


def delta_r_ecal(reco_obj, gen_obj):
    ### compute deltaR using coords at the ecal surface.
    ### this is specific to pion as it uses at_ecal_pion

    dphi = np.abs(reco_obj.phi_at_ecal - gen_obj.phi_at_ecal_pion)
    dphi = ak.where(dphi > np.pi, 2*np.pi - dphi, dphi)
    deta = reco_obj.eta_at_ecal - gen_obj.eta_at_ecal_pion
    return np.sqrt(deta**2 + dphi**2)


dxy_bins_low = np.arange(0, 8, 1)
dxy_bins_med = np.arange(8, 20, 4)
dxy_bins_high = np.arange(20, 110, 10)
dxy_bins_eff = np.concatenate([dxy_bins_low, dxy_bins_med, dxy_bins_high])

# lxy_bins_eff = np.logspace(-2, 2, 10)
lxy_bins_eff = np.arange(0, 130, 1)


new_dxy_bins_eff = np.logspace(-2, 1.7, 10)
pt_bins_eff = np.logspace(1.5, 2.7, 12)

dxy_bins = np.arange(0, 100, 1)

class TauProcessor(processor.ProcessorABC):

    def __init__(self, min_reco_pf_pt=20, min_gen_tau_pt=20, decayM=0):
        self.min_reco_pf_pt = min_reco_pf_pt
        self.min_gen_tau_pt = min_gen_tau_pt
        self.decayM = decayM

    def process(self, events):

        hpt = ( Hist.new
            .Reg(200, 0, 100, name="recoPT", label="reco pT (GeV)")
            .Reg(200, 0, 100, name="genPT" , label="gen pT (GeV)")
            .Double()
        )
        hdelta = ( Hist.new
            .Reg(200, 0, 0.7, name="deltaR", label="deltaR")
            .Reg(200, 0, 0.7, name="deltaR_custom", label="deltaR at ECAL")
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
            .Reg(80,  40, 200, name="gen_pt", label="gen_pt")
#             .Reg(120,  40, 100, name="gen_pt", label="gen_pt")
            .Double()
        )
        edges_x = [-101, -100, -99, -98, -97]
        edges_x += list(np.linspace(-3, 3, 61))  
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
            .Reg(100, 0, 10, name="deltaR", label="deltaR")
            .Double()
        )
        hdeltaR_custom = ( Hist.new
            .Reg(100, 0, 10, name="deltaR_custom", label="deltaR_custom")
            .Double()
        )
        hdeltaR_custom_zoomout = ( Hist.new
            .Reg(400, 110, 150, name="deltaR_custom", label="deltaR_custom")
            .Double()
        )
        
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
            .Var(new_dxy_bins_eff, name="reco_dxy", label="reco dxy")
            .Double()
        )
        hreco_ecal_dxy_zoom = ( Hist.new
            .Var(new_dxy_bins_eff, name="reco_ecal_dxy", label="ecal reco dxy")
            .Double()
        )
        hgen_dxy_zoom = ( Hist.new
            .Var(new_dxy_bins_eff, name="gen_dxy", label="gen dxy")
            .Double()
        )

        hreco_lxy = ( Hist.new
            .Var(lxy_bins_eff, name="reco_lxy", label="reco lxy")
            .Double()
        )
        hreco_ecal_lxy = ( Hist.new
            .Var(lxy_bins_eff, name="reco_ecal_lxy", label="ecal reco lxy")
            .Double()
        )
        hgen_lxy = ( Hist.new
            .Var(lxy_bins_eff, name="gen_lxy", label="gen lxy")
            .Double()
        )
        hreco_pt = ( Hist.new
            .Var(pt_bins_eff, name="reco_pt", label="reco pt")
            .Double()
        )
        hreco_ecal_pt = ( Hist.new
            .Var(pt_bins_eff, name="reco_ecal_pt", label="ecal reco pt")
            .Double()
        )
        hgen_pt = ( Hist.new
            .Var(pt_bins_eff, name="gen_pt", label="gen pt")
            .Double()
        )

        # select gen visible taus from the desired long-lived particle chain
        gen_vis_taus = events.GenVisTau[
            (abs(events.GenVisTau.parent.pdgId) == 15) &
            (abs(events.GenVisTau.parent.distinctParent.pdgId) == 1000015) &
            (events.GenVisTau.parent.distinctParent.hasFlags("isLastCopy")) &
            (events.GenVisTau.parent.hasFlags("fromHardProcess")) &
            (events.GenVisTau.status == self.decayM) &
            (events.GenVisTau.pt > self.min_gen_tau_pt) &
            (abs(events.GenVisTau.eta) < max_eta)
        ]

        tau_vx = gen_vis_taus.parent.distinctParent.vx - ak.firsts(gen_vis_taus.parent.distinctChildren.vx, axis =2)
        tau_vy = gen_vis_taus.parent.distinctParent.vy - ak.firsts(gen_vis_taus.parent.distinctChildren.vy, axis =2)
        Lxy = np.sqrt(tau_vx**2 + tau_vy**2)
        gen_vis_taus = ak.with_field(gen_vis_taus, Lxy, where="lxy")    
        gen_vis_taus = gen_vis_taus[abs(gen_vis_taus.lxy) < maxLxy]

        muons_from_taus = get_gen_muons_from_taus(events)
        mask = (ak.num(gen_vis_taus) == 1) & (ak.num(muons_from_taus) == 1)
        ## ************************** ##
        filter_events = events[mask]
        ## ************************** ##

        pions_from_taus = get_gen_pions_from_taus(filter_events)
        counts_pions = ak.num(pions_from_taus, axis=1)
        ## ************************** ##
        filter_events = filter_events[counts_pions > 0]
        ## ************************** ##
        if len(filter_events) == 0:
            return out  # nothing in this chunk

        pions_from_taus = get_gen_pions_from_taus(filter_events)
        muons_from_taus = get_gen_muons_from_taus(filter_events)
        calc_and_add_dxy_gen_muon(muons_from_taus, filter_events)

        # RECO side
        jets = select_jets(filter_events)
        leading_jet = get_leading_jet(jets)

        # wrap leading jet into proper shapes, handle None
        jets_one = ak.pad_none(jets, 1, axis=1)
        leading_one = jets_one[:, 0]
        empty_jet_lists = ak.Array([[[]]] * len(leading_one))
        charged_reco_pf = get_charged_pf(leading_jet, self.min_reco_pf_pt)
        pf_per_jet = ak.singletons(charged_reco_pf)

        reco_pf = ak.where(ak.is_none(leading_one), empty_jet_lists, pf_per_jet)

        pf_mask = (reco_pf.charge != 0) & (reco_pf.pt > self.min_reco_pf_pt)
        counts_charged_pf = ak.sum(pf_mask, axis=2)
        has_charged_in_leading = (ak.sum(counts_charged_pf, axis=1) > 0)
        has_charged_in_leading_event = ak.fill_none(ak.firsts(has_charged_in_leading, axis=1), False)
        ## ************************** ##
        pf_charged_events = filter_events[has_charged_in_leading_event]
        ## ************************** ##
        if len(pf_charged_events) == 0:
            return out



        # recompute with selected events
        pions_from_taus = get_gen_pions_from_taus(pf_charged_events)
        sorted_gen_pions = pions_from_taus[ak.argsort(pions_from_taus.pt, ascending=False)]
        leading_gen_pion = sorted_gen_pions[:, :1]
#         leading_gen_pion = leading_gen_pion[leading_gen_pion.eta_at_ecal_pion > -10]

        jets = select_jets(pf_charged_events)
        leading_jet = get_leading_jet(jets)
        reco_pf = get_charged_pf(leading_jet, self.min_reco_pf_pt)
        sorted_reco_pf = reco_pf[ak.argsort(reco_pf.pt, ascending=False)]
        leading_reco_pf = sorted_reco_pf[ak.argmax(sorted_reco_pf.pt, axis=2, keepdims=True)]
        leading_reco_pf = leading_reco_pf[leading_reco_pf.eta_at_ecal > -10]

        counts_gen_leading = ak.num(leading_gen_pion, axis=1)  # number of pions from reco tau per event
        counts_pf_leading = ak.num(leading_reco_pf, axis=1)  # number of pions from reco tau per event
        ## ************************** ##
        ## select only events with at least one RECO charged PFCand inside the leading (score) jet
        leading_events = pf_charged_events[(counts_pf_leading > 0) ]
        leading_events = leading_events[(counts_gen_leading > 0) ]
        ## ************************** ## 


        ##recompute everything again
        pions_from_taus = get_gen_pions_from_taus(leading_events)
        ## calculate gen level dxy for pions
        calc_and_add_dxy_gen_pion(pions_from_taus, leading_events)
        sorted_gen_pions = pions_from_taus[ak.argsort(pions_from_taus.pt, ascending=False)]
        leading_gen_pion = sorted_gen_pions[:, :1]
#         leading_gen_pion = leading_gen_pion[leading_gen_pion.eta_at_ecal_pion > -10]
        
        jets = select_jets(leading_events)
        leading_jet = get_leading_jet(jets)
        reco_pf =  get_charged_pf(leading_jet, self.min_reco_pf_pt)
        sorted_reco_pf = reco_pf[ak.argsort(reco_pf.pt, ascending=False)]
        leading_reco_pf = sorted_reco_pf[ak.argmax(sorted_reco_pf.pt, axis=2, keepdims=True)]

        ## try to get the matched pions
        matched_pf_to_pions = ak.flatten(leading_reco_pf).nearest(leading_gen_pion, threshold = 0.4)
        
        ## try to replicate with custom deltaR (at ECAL surface)
        # Compute deltaR matrix between each reco and gen candidate in an event
        dr_matrix = delta_r_ecal(
            ak.cartesian({"pf": ak.flatten(leading_reco_pf), "pion": leading_gen_pion}, nested=True).pf,
            ak.cartesian({"pf": ak.flatten(leading_reco_pf), "pion": leading_gen_pion}, nested=True).pion
        )        
        ## find the index of the gen_pion closest to each reco_pf
        ecal_nearest_idx = ak.argmin(dr_matrix, axis=2)  # per reco_pf        
        ## get the corresponding gen_pion
        ecal_matched_pf_to_pions = leading_gen_pion[ecal_nearest_idx]
        dr_min = ak.min(dr_matrix, axis=2)
        ecal_matched_pf_to_pions_masked = ak.mask(ecal_matched_pf_to_pions, dr_min < 0.4)


        # compute dR values
        delta_r_origin = leading_reco_pf[:, :, None].delta_r(leading_gen_pion[:, None, :])

        deta = leading_reco_pf.eta_at_ecal[:, :, None] - leading_gen_pion.eta_at_ecal_pion[:, None, :]
        dphi = abs(leading_reco_pf.phi_at_ecal[:, :, None] - leading_gen_pion.phi_at_ecal_pion[:, None, :])
        dphi = ak.where(dphi > np.pi, 2*np.pi - dphi, dphi)
        delta_r_custom = np.sqrt(deta**2 + dphi**2)

        flat_delta_r = ak.flatten(ak.flatten(ak.flatten(delta_r_origin)))
        flat_delta_r_custom = ak.flatten(ak.flatten(ak.flatten(delta_r_custom)))
        flat_delta_r = ak.fill_none(flat_delta_r, 999)
        flat_delta_r_custom = ak.fill_none(flat_delta_r_custom, 999)

        flat_gen_dxy = ak.flatten(leading_gen_pion.dxy)
        flat_reco_dxy = ak.flatten(matched_pf_to_pions.dxy)
        flat_ecal_reco_dxy = ak.flatten(ecal_matched_pf_to_pions_masked.dxy)

        flat_gen_pt = ak.flatten(leading_gen_pion.pt)
        flat_gen_eta = ak.flatten(leading_gen_pion.eta)
        flat_gen_eta_ecal = ak.flatten(leading_gen_pion.eta_at_ecal_pion)
        flat_reco_pt = ak.flatten(matched_pf_to_pions.pt)
        flat_ecal_reco_pt = ak.flatten(ecal_matched_pf_to_pions_masked.pt)

        flat_fail_gen_pt  = ak.flatten(leading_gen_pion[leading_gen_pion.eta_at_ecal_pion < -10].pt)
        flat_fail_gen_eta = ak.flatten(leading_gen_pion[leading_gen_pion.eta_at_ecal_pion < -10].eta)
        flat_fail_gen_dxy = ak.flatten(leading_gen_pion[leading_gen_pion.eta_at_ecal_pion < -10].dxy)
        flat_pass_gen_dxy = ak.flatten(leading_gen_pion[leading_gen_pion.eta_at_ecal_pion > -10].dxy)
        
        # fill histograms

        hdelta.fill(deltaR=ak.to_numpy(flat_delta_r), deltaR_custom=ak.to_numpy(flat_delta_r_custom))
        hdeltaR.fill(deltaR=ak.to_numpy(flat_delta_r))
        hdeltaR_custom.fill(deltaR_custom=ak.to_numpy(flat_delta_r_custom))
        hdeltaR_custom_zoomout.fill(deltaR_custom=ak.to_numpy(flat_delta_r_custom))

        hgen_dxy.fill(gen_dxy=ak.to_numpy(flat_gen_dxy))
        hreco_dxy.fill(reco_dxy=ak.to_numpy(flat_reco_dxy))
        hreco_ecal_dxy.fill(reco_ecal_dxy=ak.to_numpy(flat_ecal_reco_dxy))

        hall_dxy.fill(gen_dxy=ak.to_numpy(flat_gen_dxy))
        hpass_dxy.fill(gen_dxy=ak.to_numpy(flat_pass_gen_dxy))
        hfail_dxy.fill(gen_dxy=ak.to_numpy(flat_fail_gen_dxy))

        hgen_dxy_zoom.fill(gen_dxy=ak.to_numpy(flat_gen_dxy))
        hreco_dxy_zoom.fill(reco_dxy=ak.to_numpy(flat_reco_dxy))
        hreco_ecal_dxy_zoom.fill(reco_ecal_dxy=ak.to_numpy(flat_ecal_reco_dxy))

        hgen_pt.fill(gen_pt=ak.to_numpy(flat_gen_pt))
        hreco_pt.fill(reco_pt=ak.to_numpy(flat_reco_pt))
        hreco_ecal_pt.fill(reco_ecal_pt=ak.to_numpy(flat_ecal_reco_pt))
        
        heta_ecal_vs_dxy.fill(eta_at_ecal = flat_gen_eta_ecal, gen_dxy = ak.to_numpy(flat_gen_dxy))
        heta_ecal_vs_pt.fill (eta_at_ecal = flat_gen_eta_ecal, gen_pt  = ak.to_numpy(flat_gen_pt))
        heta_ecal_vs_eta.fill(eta_at_ecal = flat_gen_eta_ecal, gen_eta = ak.to_numpy(flat_gen_eta))
        hfail_pt_vs_eta.fill (pt = flat_fail_gen_pt, eta = ak.to_numpy(flat_fail_gen_eta))
        
        ## plot GEN lxy to check 
        gen_vis_taus = leading_events.GenVisTau[
            (abs(leading_events.GenVisTau.parent.pdgId) == 15) &
            (abs(leading_events.GenVisTau.parent.distinctParent.pdgId) == 1000015) &
            (leading_events.GenVisTau.parent.distinctParent.hasFlags("isLastCopy")) &
            (leading_events.GenVisTau.parent.hasFlags("fromHardProcess")) &
#             (leading_events.GenVisTau.status == self.decayM) &
            (leading_events.GenVisTau.pt > self.min_gen_tau_pt) &
            (abs(leading_events.GenVisTau.eta) < max_eta)
        ]
        tau_vx = gen_vis_taus.parent.distinctParent.vx - ak.firsts(gen_vis_taus.parent.distinctChildren.vx, axis =2)
        tau_vy = gen_vis_taus.parent.distinctParent.vy - ak.firsts(gen_vis_taus.parent.distinctChildren.vy, axis =2)
        Lxy = np.sqrt(tau_vx**2 + tau_vy**2)
        gen_vis_taus = ak.with_field(gen_vis_taus, Lxy, "lxy")
        gen_vis_taus = gen_vis_taus[abs(gen_vis_taus.lxy) < maxLxy]
        flat_lxy = ak.flatten(gen_vis_taus.lxy)
        flat_gentau_eta = ak.flatten(gen_vis_taus.eta)
        hgen_lxy.fill(gen_lxy = ak.to_numpy(flat_lxy))
        
        heta_ecal_vs_lxy    .fill(eta_at_ecal = flat_gen_eta_ecal, gen_lxy = ak.to_numpy(flat_lxy))
        heta_ecal_vs_tau_eta.fill(eta_at_ecal = flat_gen_eta_ecal, gen_tau_eta = ak.to_numpy(flat_gentau_eta))
        hdxy_vs_lxy.fill(gen_dxy = ak.to_numpy(flat_gen_dxy), gen_lxy = ak.to_numpy(flat_lxy))

        dataset = events.metadata["dataset"]    
        return {
                dataset: {
                    "entries": ak.num(events, axis=0),
                    "hdelta": hdelta,
                    "hdeltaR": hdeltaR,
                    "hdeltaR_custom": hdeltaR_custom,
                    "hdeltaR_custom_zoomout": hdeltaR_custom_zoomout, 
                    "hreco_dxy": hreco_dxy,
                    "hreco_ecal_dxy": hreco_ecal_dxy,
                    "hgen_lxy": hgen_lxy,
                    "hgen_dxy": hgen_dxy,
                    "hall_dxy": hall_dxy,
                    "hpass_dxy": hpass_dxy,
                    "hfail_dxy": hfail_dxy,
                    "hreco_dxy_zoom": hreco_dxy_zoom,
                    "hreco_ecal_dxy_zoom": hreco_ecal_dxy_zoom,
                    "hgen_dxy_zoom": hgen_dxy_zoom,
                    "hreco_pt": hreco_pt,
                    "hreco_ecal_pt": hreco_ecal_pt,
                    "hgen_pt": hgen_pt,
                    "hdeltaR_vs_dxy": hdeltaR_vs_dxy,
                    "heta_ecal_vs_dxy": heta_ecal_vs_dxy,
                    "heta_ecal_vs_pt": heta_ecal_vs_pt,
                    "heta_ecal_vs_eta": heta_ecal_vs_eta,
                    "hdxy_vs_lxy": hdxy_vs_lxy,
                    "hfail_pt_vs_eta": hfail_pt_vs_eta,
                    "heta_ecal_vs_lxy": heta_ecal_vs_lxy,
                    "heta_ecal_vs_tau_eta": heta_ecal_vs_tau_eta,
                    }
                }


    def postprocess(self, accumulator):
        return accumulator





import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

if __name__ == "__main__":

    iterative_run = processor.Runner(
#         executor = processor.IterativeExecutor(compression=None),
        executor=processor.FuturesExecutor(compression=None, workers = 8),        
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
      "Stau_300_1mm": {
        "files": {
          "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/summary/Run3_Summer22_chs_AK4PFCands_v12/SMS-TStauStau_MStau-300_ctau-1mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_0.root" : "Events",
          "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/summary/Run3_Summer22_chs_AK4PFCands_v12/SMS-TStauStau_MStau-300_ctau-1mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_1.root" : "Events",
          "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/summary/Run3_Summer22_chs_AK4PFCands_v12/SMS-TStauStau_MStau-300_ctau-1mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_2.root" : "Events",
          "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/summary/Run3_Summer22_chs_AK4PFCands_v12/SMS-TStauStau_MStau-300_ctau-1mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_3.root" : "Events",
         }
      },
      "Stau_300_100mm": {
        "files": {
          "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/summary/Run3_Summer22_chs_AK4PFCands_v12/SMS-TStauStau_MStau-300_ctau-1000mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_0.root" : "Events",
         }
      },
      "Stau_300_1000mm": {
        "files": {
          "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/summary/Run3_Summer22_chs_AK4PFCands_v12/SMS-TStauStau_MStau-300_ctau-1000mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_0.root" : "Events",
          "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/summary/Run3_Summer22_chs_AK4PFCands_v12/SMS-TStauStau_MStau-300_ctau-1000mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_1.root" : "Events",
          "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/summary/Run3_Summer22_chs_AK4PFCands_v12/SMS-TStauStau_MStau-300_ctau-1000mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_2.root" : "Events",
          "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/summary/Run3_Summer22_chs_AK4PFCands_v12/SMS-TStauStau_MStau-300_ctau-1000mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_3.root" : "Events",
         }
      },
      "Stau_100_100mm": {
        "files": {
          "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/summary/Run3_Summer22_chs_AK4PFCands_v12/SMS-TStauStau_MStau-100_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_0.root" : "Events",
          "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/summary/Run3_Summer22_chs_AK4PFCands_v12/SMS-TStauStau_MStau-100_ctau-100mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_1.root" : "Events",
         }
      },
      "Stau_100_1000mm": {
        "files": {
          "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/summary/Run3_Summer22_chs_AK4PFCands_v12/SMS-TStauStau_MStau-100_ctau-1000mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_0.root" : "Events",
          "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/summary/Run3_Summer22_chs_AK4PFCands_v12/SMS-TStauStau_MStau-100_ctau-1000mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_1.root" : "Events",
          "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/summary/Run3_Summer22_chs_AK4PFCands_v12/SMS-TStauStau_MStau-100_ctau-1000mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_2.root" : "Events",
         }
      },
      "Stau_500_1000mm": {
        "files": {
          "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/summary/Run3_Summer22_chs_AK4PFCands_v12/SMS-TStauStau_MStau-500_ctau-1000mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_0.root" : "Events",
          "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/summary/Run3_Summer22_chs_AK4PFCands_v12/SMS-TStauStau_MStau-500_ctau-1000mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_1.root" : "Events",
          "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/summary/Run3_Summer22_chs_AK4PFCands_v12/SMS-TStauStau_MStau-500_ctau-1000mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_2.root" : "Events",
          "root://cmsxrootd.fnal.gov//store/group/lpcdisptau/displacedTaus/nanoprod/summary/Run3_Summer22_chs_AK4PFCands_v12/SMS-TStauStau_MStau-500_ctau-1000mm_mLSP-1_TuneCP5_13p6TeV_madgraphMLM-pythia8/nano_3.root" : "Events",
         }
      }
    }
    
    decayM = 0
    out = iterative_run(
        stau_fileset,
        treename="Events",
        processor_instance=TauProcessor(min_reco_pf_pt=20, min_gen_tau_pt=50, decayM=decayM),
    )
    
    import pdb
    from matplotlib.ticker import MultipleLocator
    from coffea.util import save
    
    plot_dir = 'eff_vs_dxy'

    for sample_name in  [
#                          'Stau_300_100mm',
#                          'Stau_300_1mm',
                         'Stau_300_100mm',
                         'Stau_300_1000mm',
                         'Stau_100_100mm',
                         'Stau_100_1000mm',
                         'Stau_500_1000mm'
                        ]:

        sample_label = sample_name #'m = 300 GeV, ctau = 1000 mm'
        output = {
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
            "hdeltaR_custom": out[sample_name]['hdeltaR_custom'],
            "hdeltaR": out[sample_name]['hdeltaR'],
            "hdeltaR_custom_zoomout": out[sample_name]['hdeltaR_custom_zoomout'],
            "heta_ecal_vs_dxy": out[sample_name]['heta_ecal_vs_dxy'],
            "heta_ecal_vs_pt": out[sample_name]['heta_ecal_vs_pt'],
            "heta_ecal_vs_eta": out[sample_name]['heta_ecal_vs_eta'],
            "heta_ecal_vs_lxy": out[sample_name]['heta_ecal_vs_lxy'],
            "heta_ecal_vs_tau_eta": out[sample_name]['heta_ecal_vs_tau_eta'],
            "hdxy_vs_lxy": out[sample_name]['hdxy_vs_lxy'],
            "hfail_pt_vs_eta": out[sample_name]['hfail_pt_vs_eta'],
        }
        save(output, f"histograms_{sample_name}_dR0p4_etaLess2p4_MaxLxy{maxLxy}_decayM{decayM}.coffea")
#         save(output, f"histograms_{sample_name}_dR0p4_etaLess2p4_NoMaxLxy.coffea")
