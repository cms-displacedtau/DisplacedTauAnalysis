import awkward as ak
import numpy as np

SR = {
              "muon_pt_min":                30., ##GeV
              "muon_medium_ID_min":         0, ## > min and < max -> [0,2] to select mediumID, [-1,1] to select not-medium, [-999,999] to select any 
              "muon_medium_ID_max":         2, ## 1 is passing, 0 if not 
              "muon_tight_ID_min":         -999, ## > min and < max -> [0,2] to select tightID, [-1,1] to select not-tight, [-999,999] to select any 
              "muon_tight_ID_max":          999,  ## 
              "muon_dxy_min":               0.1, ##cm
              "muon_dxy_max":               10.,  ##cm
              "muon_iso_min":               -999,
              "muon_iso_max":               0.18,
              "muon_iso04_min":               -999,
              "muon_iso04_max":               999,
              "muon_mt_min":                -999,
              "muon_mt_max":                999,

              "jet_pt_min":                 32.,  ##GeV
              "jet_score_min":              0.9, 
              "jet_score_max":              999, 
              "jet_dxy_min":                0.1, ##cm   ## maybe, not clear if currently it's set or not
              "jet_dxy_max":                999, ##cm
              "jet_ID":                     True,
              "jet_btag_min":                   0,
              "jet_btag_max":                   999,

              "MET_pt":                     105., ##GeV
             }

## SR with the jet score cut removed, to plot the full disTauTag_score1 distribution
SR_fullScore = {j:k for j,k in SR.items()}
SR_fullScore["jet_score_min"] =  -999
SR_fullScore["jet_score_max"] =  999

## cosmic background estimate: SR cuts with the jet-score cut removed. The name has no "SR"/"PR" on purpose,
## so manual_blinding runs on data: untagged events with jet score > 0.8 are removed, cosmic-tagged events are kept
CosmicEst = {j:k for j,k in SR.items()}
CosmicEst["jet_score_min"] =  -999
CosmicEst["jet_score_max"] =  999

## N-1 selection for the muon dxy plot: every SR cut except the muon dxy cut
## (no "SR"/"PR" in the name on purpose, so selection_cuts.py still blinds data with it)
N1_muonDxy = {j:k for j,k in SR.items()}
N1_muonDxy["muon_dxy_min"] = -999
N1_muonDxy["muon_dxy_max"] = 999

## QCD CR as defined in Daniel's thesis (Table 7.3), plus the SR preselection (muon pT, jet pT, MET > 105)
QCD_CR_thesis = {j:k for j,k in SR.items()}
QCD_CR_thesis["muon_dxy_min"]  = -999
QCD_CR_thesis["muon_dxy_max"]  = 0.045
QCD_CR_thesis["muon_iso_min"]  = 0.1
QCD_CR_thesis["muon_iso_max"]  = 999
QCD_CR_thesis["jet_score_min"] = 0
QCD_CR_thesis["jet_score_max"] = 0.2
QCD_CR_thesis["jet_dxy_min"]   = -999
QCD_CR_thesis["jet_dxy_max"]   = 0.1

## defined but never used
loose_SR = {j:k for j,k in SR.items()}
loose_SR["muon_dxy_min"] =  50E-4 ##cm
loose_SR["muon_iso_max"] =  0.36
loose_SR["jet_score_min"] =  0.7

## defined but never used
loose_noIso_SR = {j:k for j,k in loose_SR.items()}
loose_noIso_SR["muon_iso_max"] =  999.

## validate w/ daniel
validation_daniel = {j:k for j,k in loose_noIso_SR.items()}
validation_daniel["muon_dxy_max"]  =  50E-4 ##cm
validation_daniel["muon_dxy_min"]  =  0 ##cm
validation_daniel["jet_score_min"] =  0 ##cm
validation_daniel["jet_score_max"] =  0.7 ##cm

## loose cuts to validate even with few events and in DY region
validation_prompt = {j:k for j,k in loose_noIso_SR.items()}
validation_prompt["jet_dxy_min"] =  -999
validation_prompt["muon_dxy_min"] = -999 ##cm
validation_prompt["muon_dxy_max"] = 999 ##cm
validation_prompt["MET_pt"] = 10 

## to be fixed, need Daniel
TT_CR = {j:k for j,k in SR.items()}
TT_CR["jet_score_min"] =  0
TT_CR["jet_score_max"] =  0.2
TT_CR["jet_dxy_min"] = 0
TT_CR["jet_dxy_max"] = 0.02
TT_CR["jet_btag_min"]    = 0.6915
TT_CR["muon_dxy_min"] = 0
TT_CR["muon_dxy_max"] = 50E-4
TT_CR["muon_iso_max"] = 0.1
TT_CR["muon_mt_min"] = 30
TT_CR["muon_mt_max"] = 999
TT_CR["MET_pt"] = 60

# TT without b-tagging
TTMinusB_CR = {j:k for j,k in SR.items()}
TTMinusB_CR["jet_score_min"] =  0
TTMinusB_CR["jet_score_max"] =  999
TTMinusB_CR["jet_dxy_min"] = 0
TTMinusB_CR["jet_dxy_max"] = 0.1
TTMinusB_CR["muon_dxy_min"] = 0
TTMinusB_CR["muon_dxy_max"] = 0.045
TTMinusB_CR["muon_iso_max"] = 999
###tight Id from muon pog https://muon-wiki.docs.cern.ch/guidelines/recommendations/?h=isolation#particle-flow-isolation
TTMinusB_CR["muon_iso04_max"] = 0.15 
TTMinusB_CR["muon_mt_min"] = 30
TTMinusB_CR["muon_mt_max"] = 999
TTMinusB_CR["MET_pt"] = 0

QCD_CR = {j:k for j,k in SR.items()}
QCD_CR["muon_dxy_min"] = 0
QCD_CR["muon_dxy_max"] = 50E-4  
QCD_CR["muon_iso_min"] =  0.1  ## invert iso cut
QCD_CR["muon_iso_max"] =  999.
QCD_CR["jet_score_min"] =  0   ## invert score cut
QCD_CR["jet_score_max"] =  0.2
QCD_CR["jet_dxy_min"] =  0     ## invert jet_dxy 
QCD_CR["jet_dxy_max"] = 0.02
QCD_CR["MET_pt"] = 0

## there was a tight_TT_CR where muon had tight ID
## to be fixed, need Daniel
W_CR = {j:k for j,k in SR.items()}
W_CR["jet_score_min"] =  0
W_CR["jet_score_max"] =  0.2
W_CR["jet_dxy_min"] = 0
W_CR["jet_dxy_max"] = 0.02
W_CR["jet_btag_max"] = 0.0499
W_CR["muon_dxy_min"] = 0
W_CR["muon_dxy_max"] = 20E-4  
W_CR["muon_iso_max"] = 0.1
W_CR["muon_mt_min"] = 30
W_CR["muon_mt_max"] = 999
W_CR["MET_pt"] = 0


### Plotting region for preselection cuts plus medium ID muon
PR = {
              "muon_pt_min":                30., ##GeV
              "muon_medium_ID_min":         0, ## > min and < max -> [0,2] to select mediumID, [-1,1] to select not-medium, [-999,999] to select any 
              "muon_medium_ID_max":         2, ## 1 is passing, 0 if not 
              "muon_tight_ID_min":         -999, ## > min and < max -> [0,2] to select tightID, [-1,1] to select not-tight, [-999,999] to select any 
              "muon_tight_ID_max":          999,  ## 
              "muon_dxy_min":               0, ##cm
              "muon_dxy_max":               999.,  ##cm
              "muon_iso_min":               -999,
              "muon_iso_max":               999,
              "muon_iso04_min":               -999,
              "muon_iso04_max":               999,
              "muon_mt_min":                -999,
              "muon_mt_max":                999,

              "jet_pt_min":                 32.,  ##GeV
              "jet_score_min":              0, 
              "jet_score_max":              999, 
              "jet_dxy_min":                0, ##cm   ## maybe, not clear if currently it's set or not
              "jet_dxy_max":                999, ##cm
              "jet_ID":                     True,
              "jet_btag_min":                   0,
              "jet_btag_max":                   999,

              "MET_pt":                     0, ##GeV
             }

PR_Mu = {
              "muon_pt_min":                30., ##GeV
              "muon_medium_ID_min":         0, ## > min and < max -> [0,2] to select mediumID, [-1,1] to select not-medium, [-999,999] to select any 
              "muon_medium_ID_max":         2, ## 1 is passing, 0 if not 
              "muon_tight_ID_min":         -999, ## > min and < max -> [0,2] to select tightID, [-1,1] to select not-tight, [-999,999] to select any 
              "muon_tight_ID_max":          999,  ## 
              "muon_dxy_min":               0, ##cm
              "muon_dxy_max":               999.,  ##cm
              "muon_iso_min":               -999,
              "muon_iso_max":               999,
              "muon_mt_min":                -999,
              "muon_mt_max":                999,

              "jet_pt_min":                 32.,  ##GeV
              "jet_score_min":              0, 
              "jet_score_max":              999, 
              "jet_dxy_min":                0, ##cm   ## maybe, not clear if currently it's set or not
              "jet_dxy_max":                999, ##cm
              "jet_ID":                     True,
              "jet_btag_min":                   0,
              "jet_btag_max":                   999,

              "MET_pt":                     0, ##GeV
             }


HPSTauMu = {
              "muon_pt_min":                26, ##GeV
              "muon_eta_max":               2.4, 
              "muon_medium_ID_min":         0., ## > min and < max -> [0,2] to select mediumID, [-1,1] to select not-medium, [-999,999] to select any 
              "muon_medium_ID_max":         2, ## 1 is passing, 0 if not 
              "muon_tight_ID_min":          -999, ## > min and < max -> [0,2] to select tightID, [-1,1] to select not-tight, [-999,999] to select any 
              "muon_tight_ID_max":          999,  ## 
              "muon_dxy_min":               -999, ##cm
              "muon_dxy_max":               0.045,  ##cm
              "muon_dz_min":                -999,
              "muon_dz_max":                0.2,
              "muon_isoid_min":             3, 
              "muon_iso_max":               0.15, 

              "tau_pt_min":                 20.,  ##GeV
              "tau_eta_max":                2.5, 
              "tau_dz_min":                -999,
              "tau_dz_max":                0.2,
              "tau_dm":                     0, ## > 0 -> pass new DM 
              "tau_vs_e_wp":                2, ## VVloose WP
              "tau_vs_jet_wp":              5, ## medium WP
              "tau_vs_mu_wp":               4, ## tight WP

             }

selections_dict = {
  'SR' : SR,
  'SR_fullScore' : SR_fullScore,
  'CosmicEst' : CosmicEst,
  'N1_muonDxy' : N1_muonDxy,
  'loose_SR_selections' : loose_SR,
  'loose_noIso_SR_selections' : loose_noIso_SR,
  'validation_prompt' : validation_prompt,
  'validation_daniel' : validation_daniel,
  'TT_CR' : TT_CR,
  'TTMinusB_CR': TTMinusB_CR,
  'QCD_CR' : QCD_CR,
  'W_CR' : W_CR,
  'PR' : PR,
  'PR_Mu' : PR_Mu,
  'HPSTauMu' : HPSTauMu,
  'QCD_CR_thesis' : QCD_CR_thesis,
}

def event_selection(events, selection):

    selections = selections_dict[selection]
    #print(selection)
    #print ('in function selections: \n', selections)
    # Method C: DisMuon can hold several muons, with the candidate at index 0, so cut on the candidate only.
    # Every event must have a candidate (selection_cuts.py drops events without one), otherwise
    # ak.flatten below would return fewer entries than there are events.
    cand = events.DisMuon[:, :1]
    good_muons  =  ak.flatten(
                     (cand.pt > selections["muon_pt_min"])  &\
                     (abs(cand.dxy) > selections["muon_dxy_min"]) &\
                     (abs(cand.dxy) < selections["muon_dxy_max"]) &\
                     (cand.pfRelIso03_all > selections["muon_iso_min"]) &\
                     (cand.pfRelIso03_all < selections["muon_iso_max"]) &\
                     (cand.pfRelIso04_all > selections["muon_iso04_min"]) &\
                     (cand.pfRelIso04_all < selections["muon_iso04_max"]) &\
                     (cand.mediumId > selections["muon_medium_ID_min"]) &\
                     (cand.mediumId < selections["muon_medium_ID_max"]) &\
                     (cand.tightId > selections["muon_tight_ID_min"]) &\
                     (cand.tightId < selections["muon_tight_ID_max"]) &\
                     (cand.mT > selections["muon_mt_min"]) &\
                     (cand.mT < selections["muon_mt_max"])
                   ) 
    
    good_jets   = ak.flatten(
                    (events.CorrectedJet.pt > selections["jet_pt_min"]) &\
                    (events.CorrectedJet.disTauTag_score1 > selections["jet_score_min"]) &\
                    (events.CorrectedJet.disTauTag_score1 < selections["jet_score_max"]) &\
                    (abs(events.CorrectedJet.dxy) > selections["jet_dxy_min"]) &\
                    (abs(events.CorrectedJet.dxy) < selections["jet_dxy_max"]) &\
                    (events.CorrectedJet.jetId == selections["jet_ID"]) &\
                    (events.CorrectedJet.btagPNetB > selections["jet_btag_min"])&\
                    (events.CorrectedJet.btagPNetB < selections["jet_btag_max"])
                  )
    
    good_MET = (events.CorrectedPFMET.pt > selections["MET_pt"])


    events = events[good_muons & good_jets & good_MET]

    return events

def prompt_muon_event_selection(events, selection):

    selections = selections_dict[selection]
    #print(selection)
    #print ('in function selections: \n', selections)
    good_muons  =  ak.flatten(
                     (events.Muon.pt > selections["muon_pt_min"])  &\
                     (abs(events.Muon.dxy) > selections["muon_dxy_min"]) &\
                     (abs(events.Muon.dxy) < selections["muon_dxy_max"]) &\
                     (events.Muon.pfRelIso03_all > selections["muon_iso_min"]) &\
                     (events.Muon.pfRelIso03_all < selections["muon_iso_max"]) &\
                     (events.Muon.pfRelIso04_all > selections["muon_iso04_min"]) &\
                     (events.Muon.pfRelIso04_all < selections["muon_iso04_max"]) &\
                     (events.Muon.mediumId > selections["muon_medium_ID_min"]) &\
                     (events.Muon.mediumId < selections["muon_medium_ID_max"]) &\
                     (events.Muon.tightId > selections["muon_tight_ID_min"]) &\
                     (events.Muon.tightId < selections["muon_tight_ID_max"]) &\
                     (events.Muon.mT > selections["muon_mt_min"]) &\
                     (events.Muon.mT < selections["muon_mt_max"])
                   ) 
    
    good_jets   = ak.flatten(
                    (events.CorrectedJet.pt > selections["jet_pt_min"]) &\
                    (events.CorrectedJet.disTauTag_score1 > selections["jet_score_min"]) &\
                    (events.CorrectedJet.disTauTag_score1 < selections["jet_score_max"]) &\
                    (abs(events.CorrectedJet.dxy) > selections["jet_dxy_min"]) &\
                    (abs(events.CorrectedJet.dxy) < selections["jet_dxy_max"]) &\
                    (events.CorrectedJet.jetId == selections["jet_ID"]) &\
                    (events.CorrectedJet.btagPNetB > selections["jet_btag_min"])&\
                    (events.CorrectedJet.btagPNetB < selections["jet_btag_max"])
                  )
    
    good_MET = (events.CorrectedPFMET.pt > selections["MET_pt"])


    events = events[good_muons & good_jets & good_MET]

    return events

## blinding box: the SR muon and jet-|dxy| cuts, but jet score > 0.8 instead of the SR's 0.9 (safety margin)
BLIND_JET_SCORE_MIN = 0.8

def manual_blinding(events):

    selections = selections_dict['SR']
    #print(selection)
    #print ('in function selections: \n', selections)
    # Method C: blind on the candidate muon (index 0) only, not on the partner muons
    cand = events.DisMuon[:, :1]
    good_muons  =  ak.flatten(
                     (abs(cand.dxy) > selections["muon_dxy_min"]) &\
                     (cand.pfRelIso03_all > selections["muon_iso_min"]) &\
                     (cand.pfRelIso03_all < selections["muon_iso_max"]) 
                   ) 
    
    good_jets   = ak.flatten(
                    (events.CorrectedJet.disTauTag_score1 > BLIND_JET_SCORE_MIN) &\
                    (events.CorrectedJet.disTauTag_score1 < selections["jet_score_max"]) &\
                    (abs(events.CorrectedJet.dxy) > selections["jet_dxy_min"]) &\
                    (abs(events.CorrectedJet.dxy) < selections["jet_dxy_max"]) 
                  )
    
    #good_MET = (events.CorrectedPFMET.pt > selections["MET_pt"])


    # cosmic-tagged events are removed by the cosmic veto, so they are not in the signal region and stay visible
    if "cosmic_tag" in events.fields:
        tagged = ak.to_numpy(events.cosmic_tag)
    else:
        tagged = np.zeros(len(events), dtype=bool)

    events = events[~(good_muons & good_jets) | tagged]

    return events

def event_selection_hpstau_mu(events, selection):

    selections = selections_dict[selection]

    good_muons  =  ak.flatten(
                     (events.Muon.pt > selections["muon_pt_min"])  &\
                     (abs(events.Muon.eta) < selections["muon_eta_max"]) &\
                     (abs(events.Muon.dxy) > selections["muon_dxy_min"]) &\
                     (abs(events.Muon.dxy) < selections["muon_dxy_max"]) &\
                     (abs(events.Muon.dz)  > selections["muon_dz_min"]) & \
                     (abs(events.Muon.dz)  < selections["muon_dz_max"]) & \
                     #(events.Muon.pfIsoId > selections["muon_isoid_min"]) &\
                     (events.Muon.pfRelIso04_all < selections["muon_iso_max"]) &\
                     (events.Muon.mediumId > selections["muon_medium_ID_min"]) &\
                     (events.Muon.mediumId < selections["muon_medium_ID_max"]) &\
                     (events.Muon.tightId > selections["muon_tight_ID_min"]) &\
                     (events.Muon.tightId < selections["muon_tight_ID_max"])
                   ) 

    
    good_taus   = ak.flatten(
                    (events.Tau.pt > selections["tau_pt_min"]) &\
                    (abs(events.Tau.eta) < selections["tau_eta_max"]) &\
                    (abs(events.Tau.dz)  > selections["tau_dz_min"]) & \
                    (abs(events.Tau.dz)  < selections["tau_dz_max"]) & \
                    (events.Tau.idDeepTau2018v2p5VSe >= selections["tau_vs_e_wp"]) &\
                    (events.Tau.idDeepTau2018v2p5VSjet >= selections["tau_vs_jet_wp"]) &\
                    (events.Tau.idDeepTau2018v2p5VSmu >= selections["tau_vs_mu_wp"]) 
                 )   
    
    events = events[good_muons & good_taus ]
    return events

#     if region == "keep_all":
#         events = events
    

def Zpeak_selection(events, selections): 
    good_muons  = (events.DisMuon.pt > selections["muon_pt"])           &\
                   (events.DisMuon.tightId == 1)                                    &\
                   (abs(events.DisMuon.dxy) > selections["muon_dxy_prompt_min"]) &\
                   (abs(events.DisMuon.dxy) < selections["muon_dxy_prompt_max"]) &\
                   (events.DisMuon.pfRelIso03_all < selections["muon_iso_max"])
                  

    good_jets   = (events.CorrectedJet.disTauTag_score1 > selections["jet_score"])   &\
                   (events.CorrectedJet.pt > selections["jet_pt"])                               &\
                   (abs(events.CorrectedJet.dxy) > selections["jet_dxy_displaced_min"])            #&\
                   #(abs(events.Jet.dxy) < selections["muon_dxy_prompt_max"])
                  
    events['DisMuon'] = ak.drop_none(events.DisMuon[good_muons])
    logger.info("Applied mask to DisMuon")
    num_good_muons = ak.count_nonzero(good_muons, axis=1)
    num_muon_mask = (num_good_muons > 0)

    events['Jet'] = ak.drop_none(events.Jet[good_jets])
    logger.info("Applied mask to Jet")
    num_good_jets = ak.count_nonzero(good_jets, axis=1)
    num_jet_mask = (num_good_jets > 0)


    events = events[num_muon_mask & num_jet_mask & good_events]

    return events


## ---------------------------------------------------------------------------------------------
## Cosmic tagging: flags only, nothing is removed. Port of the veto in
## brandiskip/DisplacedTauAnalysis@sara_samples scripts/cosmics_final.py (commit 28b7914).
## Directions are built from the stored pt/eta/phi fields, not coffea's .pt/.phi (those come from
## the low-precision px/py). Events with fewer than 2 DisMuons can never be tagged.
##   cosmic_s1: a pair that looks like a duplicate (|deta| < 0.01, |dphi| < 0.001) or is collinear
##              (inner-track cosA > 0.90) has outer-track cosA < -0.8, otherwise segment cosA < -0.8
##              (needs the outertrack/segment branches; cosmic_s1_available says if they exist)
##   cosmic_s2: cosA(candidate, any other DisMuon) < -0.99
##   cosmic_s3: highest-pT DisMuon with phi > 0 and highest-pT DisMuon with phi < 0, both
##              timeNDof > 7, and t(upper) - t(lower) < -20 ns
##   cosmic_tag = s1 | s2 | s3
## ---------------------------------------------------------------------------------------------
COSMIC_DUP_DETA = 0.01
COSMIC_DUP_DPHI = 0.001
COSMIC_COLL_COSA = 0.90
COSMIC_LEGS_COSA = -0.8
COSMIC_B2B_COSA = -0.99
COSMIC_DT = -20.
COSMIC_TIME_NDOF = 7


def _cos_angle(eta1, phi1, eta2, phi2):
    """cos of the 3D opening angle between two directions given by eta, phi"""
    return (np.cos(phi1 - phi2) + np.sinh(eta1) * np.sinh(eta2)) / (np.cosh(eta1) * np.cosh(eta2))


def _delta_phi(phi1, phi2):
    return (phi1 - phi2 + np.pi) % (2 * np.pi) - np.pi


def cosmic_tagging(events):
    mu = events.DisMuon            # candidate first (Method C), all other DisMuons kept
    eta, phi = mu["eta"], mu["phi"]

    ## stage 1: every pair of DisMuons
    ia, ib = ak.unzip(ak.argcombinations(eta, 2, axis=1))
    eta_a, eta_b, phi_a, phi_b = eta[ia], eta[ib], phi[ia], phi[ib]
    dup = (abs(eta_a - eta_b) < COSMIC_DUP_DETA) & (abs(_delta_phi(phi_a, phi_b)) < COSMIC_DUP_DPHI)
    coll = (_cos_angle(eta_a, phi_a, eta_b, phi_b) > COSMIC_COLL_COSA) & ~dup
    flag = dup | coll

    s1_available = all(f in mu.fields for f in ("outertrack_pt", "outertrack_eta", "outertrack_phi",
                                                "nSeg", "sumSegX", "sumSegY", "sumSegZ"))
    if s1_available:
        ot_pt, ot_eta, ot_phi = mu["outertrack_pt"], mu["outertrack_eta"], mu["outertrack_phi"]
        a_has, b_has = ot_pt[ia] > 0, ot_pt[ib] > 0
        both, only_a, only_b = a_has & b_has, a_has & ~b_has, ~a_has & b_has
        a_flip = (phi_a > 0) != (ot_phi[ia] > 0)
        b_flip = (phi_b > 0) != (ot_phi[ib] > 0)
        elig = both | (only_a & a_flip) | (only_b & b_flip)
        cos_out = ak.where(both, _cos_angle(ot_eta[ia], ot_phi[ia], ot_eta[ib], ot_phi[ib]),
                  ak.where(only_a & a_flip, _cos_angle(ot_eta[ia], ot_phi[ia], eta_b, phi_b),
                  ak.where(only_b & b_flip, _cos_angle(eta_a, phi_a, ot_eta[ib], ot_phi[ib]), np.nan)))

        sx, sy, sz, nseg = mu["sumSegX"], mu["sumSegY"], mu["sumSegZ"], mu["nSeg"]
        seg_dot = sx[ia] * sx[ib] + sy[ia] * sy[ib] + sz[ia] * sz[ib]
        norm_a = np.sqrt(sx[ia]**2 + sy[ia]**2 + sz[ia]**2)
        norm_b = np.sqrt(sx[ib]**2 + sy[ib]**2 + sz[ib]**2)
        seg_ok = (nseg[ia] > 0) & (nseg[ib] > 0) & (norm_a > 0) & (norm_b > 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            cos_seg = ak.where(seg_ok, seg_dot / (norm_a * norm_b), np.nan)

        rem_out = flag & elig & (cos_out < COSMIC_LEGS_COSA)
        rem_seg = flag & ~rem_out & (cos_seg < COSMIC_LEGS_COSA)
        s1 = ak.to_numpy(ak.any(rem_out | rem_seg, axis=1))
    else:
        s1 = np.zeros(len(events), dtype=bool)

    ## stage 2: candidate vs every other DisMuon
    c_eta = ak.fill_none(ak.firsts(eta), 0.)
    c_phi = ak.fill_none(ak.firsts(phi), 0.)
    cos_cand = _cos_angle(c_eta, c_phi, eta[:, 1:], phi[:, 1:])
    s2 = ak.to_numpy(ak.any(cos_cand < COSMIC_B2B_COSA, axis=1))

    ## stage 3: timing of the highest-pT upper (phi > 0) and lower (phi < 0) DisMuon
    up, lo = mu[phi > 0], mu[phi < 0]
    up_lead = ak.firsts(up[ak.argmax(up["pt"], axis=1, keepdims=True)])
    lo_lead = ak.firsts(lo[ak.argmax(lo["pt"], axis=1, keepdims=True)])
    s3 = ((up_lead["timeNDof"] > COSMIC_TIME_NDOF) & (lo_lead["timeNDof"] > COSMIC_TIME_NDOF)
          & ((up_lead["timeAtIpInOut"] - lo_lead["timeAtIpInOut"]) < COSMIC_DT))
    s3 = ak.to_numpy(ak.fill_none(s3, False))   # None = no upper/lower pair

    events = ak.with_field(events, s1, "cosmic_s1")
    events = ak.with_field(events, s2, "cosmic_s2")
    events = ak.with_field(events, s3, "cosmic_s3")
    events = ak.with_field(events, s1 | s2 | s3, "cosmic_tag")
    events = ak.with_field(events, np.full(len(events), s1_available), "cosmic_s1_available")
    return events
