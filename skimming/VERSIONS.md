# Output versions

Never reuse a version or tag. Each entry: what changed vs the previous one, the commit it was made with, where the output is.
EOS base: /store/user/lpcdisptau/bskipwor/displacedTaus/
Daniel's outputs (read-only reference): /store/group/lpcdisptau/dally/ and /store/user/dally/

## Skims  (skim/Summer22_CHS_v19/mutau/<version>/)

### v19
- Input: Run3_Summer22_chs_AK4PFCands_v19 NanoAOD
- Cuts: MET triggers, noise filters, >=1 DisMuon pT>20 |eta|<2.4, >=1 jet pT>20 |eta|<2.4
- DisMuon collection itself filtered to pT>20 |eta|<2.4 (partner muons below that are dropped)
- Commit: <fill in>

### v20
- Same as v19, but the DisMuon collection is NOT filtered: every DisMuon is kept
  (event still needs >=1 DisMuon pT>20 |eta|<2.4). Needed for Method C / cosmic partner legs.
- Commit: 4262a66c (skim change), <hash you run with>

## Selections  (selected/Summer22_CHS_v19/mutau/<skimversion>/<selection>[_<tag>]/)

### v19/SR, v19/SR_fullScore, v19/N1_muonDxy, v19/QCD_CR_thesis
- Leading-muon method: leading-pT mediumId DisMuon + highest disTauTag_score1 jetId jet, then cuts.
- One DisMuon per event in the ntuple. Note: DisMuon_dphi here is actually the jet-MET dphi (bug).
- Selection values: selection_function.py at the commit used.

### v20/SR_methodC
- Method C: candidate = highest-pT DisMuon passing ALL muon cuts of the selection; all other
  DisMuons kept with no cuts, candidate at index 0. Cuts applied to the candidate only.
- DisMuon_dphi fixed (muon-jet), dR flatten fixed. Saves mediumId, tightId, charge, timing, outer-track/segment vars.
- Commits: 68eccb7c, 489280fb, <hash you run with>