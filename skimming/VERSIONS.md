# Output versions

Never reuse a version or tag. Each entry says what changed vs the previous one, which commit made it, and where the output is.
Commit before every large run and record the hash here (`git log -1 --oneline`).

- EOS base (mine): `/store/user/lpcdisptau/bskipwor/displacedTaus/`
- Daniel's outputs (read-only reference): `/store/group/lpcdisptau/dally/` and `/store/user/dally/`
- Validation scripts: `plotting/compare_methodC.py` (event-by-event comparison + plots), `plotting/inspect_methodC.py` (one dataset in detail)

## Skims — `skim/Summer22_CHS_v19/mutau/<version>/`

### v19
- Input: Run3_Summer22_chs_AK4PFCands_v19 NanoAOD
- Event cuts: MET triggers, noise filters, >=1 DisMuon pT>20 |eta|<2.4, >=1 jet pT>20 |eta|<2.4
- The DisMuon collection itself is filtered to pT>20 |eta|<2.4, so lower-pT partner muons are dropped
- Commit: not recorded (made before this file existed)
- Known issue: skim chunk `Stau_300_1000mm/nano_3_0_18707.root` has 0 entries (the skim job failed while writing it;
  the skim never writes empty files on purpose). The same chunk in v20 has 10666 entries.
  Effect: v19/SR Stau_300_1000mm is missing 198 events (~12% of that point). All other v19 signal chunks are complete.

### v20
- Same as v19, but the DisMuon collection is NOT filtered: every DisMuon is kept
  (the event still needs >=1 DisMuon pT>20 |eta|<2.4). Needed for Method C / cosmic partner legs.
- Skim code commit: 4262a66c. Run with: 6e9eb889 (signal)
- Signal: preprocess check shows no files missing. Backgrounds: <fill in as they are done>

## Selections — `selected/Summer22_CHS_v19/mutau/<skimversion>/<selection>[_<tag>]/`

### v19/SR, v19/SR_fullScore, v19/N1_muonDxy, v19/QCD_CR_thesis
- Leading-muon method: leading-pT mediumId DisMuon + highest disTauTag_score1 jetId jet, then cuts on those.
- One DisMuon per event. DisMuon_dphi is actually the jet-MET dphi (bug, fixed in Method C code).
- v19/SR Stau_300_1000mm is missing 198 events (the empty v19 skim chunk above): its yield is ~12% low.

### v20/SR_methodC — BROKEN, deleted 2026-10-14
- DisMuon was option-typed (from ak.firsts on the jet), so utils.is_rootcompat silently dropped every DisMuon branch.

### v20/SR_methodC2
- Method C: candidate = highest-pT DisMuon passing ALL muon cuts of the selection; all other DisMuons kept with
  no cuts, candidate at index 0. Event cuts (event_selection, manual_blinding) use the candidate only.
- Also: DisMuon_dphi fixed (muon-jet), dR flatten fixed; saves mediumId, tightId, charge, timing, outer-track/segment vars.
- Commits: 68eccb7c, 489280fb, 1654027f (CorrectedJet[:, 0] fix). Run with: 1654027f (signal)
- Validation vs v19/SR (signal, `compare_methodC.py`, validation scripts committed in 8d69be1):
  every v19/SR event is in SR_methodC2 with the identical candidate muon (329327 of 329327; ref_only = 0 in all 32 points).
  Gained events: 368 in total.
  - 170 are real Method C gains (0.05%): every one has a higher-pT muon ahead of the candidate that fails the cuts.
  - 198 are in Stau_300_1000mm and are exactly the SR events of the empty v19 chunk nano_3_0_18707
    (203 Method C events come from that chunk, 0 of them in v19/SR; 198 have no muon ahead, 5 are real gains).
  Explained: 124 events (0.04%) have stored candidate pT 29.89-30.0 although pT > 30 passed. coffea's `.pt` is computed
  by scikit-hep vector from DisMuon px/py, which are stored at low precision (e.g. px=27.0, py=13.125 -> 30.021 vs
  stored pt 29.974). Affects every `.pt` cut on DisMuon since before Method C (v19 and Daniel's code too).
  Fix (cut on ["pt"] or drop px/py at load) to be decided; would be a new tagged selection.