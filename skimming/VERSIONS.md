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
- Commit: <fill in>
- Known issue: the original `samples/.../mutau/v19/signal_preprocessed.pkl` was missing 1 Stau_300_1000mm skim file
  (preprocess.py runs with skipbadfiles=True and dropped it silently). Kept as `signal_preprocessed_v19orig_missing1file.pkl`;
  re-preprocessed on <date>, check shows no files missing.

### v20
- Same as v19, but the DisMuon collection is NOT filtered: every DisMuon is kept
  (the event still needs >=1 DisMuon pT>20 |eta|<2.4). Needed for Method C / cosmic partner legs.
- Skim code commit: 4262a66c. Run with: <hash>
- Signal: preprocess check shows no files missing. Backgrounds: <fill in as they are done>

## Selections — `selected/Summer22_CHS_v19/mutau/<skimversion>/<selection>[_<tag>]/`

### v19/SR, v19/SR_fullScore, v19/N1_muonDxy, v19/QCD_CR_thesis
- Leading-muon method: leading-pT mediumId DisMuon + highest disTauTag_score1 jetId jet, then cuts on those.
- One DisMuon per event. DisMuon_dphi is actually the jet-MET dphi (bug, fixed in Method C code).
- v19/SR Stau_300_1000mm is missing 198 events (the dropped skim file above): its yield is ~12% low.

### v19/SR_leadingRerun
- Same leading-muon code as v19/SR, rerun on the re-preprocessed (complete) v19 signal input.
- Code: `skimming/selection_cuts_leadingmuon.py` = selection_cuts.py at 7cd36d9d; only the output folder suffix differs.
  It uses the current selection_function.py, whose only change (cand = DisMuon[:, :1]) does nothing when there is one muon.
- Validation vs v19/SR: <fill in, expected identical except Stau_300_1000mm +198>
- Run with: <hash>

### v20/SR_methodC — BROKEN, deleted on <date>
- DisMuon was option-typed (from ak.firsts on the jet), so utils.is_rootcompat silently dropped every DisMuon branch.

### v20/SR_methodC2
- Method C: candidate = highest-pT DisMuon passing ALL muon cuts of the selection; all other DisMuons kept with
  no cuts, candidate at index 0. Event cuts (event_selection, manual_blinding) use the candidate only.
- Also: DisMuon_dphi fixed (muon-jet), dR flatten fixed; saves mediumId, tightId, charge, timing, outer-track/segment vars.
- Commits: 68eccb7c, 489280fb, 1654027f (CorrectedJet[:, 0] fix). Run with: <hash>
- Validation vs v19/SR_leadingRerun (signal): <fill in: ref_only = 0, same candidate in all shared events, gained = N events, all with a muon ahead>
