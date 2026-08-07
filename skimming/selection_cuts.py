import awkward as ak
import uproot, os, sys
import numpy as np
import gzip, correctionlib, importlib, pickle
#sys.stdout.reconfigure(line_buffering=True)
#sys.stderr.reconfigure(line_buffering=True)
#os.environ["PYTHONUNBUFFERED"] = "1"

from coffea import processor
from coffea.nanoevents import PFNanoAODSchema
from coffea.lumi_tools import LumiData, LumiList, LumiMask

from coffea.jetmet_tools import FactorizedJetCorrector, JetCorrectionUncertainty
from coffea.jetmet_tools import JECStack, CorrectedJetsFactory, CorrectedMETFactory
from coffea.lookup_tools import extractor
from coffea.analysis_tools import PackedSelection

import fsspec_xrootd
from  fsspec_xrootd import XRootDFileSystem

import dask
from dask import config as cfg
cfg.set({'distributed.scheduler.worker-ttl': None}) # Check if this solves some dask issues
#cfg.set({'distributed.scheduler.allowed-failures': 30}) # Check if this solves some dask issues
cfg.set({"distributed.logging.distributed": "debug"})
from dask.distributed import Client, LocalCluster, wait, progress, performance_report
#from dask_lxplus import CernCluster
from lpcjobqueue import LPCCondorCluster, schedd 
from dask_jobqueue import HTCondorCluster
import socket, time
import dask_awkward as dak
import warnings
warnings.filterwarnings("ignore", module="coffea") # Suppress annoying deprecation warnings for coffea vector, c.f. https://github.com/CoffeaTeam/coffea/blob/master/src/coffea/nanoevents/methods/candidate.py
import logging

from selection_function import event_selection, event_selection_hpstau_mu, manual_blinding, prompt_muon_event_selection
from selection_function import selections_dict 
from utils import process_n_files, is_rootcompat, uproot_writeable_selected
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../input_jsons")))
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("-m"    , "--muon"    , dest = "leading_muon_type"   , help = "Leading muon variable"    , default = "pt")
parser.add_argument("-j"    , "--jet"     , dest = "leading_jet_type"    , help = "Leading jet variable"     , default = "disTauTag_score1")
parser.add_argument(
	"--sample",
	choices=['QCD', 'signal', 'WtoLNu', 'Wto2Q', 'TT', 'singleT', 'JetMET_2022', 'Muon', 'DYEMu', 'DYTau', 'VBF', 'EWK'],
	required=True,
	help='Specify the sample you want to process')
parser.add_argument(
	"--subsample",
	nargs='*',
	default='all',
	required=False,
	help='Specify the exact sample you want to process')
parser.add_argument(
	"--nfiles",
	default='-1',
	required=False,
	help='Specify the number of input files to process')
parser.add_argument(
	"--usePkl",
	default=True,
	required=False,
	help='Turn it to false to use the non-preprocessed samples')
parser.add_argument(
	"--selection",
	default='',
	required=False,
	help='Specify which CR to use')
parser.add_argument(
	"--skim",
	default='prompt_mutau',
	required=False,
	choices=['prompt_mutau','mutau', 'pmutau'],
	help='Specify input skim, which objects, and selections (Muon and HPSTau, or DisMuon and Jet)')
parser.add_argument(
	"--skimversion",
	default='v0',
	required=False,
	help='If listing skimmed files, select which version of the inputs')
parser.add_argument(
	"--nanov",
	choices=['Summer22_CHS_v10', 'Summer22_CHS_v7', 'Summer22_CHS_v19'],
	default='Summer22_CHS_v19',
	required=False,
	help='Specify the custom nanoaod version to process')
parser.add_argument(
	"--testjob",
        action="store_true",
        help="Run a test job locally")
args = parser.parse_args()


## define the folder where the input .pkl files are defined, as well as the output folder on eos for the final events
skim_folder = args.skim
selection_string = args.selection
## will change once "region" will become a flag
if skim_folder == 'prompt_mutau':
    mode_string = 'hpstau_mu' 
    selection_string = 'HPSTauMu'
elif skim_folder == 'mutau':
    mode_string = 'jet_dmu' 
    if selection_string == '':
        selection_string = 'validation_daniel'  ## FIXME
elif skim_folder == 'pmutau':
    mode_string = 'jet_pmu'
else:
    print ('make sure using the correct folder/selections')
    exit(0)
    

out_folder = f'root://cmseos.fnal.gov//store/user/lpcdisptau/bskipwor/displacedTaus/selected/{args.nanov}/{skim_folder}/{args.skimversion}/{selection_string}/'

#out_folder = f'root://cmseos.fnal.gov//store/group/lpcdisptau/bskipwor/displacedTaus/selected/{args.nanov}/{skim_folder}/{args.skimversion}/{selection_string}_JetDxy0p1cm/'


## define input samples
all_fileset = {}
if args.usePkl==True:
    ## to be made configurable
    with open(f"samples/{args.nanov}/{skim_folder}/{args.skimversion}/{args.sample}_preprocessed.pkl", "rb") as  f:
        input_dataset = pickle.load(f)
        #print(input_dataset.keys())
else:
    samples = {
        "Wto2Q": f"samples.{args.nanov}.{skim_folder}.{args.skimversion}.fileset_Wto2Q",
        "WtoLNu": f"samples.{args.nanov}.{skim_folder}.{args.skimversion}.fileset_WtoLNu",
        "QCD": f"samples.{args.nanov}.{skim_folder}.{args.skimversion}.fileset_QCD",
        "DY": f"samples.{args.nanov}.{skim_folder}.{args.skimversion}.fileset_DY",
        "DYEMu": f"samples.{args.nanov}.{skim_folder}.{args.skimversion}.fileset_DYEMu",
        "DYTau": f"samples.{args.nanov}.{skim_folder}.{args.skimversion}.fileset_DYTau",
        "signal": f"samples.{args.nanov}.{skim_folder}.{args.skimversion}.fileset_signal",
        "TT": f"samples.{args.nanov}.{skim_folder}.{args.skimversion}.fileset_TT",
        "singleT": f"samples.{args.nanov}.{skim_folder}.{args.skimversion}.fileset_singleT",
    }
    module = importlib.import_module(samples[args.sample])
    input_dataset = module.fileset  #['Stau_100_0p1mm'] 
  

## restrict to specific sub-samples
if args.subsample == 'all':
    fileset = input_dataset
else:  
    with open(f"samples/{args.nanov}/{skim_folder}/{args.skimversion}/{args.subsample[0]}_preprocessed.pkl", "rb") as  f:
        fileset = pickle.load(f)

## restrict to n files
process_n_files(int(args.nfiles), fileset)
## add an else statement to prevent empty lists if nfiles > len(fileset)            
#print("Will process {} files from the following samples:".format(args.nfiles), fileset.keys())

## branches to be included in the output files.
## tuned on prompt skim case
## will save all fields if in include_all, but only combinations of (include_prefixes,include_postfixes)
include_prefixes = ['DisMuon',  'CorrectedJet', 'Muon', 'Tau']
include_postfixes = ['pt', 'eta', 'phi', 'dxy', 'dz', 'dxyErr', 'dzErr', 'jetId',
                     'pfRelIso03_all', 'disTauTag_score1', 'numJets', 'numLooseBJets', 'numMediumBJets', 'numTightBJets', 'pfRelIso04_all',
                     'dphi', 'deta', 'dR', 'mT', 'pt_cut', 'dxy_cut', 'iso_cut', 'iso04_cut', 'mt_cut', 'score_cut', 'mediumId_cut', 'id_cut',
                     'dxySig', 'dzSig', 'eventWeight', 'btagPNetB',
                    ]
include_all = ['PFMET',  'ChsMET', 'PuppiMET',
               'nTau', 'nPFMET', 'nChsMET','nPuppiMET', 'nPV',
               'nVtx', 'event', 'run', 'luminosityBlock', 'Pileup', 'weights', 'genWeight', 'weight', 'HLT',
               'nDisMuon', 'nMuon', 'nJet', 'mT', 'PV', 'mutau_mass', 'GenVisTau',
               'CorrectedPFMET', 'n_muons', 'n_jets', "disDimuon", "JetPFCands", "PFCands" 
              ]

### FIXME: need to add Lxy and IP at GEN level                             

class SelectionProcessor(processor.ProcessorABC):
    def __init__(self, leading_muon_var, leading_jet_var, mode="jet_dmu"):
        self.leading_muon_var = args.leading_muon_type
        self.leading_jet_var  = args.leading_jet_type
        ## sara: to be better understood
        assert mode in ["hpstau_mu", "jet_dmu", "jet_pmu"]
        self._mode = mode

        self._accumulator = {}
        #for samp in skimmed_fileset:
        #    self._accumulator[samp] = dak.from_awkward(ak.Array([]), npartitions = 1)

       # Load pileup weights evaluators 
        jsonpog = "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration"
        pileup_file = jsonpog + "/POG/LUM/2022_Summer22EE/puWeights.json.gz"

        with gzip.open(pileup_file, 'rt') as f:
            self.pileup_set = correctionlib.CorrectionSet.from_string(f.read().strip())

    def get_pileup_weights(self, events, also_syst=False):
        # Apply pileup weights
        evaluator = self.pileup_set["Collisions2022_359022_362760_eraEFG_GoldenJson"]
        sf = evaluator.evaluate(events.Pileup.nTrueInt, "nominal")
#         if also_syst:
#             sf_up = evaluator.evaluate(events.Pileup.nTrueInt, "up")
#             sf_down = evaluator.evaluate(events.Pileup.nTrueInt, "down")
#         return {'nominal': sf, 'up': sf_up, 'down': sf_down}
        return {'nominal': sf}


    def process_weight_corrs_and_systs(self, events, weights):
        pileup_weights = self.get_pileup_weights(events)
        # Compute nominal weight and systematic variations by multiplying relevant factors
        # For pileup, do not multiply nominal correction factor, as it is already included in the up/down variations
        # To see this, one can reproduce the ratio in
        # https://gitlab.cern.ch/cms-nanoAOD/jsonpog-integration/-/blob/master/misc/LUM/2018_UL/puWeights.png?ref_type=heads
        # from the plain correctionset
        weight_dict = {
            'weight': weights * pileup_weights['nominal'] #* muon_weights['muon_trigger_SF'],
#             'weight_pileup_up': weights * pileup_weights['up'] * muon_weights['muon_trigger_SF'],
#             'weight_pileup_down': weights * pileup_weights['down'] * muon_weights['muon_trigger_SF'],
#             'weight_muon_trigger_up': weights * pileup_weights['nominal'] * (muon_weights['muon_trigger_SF'] + muon_weights['muon_trigger_SF_syst']),
#             'weight_muon_trigger_down': weights * pileup_weights['nominal'] * (muon_weights['muon_trigger_SF'] - muon_weights['muon_trigger_SF_syst']),
        }

        return weight_dict


    def process(self, events):

        PFNanoAODSchema.mixins["CandidateMuon"] = "Muon"
        PFNanoAODSchema.mixins["CandidateElectron"] = "Electron"
        PFNanoAODSchema.mixins["DoubleMuon"] = "Muon"
        PFNanoAODSchema.mixins["DoubleElectron"] = "Electron"
        PFNanoAODSchema.mixins["DisMuon"] = "Muon"
        PFNanoAODSchema.mixins["CorrectedJet"] = "Jet"

        n_evts = len(events)  
        logger.info(f"starting process")
        if n_evts == 0: 
            logger.info(f"no input events")
            return {"entries_written": 0}
#         out = self._accumulator.identity() 

        logger.info(f"Start process for {events.metadata['dataset']}")
        dataset = events.metadata["dataset"]    

        leading_muon_var = self.leading_muon_var
        leading_jet_var = self.leading_jet_var

        # Determine if dataset is MC or Data
        is_MC = True if hasattr(events, "GenPart") else False

       # JEC/JERC
        if is_MC:
            ext = extractor()
            ext.add_weight_sets([
                "* * ./jec/Summer22EE_22Sep2023_V2_MC_L1FastJet_AK4PFPuppi.jec.txt",
                "* * ./jec/Summer22EE_22Sep2023_V2_MC_L2Relative_AK4PFPuppi.jec.txt",
                "* * ./jec/Summer22EE_22Sep2023_V2_MC_L3Absolute_AK4PFPuppi.jec.txt",
                "* * ./jec/Summer22EE_JRV1_MC_PtResolution_AK4PFPuppi.jer.txt",
                "* * ./jec/Summer22EE_JRV1_MC_SF_AK4PFPuppi.jer.txt",
            ])
            ext.finalize()

            jet_stack_names = [
                "Summer22EE_22Sep2023_V2_MC_L1FastJet_AK4PFPuppi",
                "Summer22EE_22Sep2023_V2_MC_L2Relative_AK4PFPuppi",
                "Summer22EE_22Sep2023_V2_MC_L3Absolute_AK4PFPuppi",
                "Summer22EE_JRV1_MC_PtResolution_AK4PFPuppi",
                "Summer22EE_JRV1_MC_SF_AK4PFPuppi"
            ]

            evaluator = ext.make_evaluator()
            jec_inputs = {name: evaluator[name] for name in jet_stack_names}
            jec_stack = JECStack(jec_inputs)

            name_map = jec_stack.blank_name_map
            name_map['JetPt'] = 'pt'
            name_map['JetMass'] = 'mass'
            name_map['JetEta'] = 'eta'
            name_map['JetA'] = 'area'

            jets = events['Jet']
            jets['pt_raw'] = (1 - jets['rawFactor']) * jets['pt']
            jets['mass_raw'] = (1 - jets['rawFactor']) * jets['mass']
            jets['pt_gen'] = ak.values_astype(ak.fill_none(jets.matched_gen.pt, 0), np.float32)
            jets['Rho'] = ak.broadcast_arrays(events['Rho']['fixedGridRhoFastjetAll'], jets['pt'])[0]

            name_map['ptGenJet'] = 'pt_gen'
            name_map['ptRaw'] = 'pt_raw'
            name_map['massRaw'] = 'mass_raw'
            name_map['Rho'] = 'Rho'
            
            jet_factory = CorrectedJetsFactory(name_map, jec_stack)
            corrected_jets = jet_factory.build(jets)

            events = ak.with_field(events, corrected_jets, "CorrectedJet")

            pf_met = events.PFMET
            pf_met['pt_raw'] = events.RawPFMET.pt
            pf_met['unclustEDeltaX'] = pf_met.ptUnclusteredUp * np.cos(pf_met.phiUnclusteredUp)
            pf_met['unclustEDeltaY'] = pf_met.ptUnclusteredUp * np.sin(pf_met.phiUnclusteredUp)

            met_name_map = {}
            met_name_map['METpt'] = 'pt'
            met_name_map['METphi'] = 'phi'
            met_name_map['JetPt'] = 'pt'
            met_name_map['JetPhi'] = 'phi'
            met_name_map['ptRaw'] = 'pt_raw'
            met_name_map['UnClusteredEnergyDeltaX'] = 'unclustEDeltaX'
            met_name_map['UnClusteredEnergyDeltaY'] = 'unclustEDeltaY'

            met_factory = CorrectedMETFactory(met_name_map)
            CorrectedPFMET = met_factory.build(pf_met, corrected_jets)
            events = ak.with_field(events, CorrectedPFMET, "CorrectedPFMET")

        else:
            #print("Are we going through the data correction loop?")
            ext = extractor()
            ext.add_weight_sets([
                "* * ./jec/Summer22EE_22Sep2023_V2_MC_L1FastJet_AK4PFPuppi.jec.txt",
                "* * ./jec/Summer22EE_22Sep2023_V2_MC_L2Relative_AK4PFPuppi.jec.txt",
                "* * ./jec/Summer22EE_22Sep2023_V2_MC_L2L3Residual_AK4PFPuppi.jec.txt",
                "* * ./jec/Summer22EE_22Sep2023_V2_MC_L3Absolute_AK4PFPuppi.jec.txt",
            ])
            ext.finalize()

            jet_stack_names = [
                "Summer22EE_22Sep2023_V2_MC_L1FastJet_AK4PFPuppi",
                "Summer22EE_22Sep2023_V2_MC_L2Relative_AK4PFPuppi",
                "Summer22EE_22Sep2023_V2_MC_L2L3Residual_AK4PFPuppi",
                "Summer22EE_22Sep2023_V2_MC_L3Absolute_AK4PFPuppi",
            ]

            evaluator = ext.make_evaluator()
            jec_inputs = {name: evaluator[name] for name in jet_stack_names}
            jec_stack = JECStack(jec_inputs)

            name_map = jec_stack.blank_name_map
            name_map['JetPt'] = 'pt'
            name_map['JetMass'] = 'mass'
            name_map['JetEta'] = 'eta'
            name_map['JetA'] = 'area'

            jets = events.Jet
            jets['pt_raw'] = (1 - jets['rawFactor']) * jets['pt']
            jets['mass_raw'] = (1 - jets['rawFactor']) * jets['mass']
            jets['Rho'] = ak.broadcast_arrays(events.Rho.fixedGridRhoFastjetAll, jets.pt)[0]    

            name_map['ptRaw'] = 'pt_raw'
            name_map['massRaw'] = 'mass_raw'
            name_map['Rho'] = 'Rho'

            jet_factory = CorrectedJetsFactory(name_map, jec_stack)
            corrected_jets = jet_factory.build(jets)

            events = ak.with_field(events, corrected_jets, "CorrectedJet")
            
            pf_met = events.PFMET
            pf_met['pt_raw'] = events.RawPFMET.pt
            pf_met['unclustEDeltaX'] = pf_met.ptUnclusteredUp * np.cos(pf_met.phiUnclusteredUp)
            pf_met['unclustEDeltaY'] = pf_met.ptUnclusteredUp * np.sin(pf_met.phiUnclusteredUp)

            met_name_map = {}
            met_name_map['METpt'] = 'pt'
            met_name_map['METphi'] = 'phi'
            met_name_map['JetPt'] = 'pt'
            met_name_map['JetPhi'] = 'phi'
            met_name_map['ptRaw'] = 'pt_raw'
            met_name_map['UnClusteredEnergyDeltaX'] = 'unclustEDeltaX'
            met_name_map['UnClusteredEnergyDeltaY'] = 'unclustEDeltaY'

            met_factory = CorrectedMETFactory(met_name_map)
            CorrectedPFMET = met_factory.build(pf_met, corrected_jets)
            events = ak.with_field(events, CorrectedPFMET, "CorrectedPFMET")

        jetVeto = correctionlib.CorrectionSet.from_file("jetvetomap_2022EFG.json")
        jetVetoCorr = jetVeto['Summer22EE_23Sep2023_RunEFG_V1']
        jets = events.CorrectedJet
        jets = jets[(jets.jetId == 1) & (jets.chEmEF + jets.neEmEF < 0.9)]
        flatEta = ak.flatten(jets.eta)
        flatPhi = ak.flatten(jets.phi)
        veto = jetVetoCorr.evaluate("jetvetomap", flatEta, flatPhi)
        veto = ak.unflatten(veto, ak.num(jets))
        veto_mask = ak.all(veto == 0, axis = -1)
        events = events[veto_mask]


        ## IMPORTANT
        ## do we need to add selections before choosing the leading obj?
        if self._mode == "hpstau_mu":
            muons = events.Muon
            muons = muons[ak.argsort(muons["pfRelIso04_all"], ascending=True, axis=1)]
            muon_iso_run_lengths = ak.run_lengths(muons.pfRelIso04_all)
            muon_iso_run_lengths_first = ak.firsts(muon_iso_run_lengths)
            muon_same_leadingIso_mask = ak.where(ak.local_index(muons.pfRelIso04_all) < muon_iso_run_lengths_first, True, False)
            muons = muons[muon_same_leadingIso_mask]
            muons = muons[ak.argsort(muons.pt, ascending = False, axis =1)]
            muons = ak.singletons(ak.firsts(muons))
            events["Muon"] = muons

            taus = events.Tau
            taus = events.Tau[ak.argsort(taus["rawIso"], ascending=True, axis = 1)]
            tau_iso_run_lengths = ak.run_lengths(taus.rawIso)
            tau_iso_run_lengths_first = ak.firsts(tau_iso_run_lengths)
            tau_same_leadingIso_mask = ak.where(ak.local_index(taus.rawIso) < tau_iso_run_lengths_first, True, False)
            taus = taus[tau_same_leadingIso_mask]
            taus = taus[ak.argsort(taus.pt, ascending = False, axis =1)]
            taus = ak.singletons(ak.firsts(taus))
            events["Tau"] = taus

            #extra_electron_veto = (
            #    (ak.flatten(events.CandidateElectron.metric_table(muons), axis = 2) <= 0.5)
            #    | (ak.flatten(events.CandidateElectron.metric_table(taus), axis = 2)  <= 0.5)
            #)
            #num_extra_electron = ak.count_nonzero(extra_electron_veto, axis = 1)
            #events = events[num_extra_electron == 0]
            #muons  = muons[num_extra_electron == 0]
            #taus   = taus[num_extra_electron == 0]
 
            #extra_muon_veto = (
            #    ((ak.flatten(events.CandidateMuon.metric_table(muons), axis = 2) > 0)
            #    & (ak.flatten(events.CandidateMuon.metric_table(muons), axis = 2) <= 0.5))
            #    | (ak.flatten(events.CandidateMuon.metric_table(taus), axis = 2)  <= 0.5)
            #)
            #num_extra_muon = ak.count_nonzero(extra_muon_veto, axis = 1)
            #events = events[num_extra_muon == 0]
            #muons = muons[num_extra_muon == 0]
            #taus = taus[num_extra_muon == 0]

            #### Dilepton Veto
            #mu_pair = ak.combinations(events.DoubleMuon, 2, axis = 1)
            #el_pair = ak.combinations(events.DoubleElectron, 2, axis = 1)

            #mu1,mu2 = ak.unzip(mu_pair)
            #el1,el2 = ak.unzip(el_pair)

            #presel_mask = lambda leps1, leps2: ((leps1.charge * leps2.charge < 0) & (leps1.delta_r(leps2) > 0.15))

            #dlveto_mu_mask = presel_mask(mu1,mu2)
            #dlveto_el_mask = presel_mask(el1,el2)
        
            #dl_mu_veto = ak.sum(dlveto_mu_mask, axis=1) == 0
            #dl_el_veto = ak.sum(dlveto_el_mask, axis=1) == 0

            #dl_veto    = dl_mu_veto & dl_el_veto

            #events = events[dl_veto]    
            #muons = muons[dl_veto]
            #taus = taus[dl_veto]
            
            ## add transverse mass and mu+tau mass vars
            met = events.PFMET.pt            
            met_phi =  events.PFMET.phi     
            dphi = abs(muons.phi - met_phi)
            dphi = np.where(dphi > np.pi, 2*np.pi - dphi, dphi)  # wrap to [-pi, pi]
            mT = np.sqrt(2 * muons.pt * met * (1 - np.cos(dphi)))      
            events = ak.with_field(events, mT, "mT")
            dR = muons.metric_table(taus)

            events = events[ak.ravel(dR > 0.5)]
            taus = taus[ak.ravel(dR > 0.5)]
            muons = muons[ak.ravel(dR > 0.5)]
 
            mutau_cand = taus + muons
            events = events[ak.ravel(mutau_cand.charge == 0)]
            muons = muons[ak.ravel(mutau_cand.charge == 0)]
            taus = taus[ak.ravel(mutau_cand.charge == 0)]

            mutau_cand = mutau_cand[ak.ravel(mutau_cand.charge == 0)]
            mutau_mass = mutau_cand.mass 
            events = ak.with_field(events, mutau_mass, "mutau_mass")

            events = events[ak.ravel(mutau_mass > 40)]

            ## apply selections
            events = event_selection_hpstau_mu(events, selection_string)
        elif self._mode == "jet_pmu":
            ### To check Data vs MC
            #events = events[events.HLT.MET120_IsoTrk50]
            #events = events[events.HLT.PFMETNoMu110_PFMHTNoMu110_IDTight_FilterHF]
            ###
            muons = events["Muon"]
            muons = muons[muons.mediumId == True]
            muons = muons[ak.argsort(muons[leading_muon_var], ascending=False, axis=1)]
            muons = ak.singletons(ak.firsts(muons))
            events["Muon"] = muons
            num_muon = ak.count_nonzero(events["Muon"][leading_muon_var], axis = 1)

            muon_event_weight = ak.ones_like(events.Muon.pt)
 
            muon_sf_cset = correctionlib.CorrectionSet.from_file("ScaleFactors_Muon_Z_ID_ISO_2022_EE_schemaV2.json")
            muon_sf_id = muon_sf_cset["NUM_MediumID_DEN_TrackerMuons"]
            flat_abseta = ak.flatten(abs(muons.eta))
            flat_pt     = ak.flatten(muons.pt)
            muon_sf_id_eval = muon_sf_id.evaluate(flat_abseta, flat_pt, "nominal")

            muon_sf_id_eval = ak.unflatten(muon_sf_id_eval, ak.num(muons))
        
            per_muon = muon_sf_id_eval
            muon_event_weight = ak.prod(per_muon, axis=1)
            events["Muon"] = muons
            num_muon = ak.count_nonzero(events["Muon"][leading_muon_var], axis = 1)
            
            #events["Muon"] = ak.with_field(events.Muon, muon_event_weight, "eventWeight")

            num_corrected_jets = ak.count_nonzero(events["CorrectedJet"][leading_jet_var], axis = 1)
            loose_bjets  = events["CorrectedJet"][events["CorrectedJet"]["btagPNetB"] > 0.0499]
            medium_bjets = events["CorrectedJet"][events["CorrectedJet"]["btagPNetB"] > 0.2605]
            tight_bjets  = events["CorrectedJet"][events["CorrectedJet"]["btagPNetB"] > 0.6915]
            num_loose_bjets = ak.count_nonzero(loose_bjets[leading_jet_var], axis = 1)
            num_medium_bjets = ak.count_nonzero(medium_bjets[leading_jet_var], axis = 1)
            num_tight_bjets = ak.count_nonzero(tight_bjets[leading_jet_var], axis = 1)

            events["CorrectedJet"] = ak.with_field(events.CorrectedJet, num_corrected_jets, "numJets")
            events["CorrectedJet"] = ak.with_field(events.CorrectedJet, num_loose_bjets, "numLooseBJets")
            events["CorrectedJet"] = ak.with_field(events.CorrectedJet, num_medium_bjets, "numMediumBJets")
            events["CorrectedJet"] = ak.with_field(events.CorrectedJet, num_tight_bjets, "numTightBJets")

            correctedjets = events["CorrectedJet"]
            correctedjets = correctedjets[correctedjets.jetId == True]
            correctedjets = correctedjets[ak.argsort(correctedjets[leading_jet_var], ascending=False, axis = 1)]
            correctedjets = ak.singletons(ak.firsts(correctedjets))
            events["CorrectedJet"] = correctedjets

            num_jets = ak.count_nonzero(events["CorrectedJet"][leading_jet_var], axis = 1)
            events = events[(num_muon > 0) & (num_jets > 0)]

            met = events.CorrectedPFMET.pt
            met_phi = events.CorrectedPFMET.phi
            dphi = abs(events.Muon.phi - met_phi)
            dphi = np.where(dphi > np.pi, 2*np.pi - dphi, dphi)
            mT = np.sqrt(2 * events.Muon.pt * met * (1 - np.cos(dphi)))
            events["Muon"] = ak.with_field(events.Muon, mT, "mT")

            dphi = abs(events.CorrectedJet.phi - events.CorrectedPFMET.phi)
            dphi = np.where(dphi > np.pi, 2*np.pi - dphi, dphi)
            events["CorrectedJet"] = ak.with_field(events.CorrectedJet, dphi, "dphi")

            dxySig = events.Muon.dxy/events.Muon.dxyErr
            dzSig  = events.Muon.dz/events.Muon.dzErr
            events["Muon"] = ak.with_field(events.Muon, dxySig, "dxySig")
            events["Muon"] = ak.with_field(events.Muon, dzSig, "dzSig")

            mu_dphi = abs(events.CorrectedJet.phi - events.Muon.phi)
            mu_dphi = np.where(dphi > np.pi, 2*np.pi - dphi, dphi)
            events["Muon"] = ak.with_field(events.Muon, mu_dphi, "dphi")
            deta = abs(events.CorrectedJet.eta - events.Muon.eta)
            events["Muon"] = ak.with_field(events.Muon, deta, "deta")
            dR = events.Muon.metric_table(events.CorrectedJet)
            dR = ak.flatten(dR)
            events["Muon"] = ak.with_field(events.Muon, dR, "dR")

            ## apply selection using PackedSelections
            selections = selections_dict["TTMinusB_CR"]
            print(f"The selections are {selections}")
            event_muon_selections = { 
                        "pt_cut": ak.flatten(events.Muon.pt > selections["muon_pt_min"]),
                        "dxy_cut": ak.flatten((abs(events.Muon.dxy) > selections["muon_dxy_min"]) & (abs(events.Muon.dxy) < selections["muon_dxy_max"])),
                        "iso_cut": ak.flatten((events.Muon.pfRelIso03_all > selections["muon_iso_min"]) & (events.Muon.pfRelIso03_all < selections["muon_iso_max"])),
                        "iso04_cut": ak.flatten((events.Muon.pfRelIso04_all > selections["muon_iso04_min"]) & (events.Muon.pfRelIso04_all < selections["muon_iso04_max"])),
                        "mediumId_cut": ak.flatten((events.Muon.mediumId > selections["muon_medium_ID_min"]) &(events.Muon.mediumId < selections["muon_medium_ID_max"])),
                        "mt_cut": ak.flatten((events.Muon.mT > selections["muon_mt_min"]) & (events.Muon.mT < selections["muon_mt_max"])),
            }
            event_jet_selections = {
                        "pt_cut": ak.flatten(events.CorrectedJet.pt > selections["jet_pt_min"]),
                        "score_cut": ak.flatten((events.CorrectedJet.disTauTag_score1 > selections["jet_score_min"]) & (events.CorrectedJet.disTauTag_score1 < selections["jet_score_max"])),
                        "dxy_cut": ak.flatten((abs(events.CorrectedJet.dxy) > selections["jet_dxy_min"]) & (abs(events.CorrectedJet.dxy) < selections["jet_dxy_max"])),
                        "id_cut": ak.flatten(events.CorrectedJet.jetId == selections["jet_ID"]),
            }
            event_met_selections = {
                        "pt_cut": events.CorrectedPFMET.pt > selections["MET_pt"], 
            }
            
        
            for key, cut in event_muon_selections.items():
                events["Muon"] = ak.with_field(events.Muon, cut, key)
            for key, cut in event_jet_selections.items():
                events["CorrectedJet"] = ak.with_field(events.CorrectedJet, cut, key)
            for key, cut in event_jet_selections.items():
                events["CorrectedPFMET"] = ak.with_field(events.CorrectedPFMET, cut, key)

            ## apply selection function
            events = prompt_muon_event_selection(events, selection_string)  
            print("The number of events left is", len(events))

            if not is_MC:
                if 'SR' not in args.selection and "PR" not in args.selection:
                    events = manual_blinding(events)
            
        else:
            ### To check Data vs MC
            ### Reject events if they pass one of the L1 seeds prescaled to 0 and fail all of the un-prescaled seeds 
            #prescaled_l1 = (
            #                events.L1.ETMHF70                              |\
            #                events.L1.ETMHF80                              |\
            #                events.L1.ETMHF70_HTT60er                      |\
            #                events.L1.ETMHF80_HTT60er                      |\
            #                events.L1.ETMHF80_SingleJet55er2p5_dPhi_Min2p1 
            #               )
            #print(events.L1.ETMHF90 == False)
            #unprescaled_l1 = (
            #                  (events.L1.ETMHF90 == False)                          &\
            #                  (events.L1.ETMHF100 == False)                         &\
            #                  (events.L1.ETMHF110 == False)                         &\
            #                  (events.L1.ETMHF120 == False)                         &\
            #                  (events.L1.ETMHF130 == False)                         &\
            #                  (events.L1.ETMHF140 == False)                         &\
            #                  (events.L1.ETMHF150 == False)                         &\
            #                  (events.L1.ETM150 == False)                           &\
            #                  (events.L1.ETMHF90_HTT60er == False)                  &\
            #                  (events.L1.ETMHF100_HTT60er == False)                 &\
            #                  (events.L1.ETMHF110_HTT60er == False)                 &\
            #                  (events.L1.ETMHF120_HTT60er == False)                 &\
            #                  (events.L1.ETMHF130_HTT60er == False)                 &\
            #                  (events.L1.ETMHF90_SingleJet60er2p5_dPhi_Min2p1  == False) 
            #                  )
            #                  

            #events = events[~(prescaled_l1 & unprescaled_l1)]

            # This will need to be re-organized to use "methodC" for keeping signal muon along with all other DisMuons
            # Will need to creat a new function in selection_function.py since event_selection function is
            # written for cases when there is only one object in the array
            dismuons = events.DisMuon
            dismuons = dismuons[dismuons.mediumId == True]
            dismuons = dismuons[ak.argsort(dismuons[leading_muon_var], ascending=False, axis=1)]
            dismuons = ak.singletons(ak.firsts(dismuons))
            events["DisMuon"] = dismuons
            num_muon = ak.count_nonzero(events["DisMuon"][leading_muon_var], axis = 1)

            num_corrected_jets = ak.count_nonzero(events["CorrectedJet"][leading_jet_var], axis = 1)
            loose_bjets  = events["CorrectedJet"][events["CorrectedJet"]["btagPNetB"] > 0.0499]
            medium_bjets = events["CorrectedJet"][events["CorrectedJet"]["btagPNetB"] > 0.2605]
            tight_bjets  = events["CorrectedJet"][events["CorrectedJet"]["btagPNetB"] > 0.6915]
            num_loose_bjets = ak.count_nonzero(loose_bjets[leading_jet_var], axis = 1)
            num_medium_bjets = ak.count_nonzero(medium_bjets[leading_jet_var], axis = 1)
            num_tight_bjets = ak.count_nonzero(tight_bjets[leading_jet_var], axis = 1)

            events["CorrectedJet"] = ak.with_field(events.CorrectedJet, num_corrected_jets, "numJets")
            events["CorrectedJet"] = ak.with_field(events.CorrectedJet, num_loose_bjets, "numLooseBJets")
            events["CorrectedJet"] = ak.with_field(events.CorrectedJet, num_medium_bjets, "numMediumBJets")
            events["CorrectedJet"] = ak.with_field(events.CorrectedJet, num_tight_bjets, "numTightBJets")

            correctedjets = events["CorrectedJet"]
            correctedjets = correctedjets[correctedjets.jetId == True]
            correctedjets = correctedjets[ak.argsort(correctedjets[leading_jet_var], ascending=False, axis = 1)]
            correctedjets = ak.singletons(ak.firsts(correctedjets))
            events["CorrectedJet"] = correctedjets

            num_jets = ak.count_nonzero(events["CorrectedJet"][leading_jet_var], axis = 1)
            events = events[(num_muon > 0) & (num_jets > 0)]

            met = events.CorrectedPFMET.pt
            met_phi = events.CorrectedPFMET.phi
            dphi = abs(events.DisMuon.phi - met_phi)
            dphi = np.where(dphi > np.pi, 2*np.pi - dphi, dphi)
            mT = np.sqrt(2 * events.DisMuon.pt * met * (1 - np.cos(dphi)))
            events["DisMuon"] = ak.with_field(events.DisMuon, mT, "mT")

            dphi = abs(events.CorrectedJet.phi - events.CorrectedPFMET.phi)
            dphi = np.where(dphi > np.pi, 2*np.pi - dphi, dphi)
            events["CorrectedJet"] = ak.with_field(events.CorrectedJet, dphi, "dphi")
            mT = np.sqrt(2 * events.CorrectedJet.pt * met * (1 - np.cos(dphi)))
            events["CorrectedJet"] = ak.with_field(events.CorrectedJet, mT, "mT")

            dxySig = events.DisMuon.dxy/events.DisMuon.dxyErr
            dzSig  = events.DisMuon.dz/events.DisMuon.dzErr
            events["DisMuon"] = ak.with_field(events.DisMuon, dxySig, "dxySig")
            events["DisMuon"] = ak.with_field(events.DisMuon, dzSig, "dzSig")

            mu_dphi = abs(events.CorrectedJet.phi - events.DisMuon.phi)
            mu_dphi = np.where(dphi > np.pi, 2*np.pi - dphi, dphi)
            events["DisMuon"] = ak.with_field(events.DisMuon, mu_dphi, "dphi")
            deta = abs(events.CorrectedJet.eta - events.DisMuon.eta)
            events["DisMuon"] = ak.with_field(events.DisMuon, deta, "deta")
            dR = events.DisMuon.metric_table(events.CorrectedJet)
            dR = ak.flatten(dR)
            events["DisMuon"] = ak.with_field(events.DisMuon, dR, "dR")

            ## apply selection using PackedSelections
            selections = selections_dict["TTMinusB_CR"]
            print(f"The selections are {selections}")
            event_muon_selections = { 
                        "pt_cut": ak.flatten(events.DisMuon.pt > selections["muon_pt_min"]),
                        "dxy_cut": ak.flatten((abs(events.DisMuon.dxy) > selections["muon_dxy_min"]) & (abs(events.DisMuon.dxy) < selections["muon_dxy_max"])),
                        "iso_cut": ak.flatten((events.DisMuon.pfRelIso03_all > selections["muon_iso_min"]) & (events.DisMuon.pfRelIso03_all < selections["muon_iso_max"])),
                        "iso04_cut": ak.flatten((events.DisMuon.pfRelIso04_all > selections["muon_iso04_min"]) & (events.DisMuon.pfRelIso04_all < selections["muon_iso04_max"])),
                        "mediumId_cut": ak.flatten((events.DisMuon.mediumId > selections["muon_medium_ID_min"]) &(events.DisMuon.mediumId < selections["muon_medium_ID_max"])),
                        "mt_cut": ak.flatten((events.DisMuon.mT > selections["muon_mt_min"]) & (events.DisMuon.mT < selections["muon_mt_max"])),
            }
            event_jet_selections = {
                        "pt_cut": ak.flatten(events.CorrectedJet.pt > selections["jet_pt_min"]),
                        "score_cut": ak.flatten((events.CorrectedJet.disTauTag_score1 > selections["jet_score_min"]) & (events.CorrectedJet.disTauTag_score1 < selections["jet_score_max"])),
                        "dxy_cut": ak.flatten((abs(events.CorrectedJet.dxy) > selections["jet_dxy_min"]) & (abs(events.CorrectedJet.dxy) < selections["jet_dxy_max"])),
                        "id_cut": ak.flatten(events.CorrectedJet.jetId == selections["jet_ID"]),
            }
            event_met_selections = {
                        "pt_cut": events.CorrectedPFMET.pt > selections["MET_pt"], 
            }
            
        
            for key, cut in event_muon_selections.items():
                events["DisMuon"] = ak.with_field(events.DisMuon, cut, key)
            for key, cut in event_jet_selections.items():
                events["CorrectedJet"] = ak.with_field(events.CorrectedJet, cut, key)
            for key, cut in event_jet_selections.items():
                events["CorrectedPFMET"] = ak.with_field(events.CorrectedPFMET, cut, key)

            ## apply selection function
            events = event_selection(events, selection_string)  

            if not is_MC:
                if 'SR' not in args.selection and "PR" not in args.selection:
                    events = manual_blinding(events)

        logger.info(f"Chose leading objects & filtered events")

        weights = events.genWeight if is_MC else 1 * ak.ones_like(events.event) 
        logger.info("mc weights")
        # Handle systematics and weights
        if is_MC:
            weight_branches = self.process_weight_corrs_and_systs(events, weights)
        else:
            weight_branches = {'weight': weights}
        logger.info("all weights")
        events = ak.with_field(events, weight_branches["weight"], "weight")


        ## prevent writing out files with empty trees
        if not len(events) > 0:
            return {
                "entries_written": 0,
                 #"run_dict" : run_dict
            }

        # Write to ROOT
        events_to_write = uproot_writeable_selected(events, include_all, include_prefixes, include_postfixes)
        # unique name: dataset name + chunk range
        fname = os.path.basename(events.metadata["filename"]).replace(".root", "")
        outname = f"{out_folder}{dataset}/{fname}_{selection_string}.root"

        with uproot.recreate(outname) as fout:
            fout["Events"] = events_to_write
#         skim = ak.to_parquet(events_to_write, outname.replace('.root', '.parquet'), extensionarray=False)
        return {"entries_written": len(events_to_write)}


    def postprocess(self, accumulator):
        return accumulator
    
    
if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    tic = time.time()

    test_job = args.testjob 

    if not test_job:
        n_port = 8786
        cluster = LPCCondorCluster(
                cores=4,
                memory='10GB',
                log_directory = f"/uscmst1b_scratch/lpc1/3DayLifetime/bskipwor/log/selected/{args.skimversion}",
                transfer_input_files = ["selection_function.py", "utils.py", "/eos/uscms/store/user/lpcdisptau/json/Cert_Collisions2022_355100_362760_Golden.json", "/eos/uscms/store/user/lpcdisptau/txt/jec", "/eos/uscms/store/user/lpcdisptau/json/jetvetomap_2022EFG.json", "/eos/uscms/store/user/lpcdisptau/json/ScaleFactors_Muon_Z_ID_ISO_2022_EE_schemaV2.json", "/eos/uscms/store/user/lpcdisptau/json/ScaleFactors_Muon_Z_HLT_2022_EE_abseta_pt_schemaV2.json"],
                job_extra_directives={
                    "should_transfer_files": "YES",
                    '+JobFlavour': '"longlunch"',
                    },
                job_script_prologue=[
                    "export XRD_RUNFORKHANDLER=1",  ### enables fork-safety in the XRootD client, to avoid deadlock when accessing EOS files
                    f"export X509_USER_PROXY=$HOME/x509up_u57864",
                    "export PYTHONPATH=$PYTHONPATH:$_CONDOR_SCRATCH_DIR:$HOME",
                ],
                )
         #minimum > 0: https://github.com/CoffeaTeam/coffea/issues/465
        cluster.adapt(minimum=0, maximum=200)
        print(cluster.job_script())
    
    else:
        cluster = LocalCluster(n_workers=10, threads_per_worker=1)

    client = Client(cluster)
    lxplus_run = processor.Runner(
        executor=processor.DaskExecutor(client=client, compression=None),
        schema=PFNanoAODSchema,
        savemetrics=True,
        xrootdtimeout=600,
    )
    
    out, proc_report = lxplus_run(
        fileset,
        treename="Events",
        processor_instance=SelectionProcessor(args.leading_muon_type, args.leading_jet_type, mode_string),
        uproot_options={"allow_read_errors_with_report": (OSError, KeyError),
                        }
    )

    elapsed = time.time() - tic 
    print(f"Finished in {elapsed:.1f}s")
