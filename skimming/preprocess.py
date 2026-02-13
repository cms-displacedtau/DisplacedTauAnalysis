import argparse, importlib
import pickle, pdb
from coffea import processor
from coffea.dataset_tools import (
    apply_to_fileset,
    max_chunks,
    preprocess,
)
from dask import config as cfg
cfg.set({'distributed.scheduler.worker-ttl': None}) # Check if this solves some dask issues
from uproot.exceptions import KeyInFileError

import time, logging
from dask.distributed import Client, wait, progress, LocalCluster
# from dask_lxplus import CernCluster
from lpcjobqueue import LPCCondorCluster, schedd 
from dask_jobqueue import HTCondorCluster
import socket, time
import dask_awkward as dak
import warnings
warnings.filterwarnings("ignore", module="coffea") # Suppress annoying deprecation warnings for coffea vector, c.f. https://github.com/CoffeaTeam/coffea/blob/master/src/coffea/nanoevents/methods/candidate.py


parser = argparse.ArgumentParser(description="")
parser.add_argument(
	"--sample",
	choices=['QCD','DYEMu', 'DYTau', 'signal', 'Wto2Q', 'WtoLNu', 'TT', 'singleT', 'JetMET_2022', 'Muon', 'DYto2L-2Jets', 'DYto2Tau-2Jets_0J', 'DYto2Tau-2Jets_0J_custom'],
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
	"--skim",
	default='',
	required=False,
	help='Specify, if working on the skimmed samples, the skim name (name of the folder inside samples)')
parser.add_argument(
	"--skimversion",
	default='',
	required=False,
	help='Specify, if working on the skimmed samples, the skim name (name of the folder inside samples)')
parser.add_argument(
	"--nanov",
	choices=['Summer22_CHS_v10', 'Summer22_CHS_v7'],
	default='Summer22_CHS_v10',
	required=False,
	help='Specify the custom nanoaod version to process')
args = parser.parse_args()

outdir_p = f'{args.nanov}.'
outdir_s = f'{args.nanov}/'
if args.skim != '':
    if args.skimversion != '':
        outdir_p += f'{args.skim}.{args.skimversion}.'
        outdir_s += f'{args.skim}/{args.skimversion}/'
    else:
        outdir_p += f'{args.skim}.'
        outdir_s += f'{args.skim}/'

  
samples = {
    "Wto2Q"  : f"samples.{outdir_p}fileset_Wto2Q",
    "WtoLNu" : f"samples.{outdir_p}fileset_WtoLNu",
    "QCD"    : f"samples.{outdir_p}fileset_QCD",
    "DYEMu"     : f"samples.{outdir_p}fileset_DYEMu",
    "DYTau"     : f"samples.{outdir_p}fileset_DYTau",
    "signal" : f"samples.{outdir_p}fileset_signal",
    "TT"     : f"samples.{outdir_p}fileset_TT",
    "singleT": f"samples.{outdir_p}fileset_singleT",  ## more on this later
    "JetMET_2022": f"samples.{outdir_p}fileset_JetMET_2022",
    "Muon": f"samples.{outdir_p}fileset_Muon_2022",
    "DYto2L-2Jets": f"samples.{outdir_p}fileset_DYto2L-2Jets",
    'DYto2Tau-2Jets_0J': f"samples.{outdir_p}fileset_DYto2Tau-2Jets_0J",
    'DYto2Tau-2Jets_0J_custom': f"samples.{outdir_p}fileset_DYto2Tau-2Jets_0J_custom"
}

module = importlib.import_module(samples[args.sample])
all_fileset = module.fileset  

if args.subsample == 'all':
    fileset = all_fileset
else:  
    fileset = f"samples.{outdir_p}fileset_{args.subsample[0]}"
    subset_module = importlib.import_module(fileset)
    fileset = subset_module.fileset

nfiles = int(args.nfiles)
if nfiles != -1:
    for k in fileset.keys():
        if nfiles < len(fileset[k]['files']):
            fileset[k]['files'] = dict(list(fileset[k]['files'].items())[:nfiles])
#print("Will process {} files from the following samples:".format(nfiles), fileset.keys())

## first element of value is step_size, second is files_per_barch
pars_per_sample = {
    "Wto2Q"  : [10_000, 100], 
    "WtoLNu" : [10_000, 100],  
    "QCD"    : [10_000, 1],  
    "DY"     : [10_000, 1000],  
    "DYto2L-2Jets"     : [10_000, 1000],  
    "DYto2Tau-2Jets_0J"     : [10_000, 1000],  
    "DYto2Tau-2Jets_0J_custom": [10_000, 1000],  
    "signal" : [10_000, 1],  
    "TT"     : [10_000, 1000],  
    "singleT": [10_000, 1000],  
    "data"   : [10_000, 100], 
    "JetMET_2022": [10_000, 100],
    "Muon": [10_000, 1000],
}



if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)    
    tic = time.time()

    cluster = LPCCondorCluster(
            cores=1,
            memory='4000MB',
            log_directory = f"/uscmst1b_scratch/lpc1/3DayLifetime/condor/log/preprocess/",
            transfer_input_files = ["utils.py"],
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
    cluster.adapt(minimum=1, maximum=200)
    print(cluster.job_script())

    #cluster = LocalCluster(n_workers=10, threads_per_worker=1)
    client = Client(cluster)

    runner = processor.Runner(processor.DaskExecutor(client=client, compression=None),
                              chunksize = 50000,
                              align_clusters=False,
                              skipbadfiles=True,
                            )

    dataset_runnable = runner.preprocess(
       fileset,
       treename = "Events",
    )

    if args.subsample != 'all':
        pkl_name = f"samples/{outdir_s}{args.subsample[0]}_preprocessed.pkl"
        if nfiles > 0:
            pkl_name.replace('.pkl', f'_{nfiles}files.pkl')
        with open(pkl_name, "wb") as f:
            pickle.dump(list(dataset_runnable), f)
        #for isubsample in dataset_runnable.keys():
        #    pkl_name = f"samples/{outdir_s}{args.sample}_{isubsample}_preprocessed.pkl"
        #    if nfiles > 0:
        #        pkl_name.replace('.pkl', f'_{nfiles}files.pkl')
        #    with open(pkl_name, "wb") as f:
        #       pickle.dump({isubsample:dataset_runnable[isubsample]}, f)
    else:    
        pkl_name = f"samples/{outdir_s}{args.sample}_preprocessed.pkl"
        if nfiles > 0:
            pkl_name.replace('.pkl', f'_{nfiles}files.pkl')
        with open(pkl_name, "wb") as f:
            pickle.dump(list(dataset_runnable), f)

    elapsed = time.time() - tic 
    print(f"Preprocessing datasets finished in {elapsed:.1f}s") 

    client.shutdown()
    cluster.close()
