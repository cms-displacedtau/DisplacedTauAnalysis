#!/usr/bin/env python3

import ctypes
import json
import logging
from hepunits import cm
import hist
import numpy
import os
import re
#import yaml

from ruamel.yaml import YAML
yaml = YAML()
yaml.preserve_quotes = True
yaml.width = 1024

import ROOT
ROOT.gROOT.SetBatch(True)


logging.basicConfig(format = "[%(levelname)s] [%(asctime)s] %(message)s", level = logging.INFO)
logger = logging.getLogger("mylogger")


def exec_cmds(cmds) :
    
    for cmd in cmds :
        
        #print(cmd)
        retval = os.system(f"set -x; {cmd}")
        
        if (retval) :
            
            exit(retval)


# Factorize to a utils file later
def load_config(cfgfile) :
    
    #if (os.path.isfile(cfgfile)) :
    
    with open(cfgfile, "r") as fopen :
        
        content = fopen.read()
        
        if (cfgfile.endswith(".yml") or cfgfile.endswith(".yaml")) :
            
            logger.info(f"Loading yaml config: {cfgfile}")
            #d_loadcfg = yaml.load(content, Loader = yaml.FullLoader)
            d_loadcfg = yaml.load(content)
        
        elif (cfgfile.endswith(".json")) :
            
            logger.info(f"Loading json config: {cfgfile}")
            d_loadcfg = json.loads(content)
        
        else :
            
            logger.error(f"Invalid config provided: {cfgfile}")
            exit(1)
    
    return d_loadcfg

def parse_string_regex(s, regexp) :
    
    rgx = re.compile(regexp)
    #print(s, regexp)
    result = [m.groupdict() for m in rgx.finditer(s)][0]
    
    return result

def parse_stau_samplestring(s, regexp) :
    
    #rgx = re.compile("stau(\d+)_lsp(\d+)_ctau(\w+)")
    rgx = re.compile(regexp)
    
    #mstau, mlsp, ctau = rgx.findall(s)[0]
    #
    #result = {
    #    "mstau": float(mstau),
    #    "mlsp": float(mlsp),
    #    "ctau": ctau,
    #}
    
    result = [m.groupdict() for m in rgx.finditer(s)][0]
    
    # VERY hacky -- needs to be improved
    result["mstau"] = float(result["mstau"])
    if "mlsp" in result:
        result["mlsp"] = float(result["mlsp"])
    
    return result


def load_stau_xsec_file(xsecfile) :
    
    arr_xsec = numpy.loadtxt(xsecfile, delimiter = ",", converters = {1: eval})
    
    gr_xsec = ROOT.TGraph(len(arr_xsec[:, 0]), arr_xsec[:, 0], arr_xsec[:, 1])
    gr_xsec.SetName("gr_xsec")
    
    gr_xsec_unc_d = ROOT.TGraph(len(arr_xsec[:, 0]), arr_xsec[:, 0], arr_xsec[:, 2])
    gr_xsec_unc_d.SetName("gr_xsec_unc_d")
    
    gr_xsec_unc_u = ROOT.TGraph(len(arr_xsec[:, 0]), arr_xsec[:, 0], arr_xsec[:, 3])
    gr_xsec_unc_u.SetName("gr_xsec_unc_u")



def get_stau_xsec_dict(l_samplestr, xsecfile, regexp) :
    
    #arr_data = numpy.loadtxt(xsecfile, delimiter = ",", converters = {1: eval})
    arr_data = numpy.loadtxt(xsecfile, delimiter = ",")
    
    nelements = arr_data.shape[0]
    arr_mass = numpy.array(arr_data[:, 0], dtype = numpy.float64)
    arr_xsec = numpy.array(arr_data[:, 1], dtype = numpy.float64)
    
    #gr_xsec = ROOT.TGraph(nelements, arr_mass, arr_xsec)
    #gr_xsec.SetName("gr_xsec")
    
    #xsec = None
    #
    ## 1st column is the stau mass
    #rowidx = numpy.where(arr_xsec[:, 0] == mstau)[0]
    #
    ## Empty, i.e. mstau not present
    #if (not len(rowidx)) :
    #    
    #    logger.error(f"mstau {mstau} not found in {xsecfile}")
    #    exit(1)
    #
    #rowidx = rowidx[0]
    #
    ## 2nd colum is the xsec
    #xsec = arr_xsec[rowidx, 1]
    
    #xsec = gr_xsec.Eval(mstau)
    
    #d_xsec = {}
    #
    #for samplestr in l_samplestr :
    #    
    #    d_stau_param = parse_stau_samplestring(s = samplestr, regexp = regexp)
    #    mstau = d_stau_param["mstau"]
    #    
    #    fn = ROOT.TF1("xsec_fn", "[0] + ([1]*x**[2])", min(arr_mass), max(arr_mass))
    #    fit_result = gr_xsec.Fit(fn, "S")
    #    fitted_fn = gr_xsec.GetFunction("xsec_fn")
    #    
    #    #xsec = gr_xsec.Eval(mstau)
    #    xsec = fitted_fn.Eval(mstau)
    #    d_xsec[samplestr] = xsec
    #
    #return d_xsec
    
    d_xsec = {}
    
    for samplestr in l_samplestr :
        
        d_stau_param = parse_stau_samplestring(s = samplestr, regexp = regexp)
        mstau = d_stau_param["mstau"]
        
        # 1st column is the stau mass
        rowidx = numpy.where(arr_data[:, 0] == mstau)[0]
        
        # Empty, i.e. mstau not present
        if (not len(rowidx)) :
            
            logger.error(f"mstau {mstau} not found in {xsecfile}")
            exit(1)
        
        # 2nd colum is the xsec
        # Convert to scalar
        xsec = arr_data[rowidx, 1].item()
        
        d_xsec[samplestr] = xsec
    
    return d_xsec


#def get_stau_xsec_dict(l_samplestr, xsecfile, regexp) :
#    
#    d_xsec = {}
#    
#    l_mstaus = []
#    
#    for samplestr in l_samplestr :
#        
#        d_stau_param = parse_stau_samplestring(s = samplestr, regexp = regexp)
#        mstau = d_stau_param["mstau"]
#        l_mstaus.append(mstau)
#        
#        d_xsec[samplestr] = get_stau_xsec(samplestr = samplestr, xsecfile = xsecfile, regexp = regexp)
    
    return d_xsec


def get_hist(
    histfile,
    histname,
    samples,
    scales = [],
    rebin = None,
    underflow = True,
    overflow = True,
    set_min = None,
    set_max = None,
) :
    
    if (len(scales)) :
        
        assert(len(scales) == len(samples))
    
    opened_file = False
    if isinstance(histfile, str) :
        
        histfile = ROOT.TFile.Open(histfile)
        opened_file = True
    
    hist_result = None
    
    for isample, sample in enumerate(samples) :
        
        histname_sample = f"{sample}/{histname}"
        logger.info(f"Getting [histogram {histname_sample}] from [file {histfile.GetName()}]")
        hist_sample = histfile.Get(histname_sample).Clone()
        hist_sample.SetDirectory(0)
        #hist_sample.Print("range")
        
        if (len(scales)) :
            
            hist_sample.Scale(scales[isample])
        
        if (rebin is not None) :
            
            rebin = numpy.array(rebin, dtype = numpy.float64)
            hist_sample = hist_sample.Rebin(len(rebin)-1, "", rebin)
        
        if (hist_result is None) :
            
            hist_result = hist_sample.Clone()
            hist_result.SetDirectory(0)
        
        else :
            
            hist_result.Add(hist_sample)
    
    
    # Add under/overflow if rebinned
    if (rebin is not None) :
        
        nbins = hist_result.GetNbinsX()
        
        if (underflow) :
            
            hist_result.AddBinContent(1, hist_result.GetBinContent(0))
            hist_result.SetBinContent(0, 0.0)
            hist_result.SetBinError(0, 0.0)
        
        if (overflow) :
            
            hist_result.AddBinContent(nbins, hist_result.GetBinContent(nbins+1))
            hist_result.SetBinContent(nbins+1, 0.0)
            hist_result.SetBinError(nbins+1, 0.0)
    
    if ((set_min is not None) or (set_max is not None)) :
        
        for ibin in range(hist_result.GetNbinsX()) :
            
            val = hist_result.GetBinContent(ibin+1)
            
            if ((set_min is not None) and (val < set_min)) :
                
                hist_result.SetBinContent(ibin+1, set_min)
            
            elif ((set_max is not None) and (val > set_max)) :
                
                hist_result.SetBinContent(ibin+1, set_max)
    
    if opened_file :
        
        histfile.Close()
    
    return hist_result


def pdf_to_png(infilename, outfilename = None) :
    
    infname, _ = os.path.splitext(infilename)
    
    if (outfilename) :
        
        outfilename, _ = os.path.splitext(outfilename)
    
    else :
        
        outfilename = infname
    
    # .png is automatically added to the output file name
    retval = os.system(f"pdftoppm -cropbox -r 600 -png -singlefile {infilename} {outfilename}")
    
    return retval


def plot_fitdiagnostics_correlation(
    rootfilename,
    outfilename,
    histname,
    cov_to_corr,
    title = "",
    l_binlabel_fmt = [],
    drawopt = "colz"
    ) :
    
    """
    Include bin label format commands as strings, in l_binlabel_fmt. For e.g.:
    "{label}.replace('_', '-')"
    This string will be evaluated
    Must use the key {label} in the the string
    """
    
    #rootfilename = "fitDiagnostics.root"
    rootfile = ROOT.TFile.Open(rootfilename);
    
    inhist = rootfile.Get(histname).Clone()
    hist_correlation = inhist.Clone()
    hist_correlation.SetTitle(title)
    
    nbinsx = inhist.GetNbinsX()
    nbinsy = inhist.GetNbinsY()
    
    for ibinx in range(1, nbinsx+1) :
        
        for ibiny in range(1, nbinsy+1) :
            
            val = inhist.GetBinContent(ibinx, ibiny)
            
            if (cov_to_corr) :
                
                val /= numpy.sqrt(inhist.GetBinContent(ibinx, ibinx) * inhist.GetBinContent(ibiny, ibiny))
            
            hist_correlation.SetBinContent(ibinx, ibiny, val)
    
    ROOT.gStyle.SetOptStat(0)
    ROOT.gStyle.SetPaintTextFormat("0.2f")
    
    npix_per_bin = 100
    canvas = ROOT.TCanvas("canvas", "canvas", 1000+(npix_per_bin*nbinsx), 500+(npix_per_bin*nbinsy))
    
    canvas.SetLeftMargin(0.15)
    canvas.SetRightMargin(0.15)
    canvas.SetBottomMargin(0.2)
    
    hist_correlation.SetMarkerSize(0.6*hist_correlation.GetMarkerSize())
    
    hist_correlation.Draw(drawopt)
    
    hist_correlation.GetXaxis().SetTitle("")
    hist_correlation.GetYaxis().SetTitle("")
    
    # font = 10*font + precision
    # precision=3 allows setting font size in pixels
    hist_correlation.GetXaxis().SetLabelFont(63)
    hist_correlation.GetYaxis().SetLabelFont(63)
    
    #hist_correlation.GetXaxis().SetLabelSize(0.4*hist_correlation.GetXaxis().GetLabelSize())
    #hist_correlation.GetYaxis().SetLabelSize(0.4*hist_correlation.GetYaxis().GetLabelSize())
    
    # Set label size in pixels
    hist_correlation.GetXaxis().SetLabelSize(npix_per_bin/2)
    hist_correlation.GetYaxis().SetLabelSize(npix_per_bin/2)
    
    hist_correlation.GetXaxis().SetLabelOffset(0.4*hist_correlation.GetXaxis().GetLabelOffset())
    hist_correlation.GetYaxis().SetLabelOffset(0.3*hist_correlation.GetYaxis().GetLabelOffset())
    
    hist_correlation.GetXaxis().LabelsOption("v")
    
    for ibinx in range(1, inhist.GetNbinsX()+1) :
        
        binlabel_x = hist_correlation.GetXaxis().GetBinLabel(ibinx)
        binlabel_y = hist_correlation.GetYaxis().GetBinLabel(ibinx)
        
        for fmt in l_binlabel_fmt :
            
            binlabel_x = eval(fmt.format(**{"label": "binlabel_x"}))
            binlabel_y = eval(fmt.format(**{"label": "binlabel_y"}))
        
        hist_correlation.GetXaxis().SetBinLabel(ibinx, binlabel_x)
        hist_correlation.GetYaxis().SetBinLabel(ibinx, binlabel_y)
    
    #hist_correlation.GetXaxis().SetNdivisions(nbinsx, 0, 0, False)
    #hist_correlation.GetYaxis().SetNdivisions(nbinsy, 0, 0, False)
    
    canvas.Update()
    palette = hist_correlation.GetListOfFunctions().FindObject("palette")
    #print("X1", palette.GetX1NDC(), "X2", palette.GetX2NDC(), "Y1", palette.GetY1NDC(), "Y2", palette.GetY2NDC())
    palette.SetX1NDC(0.855)
    palette.SetX2NDC(0.900)
    palette.SetY1NDC(0.200)
    palette.SetY2NDC(0.899)
    
    #canvas.SetGridx(1)
    #canvas.SetGridy(1)
    
    canvas.Update()
    
    canvas.SaveAs(outfilename)
    
    rootfile.Close()


def get_binning(start, end, step, include_end = True) :
    
    bins = numpy.arange(start, end, step)
    
    if (include_end) :
        
        if (bins[-1] != end) :
            
            bins = numpy.append(bins, [end])
        
        else :
            
            bins = bins[0: -1]
    
    else :
        
        if (bins[-1] == end) :
            
            bins = bins[0: -1]
    
    return bins


def natural_sort(l):
    
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key = alphanum_key)


def get_decimal_places(s):
    
    return len(s.split(".")[1]) if "." in s else 0


def cms_rounding(val, unc_stat, unc_syst, N=2):
    
    min_number = min(float(val), float(unc_stat), float(unc_syst))
    threshold = 5*10**(-N)
    
    if N > 0 and min_number > threshold :
        
        return cms_rounding(val, unc_stat, unc_syst, N-1)
    
    val_str = f"{val:.{N}f}"
    unc_stat_str = f"{unc_stat:.{N}f}"
    unc_syst_str = f"{unc_syst:.{N}f}"
    
    return val_str, unc_stat_str, unc_syst_str


def root_TGraph_to_TH1(graph, binning = None, evalBins = False, setError = True) :
    
    if binning :
        hist = ROOT.TH1F(graph.GetName(), graph.GetTitle(), len(binning)-1, binning)
    else :
        hist = graph.GetHistogram().Clone()
    
    hist.SetDirectory(0)
    
    nPoint = graph.GetN()
    
    arr_x = numpy.array(graph.GetX(), dtype = numpy.float64)
    min_x = numpy.min(arr_x)
    max_x = numpy.max(arr_x)
    
    if (evalBins) :
        
        nBins = hist.GetNbinsX()
        
        for iBin in range(1, nBins+1) :
            
            pointValX = hist.GetBinCenter(iBin)
            
            if pointValX < min_x or pointValX > max_x :
                continue
            
            pointValY = graph.Eval(pointValX)
            
            if numpy.isfinite(pointValY) :
                
                hist.SetBinContent(iBin, pointValY)
            
            else :
                
                print(f"Invalid point value for bin {iBin} (x = {pointValX}): {pointValY}")
    
    else :
        for iPoint in range(0, nPoint) :
            
            if (hasattr(ROOT, "Double")) :
                
                pointValX = ROOT.Double(0)
                pointValY = ROOT.Double(0)
            
            else :
                
                pointValX = ctypes.c_double(0)
                pointValY = ctypes.c_double(0)
            
            graph.GetPoint(iPoint, pointValX, pointValY)
            #print(iPoint, pointValX, pointValY)
            
            pointErrX = graph.GetErrorX(iPoint)
            pointErrY = graph.GetErrorY(iPoint)
            
            binNum = hist.FindBin(pointValX)
            
            hist.SetBinContent(binNum, pointValY)
            
            if (setError) :
                
                hist.SetBinError(binNum, pointErrY)
    
    return hist


def get_garwood_errors(n, quantile = 0.6827, interval = False) :
    
    alpha = 1.0 - quantile
    
    d = 0 if n == 0 else ROOT.Math.gamma_quantile(alpha/2, n, 1)
    u = ROOT.Math.gamma_quantile_c(alpha/2, n+1, 1)
    
    if interval :
        
        return d, u
    
    else :
        
        return n - d, u - n
