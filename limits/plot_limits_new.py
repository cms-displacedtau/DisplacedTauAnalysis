#!/usr/bin/env python3

import argparse
import cmsstyle as CMS
import copy
import numpy
import os
import sortedcontainers

import ROOT
ROOT.gROOT.SetBatch(True)

import utils.commonutils as cmut


NPX = 500
NPY = 500
NZCONTS = 300
ROOTSPACE_0P4 = "#scale[0.4]{ }"


def get_contours_from_hist(hist, contour_value, nsmooth = 0) :
    
    if (nsmooth > 0) :

        #hist.Smooth(1, "k5a")
        #hist.Smooth(nsmooth, "k5b")

        # Works better
        for ismooth in range(0, nsmooth) :

            #hist.Smooth(1, "k5b")
            hist.Smooth(1, "kba")
            hist.Smooth(1, "k5a")
    
    canvas_temp = ROOT.TCanvas("canvas_temp", "canvas_temp")
    canvas_temp.cd()
    
    l_contours_values = [contour_value]
    
    hist_tmp = hist.Clone()
    ROOT.SetOwnership(hist_tmp, 0)
    
    # Set contours
    hist_tmp.SetContour(len(l_contours_values), numpy.array(l_contours_values, dtype = numpy.float64))
    hist_tmp.Draw("CONT Z LIST")
    canvas_temp.Update()
    
    # Get contours
    ROOT.gROOT.GetListOfSpecials().Print()
    l_contours = ROOT.gROOT.GetListOfSpecials().FindObject("contours").Clone()
    l_contours = l_contours.At(0)
    
    #if (not l_contours.GetEntries()) :
    #    
    #    print("Error: No contours found.")
    #    exit(1)
    
    canvas_temp.Clear()
    canvas_temp.Close()
    
    return l_contours


def get_contours_from_graph(graph, contour_value, nsmooth = 0) :
    
    return get_contours_from_hist(graph.GetHistogram().Clone(), contour_value, nsmooth)


def merge_contour_segments(contour_list, name="merged_contour"):
    """
    Merge multiple disjoint TGraph contour segments into a single TGraph.

    Strategy:
      1. Start with the first segment.
      2. For each remaining segment, find the one whose start or end point
         is closest to the current merged contour's end point.
      3. Append it in the correct order (forward or reversed).
      4. Repeat until all segments are consumed.

    This handles the common case where ROOT splits a horseshoe contour
    into 2–3 pieces (short-ctau arm, bottom, long-ctau arm).
    """
    n_segments = contour_list.GetEntries() if hasattr(contour_list, 'GetEntries') else len(contour_list)

    if n_segments == 0:
        return None
    if n_segments == 1:
        return contour_list.At(0).Clone(name) if hasattr(contour_list, 'At') else contour_list[0].Clone(name)

    # Extract all segments as lists of (x, y) tuples
    segments = []
    for i in range(n_segments):
        seg = contour_list.At(i) if hasattr(contour_list, 'At') else contour_list[i]
        n_pts = seg.GetN()
        if n_pts == 0:
            continue
        pts = [(seg.GetX()[j], seg.GetY()[j]) for j in range(n_pts)]
        segments.append(pts)

    if not segments:
        return None

    def dist2(p1, p2):
        """Squared distance between two points (in normalized space)."""
        # Normalize x (mass ~100-500 GeV) and y (log10(ctau) ~ -1 to 3)
        # so that distances are comparable in both dimensions
        dx = (p1[0] - p2[0]) / 400.0  # mass range
        dy = (p1[1] - p2[1]) / 4.0    # log10(ctau) range
        return dx*dx + dy*dy

    # Greedy nearest-neighbor stitching
    merged = list(segments[0])
    remaining = list(range(1, len(segments)))

    while remaining:
        end_point = merged[-1]
        start_point = merged[0]

        best_idx = None
        best_dist = float('inf')
        best_reverse = False
        best_attach_to_start = False

        for idx in remaining:
            seg = segments[idx]
            seg_start = seg[0]
            seg_end = seg[-1]

            # Try attaching to end of merged
            d = dist2(end_point, seg_start)
            if d < best_dist:
                best_dist, best_idx, best_reverse, best_attach_to_start = d, idx, False, False

            d = dist2(end_point, seg_end)
            if d < best_dist:
                best_dist, best_idx, best_reverse, best_attach_to_start = d, idx, True, False

            # Try attaching to start of merged
            d = dist2(start_point, seg_start)
            if d < best_dist:
                best_dist, best_idx, best_reverse, best_attach_to_start = d, idx, True, True

            d = dist2(start_point, seg_end)
            if d < best_dist:
                best_dist, best_idx, best_reverse, best_attach_to_start = d, idx, False, True

        seg = segments[best_idx]
        if best_reverse:
            seg = list(reversed(seg))

        if best_attach_to_start:
            merged = seg + merged
        else:
            merged = merged + seg

        remaining.remove(best_idx)

    # Build the merged TGraph
    merged_graph = ROOT.TGraph(len(merged))
    merged_graph.SetName(name)

    for i, (x, y) in enumerate(merged):
        merged_graph.SetPoint(i, x, y)

    return merged_graph


def delog_th2_yaxis(hist, logctau_min, logctau_max) :
    """
    Rebuild a TH2 whose y-axis is uniform in log10(ctau) into one with
    the equivalent (non-uniform) real-ctau bin edges, so it draws
    correctly on a canvas with SetLogy() on the ctau axis.
    """
    
    nbinsx = hist.GetNbinsX()
    nbinsy = hist.GetNbinsY()
    
    xedges = numpy.array([hist.GetXaxis().GetBinLowEdge(i) for i in range(1, nbinsx + 2)], dtype = numpy.float64)
    log_yedges = numpy.linspace(logctau_min, logctau_max, nbinsy + 1)
    yedges = numpy.power(10.0, log_yedges)
    
    hist_real = ROOT.TH2F(f"{hist.GetName()}_realctau", hist.GetTitle(), nbinsx, xedges, nbinsy, yedges)
    hist_real.GetXaxis().SetTitle(hist.GetXaxis().GetTitle())
    hist_real.GetYaxis().SetTitle(hist.GetYaxis().GetTitle())
    hist_real.GetZaxis().SetTitle(hist.GetZaxis().GetTitle())
    
    for ix in range(1, nbinsx + 1) :
        for iy in range(1, nbinsy + 1) :
            
            hist_real.SetBinContent(ix, iy, hist.GetBinContent(ix, iy))
    
    return hist_real


def get_exclusion(
    d_limits,
    d_xsecs,
    limit_key,
    xsec_key,
    name,
    axis_label,
) :
    # ctau points are log-spaced over ~3 decades (1 - 1000 mm). Doing the
    # Delaunay interpolation / contour-finding in *linear* ctau space
    # crushes nearly all the sampled points into a sliver near y=0 and
    # leaves huge unsampled linear gaps (e.g. 100 -> 1000 mm) that get
    # bridged by flat triangles -- this is what turns the expected
    # "horseshoe" into a straight line. So we interpolate in log10(ctau)
    # and convert back to real ctau afterward for display.
    logctau_min = -1.0   # log10(0.1 mm)
    logctau_max = numpy.log10(2000.0)
    NBINS_Y = 200
    
    g2_ul = ROOT.TGraph2D()
    g2_ul.SetName(f"g2_ul_{name}")
    
    h2_ul = ROOT.TH2F(f"h2_ul_{name}", f"h2_ul_{name}", 100, 0, 1000, NBINS_Y, logctau_min, logctau_max)
    h2_ul.GetXaxis().SetTitle("m(#tilde{#tau}) [GeV]")
    h2_ul.GetYaxis().SetTitle("log_{10}(c#tau_{0} [mm])")
    h2_ul.GetZaxis().SetTitle(axis_label)
    
    g2_xsecul = ROOT.TGraph2D()
    g2_xsecul.SetName(f"g2_xsecul_{name}")
    
    h2_xsecul = ROOT.TH2F(f"h2_xsecul_{name}", f"h2_xsecul_{name}", 100, 0, 1000, NBINS_Y, logctau_min, logctau_max)
    h2_xsecul.GetXaxis().SetTitle("m(#tilde{#tau}) [GeV]")
    h2_xsecul.GetYaxis().SetTitle("log_{10}(c#tau_{0} [mm])")
    h2_xsecul.GetZaxis().SetTitle(f"{axis_label} on cross section [fb]")
    
    for ipoint, ((mstau, ctau), limits) in enumerate(d_limits.items()) :
        
        if limit_key not in limits :
            
            print(f"Warning: could not find limit {limit_key} for mstau={mstau}, ctau={ctau}. Skipping")
            continue
        
        ul = limits[limit_key] # if limit_key in limits else 0
        xsec = d_xsecs[mstau][xsec_key]
        xsecul = xsec*ul
        print(limit_key, mstau, ctau, ul, xsecul)
        
        logctau = numpy.log10(ctau)
        
        g2_ul.SetPoint(ipoint, mstau, logctau, ul)
        h2_ul.Fill(mstau, logctau, ul)
        
        g2_xsecul.SetPoint(ipoint, mstau, logctau, xsecul)
        h2_xsecul.Fill(mstau, logctau, xsecul)
    
    #g2_ul.GetHistogram().GetXaxis().SetRangeUser(0, 1000)
    #g2_xsecul.GetHistogram().GetXaxis().SetRangeUser(0, 1000)
    
    g2_ul.SetNpx(NPX)
    g2_ul.SetNpy(NPY)
    
    g2_xsecul.SetNpx(NPX)
    g2_xsecul.SetNpy(NPY)
    
    h2_ul_interp = g2_ul.GetHistogram().Clone()
    h2_ul_interp.SetName(f"{h2_ul.GetName()}_interp")
    h2_ul_interp.GetXaxis().SetTitle(h2_ul.GetXaxis().GetTitle())
    h2_ul_interp.GetYaxis().SetTitle(h2_ul.GetYaxis().GetTitle())
    h2_ul_interp.GetZaxis().SetTitle(h2_ul.GetZaxis().GetTitle())
    
    contours = get_contours_from_hist(
        hist = h2_ul_interp.Clone(),
        contour_value = 1.0,
        nsmooth = 0,
    )
    print(f"[{name}] found {contours.GetEntries() if contours else 0} contour segment(s)")
    
    h2_xsecul_interp = g2_xsecul.GetHistogram().Clone()
    h2_xsecul_interp.SetName(f"{h2_xsecul.GetName()}_interp")
    h2_xsecul_interp.GetXaxis().SetTitle(h2_xsecul.GetXaxis().GetTitle())
    h2_xsecul_interp.GetYaxis().SetTitle(h2_xsecul.GetYaxis().GetTitle())
    h2_xsecul_interp.GetZaxis().SetTitle(h2_xsecul.GetZaxis().GetTitle())
    
    # Convert the interpolated colz histogram back to real (non-uniform,
    # log-spaced) ctau bin edges for display on a SetLogy() canvas.
    h2_xsecul_interp_real = delog_th2_yaxis(h2_xsecul_interp, logctau_min, logctau_max)
    h2_xsecul_interp_real.SetName(f"{h2_xsecul.GetName()}_interp_realctau")
    
    result = sortedcontainers.SortedDict()
    
    result["g2_ul"] = g2_ul
    result["h2_ul"] = h2_ul
    result["h2_ul_interp"] = h2_ul_interp
    
    result["g2_xsecul"] = g2_xsecul
    result["h2_xsecul"] = h2_xsecul
    result["h2_xsecul_interp"] = h2_xsecul_interp_real
    
    #result["d_xsecul_per_ctau"] = d_xsecul_per_ctau
    result["contour"] = None
    
    if contours and contours.GetEntries() > 0 :
        
        n_segments = contours.GetEntries()
        
        if n_segments == 1 :
            # Single contour — use it directly
            cnt = contours.At(0).Clone(f"contour_{name}")
        else :
            # Multiple segments — merge them into one continuous contour
            print(f"[{name}] Merging {n_segments} contour segments into one...")
            cnt = merge_contour_segments(contours, name=f"contour_{name}")
        
        if cnt is not None :
            # contour y-values come out in log10(ctau); convert back to ctau
            for i in range(cnt.GetN()) :
                cnt.SetPoint(i, cnt.GetX()[i], 10.0**cnt.GetY()[i])
            
            result["contour"] = cnt
            result["contours"] = contours
    
    return result




def main() :
    
    # Argument parser
    parser = argparse.ArgumentParser(formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument(
        "--jsons",
        help = "Input",
        type = str,
        nargs = "+",
        required = True,
    )
    
    parser.add_argument(
        "--extratext",
        help = "Extra text to be printed on canvas",
        type = str,
        required = False,
    )
    
    parser.add_argument(
        "--xsecfile",
        help = "Cross section csv file",
        type = str,
        required = True,
    )
    
    parser.add_argument(
        "--cmsextratext",
        help = "Will set the CMS extra text; can be empty string \"\"",
        type = str,
        required = False,
        #default = "Preliminary",
        default = "",
    )
    
    parser.add_argument(
        "--outdir",
        help = "Output directory",
        type = str,
        required = True,
    )
    
    parser.add_argument(
        "--blind",
        help = "No observed limits; will append \"_blinded\" to the output directory",
        action = "store_true",
        default = False,
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    #outdir = os.path.dirname(args.output).strip()
    outdir = f"{args.outdir}_blinded" if args.blind else args.outdir
    
    if (len(outdir)) :
        
        os.system(f"mkdir -p {outdir}")
    
    
    arr_xsec_data = numpy.loadtxt(args.xsecfile, delimiter = ",")
    
    #nelements = arr_data.shape[0]
    #arr_mass = numpy.array(arr_data[:, 0], dtype = numpy.float64)
    #arr_xsec = numpy.array(arr_data[:, 1], dtype = numpy.float64)
    
    d_xsecs = {_row[0]: {
        "nom": _row[1],
        "unc_d": _row[2],
        "unc_u": _row[3],
        "d": _row[1] + _row[2],
        "u": _row[1] + _row[3],
    } for _row in arr_xsec_data}
    
    print(d_xsecs)
    
    d_limits = sortedcontainers.SortedDict()
    d_limits_theory_u = sortedcontainers.SortedDict()
    d_limits_theory_d = sortedcontainers.SortedDict()
    
    #for ipoint, injson in enumerate(args.jsons[: 20]) :
    for ipoint, injson in enumerate(args.jsons) :
        
        sig_info = cmut.parse_stau_samplestring(
            s = injson,
            regexp = "Stau_(?P<mstau>\w+)_(?P<ctau>\w+)",
        )
        
        mstau_str = sig_info["mstau"]
        ctau_str = sig_info["ctau"]
        
        # CollectLimits keys its json by the actual mH value used for that point
        mass_key = f"{float(mstau_str)}"
        config = cmut.load_config(injson)
        
        if mass_key not in config :
            
            print(f"Warning: mass key {mass_key} not found in {injson}. Available keys: {list(config.keys())}. Skipping.")
            continue
        
        limits = config[mass_key]
        #print(limits)
        
        # Choose only finite values
        limits = {_key: _val for _key, _val in limits.items() if numpy.isfinite(_val)}
        
        limits["mstau_str"] = mstau_str
        limits["ctau_str"] = ctau_str
        
        mstau = float(mstau_str)
        ctau = float(ctau_str[0: -2].replace("p", "."))
        
        d_limits[(mstau, ctau)] = copy.deepcopy(limits)
        
        d_limits_theory_u[(mstau, ctau)] = copy.deepcopy(limits)
        d_limits_theory_d[(mstau, ctau)] = copy.deepcopy(limits)
        
        if ("obs" in limits) and not args.blind:
            
            d_limits_theory_u[(mstau, ctau)]["obs"] = d_limits_theory_u[(mstau, ctau)]["obs"] * (1.0 + d_xsecs[mstau]["unc_u"]/d_xsecs[mstau]["nom"])
            d_limits_theory_d[(mstau, ctau)]["obs"] = d_limits_theory_d[(mstau, ctau)]["obs"] * (1.0 + d_xsecs[mstau]["unc_d"]/d_xsecs[mstau]["nom"])
        
        #print("XXX", d_limits[(mstau, ctau)]["obs"], d_limits_theory_u[(mstau, ctau)]["obs"], d_limits_theory_d[(mstau, ctau)]["obs"])
    
    
    outfilename_limits = f"{outdir}/limits.root"
    outfile = ROOT.TFile.Open(outfilename_limits, "RECREATE")
    
    # Contours
    results = sortedcontainers.SortedDict()
   
    if not args.blind: 
        results["obs"] = get_exclusion(
            d_limits = d_limits,
            d_xsecs = d_xsecs,
            limit_key = "obs",
            xsec_key = "nom",
            name = "obs",
            axis_label = "Observed UL",
        )
        
        results["obs_p1"] = get_exclusion(
            d_limits = d_limits_theory_d,
            d_xsecs = d_xsecs,
            limit_key = "obs",
            xsec_key = "d",
            name = "obs_p1",
            axis_label = "Observed UL",
        )
        
        results["obs_m1"] = get_exclusion(
            d_limits = d_limits_theory_u,
            d_xsecs = d_xsecs,
            limit_key = "obs",
            xsec_key = "u",
            name = "obs_m1",
            axis_label = "Observed UL",
        )
    else:
        results["obs"] = {"contour": None}
        results["obs_p1"] = {"contour": None}
        results["obs_m1"] = {"contour": None}
    
    results["exp"] = get_exclusion(
        d_limits = d_limits,
        d_xsecs = d_xsecs,
        limit_key = "exp0",
        xsec_key = "nom",
        name = "exp",
        axis_label = "Expected UL at 95% CL",
    )
    
    results["exp_p1"] = get_exclusion(
        d_limits = d_limits,
        d_xsecs = d_xsecs,
        limit_key = "exp+1",
        xsec_key = "nom",
        name = "exp_p1",
        axis_label = "Expected UL (+1#sigma) at 95% CL",
    )
    
    results["exp_m1"] = get_exclusion(
        d_limits = d_limits,
        d_xsecs = d_xsecs,
        limit_key = "exp-1",
        xsec_key = "nom",
        name = "exp_m1",
        axis_label = "Expected UL (-1#sigma) at 95% CL",
    )
    
    results["exp_p2"] = get_exclusion(
        d_limits = d_limits,
        d_xsecs = d_xsecs,
        limit_key = "exp+2",
        xsec_key = "nom",
        name = "exp_p2",
        axis_label = "Expected UL (+2#sigma) at 95% CL",
    )
    
    results["exp_m2"] = get_exclusion(
        d_limits = d_limits,
        d_xsecs = d_xsecs,
        limit_key = "exp-2",
        xsec_key = "nom",
        name = "exp_m2",
        axis_label = "Expected UL (-2#sigma) at 95% CL",
    )
    
    
    # 1D limits
    d_xsecul_per_ctau = sortedcontainers.SortedDict()
    
    for (mstau, ctau), limits in d_limits.items() :
        
        mstau_str = limits["mstau_str"]
        ctau_str = limits["ctau_str"]
        
        ctau_key = (ctau, ctau_str)
        print(mstau_str, ctau_str)
        
        #l_limit_keys = ["obs", "exp0", "exp+1", "exp-1", "exp+2", "exp-2"]
        #skip = False
        #
        #for limit_key in l_limit_keys :
        #    if limit_key not in limits :
        #        
        #        print(f"Warning: could not find limit {limit_key} for mstau={mstau}, ctau={ctau}. Skipping this signal point.")
        #        skip = True
        #        break
        #
        #if skip :
        #    continue
        
        if ctau_key not in d_xsecul_per_ctau :
            
            d_xsecul_per_ctau[ctau_key] = {
                "g1_xsec_theory": ROOT.TGraphAsymmErrors(),
                
                "g1_xsecul_obs": ROOT.TGraph(),
                "g1_xsecul_exp": ROOT.TGraph(),
                "g1_xsecul_exp_p1": ROOT.TGraph(),
                "g1_xsecul_exp_m1": ROOT.TGraph(),
                "g1_xsecul_exp_p2": ROOT.TGraph(),
                "g1_xsecul_exp_m2": ROOT.TGraph(),
                "g1_xsecul_exp_pm1": ROOT.TGraphAsymmErrors(),
                "g1_xsecul_exp_pm2": ROOT.TGraphAsymmErrors(),
                
                "g1_rul_obs": ROOT.TGraph(),
                "g1_rul_exp": ROOT.TGraph(),
                "g1_rul_exp_p1": ROOT.TGraph(),
                "g1_rul_exp_m1": ROOT.TGraph(),
                "g1_rul_exp_p2": ROOT.TGraph(),
                "g1_rul_exp_m2": ROOT.TGraph(),
                "g1_rul_exp_pm1": ROOT.TGraphAsymmErrors(),
                "g1_rul_exp_pm2": ROOT.TGraphAsymmErrors(),
            }
            
            for key, val in d_xsecul_per_ctau[ctau_key].items() :
                
                val.SetName(f"{key}_{ctau_str}")
                val.SetTitle(f"{key}_{ctau_str}")
        
        xsec = d_xsecs[mstau]["nom"]
        xsec_u = abs(d_xsecs[mstau]["unc_u"])
        xsec_d = abs(d_xsecs[mstau]["unc_d"])
        #ipoint = d_xsecul_per_ctau[ctau_key]["g1_xsecul_exp"].GetN()
        
        cmut.logger.info(f"Filling xsec graph with: mstau({mstau}), xsec({xsec}+{xsec_u}-{xsec_d})")
        
        ipoint = d_xsecul_per_ctau[ctau_key]["g1_xsec_theory"].GetN()
        d_xsecul_per_ctau[ctau_key]["g1_xsec_theory"].SetPoint(ipoint, mstau, xsec)
        d_xsecul_per_ctau[ctau_key]["g1_xsec_theory"].SetPointError(
            ipoint,
            0.0, 0.0,
            xsec_d, xsec_u
        )
        
        if ("obs" in limits) :
            ipoint = d_xsecul_per_ctau[ctau_key]["g1_xsecul_obs"].GetN()
            
            d_xsecul_per_ctau[ctau_key]["g1_xsecul_obs"].SetPoint(ipoint, mstau, limits["obs"]*xsec)
            d_xsecul_per_ctau[ctau_key]["g1_rul_obs"].SetPoint(ipoint, mstau, limits["obs"])
        
        if ("exp0" in limits) :
            ipoint = d_xsecul_per_ctau[ctau_key]["g1_xsecul_exp"].GetN()
            
            d_xsecul_per_ctau[ctau_key]["g1_xsecul_exp"].SetPoint(ipoint, mstau, limits["exp0"]*xsec)
            d_xsecul_per_ctau[ctau_key]["g1_rul_exp"].SetPoint(ipoint, mstau, limits["exp0"])
            
            if ("exp+1" in limits and "exp-1" in limits) :
                ipoint = d_xsecul_per_ctau[ctau_key]["g1_xsecul_exp_pm1"].GetN()
                
                d_xsecul_per_ctau[ctau_key]["g1_xsecul_exp_pm1"].SetPoint(ipoint, mstau, limits["exp0"]*xsec)
                d_xsecul_per_ctau[ctau_key]["g1_xsecul_exp_pm1"].SetPointError(
                    ipoint,
                    0.0, 0.0,
                    abs(limits["exp-1"]-limits["exp0"])*xsec,
                    abs(limits["exp+1"]-limits["exp0"])*xsec
                )
                
                d_xsecul_per_ctau[ctau_key]["g1_rul_exp_pm1"].SetPoint(ipoint, mstau, limits["exp0"])
                d_xsecul_per_ctau[ctau_key]["g1_rul_exp_pm1"].SetPointError(
                    ipoint,
                    0.0, 0.0,
                    abs(limits["exp-1"]-limits["exp0"]),
                    abs(limits["exp+1"]-limits["exp0"])
                )
            
            if ("exp+2" in limits and "exp-2" in limits) :
                ipoint = d_xsecul_per_ctau[ctau_key]["g1_xsecul_exp_pm2"].GetN()
                
                d_xsecul_per_ctau[ctau_key]["g1_xsecul_exp_pm2"].SetPoint(ipoint, mstau, limits["exp0"]*xsec)
                d_xsecul_per_ctau[ctau_key]["g1_xsecul_exp_pm2"].SetPointError(
                    ipoint,
                    0.0, 0.0,
                    abs(limits["exp-2"]-limits["exp0"])*xsec,
                    abs(limits["exp+2"]-limits["exp0"])*xsec
                )
                
                d_xsecul_per_ctau[ctau_key]["g1_rul_exp_pm2"].SetPoint(ipoint, mstau, limits["exp0"])
                d_xsecul_per_ctau[ctau_key]["g1_rul_exp_pm2"].SetPointError(
                    ipoint,
                    0.0, 0.0,
                    abs(limits["exp-2"]-limits["exp0"]),
                    abs(limits["exp+2"]-limits["exp0"])
                )
        
        if ("exp+1" in limits) :
            d_xsecul_per_ctau[ctau_key]["g1_xsecul_exp_p1"].AddPoint(mstau, limits["exp+1"]*xsec)
            d_xsecul_per_ctau[ctau_key]["g1_rul_exp_p1"].AddPoint(mstau, limits["exp+1"])
        if ("exp-1" in limits) :
            d_xsecul_per_ctau[ctau_key]["g1_xsecul_exp_m1"].AddPoint(mstau, limits["exp-1"]*xsec)
            d_xsecul_per_ctau[ctau_key]["g1_rul_exp_m1"].AddPoint(mstau, limits["exp-1"])
        if ("exp+2" in limits) :
            d_xsecul_per_ctau[ctau_key]["g1_xsecul_exp_p2"].AddPoint(mstau, limits["exp+2"]*xsec)
            d_xsecul_per_ctau[ctau_key]["g1_rul_exp_p2"].AddPoint(mstau, limits["exp+2"])
        if ("exp-2" in limits) :
            d_xsecul_per_ctau[ctau_key]["g1_xsecul_exp_m2"].AddPoint(mstau, limits["exp-2"]*xsec)
            d_xsecul_per_ctau[ctau_key]["g1_rul_exp_m2"].AddPoint(mstau, limits["exp-2"])

    print(f"Writing to: {outfilename_limits}")
    outfile.cd()
    
    for result_key, result in results.items() :
        
        outfile.cd()
        outfile.mkdir(result_key)
        outfile.cd(result_key)
        
        for key, val in result.items() :
            
            if (val is None) :
                
                continue
            
            print(f"Writing [{result_key}] [{key}]")
            val.Write()
    
    
    # 2D exclusions
    canvas_name = "limits-vs-ctau0-mstau"
    
    CMS.setCMSStyle()
    CMS.SetExtraText(args.cmsextratext)
    CMS.SetLumi(324.75, "fb", "Run 3", 1)
    CMS.SetEnergy(13.6)
    CMS.SetCMSPalette()
    CMS.ResetAdditionalInfo()
    CMS.getCMSStyle().SetNumberContours(NZCONTS)
    #CMS.getCMSStyle().SetPadTickX(0)
    
    ymin = 1
    ymax = max([
        max(numpy.array(results[_key]["contour"].GetY()))
        if results[_key]["contour"] else ymin
        for _key in ["exp", "exp_m1", "exp_p1"]
    ])
    #ymax = (round(ymax/100)+4)*100
    #ymax = (round(ymax/100)+4)*1000
    #ymax = 10**(numpy.log10(ymax)+2)
    ymax = 10**5
    
    canvas = CMS.cmsCanvas(
        canvName = canvas_name,
        x_min = 90, x_max = 510,
        y_min = ymin, y_max = ymax,
        nameXaxis = "m_{#tilde{#tau}} [GeV]",
        #nameYaxis = "c#tau_{0}(#tilde{#tau}) [mm]",
        nameYaxis = "c#tau_{0} [mm]",
        #square = CMS.kSquare,
        square = False,
        iPos = 0,
        extraSpace = 0.03,
        with_z_axis = True,
        scaleLumi = 1.0,
        yTitOffset = 1.05
    )
    
    hframe = canvas.FindObject("hframe")
    #hframe = CMS.GetcmsCanvasHist(canvas)
    
    canvas.SetRightMargin(0.175)
    #CMS.SetLumi(324.75, round_lumi = True)
    #CMS.SetEnergy(13)
    #CMS.CMS_lumi(canvas, iPosX = 0)
    CMS.UpdatePad(canvas)
    
    h2_colz = results["exp"]["h2_xsecul_interp"] if args.blind else results["obs"]["h2_xsecul_interp"]
    
    zmin = 10**-1
    #zmin = h2_colz.GetMinimum()
    #zmin = 10**(round(numpy.log10(zmin))-1)
    
    zmax = 10**3
    #zmax = h2_colz.GetMaximum()
    #zmax = 10**(round(numpy.log10(zmax))+1)
    
    hframe.GetXaxis().CenterTitle(True)
    #hframe.GetXaxis().SetTicks("-")
    hframe.GetYaxis().CenterTitle(True)
    
    #h2_colz.GetXaxis().SetTicks("-")
    h2_colz.GetZaxis().SetRangeUser(zmin, zmax)
    h2_colz.GetZaxis().CenterTitle(True)
    h2_colz.GetZaxis().SetTitle("95% CL upper limit on cross section [fb]")
    h2_colz.GetZaxis().SetTitleSize(0.05)
    h2_colz.GetZaxis().SetTitleOffset(1.2)
    h2_colz.GetZaxis().SetLabelSize(0.05)
    
    legend = CMS.cmsLeg(canvas.GetLeftMargin(), 0.61, 1-canvas.GetRightMargin(), 1-canvas.GetTopMargin(), textSize = 0.0425, columns = 2)
    #legend = ROOT.TLegend(canvas.GetLeftMargin(), 0.65, 1-canvas.GetRightMargin(), 1-canvas.GetTopMargin())
    legend.SetFillColor(0)
    legend.SetFillStyle(1001)
    legend.SetBorderSize(1)
    #legend.SetMargin(0.25)
    #legend.SetTextAlign(22)
    legend.SetTextAlign(12)
    
    CMS.cmsDraw(h2_colz, "colz")
    
    if not args.blind :
        
        legend.AddEntry(0, "", "")
        legend.AddEntry(0, "", "")
        
        CMS.cmsDraw(results["obs"]["contour"], "sameL", lstyle = ROOT.kSolid, lcolor = ROOT.kBlack, lwidth = 3)
        legend.AddEntry(results["obs"]["contour"], "Observed", "L")
        
        CMS.cmsDraw(results["obs_p1"]["contour"], "sameL", lstyle = ROOT.kDashed, lcolor = ROOT.kBlack, lwidth = 3)
        CMS.cmsDraw(results["obs_m1"]["contour"], "sameL", lstyle = ROOT.kDashed, lcolor = ROOT.kBlack, lwidth = 3)
        
        legend.AddEntry(results["obs_p1"]["contour"], f"Observed{ROOTSPACE_0P4}#pm 1#sigma_{{theory}}", "L")
    
    CMS.cmsDraw(results["exp"]["contour"], "sameL", lstyle = ROOT.kSolid, lcolor = ROOT.kRed+1, lwidth = 3)
    legend.AddEntry(results["exp"]["contour"], "Expected", "L")
    
    #colors = [ROOT.kBlue, ROOT.kGreen+2, ROOT.kMagenta, ROOT.kBlack]
    #for i in range(len(results["exp"]["contours"])):
    #    seg = results["exp"]["contours"][i]
    #    CMS.cmsDraw(seg, "sameL", lstyle = ROOT.kSolid, lcolor =colors[i % len(colors)], lwidth = 3)

    if (results["exp_p1"]["contour"] and results["exp_m1"]["contour"]) :
        
        CMS.cmsDraw(results["exp_p1"]["contour"], "sameL", lstyle = ROOT.kDashed, lcolor = ROOT.kRed+1, lwidth = 3)
        CMS.cmsDraw(results["exp_m1"]["contour"], "sameL", lstyle = ROOT.kDashed, lcolor = ROOT.kRed+1, lwidth = 3)
        
        legend.AddEntry(results["exp_p1"]["contour"], f"Expected{ROOTSPACE_0P4}#pm 1#sigma_{{experiment}}", "L")
    #    for i in range(len(results["exp_p1"]["contours"])):
    #        seg = results["exp_p1"]["contours"][i]
    #        CMS.cmsDraw(seg, "sameL", lstyle = ROOT.kDashed, lcolor =colors[i % len(colors)], lwidth = 3)
    #    for i in range(len(results["exp_m1"]["contours"])):
    #        seg = results["exp_m1"]["contours"][i]
    #        CMS.cmsDraw(seg, "sameL", lstyle = ROOT.kDashed, lcolor =colors[i % len(colors)], lwidth = 3)
    
    sign_exp_pm2 = ""
    cont_legend = None
    
    #if (results["exp_p2"]["contour"]) :
    #    
    #    sign_exp_pm2 += "#plus"
    #    cont_legend = results["exp_p2"]["contour"]
    #    CMS.cmsDraw(results["exp_p2"]["contour"], "sameL", lstyle = ROOT.kDotted, lcolor = ROOT.kRed+1, lwidth = 3)
    #    
    #    #legend.AddEntry(0, "", "")
    #    #legend.AddEntry(results["exp_p2"]["contour"], f"Expected{ROOTSPACE_0P4}#pm 2#sigma_{{experiment}}", "L")
    #
    #if (results["exp_m2"]["contour"]) :
    #    
    #    sign_exp_pm2 += "#minus"
    #    cont_legend = results["exp_m2"]["contour"]
    #    CMS.cmsDraw(results["exp_m2"]["contour"], "sameL", lstyle = ROOT.kDotted, lcolor = ROOT.kRed+1, lwidth = 3)
    #    
    #    #legend.AddEntry(0, "", "")
    #    #legend.AddEntry(results["exp_p2"]["contour"], f"Expected{ROOTSPACE_0P4}#pm 2#sigma_{{experiment}}", "L")
    #
    #if sign_exp_pm2 :
    #    
    #    sign_exp_pm2 = "#pm" if (sign_exp_pm2 == "#plus#minus") else sign_exp_pm2
    #    
    #    legend.AddEntry(0, "", "")
    #    legend.AddEntry(cont_legend, f"Expected{ROOTSPACE_0P4}{sign_exp_pm2} 2#sigma_{{experiment}}", "L")
    
    
    hframe.GetXaxis().SetNdivisions(6, 5, 0)
    
    canvas.SetLogy()
    
    label = (
        "pp#rightarrow#tilde{#tau}#bar{#tilde{#tau}} , "
        "#tilde{#tau}#rightarrow#tau#tilde{G} , "
        "m_{#tilde{G}} = 1 GeV"
    )
    
    if args.extratext :
        
        label = f"#splitline{{{args.extratext}}}{{{label}}}"
    
    label = f"#lower[0.2]{{{label}}}"
    
    CMS.cmsHeader(
        leg = legend,
        legTitle = label,
        textAlign = 22,
        textSize = 0.0425,
        textFont = 62,
        #textColor = ROOT.kBlack,
        isToRemove = True
    )
    
    #CMS.getCMSStyle().SetPadTickX(0)
    canvas.RedrawAxis()
    #legend.Draw()
    #canvas.Update()
    
    canvas.SetLogz()
    
    legend.Draw()
    
    CMS.UpdatePad(canvas)
    
    outfile.cd()
    canvas.Write()
    
    canvas_outfile = f"{outdir}/{canvas_name}"
    CMS.SaveCanvas(canvas, f"{canvas_outfile}.pdf")
    #os.system(f"pdftoppm -cropbox -r 600 -png -singlefile {canvas_outfile}.pdf {canvas_outfile}")
    cmut.pdf_to_png(f"{canvas_outfile}.pdf")
    
    
    for (ctau, ctau_str), d_xsecul in d_xsecul_per_ctau.items() :
        
        outfile.cd()
        outfile.mkdir(ctau_str)
        outfile.cd(ctau_str)
        
        for key, val in d_xsecul.items() :
            
            print(f"Writing [{ctau_str}] [{key}]")
            #print(numpy.array(val.GetY(), dtype = numpy.float64))
            val.Write()
        
        g1_xsec_theory = d_xsecul["g1_xsec_theory"]
        g1_xsecul_obs = d_xsecul["g1_xsecul_obs"]
        g1_xsecul_exp = d_xsecul["g1_xsecul_exp"]
        g1_xsecul_exp_pm1 = d_xsecul["g1_xsecul_exp_pm1"]
        g1_xsecul_exp_pm2 = d_xsecul["g1_xsecul_exp_pm2"]
        
        xmin = 50
        xmax = 750
        
        #ymin = min([_ele.GetHistogram().GetMinimum() for _ele in d_xsecul.values()])
        #ymin = g1_xsec_theory.GetHistogram().GetMinimum()
        ymin = min(numpy.array(g1_xsec_theory.GetY()))
        ymin = 10**(round(numpy.log10(ymin))-2)
        #ymin = 10**-2
        
        #ymax = max([_ele.GetHistogram().GetMaximum() for _ele in d_xsecul.values()])
        #ymax = g1_xsec_theory.GetHistogram().GetMaximum()
        ymax = max(numpy.array(g1_xsec_theory.GetY()))
        ymax = 10**(round(numpy.log10(ymax))+2)
        
        canvas_name = f"limits-vs-mstau_{ctau_str}"
        
        CMS.setCMSStyle()
        CMS.SetExtraText(args.cmsextratext)
        CMS.SetLumi(324.75, "fb", "Run 3", 1)
        CMS.SetEnergy(13.6)
        CMS.ResetAdditionalInfo()
        #CMS.getCMSStyle().SetPadTickX(0)
        
        canvas = CMS.cmsCanvas(
            canvName = canvas_name,
            x_min = xmin, x_max = xmax,
            y_min = ymin, y_max = ymax,
            nameXaxis = "m_{#tilde{#tau}} [GeV]",
            nameYaxis = "Cross section [fb]",
            square = True,
            iPos = 0,
            extraSpace = 0.03,
            with_z_axis = False,
            scaleLumi = 1.0,
            yTitOffset = 1.25
        )
        
        CMS.cmsDraw(g1_xsecul_exp_pm2, "3L", fcolor = ROOT.TColor.GetColor("#85D1FBff"))
        CMS.cmsDraw(g1_xsecul_exp_pm1, "same3L", fcolor = ROOT.TColor.GetColor("#FFDF7Fff"))
        CMS.cmsDraw(g1_xsecul_exp, "sameL", lstyle = ROOT.kDashed, lcolor = ROOT.kBlack, lwidth = 3)
        
        if not args.blind :
            CMS.cmsDraw(g1_xsecul_obs, "sameL", lstyle = ROOT.kSolid, lcolor = ROOT.kBlack, lwidth = 3)
        
        CMS.cmsDraw(g1_xsec_theory, "same3L", lcolor = ROOT.kRed, lwidth = 2, fcolor = ROOT.kRed, alpha = 0.33)
        #CMS.cmsDraw(g1_xsecul_obs, "LP")
        
        g1_xsecul_exp_pm1.SetLineWidth(0)
        g1_xsecul_exp_pm2.SetLineWidth(0)
        
        legend = CMS.cmsLeg(0.45, 0.60, 0.95, 0.90, textSize = 0.04, columns = 1)
        #leg.AddEntry(g1_xsecul_obs, "Observed", "LP")
        legend.AddEntry(g1_xsec_theory, "Theory (NLO+NLL)", "LF")
        legend.AddEntry(0, "#kern[-0.3]{#bf{95% CL upper limits}}", "")
        
        if not args.blind :
            legend.AddEntry(g1_xsecul_obs, "Observed", "L")
        
        legend.AddEntry(g1_xsecul_exp, "Expected", "L")
        legend.AddEntry(g1_xsecul_exp_pm1, f"Expected{ROOTSPACE_0P4}#pm 1#sigma_{{experiment}}", "F")
        legend.AddEntry(g1_xsecul_exp_pm2, f"Expected{ROOTSPACE_0P4}#pm 2#sigma_{{experiment}}", "F")
        
        canvas.SetLogy()
        
        label = (
            "pp#rightarrow#tilde{#tau}#bar{#tilde{#tau}} , "
            "#tilde{#tau}#rightarrow#tau#tilde{G} , "
            "m_{#tilde{G}} = 1 GeV , "
            #"c#tau_{0}(#tilde{#tau}) = " + ctau_str[0:-2].replace("p", ".") + " mm"
            "c#tau_{0} = " + ctau_str[0:-2].replace("p", ".") + " mm"
        )
        
        if args.extratext :
            
            label = f"#splitline{{{args.extratext}}}{{{label}}}"
        
        model_text = ROOT.TLatex()
        model_text.SetTextSize(0.04)
        model_text.DrawLatex(100, 5*ymin, label)
        
        canvas.Write()
        #canvas.SaveAs(f"{outdir}/{canvas_name}.pdf", close = False)
        #canvas.SaveAs(f"{outdir}/{canvas_name}.png")
        
        canvas_outfile = f"{outdir}/{canvas_name}"
        CMS.SaveCanvas(canvas, f"{canvas_outfile}.pdf")
        
        # ROOT fucks up when saving as png
        # Convert from the pdf instead
        #os.system(f"pdftoppm -cropbox -r 600 -png -singlefile {canvas_outfile}.pdf {canvas_outfile}")
        cmut.pdf_to_png(f"{canvas_outfile}.pdf")
    
    return 0


if (__name__ == "__main__") :
    
    main()

