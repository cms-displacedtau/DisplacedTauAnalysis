from coffea.util import load
from matplotlib.ticker import MultipleLocator
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import hist, pdb
from hist import Hist, intervals 
import itertools

plot_dir = 'eff_vs_dxy'


def get_ratio_histogram(passing_probes, denominator):
    """Get the ratio (efficiency) of the passing over passing + failing probes.
    NaN values are replaced with 0.

    Parameters
    ----------
        passing_probes : hist.Hist
            The histogram of the passing probes.
         : hist.Hist
            The histogram of the denominator.

    Returns
    -------
        ratio : hist.Hist
            The ratio histogram.
        yerr : numpy.ndarray
            The y error of the ratio histogram.
    """

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio_values = passing_probes.values(flow=True) / denominator.values(flow=True)
    ratio = hist.Hist(hist.Hist(*passing_probes.axes))
    ratio[:] = np.nan_to_num(ratio_values)
    yerr = intervals.ratio_uncertainty(passing_probes.values(), denominator.values(), uncertainty_type="efficiency")

    return ratio, yerr


def plot_efficiency(passing_probes, denominator, log=False, **kwargs):
    """Plot the efficiency using the ratio of passing to passing + failing probes.

    Parameters
    ----------
        passing_probes : hist.Hist
            The histogram of the passing probes.
        denominator : hist.Hist
            The histogram of the denominator
        **kwargs
            Keyword arguments to pass to hist.Hist.plot1d.

    Returns
    -------
        List[Hist1DArtists]

    """
    ratio_hist, yerr = get_ratio_histogram(passing_probes, denominator)
    plt.ylabel('efficiency')
    if log:  plt.xscale('log')
    return ratio_hist.plot1d(histtype="errorbar", yerr=yerr, xerr=True, flow="none", **kwargs)

def clear_and_set_name(sample_name):
    plt.clf()
    fig, ax = plt.subplots()
    ax.title.set_text(sample_name)
    return fig,ax

def set_common_labels_scale_legend_grid(ax):
    ax.set_ylabel("events")
    plt.yscale('log')
    plt.legend()
    plt.grid(True)


decayM = 0
min_gen_tau_pt = 20
maxLxy = 100


dr_thresholds = [ 0.3 ]
# dr_thresholds = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
pion_pt_thresholds = [0]
# pion_pt_thresholds = [0, 1, 2, 5, 10]
    
for min_gen_pion_pt,delta_r_matching_threshold in itertools.product(pion_pt_thresholds, dr_thresholds):

    delta_r_matching_threshold_string = str(delta_r_matching_threshold).replace('.','p')
#     tag = f'_dR{delta_r_matching_threshold_string}_MaxLxy{maxLxy}_decayM{decayM}_pionPt{min_gen_pion_pt}_genTauPt{min_gen_tau_pt}_noTaggerScore_goodRecoProp'
    tag = f'_dR{delta_r_matching_threshold_string}_MaxLxy{maxLxy}_decayM{decayM}_pf{min_gen_pion_pt}_genPionPt0_genTauPt{min_gen_tau_pt}_noTaggerScore_goodRecoProp'
    
    for sample_name in  [
    #                      'Stau_300_1mm',
                             'Stau_300_100mm',
    #                          'Stau_300_1000mm',
    #                          'Stau_100_100mm',
    #                          'Stau_100_1000mm',
    #                          'Stau_500_1000mm'
                            ]:
    
        sample_label = sample_name #'m = 300 GeV, ctau = 1000 mm'
        print(f"#### Running on deltaR threshold = {delta_r_matching_threshold} and pion pT {min_gen_pion_pt} ####")
        
    #     input_file = load(f"inputs_study_propagation/histograms_{sample_name}_dR{delta_r_matching_threshold_string}_MaxLxy{maxLxy}_decayM{decayM}_pionPt{min_gen_pion_pt}_genTauPt{min_gen_tau_pt}_noTaggerScore.coffea")
#         input_file = load(f"inputs_study_propagation/histograms_{sample_name}_dR{delta_r_matching_threshold_string}_MaxLxy{maxLxy}_decayM{decayM}_pionPt{min_gen_pion_pt}_genTauPt{min_gen_tau_pt}_noTaggerScore_goodRecoProp.coffea")
        input_file = load(f"inputs_study_propagation/histograms_{sample_name}_dR{delta_r_matching_threshold_string}_MaxLxy{maxLxy}_decayM{decayM}_pfPt{min_gen_pion_pt}_genPionPt0_genTauPt{min_gen_tau_pt}_noTaggerScore_goodRecoProp.coffea")
       
        h_den_dxy = input_file["h_den_dxy"]
        h_num_dxy = input_file["h_num_dxy"]
        h_num_ecal_dxy = input_file["h_num_ecal_dxy"]
    
        h_fail_dxy = input_file["h_fail_dxy"]
        h_pass_dxy = input_file["h_pass_dxy"]
    
        h_den_dxy_zoom = input_file["h_den_dxy_zoom"]
        h_num_dxy_zoom = input_file["h_num_dxy_zoom"]
        h_num_ecal_dxy_zoom = input_file["h_num_ecal_dxy_zoom"]
    
        h_den_pt = input_file["h_den_pt"]
        h_num_pt = input_file["h_num_pt"]
        h_num_ecal_pt = input_file["h_num_ecal_pt"]
        h_num_jet_pion_pt = input_file["h_num_jet_pion_pt"]
        h_num_jet_gvt_pt = input_file["h_num_jet_gvt_pt"]
        h_num_jet_dxy = input_file["h_num_jet_dxy"]
        
        h_eta_ecal_vs_dxy = input_file['h_eta_ecal_vs_dxy']
        h_eta_ecal_vs_pt = input_file['h_eta_ecal_vs_pt']
        h_eta_ecal_vs_eta = input_file['h_eta_ecal_vs_eta']
        h_eta_ecal_vs_lxy = input_file['h_eta_ecal_vs_lxy']
        h_eta_ecal_vs_tau_eta  = input_file['h_eta_ecal_vs_tau_eta']
        h_dxy_vs_lxy = input_file['h_dxy_vs_lxy']
        h_fail_pt_vs_eta = input_file['h_fail_pt_vs_eta']
    
        h_gen_lxy = input_file["h_gen_lxy"]
        h_gen_tau_pt = input_file["h_gen_tau_pt"]
        
    #     h_reco_lxy = input_file["h_reco_lxy"]
    #     h_reco_ecal_lxy = input_file["h_reco_ecal_lxy"]
    
        h_deltaR_ecal_zoomout = input_file["h_deltaR_ecal_zoomout"]
        h_deltaR_ecal = input_file["h_deltaR_ecal"]
        h_deltaR = input_file["h_deltaR"]
        h_delta = input_file["h_delta"]
        
        h_allgenpions_eta_ecal_vs_eta = input_file["h_allgenpions_eta_ecal_vs_eta"]
        h_gen_pt = input_file["h_gen_pt"]
        h_gen_pt_zoom = input_file["h_gen_pt_zoom"]
        h_reco_pt = input_file["h_reco_pt"]
        h_reco_pt_zoom = input_file["h_reco_pt_zoom"]
    
        h_deta_bump = input_file["h_deta_bump"]
        h_dphi_bump = input_file["h_dphi_bump"]
        h_deta_ecal_bump = input_file["h_deta_ecal_bump"]
        h_dphi_ecal_bump = input_file["h_dphi_ecal_bump"]
        h_dpt_bump = input_file["h_dpt_bump"]
        h_dpt_all = input_file["h_dpt_all"]
        h_dcharge_bump = input_file["h_dcharge_bump"]
        h_dcharge_all = input_file["h_dcharge_all"]
        h_lxy_bump = input_file["h_lxy_bump"]
        h_lxy_all = input_file["h_lxy_all"]
    
        h_phi_ecal_vs_phi_bump = input_file["h_phi_ecal_vs_phi_bump"]
        h_genphi_ecal_vs_genphi_bump = input_file["h_genphi_ecal_vs_genphi_bump"]
        h_pdgid_bump = input_file["h_pdgid_bump"]
        h_geneta_bump = input_file["h_geneta_bump"]
    
        h_phi_ecal_vs_phi_worse = input_file["h_phi_ecal_vs_phi_worse"]
        h_genphi_ecal_vs_genphi_worse = input_file["h_genphi_ecal_vs_genphi_worse"]
        h_deta_worse = input_file["h_deta_worse"]
        h_dphi_worse = input_file["h_dphi_worse"]
        h_deta_ecal_worse = input_file["h_deta_ecal_worse"]
        h_dphi_ecal_worse = input_file["h_dphi_ecal_worse"]
        h_dpt_worse = input_file["h_dpt_worse"]
        h_dcharge_worse = input_file["h_dcharge_worse"]
        h_lxy_worse = input_file["h_lxy_worse"]
        h_pt_vs_genpt_worse = input_file["h_pt_vs_genpt_worse"]
        h_dxy_vs_gendxy_worse = input_file["h_dxy_vs_gendxy_worse"]

        h_num_gvt_pt_for_alljet = input_file["h_num_gvt_pt_for_alljet"]
        h_den_gvt_pt_for_alljet = input_file["h_den_gvt_pt_for_alljet"]
        
        h_num_gvt_pt = input_file["h_num_gvt_pt"]
        h_num_ecal_gvt_pt = input_file["h_num_ecal_gvt_pt"]
        h_den_gvt_pt = input_file["h_den_gvt_pt"]
         
    



        fig, ax  = clear_and_set_name(sample_name)
        h_gen_pt.plot()
        plt.savefig(f"{plot_dir}/pt_genpions_{sample_name}{tag}.pdf")
        plt.close()
    
        fig, ax  = clear_and_set_name(sample_name)
        h_gen_pt_zoom.plot()
        plt.savefig(f"{plot_dir}/pt_zoom_genpions_{sample_name}{tag}.pdf")
        plt.close()
    
        fig, ax  = clear_and_set_name(sample_name)
        h_reco_pt.plot()
        plt.savefig(f"{plot_dir}/pt_recopions_{sample_name}{tag}.pdf")
        plt.close()
    
        fig, ax  = clear_and_set_name(sample_name)
        h_reco_pt_zoom.plot()
        plt.savefig(f"{plot_dir}/pt_zoom_recopions_{sample_name}{tag}.pdf")
        plt.close()
        
        fig, ax  = clear_and_set_name(sample_name)
        h_gen_tau_pt.plot()
        plt.savefig(f"{plot_dir}/pt_genvistau_{sample_name}{tag}.pdf")
        plt.close()
        
    
        plt.clf()
        split = 3  # number of sentinel bins before the linspace(-3,3) region starts
        fig, (ax_lo, ax_hi) = plt.subplots(
            1, 2, figsize=(9, 5), sharey=True,
            gridspec_kw={"width_ratios": [1, 6], "wspace": 0.06}
        )
    
        x_edges = h_allgenpions_eta_ecal_vs_eta.axes[0].edges   # gen_eta_at_ecal, variable width
        y_edges = h_allgenpions_eta_ecal_vs_eta.axes[1].edges   # gen_eta
        values = h_allgenpions_eta_ecal_vs_eta.values()         # shape (n_x, n_y)
        cmap = plt.cm.viridis.copy(); cmap.set_under("white")
        norm = mcolors.Normalize(vmin=1e-9, vmax=values.max())
        ax_lo.pcolormesh(x_edges[:split+1], y_edges, values[:split].T, cmap=cmap, norm=norm)
        ax_lo.set_xlim(x_edges[0], x_edges[split])
        ax_lo.set_ylabel("gen_eta")
        ax_lo.set_xlabel("fail flags")
        mesh = ax_hi.pcolormesh(x_edges[split:], y_edges, values[split:].T, cmap=cmap, norm=norm)
        ax_hi.set_xlim(-3, 3)
        ax_hi.set_xlabel("gen_eta_at_ecal")
        ax_hi.tick_params(labelleft=False)
        fig.colorbar(mesh, ax=ax_hi, label="entries")
        plt.savefig(f"{plot_dir}/2D_allgenpions_eta_ecal_vs_eta_broken_axis{sample_name}{tag}.pdf")
    
    #     exit(0)
    
#         fig, ax  = clear_and_set_name(sample_name)
#         plot_efficiency(h_num_dxy, 
#                         h_den_dxy, 
#                         label='deltaR')
#         plot_efficiency(h_num_ecal_dxy, 
#                         h_den_dxy, 
#                         label='deltaR ecal')
#         ax.set_ylabel("Matched GVT/Total GVT")
#         ax.set_xlabel("GVT_dxy [cm]")
#         ax.set_title(sample_label)
#         ax.set_ylim(0,1)
#         plt.legend()
#         plt.grid(True)
#         plt.savefig(f"{plot_dir}/eff_dxy_{sample_name}{tag}.pdf")
#         plt.close()
    
        fig, ax  = clear_and_set_name(sample_name)
        plot_efficiency(h_num_dxy_zoom, 
                        h_den_dxy_zoom, 
                        label='deltaR')
        plot_efficiency(h_num_ecal_dxy_zoom, 
                        h_den_dxy_zoom, 
                        label='deltaR ecal')
        plot_efficiency(h_num_jet_dxy, 
                        h_den_dxy_zoom, 
                        label='jet matching')
    
        ax.set_ylabel("Matched GVT/Total GVT")
        ax.set_xlabel("leading pion dxy [cm]")
        ax.set_ylim(0,1)
        ax.set_title(sample_label)
        plt.legend()
        plt.grid(True)
#         plt.savefig(f"{plot_dir}/eff_dxy_zoom_{sample_name}{tag}.pdf")
        plt.xscale('log')
        plt.savefig(f"{plot_dir}/eff_dxy_zoom_{sample_name}{tag}_log.pdf")
        plt.close()
    
    
        fig, ax  = clear_and_set_name(sample_name)
        plot_efficiency(h_num_pt, 
                        h_den_pt, 
                        label='deltaR')
        plot_efficiency(h_num_ecal_pt, 
                        h_den_pt, 
                        label='deltaR ecal')
        plot_efficiency(h_num_jet_pion_pt, 
                        h_den_pt, 
                        label='jet matching')
        ax.set_ylabel("Matched GVT/Total GVT")
        ax.set_xlabel("leading pion pt [GeV]")
        ax.set_title(sample_label)
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{plot_dir}/eff_pt_{sample_name}{tag}.pdf")
        plt.close()
        
        
        fig, ax  = clear_and_set_name(sample_name)
        plot_efficiency(h_num_gvt_pt, 
                        h_den_gvt_pt,
                        label='delta R')
        plot_efficiency(h_num_ecal_gvt_pt, 
                        h_den_gvt_pt,
                        label='delta R ecal')
        plot_efficiency(h_num_gvt_pt_for_alljet, 
                        h_den_gvt_pt_for_alljet, 
                        label='all jets')
                        
        ax.set_ylabel("Matched GVT/Total GVT")
        ax.set_xlabel("GVT pt [GeV]")
        ax.set_title(sample_label)
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{plot_dir}/eff_gvt_pt_{sample_name}{tag}.pdf")
        plt.close()

        
    
    
        fig, ax  = clear_and_set_name(sample_name)
        h_deltaR_ecal.plot(ax=ax, label = 'deltaR ecal')
        h_deltaR.plot(ax=ax, label = 'deltaR')
        print (sample_label, ' ->  n entries in deltaR: ', h_deltaR.sum())
        set_common_labels_scale_legend_grid(ax)
        ax.set_xlabel("deltaR")
        plt.savefig(f"{plot_dir}/deltaR_{sample_name}{tag}.pdf")
        plt.close()
    
        fig, ax  = clear_and_set_name(sample_name)
        print (sample_label, ' ->  n entries in deltaR OUT: ', h_deltaR_ecal_zoomout.sum())
        h_deltaR_ecal_zoomout.plot(ax=ax, label = 'deltaR ecal')
        ax.set_ylabel("events")
        ax.set_xlabel("deltaR")
    #     plt.yscale('log')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{plot_dir}/deltaR_out_{sample_name}{tag}.pdf")
        plt.close()
       
        import matplotlib.colors as colors
    
        fig, ax  = clear_and_set_name(sample_name)
        h_delta.plot(    norm=colors.LogNorm(vmin=1, vmax=10000))  
        plt.savefig(f"{plot_dir}/2D_deltaR_{sample_name}{tag}.pdf")
        plt.close()
    
    
        ### study events with large deltaR ecal
        fig, ax  = clear_and_set_name(sample_name)
        h_phi_ecal_vs_phi_worse.plot()
        plt.savefig(f"{plot_dir}/worse_2D_reco_phi_{sample_name}{tag}.pdf")
        plt.close()
    
        fig, ax  = clear_and_set_name(sample_name)
        h_genphi_ecal_vs_genphi_worse.plot()
        plt.savefig(f"{plot_dir}/worse_2D_gen_phi_{sample_name}{tag}.pdf")
        plt.close()
    
        fig, ax  = clear_and_set_name(sample_name)
        h_pt_vs_genpt_worse.plot()
        plt.savefig(f"{plot_dir}/worse_2D_pt_vs_genpt_{sample_name}{tag}.pdf")
        plt.close()

        fig, ax  = clear_and_set_name(sample_name)
        h_dxy_vs_gendxy_worse.plot()
        plt.savefig(f"{plot_dir}/worse_2D_dxy_vs_gendxy_{sample_name}{tag}.pdf")
        plt.close()
        
        fig, ax  = clear_and_set_name(sample_name)
        h_deta_ecal_worse.plot(ax=ax, label = 'deltaEta ecal')
        h_deta_worse.plot(ax=ax, label = 'deltaEta')
        set_common_labels_scale_legend_grid(ax)
        ax.set_xlabel("deltaEta")
        plt.savefig(f"{plot_dir}/worse_deltaEta_{sample_name}{tag}.pdf")
        plt.close()
    
        fig, ax  = clear_and_set_name(sample_name)
        h_dphi_ecal_worse.plot(ax=ax, label = 'deltaPhi ecal')
        h_dphi_worse.plot(ax=ax, label = 'deltaPhi')
        set_common_labels_scale_legend_grid(ax)
        ax.set_xlabel("deltaPhi")
        plt.savefig(f"{plot_dir}/worse_deltaPhi_{sample_name}{tag}.pdf")
        plt.close()
    
        fig, ax  = clear_and_set_name(sample_name)
        h_dpt_worse.plot(ax=ax, label = 'worse')
        h_dpt_all.plot(ax=ax, label = 'all')
        set_common_labels_scale_legend_grid(ax)
        ax.set_xlabel("deltaPt")
        plt.savefig(f"{plot_dir}/worse_deltaPt_{sample_name}{tag}.pdf")
        plt.close()
    
        fig, ax  = clear_and_set_name(sample_name)
        h_dcharge_worse.plot(ax=ax, label = 'worse')
        h_dcharge_all.plot(ax=ax, label = 'all')
        set_common_labels_scale_legend_grid(ax)
        ax.set_xlabel("deltaCharge")
        plt.savefig(f"{plot_dir}/worse_deltaCharge_{sample_name}{tag}.pdf")
        plt.close()
    
        fig, ax  = clear_and_set_name(sample_name)
        h_lxy_worse.plot(ax=ax, label = 'worse')
        h_lxy_all.plot(ax=ax, label = 'all')
        set_common_labels_scale_legend_grid(ax)
        ax.set_xlabel("lxy")
        plt.savefig(f"{plot_dir}/worse_lxy_{sample_name}{tag}.pdf")
        plt.close()
    
    
        ### study events at deltaR_ecal around 3
#         fig, ax  = clear_and_set_name(sample_name)
#         h_pdgid_bump.plot()
#         plt.savefig(f"{plot_dir}/bump_pdgid_{sample_name}{tag}.pdf")
#         plt.close()
#     
#         fig, ax  = clear_and_set_name(sample_name)
#         h_geneta_bump.plot()
#         plt.savefig(f"{plot_dir}/bump_eta_{sample_name}{tag}.pdf")
#         plt.close()
#         
#         fig, ax  = clear_and_set_name(sample_name)
#         h_phi_ecal_vs_phi_bump.plot()
#         plt.savefig(f"{plot_dir}/bump_2D_reco_phi_{sample_name}{tag}.pdf")
#         plt.close()
#     
#         fig, ax  = clear_and_set_name(sample_name)
#         h_genphi_ecal_vs_genphi_bump.plot()
#         plt.savefig(f"{plot_dir}/bump_2D_gen_phi_{sample_name}{tag}.pdf")
#         plt.close()
#     
#     
#         fig, ax  = clear_and_set_name(sample_name)
#         h_deta_ecal_bump.plot(ax=ax, label = 'deltaEta ecal')
#         h_deta_bump.plot(ax=ax, label = 'deltaEta')
#         set_common_labels_scale_legend_grid(ax)
#         ax.set_xlabel("deltaEta")
#         plt.savefig(f"{plot_dir}/bump_deltaEta_{sample_name}{tag}.pdf")
#         plt.close()
#     
#         fig, ax  = clear_and_set_name(sample_name)
#         h_dphi_ecal_bump.plot(ax=ax, label = 'deltaPhi ecal')
#         h_dphi_bump.plot(ax=ax, label = 'deltaPhi')
#         set_common_labels_scale_legend_grid(ax)
#         ax.set_xlabel("deltaPhi")
#         plt.savefig(f"{plot_dir}/bump_deltaPhi_{sample_name}{tag}.pdf")
#         plt.close()
#     
#         fig, ax  = clear_and_set_name(sample_name)
#         h_dpt_bump.plot(ax=ax, label = 'bump')
#         h_dpt_all.plot(ax=ax, label = 'all')
#         set_common_labels_scale_legend_grid(ax)
#         ax.set_xlabel("deltaPt")
#         plt.savefig(f"{plot_dir}/bump_deltaPt_{sample_name}{tag}.pdf")
#         plt.close()
#     
#         fig, ax  = clear_and_set_name(sample_name)
#         h_dcharge_bump.plot(ax=ax, label = 'bump')
#         h_dcharge_all.plot(ax=ax, label = 'all')
#         set_common_labels_scale_legend_grid(ax)
#         ax.set_xlabel("deltaCharge")
#         plt.savefig(f"{plot_dir}/bump_deltaCharge_{sample_name}{tag}.pdf")
#         plt.close()
#     
#         fig, ax  = clear_and_set_name(sample_name)
#         h_lxy_bump.plot(ax=ax, label = 'bump')
#         h_lxy_all.plot(ax=ax, label = 'all')
#         set_common_labels_scale_legend_grid(ax)
#         ax.set_xlabel("lxy")
#         plt.savefig(f"{plot_dir}/bump_lxy_{sample_name}{tag}.pdf")
#         plt.close()
    
    
        fig, ax  = clear_and_set_name(sample_name)
        h_eta_ecal_vs_dxy.plot()
        plt.savefig(f"{plot_dir}/2D_eta_ecal_vs_dxy_{sample_name}{tag}.pdf")
        plt.close()
    
        fig, ax  = clear_and_set_name(sample_name)
        h_eta_ecal_vs_pt.plot()
        plt.savefig(f"{plot_dir}/2D_eta_ecal_vs_pt_{sample_name}{tag}.pdf")
        plt.close()
    
        fig, ax  = clear_and_set_name(sample_name)
        h_eta_ecal_vs_eta.plot()
        plt.savefig(f"{plot_dir}/2D_eta_ecal_vs_eta_{sample_name}{tag}.pdf")
        plt.close()
    
#         fig, ax  = clear_and_set_name(sample_name)
#         h_eta_ecal_vs_lxy.plot()
#         plt.savefig(f"{plot_dir}/2D_eta_ecal_vs_lxy_{sample_name}{tag}.pdf")
#         plt.close()
    
#         fig, ax  = clear_and_set_name(sample_name)
#         h_eta_ecal_vs_tau_eta.plot()
#         plt.savefig(f"{plot_dir}/2D_eta_ecal_vs_gentau_eta_{sample_name}{tag}.pdf")
#         plt.close()
        
        fig, ax  = clear_and_set_name(sample_name)
        h_dxy_vs_lxy.plot(ax=ax)#, cmin=1)
        plt.savefig(f"{plot_dir}/2D_dxy_vs_lxy_{sample_name}{tag}.pdf")
        plt.close()    
        
        fig, ax  = clear_and_set_name(sample_name)
        h_fail_pt_vs_eta.plot()
        plt.savefig(f"{plot_dir}/2D_fail_pt_vs_eta_{sample_name}{tag}.pdf")
        plt.close()
    
        fig, ax  = clear_and_set_name(sample_name)
        h_fail_dxy.plot(label='fail')
        h_pass_dxy.plot(label='pass')
        h_den_dxy.plot(label='all')
        plt.legend()
        plt.savefig(f"{plot_dir}/dxy_compare_{sample_name}{tag}.pdf")
        plt.close()
    
#         fig, ax  = clear_and_set_name(sample_name)
#         h_gen_lxy.plot(label='all')
#         plt.legend()
#         plt.savefig(f"{plot_dir}/lxy_gen_{sample_name}{tag}.pdf")
#         plt.close()
    
    
        fig, ax = clear_and_set_name(sample_name)
#         h_den_dxy.plot(label='gen')
        h_num_dxy.plot(label='gen matched to reco')
        h_num_ecal_dxy.plot(label='gen matched to reco ecal')
        plt.legend()
        plt.savefig(f"{plot_dir}/dxy_gen_{sample_name}{tag}.pdf")
        plt.close()
    
    
    
    

    ## Daniel's style plots
#     plt.clf()
#     fig, ax = plt.subplots(2, 1, figsize=(8,10))
#     ax[0].title.set_text(sample_name)
#     h_num_ecal_dxy.plot_ratio(h_gen_dxy,
#                               rp_num_label = "Matched GVT",
#                               rp_denom_label = "All GVT",
#                               rp_uncertainty = 'efficiency',
#                               ax_dict = {"main_ax": ax[0], "ratio_ax": ax[1]}
#                              )
#     ax[1].set_ylim((0., 1.5))
#     ax[1].set_ylabel("Matched GVT/Total GVT")
#     ax[1].set_xlabel("GVT_dxy [cm]")
#     ax[0].set_ylabel("Number of GVT")
#     ax[1].xaxis.set_major_locator(MultipleLocator(10))
#     ax[1].xaxis.set_minor_locator(MultipleLocator(5))
#     ax[0].xaxis.set_major_locator(MultipleLocator(10))
#     ax[0].xaxis.set_minor_locator(MultipleLocator(5))
#     plt.savefig(f"{plot_dir}/gvt_dxy_ecalDeltaR_eff_{sample_name}.pdf")
    
