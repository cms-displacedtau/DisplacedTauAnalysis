from coffea.util import load
from matplotlib.ticker import MultipleLocator
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import hist, pdb
from hist import Hist, intervals 

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

tag = '_dR0p4_2p4_fixLxy_MaxLxy100_decayM100'
# tag = '_dR0p4_2p4_fixLxy_NOMaxLxy'

for sample_name in  [
#                      'Stau_300_1mm',
                         'Stau_300_100mm',
                         'Stau_300_1000mm',
                         'Stau_100_100mm',
                         'Stau_100_1000mm',
                         'Stau_500_1000mm'
#                      'Stau_100_1000mm',
#                      'Stau_500_1000mm'
                        ]:

    sample_label = sample_name #'m = 300 GeV, ctau = 1000 mm'
    
#     input_file = load(f"histograms_{sample_name}_dR0p4_etaLess2p4.coffea")
#     input_file = load(f"histograms_{sample_name}_dR0p4_etaLess2p4_NoMaxLxy.coffea")
    input_file = load(f"histograms_{sample_name}_dR0p4_etaLess2p4_MaxLxy100_decayM100.coffea")
#     input_file = load(f"histograms_{sample_name}_dR0p4_etaLess2p4_vtxPosition.coffea")
    
    hgen_dxy = input_file["hgen_dxy"]
    hreco_dxy = input_file["hreco_dxy"]
    hreco_ecal_dxy = input_file["hreco_ecal_dxy"]

    hfail_dxy = input_file["hfail_dxy"]
    hpass_dxy = input_file["hpass_dxy"]
    hall_dxy = input_file["hall_dxy"]

    hgen_dxy_zoom = input_file["hgen_dxy_zoom"]
    hreco_dxy_zoom = input_file["hreco_dxy_zoom"]
    hreco_ecal_dxy_zoom = input_file["hreco_ecal_dxy_zoom"]

    hgen_pt = input_file["hgen_pt"]
    hreco_pt = input_file["hreco_pt"]
    hreco_ecal_pt = input_file["hreco_ecal_pt"]
    
    heta_ecal_vs_dxy = input_file['heta_ecal_vs_dxy']
    heta_ecal_vs_pt = input_file['heta_ecal_vs_pt']
    heta_ecal_vs_eta = input_file['heta_ecal_vs_eta']
    heta_ecal_vs_lxy = input_file['heta_ecal_vs_lxy']
    heta_ecal_vs_tau_eta  = input_file['heta_ecal_vs_tau_eta']
    hdxy_vs_lxy = input_file['hdxy_vs_lxy']
    hfail_pt_vs_eta = input_file['hfail_pt_vs_eta']

    hgen_lxy = input_file["hgen_lxy"]
#     hreco_lxy = input_file["hreco_lxy"]
#     hreco_ecal_lxy = input_file["hreco_ecal_lxy"]

    hdeltaR_custom_zoomout = input_file["hdeltaR_custom_zoomout"]
    hdeltaR_custom = input_file["hdeltaR_custom"]
    hdeltaR = input_file["hdeltaR"]



    clear_and_set_name(sample_name)
    plot_efficiency(hreco_dxy, 
                    hgen_dxy, 
                    label='deltaR')
    plot_efficiency(hreco_ecal_dxy, 
                    hgen_dxy, 
                    label='deltaR ecal')
    ax.set_ylabel("Matched GVT/Total GVT")
    ax.set_xlabel("GVT_dxy [cm]")
    ax.set_title(sample_label)
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{plot_dir}/eff_dxy_{sample_name}{tag}.pdf")
    plt.close()

    clear_and_set_name(sample_name)
    plot_efficiency(hreco_dxy_zoom, 
                    hgen_dxy_zoom, 
                    label='deltaR')
    plot_efficiency(hreco_ecal_dxy_zoom, 
                    hgen_dxy_zoom, 
                    label='deltaR ecal')

    ax.set_ylabel("Matched GVT/Total GVT")
    ax.set_xlabel("GVT_dxy [cm]")
    ax.set_title(sample_label)
    plt.legend()
    plt.grid(True)
    plt.xscale('log')
    plt.savefig(f"{plot_dir}/eff_dxy_zoom_{sample_name}{tag}.pdf")
    plt.close()


    clear_and_set_name(sample_name)
    plot_efficiency(hreco_pt, 
                    hgen_pt, 
                    label='deltaR')
    plot_efficiency(hreco_ecal_pt, 
                    hgen_pt, 
                    label='deltaR ecal')

    ax.set_ylabel("Matched GVT/Total GVT")
    ax.set_xlabel("pion pt [GeV]")
    ax.set_title(sample_label)
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{plot_dir}/eff_pt_{sample_name}{tag}.pdf")
    plt.close()


    clear_and_set_name(sample_name)
    hdeltaR_custom.plot(ax=ax, label = 'deltaR ecal')
    hdeltaR.plot(ax=ax, label = 'deltaR')
    print (sample_label, ' ->  n entries in deltaR: ', hdeltaR.sum())
    ax.set_ylabel("events")
    ax.set_xlabel("deltaR")
#     ax.set_title(sample_label)
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{plot_dir}/deltaR_{sample_name}{tag}.pdf")

    clear_and_set_name(sample_name)
    print (sample_label, ' ->  n entries in deltaR OUT: ', hdeltaR_custom_zoomout.sum())
    hdeltaR_custom_zoomout.plot(ax=ax, label = 'deltaR ecal')
    ax.set_ylabel("events")
    ax.set_xlabel("deltaR")
#     plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{plot_dir}/deltaR_out_{sample_name}{tag}.pdf")
   

    clear_and_set_name(sample_name)
    heta_ecal_vs_dxy.plot()
    plt.savefig(f"{plot_dir}/eta_ecal_vs_dxy_{sample_name}{tag}.pdf")

    clear_and_set_name(sample_name)
    heta_ecal_vs_pt.plot()
    plt.savefig(f"{plot_dir}/eta_ecal_vs_pt_{sample_name}{tag}.pdf")

    clear_and_set_name(sample_name)
    heta_ecal_vs_eta.plot()
#     heta_ecal_vs_eta.plot(ax=ax, cmin=1)#, cmax=2)
    plt.savefig(f"{plot_dir}/eta_ecal_vs_eta_{sample_name}{tag}.pdf")
    plt.close()

    clear_and_set_name(sample_name)
    heta_ecal_vs_lxy.plot()
    plt.savefig(f"{plot_dir}/eta_ecal_vs_lxy_{sample_name}{tag}.pdf")
    plt.close()

    clear_and_set_name(sample_name)
    heta_ecal_vs_tau_eta.plot()
#     heta_ecal_vs_tau_eta.plot( ax=ax, cmin=0, cmax=2)
    plt.savefig(f"{plot_dir}/eta_ecal_vs_gentau_eta_{sample_name}{tag}.pdf")
    plt.close()
    
    clear_and_set_name(sample_name)
    hdxy_vs_lxy.plot(ax=ax, cmin=1)
    plt.savefig(f"{plot_dir}/dxy_vs_lxy_{sample_name}{tag}.pdf")
    plt.close()    
    
    clear_and_set_name(sample_name)
    hfail_pt_vs_eta.plot()
    plt.savefig(f"{plot_dir}/fail_pt_vs_eta_{sample_name}{tag}.pdf")
    plt.close()

    clear_and_set_name(sample_name)
    hfail_dxy.plot(label='fail')
    hpass_dxy.plot(label='pass')
    hall_dxy.plot(label='all')
    plt.legend()
    plt.savefig(f"{plot_dir}/compare_dxy_{sample_name}{tag}.pdf")
    plt.close()

    clear_and_set_name(sample_name)
    hgen_lxy.plot(label='all')
    plt.legend()
    plt.savefig(f"{plot_dir}/gen_lxy_{sample_name}{tag}.pdf")
    plt.close()


    clear_and_set_name(sample_name)
    hgen_dxy.plot(label='gen')
    hreco_dxy.plot(label='gen matched to reco')
    hreco_ecal_dxy.plot(label='gen matched to reco ecal')
    plt.legend()
    plt.savefig(f"{plot_dir}/gen_dxy_{sample_name}{tag}.pdf")
    plt.close()

    ## Daniel's style plots
#     plt.clf()
#     fig, ax = plt.subplots(2, 1, figsize=(8,10))
#     ax[0].title.set_text(sample_name)
#     hreco_ecal_dxy.plot_ratio(hgen_dxy,
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
    
