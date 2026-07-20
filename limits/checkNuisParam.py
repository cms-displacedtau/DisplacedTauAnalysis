import uproot

f = uproot.open('higgsCombine_bestfit_500_50mm.MultiDimFit.mH120.root')
t = f['limit']

params = ['r', 'CMS_pileup', 'CMS_res_j', 'CMS_scale_j', 'lumi_13p6TeV', 'xsec_Stau_500_50mm']
for p in params:
    try:
        val = t[p].array()
        print('%s: %.4f' % (p, val[0]))
    except:
        print('%s: not found' % p)
