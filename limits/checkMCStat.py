import uproot
import numpy as np

f = uproot.open('higgsCombine_scanMCstat.MultiDimFit.mH120.root')
t = f['limit']

r = t['r'].array()
mcstat = t['QCD_mcstat_SR'].array()

print('r              QCD_mcstat_SR')
print('-'*30)
for ri, mi in sorted(zip(r, mcstat)):
    print('%.2f           %.4f' % (ri, mi))
