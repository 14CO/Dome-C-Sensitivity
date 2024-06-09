#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generate null distributions of Bayes Factors for variable GCR models compared to constant (steady-state models).
"""

import re
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import os
from glob import glob
from cycler import cycler
from tqdm import tqdm

from astropy.io import fits

from scipy.integrate import trapz, cumtrapz
from scipy.stats import anderson_ksamp, ks_2samp, kstest, norm
from scipy.optimize import minimize
from scipy.interpolate import PchipInterpolator

from coprofile import COProfile
from likelihood import ModelLikelihood

from argparse import ArgumentParser

import random

fmu_neg, fmu_fast = 0.068, 0.076

if __name__ == '__main__':
    p = ArgumentParser(description='BF null distribition generator')
    p.add_argument('-d', '--depthavg', type=float, default=20.,
                   help='Depth average of CO profile, in m.')
    p.add_argument('-u', '--reluncertainty', type=float, default=0.02,
                   help='Depth average of CO profile, in m.')
    p.add_argument('-n', '--number', type=int, default=5000000,
                   help='Number of trials to generate')
    p.add_argument('-f', '--fixed', type=bool, default=False,
                   help='Number of trials to generate')
    p.add_argument('-s', '--sample', type=bool, default=False,
                   help='Number of trials to generate')
    p.add_argument('-p', '--priors', type=bool, default=False,
                   help='Number of trials to generate')
    p.add_argument('-fo', '--fofactors', type=bool, default=False,
                   help='Number of trials to generate')
    p.add_argument('-nl', '--null', type=bool, default=False,
                   help='Number of trials to generate')
    p.add_argument('-i', '--id', type=int, default=1,
                   help='Simulation number')
    args = p.parse_args()
    
    print(args)

    fixed = 'pres' if args.fixed else 'past'
    sample = 'mult' if args.sample else 'one'
    priors = 'non-unif' if args.priors else 'unif'
    f_factors = 'all' if args.fofactors else 'const'
    null = 'null-out' if args.null else 'null-in'
    print(fixed, sample, priors, f_factors, null)
    
    print('Loading model data...')

    const_models = ModelLikelihood('FITS_models/balco_14co_const_models.fits', depth_avg=args.depthavg)
    linear_models = ModelLikelihood('FITS_models/balco_14co_linear_models_{}_{}_{}.fits'.format(fixed, f_factors, null), 
                                    depth_avg=args.depthavg)
    step_models = ModelLikelihood('FITS_models/balco_14co_step_models_{}_{}_{}.fits'.format(fixed, f_factors, null), 
                                  depth_avg=args.depthavg)
    burst_models = ModelLikelihood('FITS_models/balco_14co_burst_models_{}.fits'.format(f_factors), depth_avg=args.depthavg)
    
    if args.priors:
        print('Generating priors...')

        const_models.import_prior('14CO_f_all_ChiSq.csv')
        linear_models.import_prior('14CO_f_all_ChiSq.csv')
        step_models.import_prior('14CO_f_all_ChiSq.csv')
        burst_models.import_prior('14CO_f_all_ChiSq.csv')
    
    print('Calculating Bayes factors')
    
    dist = (const_models.fofactors['FOMUNEG'] - fmu_neg)**2 + (const_models.fofactors['FOMUFAST'] - fmu_fast)**2
    j = dist.argmin()
    data = const_models.models[j]
    
    data_mult = random.choices(const_models.models, weights=np.exp(const_models.logprior),k=args.number)

    BF_lin_null = np.zeros(args.number)
    BF_step_null = np.zeros(args.number)
    BF_100yr_null = np.zeros(args.number)
    
    if args.sample:
        data_mult = random.choices(const_models.models, weights=np.exp(const_models.logprior),k=args.number)
        
        for i in tqdm(range(args.number)):
            z_samp, CO_samp, dCO_samp = data_mult[i].sample_z()

            const_like = const_models.likelihood(z_samp, CO_samp, dCO_samp)

            BF_lin_null[i] = const_like / linear_models.likelihood(z_samp, CO_samp, dCO_samp)

            BF_step_null[i] = const_like / step_models.likelihood(z_samp, CO_samp, dCO_samp)

            BF_100yr_null[i] = const_like / burst_models.likelihood(z_samp, CO_samp, dCO_samp)
            
    else:
        dist = (const_models.fofactors['FOMUNEG'] - fmu_neg)**2 + (const_models.fofactors['FOMUFAST'] - fmu_fast)**2
        j = dist.argmin()
        data = const_models.models[j]

        for i in tqdm(range(args.number)):
            z_samp, CO_samp, dCO_samp = data.sample_z()

            const_like = const_models.likelihood(z_samp, CO_samp, dCO_samp)

            BF_lin_null[i] = const_like / linear_models.likelihood(z_samp, CO_samp, dCO_samp)

            BF_step_null[i] = const_like / step_models.likelihood(z_samp, CO_samp, dCO_samp)

            BF_100yr_null[i] = const_like / burst_models.likelihood(z_samp, CO_samp, dCO_samp)
    np.savez(f'FITS_models/bf_null_{args.depthavg:g}m_{100*args.reluncertainty:g}pct_{args.number:06d}_{fixed}_{sample}_{priors}_{f_factors}_{null}_{args.id:02d}.npz',
             BF_lin=BF_lin_null,
             BF_step=BF_step_null,
             BF_100yr=BF_100yr_null)
