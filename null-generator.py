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

fmu_neg, fmu_fast = 0.066, 0.072

if __name__ == '__main__':
    p = ArgumentParser(description='BF null distribition generator')
    p.add_argument('-d', '--depthavg', type=float, default=20.,
                   help='Depth average of CO profile, in m.')
    p.add_argument('-u', '--reluncertainty', type=float, default=0.02,
                   help='Depth average of CO profile, in m.')
    p.add_argument('-n', '--number', type=int, default=10000,
                   help='Number of trials to generate')
    p.add_argument('-i', '--id', type=int, default=1,
                   help='Simulation number')
    args = p.parse_args()

    const_models = ModelLikelihood('models/balco_14co_const_models.fits', depth_avg=args.depthavg)
    linear_models = ModelLikelihood('models/balco_14co_linear_models_NEW.fits', depth_avg=args.depthavg)
    step_models = ModelLikelihood('models/balco_14co_step_models_NEW.fits', depth_avg=args.depthavg)
    burst_models = ModelLikelihood('models/balco_14co_burst_models_NEW.fits', depth_avg=args.depthavg)
    
    #dist = (const_models.fofactors['FOMUNEG'] - fmu_neg)**2 + (const_models.fofactors['FOMUFAST'] - fmu_fast)**2
    #j = dist.argmin()
    #data = const_models.models[j]
    
    data_mult = random.choices(const_models.models, weights=np.exp(const_models.logprior),k=args.number)

    BF_lin_null = []
    BF_step_null = []
    BF_100yr_null = []
    BF_1kyr_null = []

    for i in tqdm(range(args.number)):
        #z_samp, CO_samp, dCO_samp = data.sample_z()
        z_samp, CO_samp, dCO_samp = data_mult[i].sample_z()

        B = const_models.likelihood(z_samp, CO_samp, dCO_samp) / linear_models.likelihood(z_samp, CO_samp, dCO_samp)
        BF_lin_null.append(B)

        B = const_models.likelihood(z_samp, CO_samp, dCO_samp) / step_models.likelihood(z_samp, CO_samp, dCO_samp)
        BF_step_null.append(B)

        B = const_models.likelihood(z_samp, CO_samp, dCO_samp) / burst_models.likelihood(z_samp, CO_samp, dCO_samp)
        BF_100yr_null.append(B)

    np.savez(f'models/bf_null_{args.depthavg:g}m_{100*args.reluncertainty:g}pct_{args.number:06d}_{args.id:02d}.npz',
             BF_lin=BF_lin_null,
             BF_step=BF_step_null,
             BF_100yr=BF_100yr_null)
