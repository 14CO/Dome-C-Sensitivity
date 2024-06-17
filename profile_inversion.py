# -*- coding: utf-8 -*-
"""Implementation of 14CO profile models which can be used to compute the likelihood of a given dataset, given the model."""

import numpy as np

from abc import ABC
from astropy.io import fits

from coprofile import COProfile
from coprofile import COGenerator

import matplotlib.pyplot as plt

import pandas as pd

from scipy import integrate

class Inverter(ABC):
    
    
    def __init__(self, model = None, f_ratio = 72/66, mu_neg_file = 'models/balco_14co_delta_neg_models.fits', mu_fast_file = 'models/balco_14co_delta_fast_models.fits'):
        """
        Parameters
        ----------
        model : array ( axis0: time; axis1: function parameters )
            
        f_ratio : float
            
        mu_neg_file : str
            
        mu_fast_file : str
            
        
        Internal Values
        ---------------
        self.f_ratio : float
            
        self.z : array ( axis0 : depth )
            
        self.z_samp : array ( axis0 : sample depth )
            
        self.t : array ( axis0: time )
            
        self.t_samp : array ( axis0: function parameters )
            
        self.comp : array ( axis0: sample depths ; axis1: depth )
            
        self.model : array ( axis0: time; axis1: function parameters )
            
        self.G : array ( axis0: depth ; axis1: time )
            
        self.G_inv : array ( axis0: time ; axis1: depth )
            
        self.G_comp : array ( axis0: sample depth ; axis1: function parameters )
            
        self.G_comp_inv : array ( axis0: function parameters ; axis1: sample depth )
            
        """
        
        self.f_ratio = f_ratio
        
        hdus_neg = fits.open(mu_neg_file)

        # Set up mu_neg 14CO profiles.
        z_neg = hdus_neg['DEPTH'].data
        co14_neg = hdus_neg['CO14'].data
        t_spike_neg = hdus_neg['T_SPIKE'].data['T_SPIKE']

        order = np.argsort(-t_spike_neg)

        z = z_neg
        t = np.array(t_spike_neg[order])
        CO_neg = np.array(co14_neg[order])[:,1:]


        hdus_fast = fits.open(mu_fast_file)

        # Set up mu_fast 14CO profiles.
        z_fast = hdus_fast['DEPTH'].data
        co14_fast = hdus_fast['CO14'].data
        t_spike_fast = hdus_fast['T_SPIKE'].data['T_SPIKE']

        order = np.argsort(-t_spike_fast)

        # axis0 = time, axis1 = depth
        CO_fast = np.array(co14_fast[order])[:,1:]
        
        self.z = z
        self.t = t
        
        # axis0: depth, axis1: time
        self.G = (CO_neg + f_ratio * CO_fast).T
        self.G_inv = np.linalg.inv(self.G)
        
        step = int(20 / np.diff(self.z)[0])
        i = np.arange(0, len(self.G), step)
        i = np.append(i,len(self.G))

        self.comp = np.zeros((len(i)-1, len(self.G)))
        for x in range(len(i)-1):
            self.comp[x, i[x]:i[x+1]] = 1/(i[x+1]-i[x])
            
        self.z_samp = np.matmul(comp, z[1:])
        self.t_samp = np.matmul(comp, t)
        
        if model == None:
            model = np.zeros((len(self.G),len(i)-1))
            for x in range(len(i)-1):
                model[i[x]:i[x+1], x] = 1
        self.model = np.array(model)
        
        self.G_comp = np.matmul(self.comp, np.matmul(self.G, self.model))
        
        self.G_comp_inv = np.linalg.inv(self.G_comp)
        
    def solve(self, CO_samp, dCO_samp):
        """
        Parameters
        ----------
        CO_samp : array ( axis0: sample depth )
            
        dCO_samp : array ( axis0: sample depth )
            
        
        Returns
        -------
        f_solve : array ( axis0: function parameters )
            
        df : array ( axis0: function parameters )
            
        """
        
        f_solve = np.matmul(self.G_comp_inv, CO_samp)
        df = np.sqrt(np.sum((self.G_comp_inv * dCO_samp)**2,axis=1))
        return f_solve, df
    
    def likelihood(self, params, CO_samp, dCO_samp):
        """
        Parameters
        ----------
        params : array ( axis0: function parameters )
            
        CO_samp : array ( axis0: sample depth )
            
        dCO_samp : array ( axis0: sample depth )
            
        
        Returns
        -------
        Pr : float
            
        """
        
        CO_func = np.matmul(self.G_comp, params)
        
        N = len(CO_samp)
        
        # axis0: depth
        loglike = -0.5*np.sum(((CO_func - CO_samp) / dCO_samp)**2) - 0.5*N*np.log(2*np.pi) - np.sum(np.log(dCO_samp))
        
        Pr = np.exp(loglike)
        
        return Pr