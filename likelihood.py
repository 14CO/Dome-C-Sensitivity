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

class ModelLikelihood(ABC):
    """Model likelihood base class.
    """
    
    def __init__(self, model_fits, depth_avg=10, rel_uncertainty=0.02):
        """Construct a model likelihood.
        
        Parameters
        ----------
        model_fits_file : str
            Path to FITS file containing model parameters and 14CO profiles.
        depth_avg: float
            Profile depth averaging, in meters.
        rel_uncertainty: float
            Relative statistical uncertainty in measurements (0..1).
            
        Internal Values
        ---------------
        self.models : array ( axis0: parameter space ) of COProfiles
            COProfiles of simulated data for different sets of parameters
        self.CO_mods : array ( axis0: parameter space, axis1: depth )
            14CO densities collected into a single array
        self.amplitudes : columns ( columns: 1, axis0: parameter space )
            Amplitude parameters
        self.fofactors : columns ( columns: 2, axis0: parameter space )
            14CO baseline production rates
        self.parameters : array ( axis0: parameter space ; axis1: parameter type )
            All parameters
        self.volume : float
            total volume in parameter space
        self.dtheta : float
            unit volume in parameter space
        self.logprior : array ( axis0: parameter space )
            log of Bayesian prior on parameters
        """
        hdus = fits.open(model_fits)

        # Set up 14CO profiles.
        z = hdus['DEPTH'].data
        co14 = hdus['CO14'].data
        self.models = [COProfile(z, co, depth_avg, rel_uncertainty) for co in co14]
        
        self.CO_mods = np.array([model.CO_model for model in self.models])
            
        # Input model parameters and set up uniform prior volume.
        params = []
        
        if 'AMPL' in hdus: # Check for amplitudes
            self.amplitudes = hdus['AMPL'].data
            params.append(self.amplitudes['AMPL'])
            
            hdr = hdus['AMPL'].header
            
            n_amp = hdr['N_AMPL']
            ampWidth = hdr['AMPLMAX'] - hdr['AMPLMIN'] if n_amp>1 else 1.
        else:
            n_amp = 1
            ampWidth = 1.
            
        if 'FOFACTOR' in hdus: # Check for fofactors
            self.fofactors = hdus['FOFACTOR'].data
            params.append(self.fofactors['FOMUNEG'])
            params.append(self.fofactors['FOMUFAST'])
            
            hdr = hdus['FOFACTOR'].header
            
            n_foneg = hdr['N_FONEG']
            n_fofast = hdr['N_FOFAST']
            negWidth = hdr['FNEGMAX'] - hdr['FNEGMIN'] if n_foneg>1 else 1.
            fastWidth = hdr['FFASTMAX'] - hdr['FFASTMIN'] if n_fofast>1 else 1.
        else:    
            n_foneg = 1
            n_fofast = 1
            negWidth = 1.
            fastWidth = 1.
        
        self.parameters = np.array(params).T
        
        ncells = len(self.models)
        
        #volfrac = ncells / (n_ampl * n_foneg * n_fofast) # fraction of a full cube taken up in parameter space
        #self.volume = volfrac * ampWidth * negWidth * fastWidth # total volume in parameter space
        
        self.volume = 1 #If all we care about is calculating likelihood, we can always choose units so that volume = 1
            
        self.dtheta = self.volume / ncells
        self.logprior = np.zeros(ncells) # log of uniform prior
        self.logprior -= np.log(np.sum(np.exp(self.logprior) * self.dtheta)) # Normalize prior
        
    def import_prior(self, file):
        # reads chi^2 data from csv and converts it to a prior
        
        chiData = pd.read_csv(file) # chi squared fit data for parameters
        f_neg = np.expand_dims(np.array(chiData['f_neg']),axis=1)
        f_fast = np.expand_dims(np.array(chiData['fast']),axis=1)
        Chi_sq = np.array(chiData['Chi_sq'])
        
        # parameters used in model
        mod_neg = self.fofactors['FOMUNEG']
        mod_fast = self.fofactors['FOMUFAST']

        # find closest data points to values used in model
        dist = (mod_neg-f_neg)**2 + (mod_fast-f_fast)**2
        j = dist.argmin(axis=0)

        chi2 = Chi_sq[j]

        logprior = -chi2/2 # log likelihood = -1/2 chi^2
        self.logprior = logprior - np.log(np.sum(np.exp(logprior) * self.dtheta)) # Normalize prior
    
    def likelihood(self, z_samp, CO_samp, dCO_samp):        
        """Compute likelihood of a given realized 14CO profile given this model.

        Parameters
        ----------
        z_samp : list or ndarray
            Sampled depths of a realized 14CO profile.
        CO_samp: list or ndarray
            Realization of a 14CO profile data set.
        dCO_samp: list or ndarray
            Statistical uncertainties in realization of a 14CO profile.

        Returns
        -------
        Pr : float
            Likelihood of observing a given realized 14CO profile given this
            model, marginalizing over allowed model parameters.
        """
        N = len(CO_samp)
        
        # axis0: parameter space; axis1: depth
        loglike = -0.5*np.sum(((self.CO_mods - CO_samp) / dCO_samp)**2, axis=1) - 0.5*N*np.log(2*np.pi) - np.sum(np.log(dCO_samp))
        #Pr = np.sum(np.exp(loglike + self.logprior) * self.dtheta)
        Pr = np.mean(np.exp(loglike + self.logprior)) #ASSUMING VOLUME = 1; dtheta = 1/N
        
        return Pr
    
    def likelihood_mult(self, CO_samps, dCO_samp): #multiple likelihood calculations at once
        
        """Compute likelihoods of multiple 14CO profiles given this model.

        Parameters
        ----------
        CO_samps: array; axis0 = depth; axis1 = iterations
            Realization of a 14CO profile data set.
        dCO_samps: array; axis0 = depth
            Statistical uncertainties in realization of a 14CO profile.

        Returns
        -------
        Pr : array
            Likelihood of observing a given realized 14CO profile given this
            model, marginalizing over allowed model parameters.
        """
        
        N = np.shape(CO_samps)[0]
        
        # self.CO_mods: axis0 = parameter space; axis1 = depth
        # self.logprior: axis0 = parameter space
        
        # axis0 = parameter space; axis1 = depth; axis2 = iterations
        CO_th = np.expand_dims(self.CO_mods, axis=2)
        CO_exp = np.expand_dims(CO_samps, axis=0)
        dCO = np.expand_dims(dCO_samp, axis=(0,2))
        
        # sum over axis1 (depth)
        loglike = -0.5*np.sum(((CO_th - CO_exp)/dCO)**2, axis=1) - 0.5*N*np.log(2*np.pi) - np.sum(np.log(dCO), axis=1)
        
        # axis0 = parameter space; axis1 = iterations
        logprior = np.expand_dims(self.logprior, axis=1)
        
        # sum over axis0 (parameter space)
        Pr = np.average(np.exp(loglike + logprior), axis=0)
        
        # axis0 = iterations
        return Pr