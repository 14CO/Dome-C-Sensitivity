# -*- coding: utf-8 -*-
"""Implementation of 14CO profile models which can be used to compute the likelihood of a given dataset, given the model."""

import numpy as np
from coprofile import COProfile

class ModelLikelihood(ABC):
    """Model likelihood base class.
    """
    
    def __init__(self, model_fits_file, depth_avg=10, rel_uncertainty=0.02):
        """Construct a model likelihood.
        
        Parameters
        ----------
        model_fits_file : str
            Path to FITS file containing model parameters and 14CO profiles.
        depth_avg: float
            Profile depth averaging, in meters.
        rel_uncertainty: float
            Relative statistical uncertainty in measurements (0..1).
        """
        pass
    
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
        Pr = 0.
        for model in self.models:
            loglike = -0.5*N*np.log(2*np.pi) - np.sum(0.5*((model.CO_model - CO_samp)/dCO_samp)**2 + np.log(dCO_samp))
            logprior = -np.log(self.volume)
            Pr += np.exp(loglike + logprior) * self.dtheta
        
        return Pr
    

class ConstModelLikelihood(ModelLikelihood):
    """Model likelihood for GCR fluxes that are constant in time but have a systematic uncertainty in the cosmogenic 14CO production rates.
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
        """
        hdus = fits.open(model_fits)

        # Set up 14CO profiles.
        z = hdus['DEPTH'].data
        co14 = hdus['CO14'].data
        self.models = []
        
        for co in co14:
            cp = COProfile(z, co, depth_avg, rel_uncertainty)
            self.models.append(cp)
            
        # Input model parameters and set up uniform prior volume.
        self.fofactors = hdus['FOFACTOR'].data
        hdr = hdus['FOFACTOR'].header
        
        ncells = len(self.fofactors)
        volfrac = ncells / (hdr['N_FONEG']*hdr['N_FOFAST'])
        self.volume = volfrac * (hdr['FNEGMAX']-hdr['FNEGMIN']) * (hdr['FFASTMAX']-hdr['FFASTMIN'])
        self.dtheta = self.volume / ncells
    

class VariableModelLikelihood(ModelLikelihood):
    """Model likelihood for GCR fluxes that vary in time.
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
        """
        hdus = fits.open(model_fits)

        # Set up 14CO profiles.
        z = hdus['DEPTH'].data
        co14 = hdus['CO14'].data
        self.models = []
        
        for co in co14:
            cp = COProfile(z, co, depth_avg, rel_uncertainty)
            self.models.append(cp)
            
        # Input model parameters and set up uniform prior volume.
        self.amplitudes = hdus['AMPL'].data
        
        hdr = hdus['AMPL'].header
        self.volume = hdr['AMPLMAX'] - hdr['AMPLMIN']
        self.dtheta = hdr['DELTAMPL']
