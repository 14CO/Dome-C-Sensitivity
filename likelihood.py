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
            
        self.CO_mods : array ( axis0: parameter space, axis1: depth )
            
        self.amplitudes : columns ( columns: 1, axis0: parameter space )
            
        self.fofactors : columns ( columns: 2, axis0: parameter space )
            
        self.parameters : array ( axis0: parameter space ; axis1: parameter type )
            
        self.volume : float
            
        self.dtheta : float
            
        self.logprior : array ( axis0: parameter space )
            
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
        
        """Compute likelihood of a given realized 14CO profile given this model.

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
    
    
class FunctionModelLikelihood(ABC):
    """Model likelihood built from functional dependence of f_mu_neg and f_mu_fast over time
    """
    
    def __init__(self, generator, f_func, bounds, prior_func = None):
        """Construct a model likelihood
        
        Parameters
        ----------
        generator : COGenerator
            
        f_func : function
            Functional dependence of f_mu_neg and f_mu_fast over time
        
            Parameters
            ----------
            t : array ( axis0: time )
            
            theta : tuple
                
            
            Returns
            -------
            f_mu_neg : array ( axis0: time )
                
            f_mu_fast : array ( axis0: time )
            
        bounds : array ( axis0: params ) of tuples (min, max)
            
        prior_func : function
            Bayesian prior on params
            
            Parameters
            ----------
            theta : tuple
                
                
            Returns
            -------
            logprior : float
                
                
        Internal Values
        ---------------
        
            
        """
        self.t = generator.t
        self.generator = generator
        self.f_func = f_func
        self.bounds = bounds
        if prior_func == None:
            self.prior_func = self.unif_prior
        else:
            self.prior_func = prior_func
        self.norm = 0.
        self.normalize()
        
    def unif_prior(self, theta):
        return 0.
    
    def logprior(self, theta):
        return self.prior_func(theta) + self.norm
    
    def prior(self, *theta):
        return np.exp(self.logprior(theta))
        
    def normalize(self):
        self.norm = -np.log( integrate.nquad(self.prior, self.bounds)[0] )
        return
        
    def CO(self, theta):
        """
        Parameters
        ----------
        z_samp : array ( axis0 : depth )
        
        theta : tuple
        
        Returns
        -------
        CO : array ( axis0 : depth )
        """
        return self.generator.generate(*self.f_func(self.t, theta)).CO_model
        
    def prob(self, *theta):
        N = len(self.CO_samp)
        
        loglike = -0.5*np.sum(((self.CO(theta) - self.CO_samp) / self.dCO_samp)**2) - 0.5*N*np.log(2*np.pi) - np.sum(np.log(self.dCO_samp))
        
        return np.exp(loglike + self.logprior(theta))
        
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
        # this is a really janky hack, but it should work
        self.z_samp = z_samp
        self.CO_samp = CO_samp
        self.dCO_samp = dCO_samp
        return integrate.nquad(self.prob, self.bounds)
    
    def plot_factors(self, thetas):
        neg = np.zeros((len(thetas),len(self.t)))
        fast = np.zeros((len(thetas),len(self.t)))
        for i,theta in enumerate(thetas):
            n, f = self.f_func(self.t,theta)
            neg[i] = n
            fast[i] = f

        fig, ax = plt.subplots(1,1, figsize=(6,5), tight_layout=True)
        for i,n in enumerate(neg):
            ax.plot(self.t, n, label='{}'.format(thetas[i]))
        ax.set(xlim=(min(self.t),max(self.t)),
               xlabel='time [yrs]',
                ylim=(0.02,0.12),
                ylabel='f_mu_neg')
        ax.legend(loc='lower right', fontsize=10)
        plt.title('f_mu_neg over time')

        fig, ax = plt.subplots(1,1, figsize=(6,5), tight_layout=True)
        for i,f in enumerate(fast):
            ax.plot(self.t, f, label='{}'.format(thetas[i]))
        ax.set(xlim=(min(self.t),max(self.t)),
                xlabel='time [yrs]',
                ylim=(0.02,0.12),
                ylabel='f_mu_fast')
        ax.legend(loc='lower right', fontsize=10)
        plt.title('f_mu_fast over time')
        
    def plot_profiles(self, thetas):
        p = [self.generator.generate(*self.f_func(self.t, theta)) for theta in thetas]

        fig, ax = plt.subplots(1,1, figsize=(6,5), tight_layout=True)
        for i,profile in enumerate(p):
            ax.plot(profile.z, profile.CO, label='{}'.format(thetas[i]))
        ax.set(xlim=(90,300),
                xlabel='depth [m]',
                ylim=(0,30),
                ylabel=r'$^{14}$CO concentration [molecule g$^{-1}$]')
        ax.legend(loc='lower right', fontsize=10)
        plt.title('CO Profiles')
        
class SampleInverter(ABC):
    
    
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
            
        self.z_samp = np.matmul(self.comp, z[1:])
        self.t_samp = np.matmul(self.comp, t)
        
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