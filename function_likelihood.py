import numpy as np

from abc import ABC
from astropy.io import fits

from coprofile import COProfile
from coprofile import COGenerator

import matplotlib.pyplot as plt

import pandas as pd

from scipy import integrate

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
    
    
    def __init__(self, m = 'step', f_ratio = 72/66, mu_neg_file = 'models/balco_14co_delta_neg_models.fits', mu_fast_file = 'models/balco_14co_delta_fast_models.fits'):
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
        
        self.i = i

        self.comp = np.zeros((len(i)-1, len(self.G)))
        for x in range(len(i)-1):
            self.comp[x, i[x]:i[x+1]] = 1/(i[x+1]-i[x])
            
        self.z_samp = np.matmul(self.comp, z[1:])
        
        if m == 'step':
            c = i
            model = np.zeros((len(self.G),len(i)-1))
            for x in range(len(i)-1):
                model[i[x]:i[x+1], x] = 1
                
            self.model = np.array(model)
            self.tcomp = self.model.T/np.expand_dims(np.sum(self.model,axis=0),axis=1)
            self.t_samp = np.matmul(self.tcomp, t)
        
        elif m == 'slope':
            c = i
            b = np.array((c[1:]+c[:-1])/2, dtype=int)
            
            model = np.zeros((len(self.G),len(i)-1))
            
            model[0:b[0],0] += (t[0:b[0]] - t[b[1]]) / (t[b[0]] - t[b[1]])
            model[0:b[0],1] += (t[0:b[0]] - t[b[0]]) / (t[b[1]] - t[b[0]])
            
            model[b[-1]:,-1] += (t[b[-1]:] - t[b[-2]]) / (t[b[-1]] - t[b[-2]])
            model[b[-1]:,-2] += (t[b[-1]:] - t[b[-1]]) / (t[b[-2]] - t[b[-1]])
            
            for j in range(len(i)-2):
                model[b[j]:b[j+1],j] += (t[b[j]:b[j+1]] - t[b[j+1]]) / (t[b[j]] - t[b[j+1]])
                model[b[j]:b[j+1],j+1] += (t[b[j]:b[j+1]] - t[b[j]]) / (t[b[j+1]] - t[b[j]])
            
            self.model = np.array(model)
            self.tcomp = self.model.T/np.expand_dims(np.sum(self.model,axis=0),axis=1)
            self.t_samp = t[b]
        
        elif m == 'even':
            a = np.linspace(0,np.sum(self.G),len(i))
            b = np.cumsum(np.sum(self.G, axis=0))

            c = np.zeros(len(i), dtype=int)
            j = 0
            for i,x in enumerate(b):
                if x>=a[j]:
                    c[j] = i
                    j+= 1
            c[-1] = len(b)
            
            model = np.zeros((len(self.G),len(c)-1))
            for x in range(len(c)-1):
                model[c[x]:c[x+1], x] = 1
                
            self.model = np.array(model)
            self.tcomp = self.model.T/np.expand_dims(np.sum(self.model,axis=0),axis=1)
            self.t_samp = np.matmul(self.tcomp, t)
        
        self.G_comp = np.matmul(self.comp, np.matmul(self.G, self.model))
        
        self.G_comp_inv = np.linalg.inv(self.G_comp)
        
        self.interp = np.zeros((len(self.z[1:]), len(self.z_samp)))

        for j in range(len(self.z_samp)+1):
            if j == 0:
                bound = self.z[1:]<=self.z_samp[j]
                self.interp[bound, j] += (self.z[1:][bound]-self.z[0])/(self.z_samp[j]-self.z[0])

                #interp[bound, j+1] += (z_samp[j]-z[1:][bound])/(z_samp[j]-z_samp[j+1])
            elif j == len(self.z_samp):
                bound = self.z[1:]>=self.z_samp[j-1]
                self.interp[bound, j-1] += (self.z_samp[j-2]-self.z[1:][bound])/(self.z_samp[j-2]-self.z_samp[j-1])

                self.interp[bound, j-2] += (self.z_samp[j-1]-self.z[1:][bound])/(self.z_samp[j-1]-self.z_samp[j-2])
            else:
                bound = np.logical_and(self.z[1:]<=self.z_samp[j], self.z[1:]>=self.z_samp[j-1])
                self.interp[bound, j-1] += (self.z_samp[j]-self.z[1:][bound])/(self.z_samp[j]-self.z_samp[j-1])

                bound = np.logical_and(self.z[1:]>=self.z_samp[j-1], self.z[1:]<=self.z_samp[j])
                self.interp[bound, j] += (self.z_samp[j-1]-self.z[1:][bound])/(self.z_samp[j-1]-self.z_samp[j])
                
        self.resolve = np.matmul(self.tcomp, np.matmul(self.G_inv, self.interp))
        
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
    
class Unfolder(ABC):
    
    def __init__(self, mu_neg_file = 'models/balco_14co_delta_neg_models.fits', mu_fast_file = 'models/balco_14co_delta_fast_models.fits'):
        """
        Parameters
        ----------
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
        
        hdus_neg = fits.open(mu_neg_file)

        # Set up mu_neg 14CO profiles.
        z_neg = hdus_neg['DEPTH'].data
        co14_neg = hdus_neg['CO14'].data
        t_spike_neg = hdus_neg['T_SPIKE'].data['T_SPIKE']

        order = np.argsort(-t_spike_neg)

        z = z_neg
        t = np.array(t_spike_neg[order])
        CO_neg = np.array(co14_neg[order])#[:,1:]


        hdus_fast = fits.open(mu_fast_file)

        # Set up mu_fast 14CO profiles.
        z_fast = hdus_fast['DEPTH'].data
        co14_fast = hdus_fast['CO14'].data
        t_spike_fast = hdus_fast['T_SPIKE'].data['T_SPIKE']

        order = np.argsort(-t_spike_fast)

        # axis0 = time, axis1 = depth
        CO_fast = np.array(co14_fast[order])#[:,1:]
        
        self.z = z
        self.t = t
        
        # axis0: depth, axis1: time
        self.G = np.append(CO_neg, CO_fast, axis=0).T
        
        step = int(20 / np.diff(self.z)[0])
        i = np.arange(0, len(self.G), step)
        i = np.append(i,len(self.G))

        self.comp = np.zeros((len(i)-1, len(self.G)))
        for x in range(len(i)-1):
            self.comp[x, i[x]:i[x+1]] = 1/(i[x+1]-i[x])

        self.z_samp = np.matmul(self.comp, self.z)
        self.t_samp = np.matmul(self.comp, np.append(self.t,[0]))

        self.G_comp = np.matmul(self.comp, self.G)