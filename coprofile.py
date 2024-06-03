# -*- coding: utf-8 -*-
"""This module implements a representation of a 14CO vertical profile and can be used to generate realizations of a measurement with a particular depth average and statistical uncertainty."""

import numpy as np

from astropy.io import fits

class COGenerator:
    """Generates CO Profiles from time-variance of muon reactions"""
    
    def __init__(self, mu_neg_file, mu_fast_file):
        """Initialize CO Profile Generator
        
        Parameters
        ----------
        mu_neg_file : str
            address of FITS file for mu_neg CO14 data.
        mu_fast_file : str
            address of FITS file for mu_fast CO14 data.
        
        Internal Values
        ----------
        self.t : ndarray ( axis0: time )
            Time array for 14CO production [years].
        self.z : ndarray ( axis0: depth )
            True or ice-equivalent depth array [m].
        self.CO_neg : ndarray ( axis0: time ; axis1: depth )
            14CO concentration profile from negative muon reactions [molecules/g].
        self.CO_fast : ndarray ( axis0: time ; axis1: depth )
            14CO concentration profile from fast muon reactions [molecules/g].
        """
        
        hdus_neg = fits.open(mu_neg_file)

        # Set up mu_neg 14CO profiles.
        z_neg = hdus_neg['DEPTH'].data
        co14_neg = hdus_neg['CO14'].data
        t_spike_neg = hdus_neg['T_SPIKE'].data['T_SPIKE']

        order = np.argsort(t_spike_neg)
        
        self.z = z_neg
        self.t = np.array(t_spike_neg[order])
        self.CO_neg = np.array(co14_neg[order])
        
        
        hdus_fast = fits.open(mu_fast_file)

        # Set up mu_fast 14CO profiles.
        z_fast = hdus_fast['DEPTH'].data
        co14_fast = hdus_fast['CO14'].data
        t_spike_fast = hdus_fast['T_SPIKE'].data['T_SPIKE']

        order = np.argsort(t_spike_fast)
        
        self.CO_fast = np.array(co14_fast[order])
        
    def generate(self, f_neg = 0.066, f_fast = 0.072, depth_avg=20, rel_uncertainty=0.02):
        """Generate COProfile
        
        Parameters
        ----------
        f_neg : float or ndarray ( axis0: time )
            f_mu_neg value over time.
        f_fast : float or ndarray ( axis0: time )
            f_mu_fast value over time.
        depth_avg : int or float
            Depth averaging parameter in meters (force use of 10 or 20 m only).
        rel_uncertainty: float
            Relative fractional uncertainty in concentration measurement.
        """
        if len(np.shape(np.array(f_neg))) == 0:
            neg = float(f_neg)
        else:
            neg = np.expand_dims(np.array(f_neg), axis=1)
        
        if len(np.shape(np.array(f_fast))) == 0:
            fast = float(f_fast)
        else:
            fast = np.expand_dims(np.array(f_fast), axis=1)
            
        # axis0: time; axis1: depth
        # sum over axis0
        CO = np.sum(self.CO_neg * neg + self.CO_fast * fast, axis = 0)
        
        return COProfile(self.z, CO, depth_avg, rel_uncertainty)

class COProfile:
    """Storage of a 14CO profile calculation."""
    
    def __init__(self, z, CO, depth_avg=20, rel_uncertainty=0.02):
        """Initialize at 14CO profile.
        
        Parameters
        ----------
        z : ndarray
            True or ice-equivalent depth array [m].
        CO : ndarray
            14CO concentration profile [molecules/g].
        depth_avg : int or float
            Depth averaging parameter in meters (force use of 10 or 20 m only).
        rel_uncertainty: float
            Relative fractional uncertainty in concentration measurement.
        """
        self.z, self.CO = z, CO
        if depth_avg == 10:
            dz = 10.
        elif depth_avg == 20:
            dz = 20.
        else:
            raise ValueError('depth_avg = {} m invalid; only 10 m and 20 m supported.'.format(depth_avg))
        self.dCOrel = rel_uncertainty
          
        # Resample to block size required by the depth averageing parameter.
        blk = int(dz / np.diff(z)[0])
        z_reduced = []
        co_reduced = []
        
        i, n = 0, len(z)
        while i < n:
            j = i+blk if i+blk < n else n
            z_reduced.append(np.mean(z[i:j]))
            co_reduced.append(np.mean(CO[i:j]))
            i += blk
        
        self.z_samp = np.asarray(z_reduced)
        self.CO_model = np.asarray(co_reduced)
        
    def sample_z(self):
        """Return a sampled realization of the profile."""
        #CO_samp = [np.random.normal(co, self.dCOrel*co) for co in self.CO_model]
        CO_samp = np.random.normal(self.CO_model, self.dCOrel*self.CO_model)
        return self.z_samp, CO_samp, self.dCOrel*self.CO_model
    
    def sample_z_mult(self, N=1): # multiple samples at once
        COs = np.expand_dims(self.CO_model, axis=1)
        dCOs = np.expand_dims(self.dCOrel*self.CO_model, axis=1)
        
        CO_samp = np.random.normal(COs, dCOs, size = (len(self.CO_model), N))
        return self.z_samp, CO_samp, self.dCOrel*self.CO_model
