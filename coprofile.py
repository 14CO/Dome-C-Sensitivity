# -*- coding: utf-8 -*-
"""This module implements a representation of a 14CO vertical profile and can be used to generate realizations of a measurement with a particular depth average and statistical uncertainty."""

import numpy as np

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
        CO_samp = [np.random.normal(co, self.dCOrel*co) for co in self.CO_model]
        return self.z_samp, CO_samp, self.dCOrel*self.CO_model
