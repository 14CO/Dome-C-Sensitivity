%% Muonic 14C production rate calculation
% Based on P_mu_total bade by greg balco (2016)
% Updates by bhmiel

function [P_neg,P_fast] = Balco_P_mu_total(z,h)

%% Original code introduction by Balco
% Calculates the production rate of Al-26 or Be-10 by muons
% as a function of depth below the surface z (g/cm2) and
% site atmospheric pressure h (hPa).
% 
% syntax out = P_mu_total(z,h,consts,dflag);
%
% consts is a structure containing nuclide-specific constants, as follows:
% consts.Natoms -  atom number density of the target atom (atoms/g)
% consts.k_neg - summary cross-section for negative muon capture (atoms/muon)
% consts.sigma190 - 190 GeV x-section for fast muon production (cm2)
%
% In version 1.2 you can enter consts.sigma0, which is the x-section
% for fast muon production at 1 Gev (see Heisinger), which is equivalent to
% sigma190 if you know alpha. Enter either consts.sigma190 or
% consts.sigma0, not both. 
%
% dflag = 'no' causes it to return only the nuclide production rate
% from muon reactions. Note that this is the total production rate from
% muons, that is, the sum of production by negative muon capture and 
% production by fast muon reactions. 
%
% dflag = 'yes' causes it to return a structure containing the complete
% breakdown of fluxes, stopping rates, production rates, etc. --
% 
% out.phi_vert_slhl muons/cm2/s/sr
% out.R_vert_slhl muons/g/s/sr
% out.R_vert_site muons/g/s/sr
% out.phi_vert_site muons/cm2/s/sr
% out.phi muons/cm2/yr
% out.R muons/g/yr
% out.P_fast atoms/g/yr
% out.P_neg atoms/g/yr
% out.Beta nondimensional
% out.Ebar GeV
% out.H g/cm2
% out.LZ g/cm2
%
% See the hard-copy documentation for more details. 
%
% This uses the scheme in Heisinger and others (2002, 2 papers). The
% vertically traveling muon flux is scaled to the site elevation using
% energy-dependent attenuation lengths from Boezio et al. (2000). See the 
% hard-copy documentation for detailed citations and a full discussion of
% the calculation. 
%
% Note that some constants are internal to the function. The only ones that
% get passed from upstream are the ones that a) are nuclide-specific, or b) 
% actually have quoted uncertainties in Heisinger's papers. 
% The fraction of muons that are negative is internal; so is the
% energy-dependence exponent alpha.
% 
%
% Written by Greg Balco -- UW Cosmogenic Nuclide Lab
% balcs@u.washington.edu
% March, 2006
% Part of the CRONUS-Earth online calculators: 
%      http://hess.ess.washington.edu/math
%
% Copyright 2001-2007, University of Washington
% Subsequently Greg Balco, Berkeley Geochronology Center
% All rights reserved
% Developed in part with funding from the National Science Foundation.
%
% This is version 1.2 which permits entering either sigma0 or sigma190.
% This version by Greg Balco, September 2016.
%
% This program is free software; you can redistribute it and/or modify
% it under the terms of the GNU General Public License, version 2,
% as published by the Free Software Foundation (www.fsf.org).

% what version is this?

ver = '1.3';

% Hardcoded constants specific to 14C production
% Michael's Version for silicon
% consts.Natoms = [2.00600000000000e+22];
% consts.k_neg = [0.0176306944000000];
% consts.sigma190 = [4.50000000000000e-28]; 

% My correct numbers
consts.Natoms = 6.022E23./18.02; %Avogadro's Number ./MW of H2O
consts.k_neg = .137 .* .1828; % See heisenger 02, product of f_star (.137), f_C (1) & f_D (.1828)
consts.sigma190 = 4.50000000000000e-28; 

% remember what direction the z vector came in

in_size = size(z);

% standardize direction

if size(z,1) > 1
    z = z';
end

% figure the difference in atmospheric depth from sea level in g/cm2

H = (1013.25 - h).*1.019716;   %the 1.019716 number is basically just 1/g accounting for needed unit conversions

% find the vertical flux at SLHL

a = 258.5*(100.^2.66);
b = 75*(100.^1.66);

phi_vert_slhl = (a./((z+21000).*(((z+1000).^1.66) + b))).*exp(-5.5e-6 .* z);

% The above expression is only good to 2e5 g/cm2. We don't ever consider production
% below that depth. The full-depth scheme appears in the comments below.
% ------ begin full-depth flux equations -------
%phiz_1 = (a./((z+21000).*(((z+1000).^1.66) + b))).*exp(-5.5e-6 .* z);
%phiz_2 = 1.82e-6.*((121100./z).^2).*exp(-z./121100) + 2.84e-13;
%out(find(z<200000)) = phiz_1(find(z<200000));
%out(find(z>=200000)) = phiz_2(find(z>=200000));
% ------ end full-depth flux equations -------

% find the stopping rate of vertical muons at SLHL
% this is done in a subfunction Rv0, because it gets integrated later.

R_vert_slhl = Rv0(z);

% find the stopping rate of vertical muons at site

R_vert_site = R_vert_slhl.*exp(H./LZ(z));

% find the flux of vertical muons at site

for a = 1:length(z)
    % integrate
    % ends at 200,001 g/cm2 to avoid being asked for an zero
    % range of integration -- 
    % get integration tolerance -- want relative tolerance around
    % 1 part in 10^4. 
    tol = phi_vert_slhl(a) * 1e-4;
    [temp,fcnt] = quad(@(x) Rv0(x).*exp(H./LZ(x)),z(a),(2e5+1),tol);
    % second variable assignment here to preserve fcnt if needed
    phi_vert_site(a) = temp;
end
   
% invariant flux at 2e5 g/cm2 depth - constant of integration
% calculated using commented-out formula above
phi_200k = (a./((2e5+21000).*(((2e5+1000).^1.66) + b))).*exp(-5.5e-6 .* 2e5);
phi_vert_site = phi_vert_site + phi_200k;

% find the total flux of muons at site

% angular distribution exponent
nofz = 3.21 - 0.297.*log((z+H)./100 + 42) + 1.21e-5.*(z+H);
% derivative of same
dndz = (-0.297./100)./((z+H)./100 + 42) + 1.21e-5;

phi_temp = phi_vert_site .* 2 .* pi ./ (nofz+1);

% that was in muons/cm2/s
% convert to muons/cm2/yr

phi = phi_temp*60*60*24*365.25;

% find the total stopping rate of muons at site

R_temp = (2.*pi./(nofz+1)).*R_vert_site ... 
    - phi_vert_site.*(-2.*pi.*((nofz+1).^-2)).*dndz;
    
% that was in total muons/g/s
% convert to negative muons/g/yr

% R = R_temp*0.44*60*60*24*365.25;
R = R_temp*1/(1+1.268)*60*60*24*365.25;
% Now calculate the production rates. 

% Depth-dependent parts of the fast muon reaction cross-section

Beta = 0.846 - 0.015 .* log((z./100)+1) + 0.003139 .* (log((z./100)+1).^2);
Ebar = 7.6 + 321.7.*(1 - exp(-8.059e-6.*z)) + 50.7.*(1-exp(-5.05e-7.*z)); % in GeV

% internally defined constants

aalpha = 0.75;  %0.75 is default
%aalpha = 1.0;
%display('Alert: alpha set to 1.0 for fast muons')

% Figure out which cross-section entered
if isfield(consts,'sigma0')
    sigma0 = consts.sigma0;
else
    sigma0 = consts.sigma190./(190.^aalpha);
end

% fast muon production

P_fast = phi.*Beta.*(Ebar.^aalpha).*sigma0.*consts.Natoms;

% negative muon capture

P_neg = R.*consts.k_neg;
 
% Old output from original code no longer needed
% % return 
% 
% if strcmp(dflag,'no');
%     out = P_fast + P_neg;
% elseif strcmp(dflag,'yes');
%     out.phi_vert_slhl = phi_vert_slhl;
%     out.R_vert_slhl = R_vert_slhl;
%     out.phi_vert_site = phi_vert_site;
%     out.R_vert_site = R_vert_site;
%     out.phi= phi;
%     out.R = R;
%     out.Beta = Beta;
%     out.Ebar = Ebar;
%     out.P_fast = P_fast;
%     out.P_neg = P_neg;
%     out.H = H;
%     out.LZ = LZ(z);
%     out.ver = ver;
% end;

% -------------------------------------------------------------------------


function out = Rv0(z);
%% Muonic Stopping rate Subfunction
% this subfunction returns the stopping rate of vertically traveling muons
% as a function of depth z at sea level and high latitude.


a = exp(-5.5e-6.*z);
b = z + 21000;
c = (z + 1000).^1.66 + 1.567e5;
dadz = -5.5e-6 .* exp(-5.5e-6.*z);
dbdz = 1;
dcdz = 1.66.*(z + 1000).^0.66;

out = -5.401e7 .* (b.*c.*dadz - a.*(c.*dbdz + b.*dcdz))./(b.^2 .* c.^2);

% full depth calculation appears in comments below
%R_1 = -5.401e7 .* (b.*c.*dadz - a.*(c.*dbdz + b.*dcdz))./(b.^2 .* c.^2);
%f = (121100./z).^2;
%g = exp(-z./121100);
%dfdz = (-2.*(121100.^2))./(z.^3);
%dgdz = -exp(-z./121100)./121100;
%R_2 = -1.82e-6.*(g.*dfdz + f.*dgdz);
%out(find(z<200000)) = R_1(find(z<200000));
%out(find(z>=200000)) = R_2(find(z>=200000));

% -------------------------------------------------------------------------

function out = LZ(z);
%% Attenuation length subfunction
% this subfunction returns the effective atmospheric attenuation length for
% muons of range Z

% define range/momentum relation
% table for muons in standard rock in Groom and others 2001

data = [4.704e1 8.516e-1
    5.616e1 1.542e0
    6.802e1 2.866e0
    8.509e1 5.698e0
    1.003e2 9.145e0
    1.527e2 2.676e1
    1.764e2 3.696e1
    2.218e2 5.879e1
    2.868e2 9.332e1
    3.917e2 1.524e2
    4.945e2 2.115e2
    8.995e2 4.418e2
    1.101e3 5.534e2
    1.502e3 7.712e2
    2.103e3 1.088e3
    3.104e3 1.599e3
    4.104e3 2.095e3
    8.105e3 3.998e3
    1.011e4 4.920e3
    1.411e4 6.724e3
    2.011e4 9.360e3
    3.011e4 1.362e4
    4.011e4 1.776e4
    8.011e4 3.343e4
    1.001e5 4.084e4
    1.401e5 5.495e4
    2.001e5 7.459e4
    3.001e5 1.040e5
    4.001e5 1.302e5
    8.001e5 2.129e5];

% ignore ranges bigger than 2e5 g/cm2
%    1.000e6 2.453e5
%    1.4e6   2.990e5
%    2.0e6   3.616e5
%    3.0e6   4.384e5
%    4.0e6   4.957e5
%    8.0e6   6.400e5
%    1.0e7   6.877e5
%    1.4e7   7.603e5
%    2.0e7   8.379e5
%    3.0e7   9.264e5
%    4.0e7   9.894e5
%    8.0e7   1.141e6
%    1.0e8   1.189e6];

% units are range in g cm-2 (column 2)
% momentum in MeV/c (column 1)

% deal with zero situation

too_low = find(z < 1);
z(too_low) = ones(size(too_low));   %just sets all depths that are considered too low to 1 g/cm^2

% obtain momenta
% use log-linear interpolation

P_MeVc = exp(interp1q(log(data(:,2)),log(data(:,1)),log(z')))';

% obtain attenuation lengths

out = 263 + 150 .* (P_MeVc./1000);

% -------------------------------------------------------------------------

