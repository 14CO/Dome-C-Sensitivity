% Simple model for Dome C cosmogenic 14CO accumulation.
% This is a more standalone version that does not require the firn model
% to be run immediately before this model is run. Still calls the
% Balco_P_mu_total script
% The model calculates 14CO below ≈lock-in depth using the Balco muon 
% parameterization adjustable tuning factors (F*Omega) that account for the
% fractional reduction (F) in total 14C production rate as compared to the 
% original Balco / Heisinger parameterizations, and for the fraction of
% total produced 14C that forms 14CO (Omega). These tuning factors are
% based on the range of values that seem plausible given our TG and Summit
% 14CO results
% This  version of the model moves the ice layers (model boxes) down
% each year as prescribed by a depth - ice layer age scale
% Also allows to investigate effects of production rate that is changing with time
% By Vas Petrenko. Modified from "Deep_Ice_14CO_standalone.m"
% Most recent update April 10 2023

clear all
%close all

% load the desired depth-age scale (real depths)
% note that this script assumes that the age scale has annual resolution
% i.e. exactly 1-year increments

%load('Trial_age_scale.mat') % This is a depth-age scale that simulates
% previous version of this model that assumed a constant accumulation and
% no ice thinning. Use for testing purposes only

load('DomeC_age_scale_Apr2023.mat')   %This is the version of the depth-age scale that should be used for new 14CO calculations

% load the coversion table between real and ice equivalent depths at Dome C
% and convert age scale to ice equivalent depths 
load('Real_vs_ice_eq_depth.mat') 
depths_ice_eq = interp1(z, ice_eq_z, depths_real);

% Set the desired start depth:
z_start = 96.5;   % this is the starting real depth of this model in m
% choosing 96.5m for z_end produces very good agreement for 14CO with the
% firn model at the deepest levels of that model (below 100 m) 
z_start_offset = 0; %this is for investigating the effect of uncertainty in lock-in depth (in real depth)
z_start = z_start + z_start_offset;
ice_eq_z_start = interp1(z, ice_eq_z, z_start); %this is the starting depth in m ice equivalent;
%find the index of the age scale array that corresponds to this start
%depth:
[~, start_index] = min(abs(depths_ice_eq - ice_eq_z_start));

% Set the desired end depth:
z_deep_actual = 300;  % this is the end real depth in m
z_deep_ice_eq = interp1(z, ice_eq_z, z_deep_actual); % in ice equivalent depth
%find the index of the age scale array that corresponds to this end
%depth:
[min_diff, end_index] = min(abs(depths_ice_eq - z_deep_ice_eq));

% define the depth grid (m ice equiv depth); this corresponds to annual
% layers from depth-age scale
ice_z = depths_ice_eq(start_index : end_index);

% define some parameters needed for the in situ 14CO calculations
rho_ice = 0.9239; % density of solid ice at Dome C
pressure = 65800; %surface pressure in Pa, should be 65800 for Dome C
lambda = 1.21e-4; %14C decay constant

% Calculate 14CO production rates by negative muon capture and fast muons
% for the depths of interest
% This uses Greg Balco's code for 14C production by muons (Heisenger et al parameterizations) 
[Balco_mu_neg,Balco_mu_fast] = Balco_P_mu_total(100*ice_z*rho_ice,pressure./100);        
% Need to transpose output... 
Balco_mu_neg = Balco_mu_neg';
Balco_mu_fast = Balco_mu_fast';

% Loop over F*Omega factors allowed by Dyonisius et al.
factors = readmatrix('factors_2sigma_hull.csv');

for ifac=1 : size(factors)
% for ifac=1 : 1
    factor = factors(ifac,:);
    negfac = factor(1);
    fstfac = factor(2);

    % negfac = 0.066;
    % fstfac = 0.072;

    P_mu_neg_14CO = Balco_mu_neg.*negfac; %this applies the F*Omega tuning factor discussed in intro
    P_mu_fast_14CO = Balco_mu_fast.*fstfac;    
      
    % Loop over step change values.
    P_changes = linspace(0.5, 1.5, 101);
    for k=1 : length(P_changes)
    
        outf = sprintf('step_models_pres/co14_step_%.4f_%.4f_%.4f.csv', P_changes(k), negfac, fstfac);
        fprintf('%3d %.4f %.4f %3d %.4f %s\n', ifac, negfac, fstfac, k, P_changes(k), outf);
    
        % Define the time array - needed for time-variable production rates
        t_step = 1.0; %years. Note: I am not sure that the current version would work properly if this is not 1; would need to look into it
        duration = ages(end_index) - ages(start_index); % duration of model run in years
        no_t_pts = int16(duration); % convert to integer
        t = zeros(1,no_t_pts);
        
        % Define the results array
        no_depths = length(ice_z);  % for an annual age scale this will be the same as number of time points
        C14_CO = zeros(1,no_depths);  % this will be in molecules/g
        C14_CO_temp = zeros(1,no_depths);
        
        % Production rate matrices (in time and depth):
        Pmn = ones(no_t_pts, no_depths); %first index is the time dimension, 2nd index is the depth dimension
        Pmf = ones(no_t_pts, no_depths);
        
        %Assign the 14CO value to the top box of this model:
        C14_CO(1) = 0;
        
        % Assign constant production rates:
        for i=1 : no_depths
            Pmn(:,i) = P_mu_neg_14CO(i);
            Pmf(:,i) = P_mu_fast_14CO(i);
        end
        
        
        % Now calculate the time-evolving 14CO concentration:
        % CASE 1:  a step change in production rates halfway through run:
        % --------------------------------------
        %P_change = P_changes(k); % magnitude (factor) of step change in production rate
        %half_way = int16(no_t_pts/2);
        %Pmn(half_way:no_t_pts,:) = P_change.*Pmn(half_way:no_t_pts,:);
        %Pmf(half_way:no_t_pts,:) = P_change.*Pmf(half_way:no_t_pts,:);
        %-----------------------------------------------------

        % CASE 1A:  a step change in production rates halfway through run
        % WITH fixed point at present day
        % --------------------------------------
        P_change = P_changes(k); % magnitude (factor) of step change in production rate
        half_way = int16(no_t_pts/2);
        Pmn(1:half_way,:) = Pmn(1:half_way,:)/P_change;
        Pmf(1:half_way,:) = Pmf(1:half_way,:)/P_change;
        %-----------------------------------------------------
        
        %CASE 2: a gradual linear transition in production rates:
        %------------------------------------------
        % P_change = P_changes(k);  %values above one mean production rates increase with time
        % P_change_step = (P_change - 1)/double(no_t_pts - 1);
        % 
        % for i=2 : no_t_pts  % i is serving as the counter for time
        %     Pmn(i,:) = double(1+ P_change_step*double(i-1))*Pmn(i,:);
        %     Pmf(i,:) = double(1+ P_change_step*double(i-1))*Pmf(i,:);
        % end
        %-------------------------------------------
        
        % CASE 3: A transient spike in production rates:
        %------------------------------------------
        % P_change = P_changes(k); % magnitude (factor) of transient change in production rate
        % spike_start = 3000; %timestep at which the production rate spike begins
        % spike_dur = 100; %number of time steps for which production rate spike persists
        % spike_end = spike_start + spike_dur;
        % Pmn(spike_start:spike_end,:) = P_change*Pmn(spike_start:spike_end,:);
        % Pmf(spike_start:spike_end,:) = P_change*Pmf(spike_start:spike_end,:);
        %-------------------------------------------
        
        Pmf_top = Pmf(:,1); % this is a diagnostic to ensure production rate really changes with time
        
        % Now calculate 14CO:
        for i=1 : no_t_pts % i is serving as the counter for time 
            delta_C14 = (Pmn(i,:) + Pmf(i,:) - lambda*C14_CO)*t_step; %14C change at every depth level
            for j=2 : no_depths % j is serving as the counter for depth
                C14_CO_temp(j) = C14_CO(j-1) + delta_C14(j-1);  %update the 14CO concentration in each box and shift boxes down
            end
            C14_CO = C14_CO_temp;
        end
        
        %Avg14CO = mean(C14_CO)
        
        % convert back to real depth
        real_z = interp1(ice_eq_z,z,ice_z);
        tab = array2table([real_z, transpose(C14_CO)]);
        tab.Properties.VariableNames(1:2) = {'z', 'co14_co'};
        writetable(tab, outf);
    
        %figure
        plot(real_z, C14_CO)
        xlabel('depth [m]')
        ylabel('^{14}CO molecules/g ice')
        hold on
    %plot(ice_z, C14_CO)
    end
end
