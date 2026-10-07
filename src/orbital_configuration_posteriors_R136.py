#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
orbital_configuration_posteriors_R136.py

Monte Carlo simulation of the orbital configuration posterior
distributions from Tramper+2026 adapted for the massive stellar
systems R136-A1 and R136-A2 nd teh GRAVITY+ observations.

This code computes posterior probability density functions for
projected separations, semi-major axes, orbital periods,
radial velocities, and tangential velocities under assumptions
about the distributions of eccentricity, inclination, and
argument of periastron.

Author
------
Hugues Sana

Affiliation
-----------
Institute of Astronomy
KU Leuven
Belgium

Repository
----------
https://github.com/hsana/<repository>R136_2026

Associated publication
----------------------
Tramper, Sana, de Koter, Pauwels, A&A 711, A236 (2026), doi:10.1051/0004-6361/202557503
Gravity Collaboration et al., Science, submitted, 2026.

Citation
--------
If you use this software in scientific work, please cite:

Tramper, F. et al. (2026),
Southern massive stars at high angular resolution: Physical separations and mass ratios,
Astronomy and Astrophysics,
Volume 711, id.A236, 14 pp.
DOI:  10.1051/0004-6361/202557503

Gravity Collaboration et al. (2026),
Observational Evidence for Individual Stars Above 200 Solar Masses in the Local Universe
Science
in press
DOI:

License
-------
BSD 3-Clause License

Copyright (c) 2026 Hugues Sana

"""

import numpy as np
import matplotlib.pyplot as plt

# --- Utilities ---
def asymmetric_gaussian(n, median=0.0, sigma_left=1.0, sigma_right=1.0, rng=None):
    """
    Draw N samples from an asymmetric Gaussian distribution using a NumPy Generator.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Randomly choose left/right side
    signs = rng.choice([-1, 1], size=n)

    # Half-normal magnitudes
    magnitudes = np.abs(rng.normal(loc=0.0, scale=1.0, size=n))

    # Scale depending on side
    scaled = np.where(signs < 0,
                      magnitudes * sigma_left,
                      magnitudes * sigma_right)

    return median + scaled * signs


def solve_kepler_E(M, e, tol=1e-12, maxiter=50):
    M = np.asarray(M)
    Mwrap = (M + np.pi) % (2*np.pi) - np.pi
    e_arr = np.full_like(Mwrap, e, dtype=float)
    E = Mwrap + e_arr * np.sin(Mwrap) / (1 - np.sin(Mwrap + e_arr) + np.sin(Mwrap))
    for _ in range(maxiter):
        f = E - e_arr*np.sin(E) - Mwrap
        fp = 1 - e_arr*np.cos(E)
        dE = -f / fp
        E += dE
        if np.max(np.abs(dE)) < tol:
            break
    return E

def sample_sep_over_a(
        n,
        e_spec=None,
        i_spec=None,
        omega_spec=None,
        e_low=0.0,
        e_high=0.9,
        rng=None,
        object='A1',
        output='Period'):
    """
    Generate Monte Carlo realizations of orbital configurations.

    Parameters
    ----------
    n : int
        Number of Monte Carlo samples.

    e_spec : float, optional
        Fixed eccentricity. If None, eccentricities are drawn from
        a uniform distribution between e_low and e_high.

    i_spec : float, optional
        Fixed inclination in radians. If None, inclinations are
        sampled isotropically.

    omega_spec : float, optional
        Fixed argument of periastron in radians. If None, values are
        drawn uniformly between 0 and 2π.

    e_low, e_high : float
        Bounds of the eccentricity prior.

    rng : numpy.random.Generator, optional
        Random number generator.

    object : {'A1','A2'}
        Target stellar system.

    output : str
        Quantity to return. Possible values are:

        - 'Ratio'
        - 'Proj_Separation'
        - 'Semi-major_axis'
        - 'Period'
        - 'RV'
        - 'Vtan_mas'
        - 'Vtan_micas'

    Returns
    -------
    ndarray
        Monte Carlo samples of the requested quantity.
    """
    if rng is None:
        rng = np.random.default_rng()
    if e_spec is None:
        e = rng.uniform(e_low, e_high, size=n)
    else:
        e = np.full(n, float(e_spec))
    if i_spec is None:
        cosi = rng.uniform(-1.0, 1.0, size=n)
        i = np.arccos(cosi)
    else:
        i = np.full(n, float(i_spec))
        cosi=np.full(n,np.cos(i))
    if omega_spec is None:
        omega = rng.uniform(0.0, 2*np.pi, size=n)
    else:
        omega = np.full(n, float(omega_spec))

    sini=np.sqrt(1-cosi**2)
    M = rng.uniform(0.0, 2*np.pi, size=n)
    E = M.copy()
    for _ in range(7):
        f = E - e*np.sin(E) - M
        fp = 1 - e*np.cos(E)
        E -= f / fp
    beta = np.sqrt((1+e)/(1-e))
    nu = 2*np.arctan2(beta * np.sin(E/2), np.cos(E/2))
    r_over_a = 1 - e*np.cos(E)
    k = np.sqrt(1 - (np.sin(i)**2) * (np.sin(nu + omega)**2))
    sep_over_a = r_over_a * k


    if object == 'A1':
        m1=asymmetric_gaussian(n=n,median=237,sigma_left=-38 , sigma_right=+59, rng=rng) #Msun
        m2=asymmetric_gaussian(n=n,median=135,sigma_left=-21 , sigma_right=+18, rng=rng) #Msun
        x2 = 2.2613+rng.standard_normal(size=n)* 0.0066 #mas
        y2 = 14.224+rng.standard_normal(size=n)*0.0121  #mas
             
    else:
        #OBJECT A2
        m1=asymmetric_gaussian(n=n,median=208,sigma_left=-30 , sigma_right=+36, rng=rng) #Msun
        m2=asymmetric_gaussian(n=n,median=78 ,sigma_left=-8  , sigma_right=+10, rng=rng) #Msun
        x2 = -19.623+rng.standard_normal(size=n)*0.00669 #mas
        y2 = -12.865+rng.standard_normal(size=n)*0.0128  #mas
    mass_msun=m1+m2
    sep_mas=np.sqrt(x2**2+y2**2)
    
    dist_kpc=49.59+rng.standard_normal(size=n)*0.09+rng.standard_normal(size=n)*0.54
    
    sep_au=sep_mas*dist_kpc

    
    pc2au=206265.
    au2km=149597828.68
    yr2sec=365.25*24*3600. 
    Msun_kg=1.989E30 #kg
    GM_kms=1.3271244E11 #km3/s2
    G_auMsun=GM_kms/au2km # pc/Msun/(km/s)^2
    
    a_au=sep_au/sep_over_a
    a1_km=a_au*m2/(m1+m2)*au2km
    P_yr=sep_au**1.5/mass_msun**0.5 / sep_over_a**1.5
    P_sec=P_yr*yr2sec
     
    K1_kms=2*np.pi*sini/P_sec/np.sqrt(1-e*e)*a1_km
    rv = K1_kms * (np.cos(nu+omega)+e*np.cos(omega))

    mu_au=G_auMsun*mass_msun # au/(km/s)^2
    p_au=a_au*(1-e*e)
    v_radial=np.sqrt(mu_au/p_au)*e*np.sin(nu) #kms/s
    v_tan   =np.sqrt(mu_au/p_au)*(1+e*np.cos(nu)) #km/s

    v_x= v_radial*np.cos(nu+omega)-v_tan*np.sin(nu+omega)       #km/s
    v_y=(v_radial*np.sin(nu+omega)+v_tan*np.cos(nu+omega))*cosi #km/s
    v_sky=np.sqrt(v_x*v_x+v_y*v_y)    #km/s    
    
    if output ==  'Ratio':
        results=sep_over_a
    elif output == 'Proj_Separation':
        results=sep_au
    elif output == 'Semi-major_axis':
        results=a_au
    elif output == 'Period':
        results=P_yr
    elif output == 'RV':
        results = rv  #km/s
    elif output == 'Vtan_mas':
        results=v_sky / au2km / dist_kpc * yr2sec  #mas
    elif output == 'Vtan_micas':
        results=v_sky / au2km / dist_kpc * yr2sec *1000 #micro-as
    return results

def hdi_bounds(data, cred_mass):
    data = np.sort(data)
    interval_idx_inc = int(np.floor(cred_mass * len(data)))
    n_intervals = len(data) - interval_idx_inc
    interval_width = data[interval_idx_inc:] - data[:n_intervals]
    min_idx = np.argmin(interval_width)
    return data[min_idx], data[min_idx + interval_idx_inc]

# --- Plotting ---
rng = np.random.default_rng(42)
nbins = 200
#xmax = 200
#xmin = 2.

object='A1'
output='Period'
output='Semi-major_axis'
#output='Ratio'
#output='RV'
##output='Vtan_mas'
#output='Vtan_micas'
if output == 'RV':
    xmax=15
    xmin=-15
    scale_choice = 'linear'
    Ndigit=3
if output == 'Vtan_mas':
    xmax=.1
    xmin=0.001
    scale_choice = 'log'
    Ndigit=3
if output == 'Vtan_micas':
    xmax=125
    xmin=0
    scale_choice = 'linear'
    Ndigit=3
if output == 'Ratio':
    xmax=2
    xmin=0
    scale_choice = 'linear'
    Ndigit=3
elif output == 'Proj_Separation':
    xmax=2000
    xmin=200
    scale_choice = 'log'
    Ndigit=0
elif output ==  'Semi-major_axis':
    xmax=10000
    xmin=200
    scale_choice = 'log'
    Ndigit=0
elif output ==  'Period':
    xmax=20000
    xmin=200
    scale_choice = 'log'
    Ndigit=0

#scale_choice = 'linear'
bins = np.linspace(xmin, xmax, nbins + 1)
cent = 0.5 * (bins[:-1] + bins[1:])

fig_width_cm = 8.8   # 80% of 8.8cm
fig_width = fig_width_cm / 2.54  # in inches
fig_height = 22 / 2.54  # 22cm in inches

fig, axes = plt.subplots(4, 1, figsize=(fig_width, fig_height), sharex=True)
fig.subplots_adjust(left=0.1, right=0.98, top=0.7, bottom=0.1)
# Panel 1: p(Sep/a | e)
ax = axes[0]
E_vals = [0.0, 0.3, 0.6, 0.9]
colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(E_vals)))
for e, c in zip(E_vals, colors):
    x = sample_sep_over_a(300_000, e_spec=e, rng=rng,object=object,output=output)
    h, _ = np.histogram(x, bins=bins, density=False)
    pdf = h / np.trapz(h, x=cent)
    ax.plot(cent, pdf, color=c, lw=2.2, label=f"e={e:.1f}")
ax.set_ylabel('PDF')
ax.set_title('Conditional on e  (i isotropic, ω uniform)', fontsize=9)
ax.legend(frameon=False, ncol=1, loc='upper right', fontsize=8)
ax.set_xlim(xmin, xmax)
ax.set_ylim(bottom=0)
ax.set_xscale(scale_choice)

# Panel 2: p(Sep/a | i)
ax = axes[1]
I_vals_deg = [0, 30, 60, 80]
colors = plt.cm.cividis(np.linspace(0.1, 0.9, len(I_vals_deg)))
for ideg, c in zip(I_vals_deg, colors):
    x = sample_sep_over_a(300_000, i_spec=np.deg2rad(ideg), rng=rng,object=object,output=output)
    h, _ = np.histogram(x, bins=bins, density=False)
    pdf = h / np.trapz(h, x=cent)
    ax.plot(cent, pdf, color=c, lw=2.2, label=f"i={ideg}°")
ax.set_ylabel('PDF')
ax.set_title('Conditional on i  (e uniform in [0,0.9], ω uniform)', fontsize=9)
ax.legend(frameon=False, ncol=1, loc='upper right', fontsize=8)
ax.set_xlim(xmin, xmax)
ax.set_ylim(bottom=0)
ax.set_xscale(scale_choice)

# Panel 3: p(Sep/a | ω)
ax = axes[2]
W_vals_deg = [0, 30, 60, 90]
colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(W_vals_deg)))
for wdeg, c in zip(W_vals_deg, colors):
    x = sample_sep_over_a(300_000, omega_spec=np.deg2rad(wdeg), rng=rng,object=object,output=output)
    h, _ = np.histogram(x, bins=bins, density=False)
    pdf = h / np.trapz(h, x=cent)
    ax.plot(cent, pdf, color=c, lw=2.2, label=f"ω={wdeg}°")
ax.set_ylabel('PDF')
ax.set_title('Conditional on ω  (e uniform, i isotropic)', fontsize=9)
ax.legend(frameon=False, ncol=1, loc='upper right', fontsize=8)
ax.set_xlim(xmin, xmax)
ax.set_ylim(bottom=0)
ax.set_xscale(scale_choice)

# Panel 4: Fully marginalised p(Sep/a)
ax = axes[3]
x = sample_sep_over_a(900_000, rng=rng,object=object,output=output)
h, _ = np.histogram(x, bins=bins, density=False)
pdf = h / np.trapz(h, x=cent)
ax.plot(cent, pdf, color='k', lw=2.8)
ax.set_xscale(scale_choice)

# Compute HDIs and median
x_sorted = np.sort(x)
median = np.median(x_sorted)
#hdi_50 = hdi_bounds(x_sorted, 0.50)
hdi_68 = hdi_bounds(x_sorted, 0.68)
hdi_95 = hdi_bounds(x_sorted, 0.95)

# Shade HDIs
# Shade 68% HDI under the curve
mask_95 = (cent >= hdi_95[0]) & (cent <= hdi_95[1])
ax.fill_between(cent[mask_95], 0, pdf[mask_95], color='blue', alpha=0.08)

mask_68 = (cent >= hdi_68[0]) & (cent <= hdi_68[1])
ax.fill_between(cent[mask_68], 0, pdf[mask_68], color='orange', alpha=0.25)

# Shade 50% HDI under the curve
#mask_50 = (cent >= hdi_50[0]) & (cent <= hdi_50[1])
#ax.fill_between(cent[mask_50], 0, pdf[mask_50], color='orange', alpha=0.2)
#ax.axvspan(hdi_68[0], hdi_68[1], color='blue', alpha=0.2)
#ax.axvspan(hdi_50[0], hdi_50[1], color='orange', alpha=0.3)

# Draw vertical lines
#ax.axvline(median, color='red', lw=1.8,ls='--')
#ax.axvline(hdi_50[0], color='orange', lw=1.5, ls='--')
#ax.axvline(hdi_50[1], color='orange', lw=1.5, ls='--')
#ax.axvline(hdi_68[0], color='blue', lw=1.5, ls='--')
#ax.axvline(hdi_68[1], color='blue', lw=1.5, ls='--')
# Shade 68% HDI under the curve

# Draw vertical lines from 0 up to the PDF curve
def draw_vertical_fill(xval, color,label=None):
    idx = np.argmin(np.abs(cent - xval))
 #   ax.fill_between([cent[idx]-0.002, cent[idx]+0.002], 0, pdf[idx], color=color, alpha=0.8)
    ax.plot([cent[idx], cent[idx]], [0, pdf[idx]], color=color, linestyle='--', lw=1.5,label=label)
    
draw_vertical_fill(median, 'red',label='Median')
#draw_vertical_fill(hdi_50[0], 'orange',label='50% HDI')
#draw_vertical_fill(hdi_50[1], 'orange')
draw_vertical_fill(hdi_68[0], 'orange',label='68% HDI')
draw_vertical_fill(hdi_68[1], 'orange')
draw_vertical_fill(hdi_95[0], 'blue',label='95% HDI')
draw_vertical_fill(hdi_95[1], 'blue')

# Plot the PDF curve last (so it's always on top)
ax.plot(cent, pdf, color='k', lw=2.8)


# Annotate values
ymax = ax.get_ylim()[1]
if Ndigit == 3:
    if output ==  'Vtan_mas':
        ysh68=.2
    else:
        ysh68=.02
    ax.text(median   , ymax*0.05, f'{median:.3f}'   , color='red'   , ha='right', va='bottom', fontsize=10,rotation=90)
    ax.text(hdi_68[0], ymax*ysh68, f'{hdi_68[0]:.3f}', color='orange', ha='right' , va='bottom', fontsize=10,rotation=90)
    ax.text(hdi_68[1], ymax*ysh68, f'{hdi_68[1]:.3f}', color='orange', ha='left'  , va='bottom', fontsize=10,rotation=90)
    ax.text(hdi_95[0], ymax*0.05, f'{hdi_95[0]:.3f}', color='blue'  , ha='right' , va='bottom', fontsize=10,rotation=90)
    ax.text(hdi_95[1], ymax*0.05, f'{hdi_95[1]:.3f}', color='blue'  , ha='left'  , va='bottom', fontsize=10,rotation=90)
else:
    ax.text(median   , ymax*0.05, f'{median:.0f}'   , color='red'   , ha='right', va='bottom', fontsize=10,rotation=90)
    ax.text(hdi_68[0], ymax*0.02, f'{hdi_68[0]:.0f}', color='orange', ha='left' , va='bottom', fontsize=10,rotation=90)
    ax.text(hdi_68[1], ymax*0.02, f'{hdi_68[1]:.0f}', color='orange', ha='left'  , va='bottom', fontsize=10,rotation=90)
    ax.text(hdi_95[0], ymax*0.05, f'{hdi_95[0]:.0f}', color='blue'  , ha='right' , va='bottom', fontsize=10,rotation=90)
    ax.text(hdi_95[1], ymax*0.05, f'{hdi_95[1]:.0f}', color='blue'  , ha='left'  , va='bottom', fontsize=10,rotation=90)

if output ==  'Period':
    ax.set_xlabel('Period (year)')
elif output ==  'Semi-major_axis':
    ax.set_xlabel('a (au)')
elif output ==  'Proj_Separation':
    ax.set_xlabel('Proj. Separation (au)')
elif output ==  'Ratio':
    ax.set_xlabel('Sep/a')
elif output ==  'RV':
    ax.set_xlabel('RV (km/s)')
elif output ==  'Vtan_mas':
    ax.set_xlabel('Tangential velocity (mas/yr)')
elif output ==  'Vtan_micas':
    ax.set_xlabel('Tangential velocity (1e-3mas/yr)')
        
ax.set_ylabel('PDF')
ax.set_title('Fully marginalised  (e uniform, i isotropic, ω uniform)', fontsize=9)
ax.set_xlim(xmin, xmax)
ax.set_ylim(bottom=0)


ax.legend(frameon=False, ncol=1, loc='upper right', fontsize=8)


# Final layout and save
fig.tight_layout(pad=0.2)
fig.savefig(output+'_'+object+'.png', dpi=300, bbox_inches='tight')
plt.close(fig)
