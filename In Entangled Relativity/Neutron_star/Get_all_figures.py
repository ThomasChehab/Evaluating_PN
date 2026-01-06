from TOV import *
import matplotlib
import matplotlib.pyplot as plt
import scipy.constants as cst
import scipy.optimize
import numpy as np
import math
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d, splrep

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

from matplotlib.transforms import Bbox
import mplhep as hep
hep.style.use("ATLAS")

# In[2]:


def run(rho_cen):
    PhiInit = 1
    PsiInit = 0
    radiusMax_in = 40000
    radiusMax_out = 10000000
    Npoint = 1000000
    log_active = False
    dilaton_active = True
    rhoInit = rho_cen*cst.eV*10**6/(cst.c**2*cst.fermi**3)
    tov = TOV(rhoInit, PsiInit, PhiInit, radiusMax_in, radiusMax_out, Npoint, dilaton_active, log_active)
    tov.ComputeTOV()
    r = tov.radius
    a = tov.g_tt
    b = tov.g_rr
    phi = tov.Phi
    phi_dot = tov.Psi
    radiusStar = tov.radiusStar
    mass_ADM = tov.massADM / (1.989*10**30) # in solar mass
    a_dot = (-a[1:-2]+a[2:-1])/(r[2:-1]-r[1:-2])
    b_dot = (-b[1:-2]+b[2:-1])/(r[2:-1]-r[1:-2])
    f_a = -a_dot*r[1:-2]*r[1:-2]/1000
    f_b = -b_dot*r[1:-2]*r[1:-2]/1000
    f_phi = -phi_dot*r*r/1000
    SoS_c_max = np.max(tov.v_c)
    b_ = 1/(2*np.sqrt(3))
    C = f_b[-1]/f_phi[-1]
    a1 = 1
    a2 = 4*b_*(C+1)
    a3 = -1
    a4 = a2*a2-4*a1*a3
    gamma = (-a2-np.sqrt(a4))/(2*a1)
    r_m = 1000*f_phi[-1]*(1+gamma**2)/(4*b_*gamma)
    comment = '$L_m = -\\rho$'
    descr = 'rho'
    D = (1-gamma**2)/(1+gamma**2)
    ge = ((D - np.sign(gamma) * 1/np.sqrt(3) * np.sqrt(1-D**2)))/((D +  np.sign(gamma) * 1/np.sqrt(3) * np.sqrt(1-D**2)))
    # print('ge =', ge)
    ge_theta = tov.Ge_theta
    # print('ge_theta =', ge_theta)
    print(' écart pourcent theta', (ge- ge_theta)/ge_theta *100, '\n')

    delta = 4/3 * ( ge**2 - 1/4 * (D + np.sign(gamma) * np.sqrt((1-D**2 )/(3)))**(-2) )

    delta_theta = tov.Delta_theta
    print('écart pourcent delta', (delta - delta_theta)/delta_theta * 100, '\n')



    return (b_, D, rho_cen, gamma, ge_theta, delta_theta, SoS_c_max)

# run(1500)

def make_gamma_beta_plots(n):
    nspace = n

    den_space = np.linspace(100,2000,num=n)
    beta_vec = np.array([])
    vsurc_a = np.array([])
    gamma_edd_a = np.array([])
    delta_edd_a = np.array([])


    for den in den_space:
        b_, D, rho_cen, gamma, ge, de, vsurc  = run(den)

        vsurc_a = np.append(vsurc_a,vsurc )
        gamma_edd_a = np.append(gamma_edd_a,ge )
        delta_edd_a = np.append(delta_edd_a,de )

    index_max = np.where(vsurc_a > 1/np.sqrt(3))[0][0]
    den_space = den_space[0:index_max]
    # print('densité max', den_space[-1])
    gamma_edd_a = gamma_edd_a[0:index_max]
    delta_edd_a = delta_edd_a[0:index_max]

    # comment = '$L_m = -\\epsilon$'#-\\rho$'
    comment = '$L_m = -\\rho$'


    fig,ax = plt.subplots()
    plt.xlabel('Central density ($MeV/fm^3$)')
    ax.plot(den_space,(1-gamma_edd_a), label=f'$1-\\gamma_e$ ({comment})', color = 'tab:blue')
    ax.set_ylabel('$1-\\gamma_e$', fontsize=18)#, color = 'tab:blue')
    fig.legend( bbox_to_anchor=(0.9, 0.6))
    plt.savefig(f'./gamma_exact_m.png', dpi = 200,bbox_inches='tight')
    # plt.show()
    plt.close()

    # Plot de delta :
    fig,ax = plt.subplots()
    plt.xlabel('Central density ($MeV/fm^3$)')
    ax.plot(den_space,(1-delta_edd_a), label=f'$1-\\delta_e$ ({comment})', color = 'tab:blue')
    ax.set_ylabel('$1-\\delta_e$', fontsize=18)#, color = 'tab:blue')
    fig.legend( bbox_to_anchor=(0.9, 0.6))
    plt.savefig(f'./delta_exact_m.png', dpi = 200,bbox_inches='tight')
    # plt.show()
    plt.close()


make_gamma_beta_plots(150)
