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
    PhiInit = tov.find_dilaton_center()[0]
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
    ge = ((D - np.sign(gamma) * 2/np.sqrt(3) * np.sqrt(1-D**2)/2))/((D +  np.sign(gamma) * 2/np.sqrt(3) * np.sqrt(1-D**2)/2))
    print('ge =', ge)
    ge_theta = tov.Ge_theta
    print('ge_theta =', ge_theta)

    print(' écart pourcent theta', (ge- ge_theta)/ge_theta *100, '\n')


    return (b_, D, rho_cen, gamma, ge, SoS_c_max)

# run(1500)

def make_gamma_beta_plots(n):
    nspace = n

    den_space = np.linspace(100,2000,num=n)
    beta_vec = np.array([])
    # delta_PN_vec = np.array([])


    vsurc_a = np.array([])
    gamma_edd_a = np.array([])
    gamma_edd_a_m = np.array([])
    # delta_PN = np.array([])
#

    for den in den_space:
        b_, D, rho_cen, gamma, ge, vsurc  = run(den)
#
        vsurc_a = np.append(vsurc_a,vsurc )
#
#
        gamma_edd_a = np.append(gamma_edd_a,ge )
#
        # delta = 4/3 * (( 1 - gamma**2 - 4 * gamma * xi)**2 - (1/4) * (1 + gamma**2)**2 ) * 1/(1 - gamma**2 + 4 * gamma * xi)**2
        # delta_PN = np.append(delta_PN,delta)


    index_max = np.where(vsurc_a > 1/np.sqrt(3))[0][0]
    den_space = den_space[0:index_max]
    # print('densité max', den_space[-1])
    gamma_edd_a = gamma_edd_a[0:index_max]
    # comment = '$L_m = -\\epsilon$'#-\\rho$'
    comment = '$L_m = -\\rho$'

    # delta_a = delta_PN[0:index_max]


    # fig,ax = plt.subplots()
    # plt.xlabel('Central density ($MeV/fm^3$)')
    # ax.plot(den_space,gamma_vec, label=f'$\\alpha$ ({comment})', color = 'tab:blue')
    # ax.set_ylabel('$\\alpha$', fontsize=12, color = 'tab:blue')
    #
    # ax2=ax.twinx()
    # ax2.plot(den_space,beta_vec, label=f'$\\beta$ ({comment})', color = 'tab:green', linestyle = 'dashed')
    # ax2.set_ylabel('$\\beta$', fontsize=12, color = 'tab:green')
    #
    # fig.legend( bbox_to_anchor=(0.9, 0.7))
    # # plt.savefig(f'figures/alpha_beta_{nspace}_{descr}.png', dpi = 200,bbox_inches='tight')
    # plt.show()

    fig,ax = plt.subplots()
    plt.xlabel('Central density ($MeV/fm^3$)')
    ax.plot(den_space,1-gamma_edd_a, label=f'$1-\\gamma_e$ ({comment})', color = 'tab:blue')
    ax.set_ylabel('$1-\\gamma_e$', fontsize=18)#, color = 'tab:blue')

    fig.legend( bbox_to_anchor=(0.9, 0.6))
    plt.savefig(f'./evaluating_gamma_with_alpha.png', dpi = 200,bbox_inches='tight')
    # plt.show()
    plt.close()

    fig,ax = plt.subplots()
    plt.xlabel('Central density ($MeV/fm^3$)')
    ax.plot(den_space,gamma_edd_a, label=f'$\\gamma_e$ ({comment})', color = 'tab:blue')
    ax.set_ylabel('$\\gamma_e $', fontsize=18)#, color = 'tab:blue')
    fig.legend( bbox_to_anchor=(0.9, 0.6))
    plt.savefig(f'./evaluating_gamma_exact.png', dpi = 200,bbox_inches='tight')
    # plt.show()
    plt.close()

    # Plot de delta :
    # fig,ax = plt.subplots()
    # plt.xlabel('Central density ($MeV/fm^3$)')
    # ax.plot(den_space,delta_a-1, label=f'$\\delta_e -1$ ({comment})', color = 'tab:blue')
    # ax.set_ylabel('$\\delta_e-1$', fontsize=18)#, color = 'tab:blue')
    #
    # fig.legend( bbox_to_anchor=(0.9, 0.6))
    # plt.savefig(f'./evaluating_delta_with_alpha.png', dpi = 200,bbox_inches='tight')
    # # plt.show()
    # plt.close()


    # plt.close()
    # fig,ax = plt.subplots()
    # plt.xlabel('$\\gamma - 1$')
    # ax.plot(gamma_edd_a_m, Dipolar, label=f' dipolar emissions{comment}', color = 'tab:blue')
    # ax.set_ylabel('Dipolar constraints', fontsize=12, color = 'tab:blue')
    #
    # fig.legend( bbox_to_anchor=(0.9, 0.7))
    # # plt.savefig(f'figures/alpha_beta_{nspace}_{descr}.png', dpi = 200,bbox_inches='tight')
    # plt.show()

make_gamma_beta_plots(20)
