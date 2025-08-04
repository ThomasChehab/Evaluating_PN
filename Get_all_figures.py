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

def run(opti,rho_cen):
    dependence = 2
    retro = False
    PhiInit = 1
    PsiInit = 0
    option = opti
    radiusMax_in = 40000
    radiusMax_out = 10000000
    Npoint = 1000000

    log_active = False
    dilaton_active = True
    rhoInit = rho_cen*cst.eV*10**6/(cst.c**2*cst.fermi**3)
    tov = TOV(rhoInit, PsiInit, PhiInit, radiusMax_in, radiusMax_out, Npoint, option, dilaton_active, log_active, retro, dependence)
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

    # print('f_a at infinity ', f_a[-1])
    # print('f_b at infinity ', f_b[-1])
    # print('f_phi at infinity ', f_phi[-1])
    if option == 1:
        b_ = 1/(2*np.sqrt(3))
        C = f_b[-1]/f_phi[-1]
        a1 = 1
        a2 = 4*b_*(C+1)
        a3 = -1
        a4 = a2*a2-4*a1*a3
        gamma = (-a2-np.sqrt(a4))/(2*a1)
        r_m = 1000*f_phi[-1]*(1+gamma**2)/(4*b_*gamma)
        comment = '$L_m = -\\epsilon$'#-\\rho$'
        descr = 'rho'
    elif option == 2:
        b_ = -1/(2*np.sqrt(3))
        C = f_b[-1]/f_phi[-1]
        a1 = 1
        a2 = 4*b_*(C+1)
        a3 = -1
        a4 = a2*a2-4*a1*a3
        gamma = (-a2-np.sqrt(a4))/(2*a1)
        r_m = 1000*f_phi[-1]*(1+gamma**2)/(4*b_*gamma)
        comment = '$L_m = P$'
        descr = 'P'
    else:
        print('not a valid option, try 1 or 2')
    r_2 = np.linspace(r_m,r[-1],num=100000)
    exp_phi = -4*b_*gamma/(1+gamma**2)
    exp_rho = (gamma**2)/(1+gamma**2)-exp_phi/2
    exp_a = (1-gamma**2)/(1+gamma**2)-exp_phi
    exp_b = -(1-gamma**2)/(1+gamma**2)-exp_phi
    rho_u = r_2*(1-r_m/r_2)**exp_rho
    a_u = (1-r_m/r_2)**exp_a
    phi_u= (1-r_m/r_2)**exp_phi
    drho_dr_u = (1-r_m/r_2)**exp_rho+exp_rho*(r_m/r_2)*(1-r_m/r_2)**(exp_rho-1)
    b_u = ((1-r_m/r_2)**exp_b)*(drho_dr_u)**(-2)
    r_lim = 80000
    if option == 1:
        couleur = (0.85,0.325,0.098)
        nom = '$\\alpha_-$'
        alphag = gamma
    elif option == 2:
        couleur = (0.929,0.694,0.125)
        nom = '$\\alpha_+$'
        alphag = - gamma

    return (r,a,b,phi,rho_u,a_u,phi_u,b_u,couleur,nom, comment,radiusStar,r_lim,descr,mass_ADM,rho_cen,alphag, SoS_c_max)



def make_gamma_beta_plots(n,opti):
    nspace = n
    option = opti
    if opti == 1:
        descr = 'rho'
        comment = '$L_m = - \epsilon$'#\rho$'
    elif opti == 2:
        descr = 'P'
        comment ='$L_m = P$'
    else:
        print('not a valid option, try 1 or 2')

    den_space = np.linspace(100,3000,num=n)
    gamma_vec = np.array([])
    beta_vec = np.array([])
    delta_PN_vec = np.array([])
    xi = 1/(2*np.sqrt(3))
    vsurc_a = np.array([])
    gamma_edd_a = np.array([])
    gamma_edd_a_m = np.array([])
    delta_PN = np.array([])

    for den in den_space:
        r,a,b,phi,rho_u,a_u,phi_u,b_u,couleur,nom, comment,radiusStar,r_lim,descr,mass_ADM,rho_cen,gamma, vsurc  = run(option,den)
        gamma_edd_m = -(8 * gamma * xi)/(1-gamma**2 +4 *gamma * xi)#(1 - gamma**2 - 4 * gamma * xi)/(1 - gamma**2 + 4 * gamma * xi)
        gamma_edd = (1 - gamma**2 - 4 * gamma * xi)/(1 - gamma**2 + 4 * gamma * xi)
        vsurc_a = np.append(vsurc_a,vsurc )
        gamma_vec = np.append(gamma_vec,gamma)
        beta_vec = np.append(beta_vec,(1-gamma**2)/(1+gamma**2))
        gamma_edd_a_m = np.append(gamma_edd_a_m,gamma_edd_m )
        gamma_edd_a = np.append(gamma_edd_a,gamma_edd )
        delta = 4/3 * (( 1 - gamma**2 - 4 * gamma * xi)**2 - (1/4) * (1 + gamma**2)**2 ) * 1/(1 - gamma**2 + 4 * gamma * xi)**2
        delta_PN = np.append(delta_PN,delta)

    index_max = np.where(vsurc_a > 1/np.sqrt(3))[0][0]+1
    den_space = den_space[0:index_max]
    gamma_vec = gamma_vec[0:index_max]
    beta_vec = beta_vec[0:index_max]
    gamma_edd_a = gamma_edd_a[0:index_max]
    gamma_edd_a_m = gamma_edd_a_m[0:index_max]

    delta_a = delta_PN[0:index_max]
    fig,ax = plt.subplots()
    plt.xlabel('Central density ($MeV/fm^3$)')
    ax.plot(den_space,gamma_edd_a_m, label=f'$\\gamma -1$ ({comment})', color = 'tab:blue')
    ax.set_ylabel('$\\gamma - 1 $', fontsize=18)#, color = 'tab:blue')

    fig.legend( bbox_to_anchor=(0.9, 0.6))
    plt.savefig(f'./evaluating_gamma_with_alpha.png', dpi = 200,bbox_inches='tight')
    # plt.show()
    plt.close()



    fig,ax = plt.subplots()
    plt.xlabel('Central density ($MeV/fm^3$)')
    ax.plot(den_space,delta_a-1, label=f'$\\delta$ ({comment})', color = 'tab:blue')
    ax.set_ylabel('$\\delta-1$', fontsize=18)#, color = 'tab:blue')

    fig.legend( bbox_to_anchor=(0.9, 0.6))
    plt.savefig(f'./evaluating_delta_with_alpha.png', dpi = 200,bbox_inches='tight')
    # plt.show()
    plt.close()

# In[7]:


make_gamma_beta_plots(50,1)
