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

# function that compute JNW parameter and hence gamma exact
def run(rho_cen):
    PhiInit = 1 #definition of initial values
    PsiInit = 0
    radiusMax_in = 40000 # maximal radius of the star
    radiusMax_out = 10000000 # maximal radius outside the star
    Npoint = 1000000 #number of points
    log_active = False #True = print star's structure values
    dilaton_active = True
    rhoInit = rho_cen*cst.eV*10**6/(cst.c**2*cst.fermi**3)
    tov = TOV(rhoInit, PsiInit, PhiInit, radiusMax_in, radiusMax_out, Npoint, dilaton_active, log_active)
    tov.ComputeTOV()# introducing tov class
    r = tov.radius #recovering parameters
    a = tov.g_tt
    b = tov.g_rr
    phi = tov.Phi
    phi_dot = tov.Psi
    radiusStar = tov.radiusStar
    mass_ADM = tov.massADM / (1.989*10**30) # in solar mass
    a_dot = (-a[1:-2]+a[2:-1])/(r[2:-1]-r[1:-2]) #computing derivative
    b_dot = (-b[1:-2]+b[2:-1])/(r[2:-1]-r[1:-2])
    f_a = -a_dot*r[1:-2]*r[1:-2]/1000
    f_b = -b_dot*r[1:-2]*r[1:-2]/1000
    f_phi = -phi_dot*r*r/1000
    SoS_c_max = np.max(tov.v_c)#defining the speed of sound (should not exceed c/sqrt(3))
    b_ = 1/(2*np.sqrt(3))
    C = f_b[-1]/f_phi[-1]#parameter that tend to infinity
    #recovering gamma by solving the Second degree equation obtain by analytically solving C
    a1 = 1
    a2 = 4*b_*(C+1)
    a3 = -1
    a4 = a2*a2-4*a1*a3
    gamma = (-a2-np.sqrt(a4))/(2*a1)
    r_m = 1000*f_phi[-1]*(1+gamma**2)/(4*b_*gamma)
    comment = '$L_m = -\\rho$'
    descr = 'rho'
    D = (1-gamma**2)/(1+gamma**2)#renaming JNW parameter
    ge = ((D - np.sign(gamma) * 1/np.sqrt(3) * np.sqrt(1-D**2)))/((D +  np.sign(gamma) * 1/np.sqrt(3) * np.sqrt(1-D**2)))# gamma exact
    ge_theta = tov.Ge_theta# recovering gamma exact computed in TOV class
    gamma_dev_per = (ge- ge_theta)/ge_theta *100
    delta = 4/3 * ( ge**2 - 1/4 * (D + np.sign(gamma) * np.sqrt((1-D**2 )/(3)))**(-2) )
    delta_theta = tov.Delta_theta
    delta_dev_per = (delta - delta_theta)/delta_theta * 100
    return (b_, D, rho_cen, gamma, ge_theta, delta_theta, gamma_dev_per, delta_dev_per, SoS_c_max)

#function that compute vectors and plot them
def make_gamma_beta_plots(n):

    nspace = n #number of iteration
    den_space = np.linspace(100,2000,num=n) #min max density
    beta_vec = np.array([])
    vsurc_a = np.array([])
    gamma_edd_a = np.array([])
    delta_edd_a = np.array([])
    gamma_dev_per_a = np.array([])
    delta_dev_per_a = np.array([])

#loop that compute the paramaters
    for den in den_space:
        b_, D, rho_cen, gamma, ge, de, gamma_dev, delta_dev, vsurc  = run(den)

        vsurc_a = np.append(vsurc_a,vsurc)
        gamma_edd_a = np.append(gamma_edd_a,ge)
        delta_edd_a = np.append(delta_edd_a,de)
        gamma_dev_per_a = np.append(gamma_dev_per_a,gamma_dev)
        delta_dev_per_a = np.append(delta_dev_per_a,delta_dev)

    index_max = np.where(vsurc_a > 1/np.sqrt(3))[0][0] #imposing sound speed not to exceed c/sqrt(3)
    den_space = den_space[0:index_max]
    # print('densité max', den_space[-1])
    gamma_edd_a = gamma_edd_a[0:index_max]
    delta_edd_a = delta_edd_a[0:index_max]
    gamma_dev_per_a = gamma_dev_per_a[0:index_max]
    delta_dev_per_a = delta_dev_per_a[0:index_max]
    comment = '$L_m = -\\rho$'

    print(f'maximal deviation in percent between both gamma computation = {max(abs(gamma_dev_per_a)):.3f}%')
    print(f'maximal deviation in percent between both delta computation = {max(abs(delta_dev_per_a)):.3f}%\n')

    print(f'minimal deviation from unity of gamma = {min(abs(1 - (gamma_edd_a)))*100:.3f}%')
    print(f'minimal deviation from unity of delta = {min(abs(1 - (delta_edd_a)))*100:.3f}%\n')

    print(f'maximal deviation from unity of gamma = {max(abs(1 - (gamma_edd_a)))*100:.3f}%')
    print(f'maximal deviation from unity of delta = {max(abs(1 - (delta_edd_a)))*100:.3f}%')

    #plot 1 - gamma
    fig,ax = plt.subplots()
    plt.xlabel('Central density ($MeV/fm^3$)')
    ax.plot(den_space,(1-gamma_edd_a), label=f'$1-\\gamma_e$ ({comment})', color = 'tab:blue')
    ax.set_ylabel('$1-\\gamma_e$', fontsize=18)#, color = 'tab:blue')
    fig.legend( bbox_to_anchor=(0.9, 0.6))
    plt.savefig(f'./gamma_exact_m.png', dpi = 200,bbox_inches='tight')
    # plt.show()
    plt.close()

    #plot 1 - delta :
    fig,ax = plt.subplots()
    plt.xlabel('Central density ($MeV/fm^3$)')
    ax.plot(den_space,(1-delta_edd_a), label=f'$1-\\delta_e$ ({comment})', color = 'tab:blue')
    ax.set_ylabel('$1-\\delta_e$', fontsize=18)#, color = 'tab:blue')
    fig.legend( bbox_to_anchor=(0.9, 0.6))
    plt.savefig(f'./delta_exact_m.png', dpi = 200,bbox_inches='tight')
    # plt.show()
    plt.close()

make_gamma_beta_plots(150)
