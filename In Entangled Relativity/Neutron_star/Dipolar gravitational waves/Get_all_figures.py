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
def run(rho_cen, EoS):
    PhiInit = 1 #definition of initial values
    PsiInit = 0
    radiusMax_in = 40000 # maximal radius of the star
    radiusMax_out = 10000000 # maximal radius outside the star
    Npoint = 1000000 #number of points
    log_active = False #True = print star's structure values
    dilaton_active = True
    rhoInit = rho_cen*cst.eV*10**6/(cst.c**2*cst.fermi**3)
    Lagrangian = 1 #0: Lm = T, 1: Lm = -rho c^2, 2: Lm = P
    #EoS = 1 # 0 = polytropic, 1 = SLy
    tov = TOV(rhoInit, PsiInit, PhiInit, radiusMax_in, radiusMax_out, Npoint,Lagrangian, dilaton_active, log_active, EoS)# introducing tov class
    tov.ComputeTOV() #Computing star's parameters
    r = tov.radius #Recovering parameters
    a = tov.g_tt
    b = tov.g_rr
    phi = tov.Phi
    phi_dot = tov.Psi
    radiusStar = tov.radiusStar
    mass_ADM = tov.massADM / (1.989*10**30) # in solar mass
    a_dot = (-a[1:-2]+a[2:-1])/(r[2:-1]-r[1:-2]) #computing derivatives
    b_dot = (-b[1:-2]+b[2:-1])/(r[2:-1]-r[1:-2])
    f_a = -a_dot*r[1:-2]*r[1:-2]/1000
    f_b = -b_dot*r[1:-2]*r[1:-2]/1000
    f_phi = -phi_dot*r*r/1000
    SoS_c_max = np.max(tov.v_c)#defining the speed of sound (should not exceed c/sqrt(3))
    b_ = 1/(2*np.sqrt(3)) # Conformal factor
    C = f_b[-1]/f_phi[-1]#parameter that tend to infinity
    #recovering gamma by solving the Second degree equation obtain by analytically solving C
    a1 = 1
    a2 = 4*b_*(C+1)
    a3 = -1
    a4 = a2*a2-4*a1*a3
    gamma = (-a2-np.sqrt(a4))/(2*a1) # = alpha from Janis Newman Winicour
    D = (1-gamma**2)/(1+gamma**2)#renaming JNW parameter
    ge = ((D - np.sign(gamma) * 1/np.sqrt(3) * np.sqrt(1-D**2)))/((D +  np.sign(gamma) * 1/np.sqrt(3) * np.sqrt(1-D**2)))# gamma exact depending in the scalar charge of JNW solution, Eq(11) (with w=0) of : On the numerical evaluation of the exact PN parameters in BD and ER
    ge_theta = tov.Ge_theta# Definition depending in the structure of the star. From Eq(70) of : On the numerical evaluation of the exact PN parameters in BD and ER
    gamma_dev_per = (ge- ge_theta)/ge_theta *100 # deviation between both definition in percent of gamma exact
    delta = 4/3 * ( ge**2 - 1/4 * (D + np.sign(gamma) * np.sqrt((1-D**2 )/(3)))**(-2) )# delta exact depending in the scalar charge of JNW solution, Eq(11) (with w=0) of : On the numerical evaluation of the exact PN parameters in BD and ER
    delta_theta = tov.Delta_theta# Definition depending in the structure of the star. From Eq(70) of : On the numerical evaluation of the exact PN parameters in BD and ER
    delta_dev_per = (delta - delta_theta)/delta_theta * 100# deviation between both definition in percent of delta exact

    # For PSRJ1738+0333, low eccentricity binary
    Mc = 0.181 * (1.989*10**30) #companion mass in solar mass
    period = 0.35479 * 24 * 3600 #per second
    n_b = 2*cst.pi/(period) # orbital frequency red/sec
    q = 8.1# mass ratio
    e = 3.4 * 10**(-7)
    Gstar = cst.G#/(np.exp(-2*phi[-1]/np.sqrt(3)) * (1+4/3)) # quel signe dans l'exp ?

    # mc = Mc/(1.989*10**30)
    T = cst.G * 1.989*10**30/cst.c**3 # solar gravitational constant of time
    alpha_p = (2*gamma)/(1-gamma**2)
    csts = - 2 * cst.pi* n_b * Gstar* Mc *cst.c**(-3) * q/(q+1) * ((1+(e**2)/2)/(1 - e**2)**(5/2))
    # csts = - 2 * cst.pi* n_b *T * mc * q/(q+1) * ((1+(e**2)/2)/(1 - e**2)**(5/2))
    pdot = csts * alpha_p**2

    return (gamma, ge_theta, delta_theta, gamma_dev_per, delta_dev_per, SoS_c_max, pdot, mass_ADM)


# run(1578)

#function that compute vectors and plot them
def evaluate_pdot(n):

    nspace = n #number of iteration
    den_space = np.linspace(100,2000,num=n) #min max density
    beta_vec = np.array([])
    vsurc_a = np.array([])
    gamma_edd_a = np.array([])
    delta_edd_a = np.array([])
    gamma_dev_per_a = np.array([])
    delta_dev_per_a = np.array([])
    pdot_a = np.array([])
    mass_ADM_a = np.array([])

    beta_vec_SLy = np.array([])
    vsurc_a_SLy = np.array([])
    gamma_edd_a_SLy = np.array([])
    delta_edd_a_SLy = np.array([])
    gamma_dev_per_a_SLy = np.array([])
    delta_dev_per_a_SLy = np.array([])
    pdot_a_SLy = np.array([])
    mass_ADM_a_SLy = np.array([])

#loop that compute the paramaters for polytropic EoS
    for den in den_space:
        gamma, ge, de, gamma_dev, delta_dev, vsurc, pdot, mass_ADM  = run(den, 0)

        vsurc_a = np.append(vsurc_a,vsurc)
        gamma_edd_a = np.append(gamma_edd_a,ge)
        delta_edd_a = np.append(delta_edd_a,de)
        gamma_dev_per_a = np.append(gamma_dev_per_a,gamma_dev)
        delta_dev_per_a = np.append(delta_dev_per_a,delta_dev)
        pdot_a = np.append(pdot_a,pdot)
        mass_ADM_a = np.append(mass_ADM_a, mass_ADM)

    index_max = np.where(vsurc_a > 1/np.sqrt(3))[0][0] #imposing sound speed not to exceed c/sqrt(3)
    den_space = den_space[0:index_max]
    gamma_edd_a = gamma_edd_a[0:index_max]
    delta_edd_a = delta_edd_a[0:index_max]
    gamma_dev_per_a = gamma_dev_per_a[0:index_max]
    delta_dev_per_a = delta_dev_per_a[0:index_max]
    pdot_a = pdot_a[0:index_max]
    mass_ADM_a = mass_ADM_a[0:index_max]

#loop that compute the paramaters for SLy EoS
    den_space_SLy = np.linspace(100,2000,num=n) #min max density

    for den in den_space_SLy:
        gamma_SLy, ge_SLy, de_SLy, gamma_dev_SLy, delta_dev_SLy, vsurc_SLy, pdot_SLy, mass_ADM_SLy  = run(den, 1)

        vsurc_a_SLy = np.append(vsurc_a_SLy,vsurc_SLy)
        gamma_edd_a_SLy = np.append(gamma_edd_a_SLy,ge_SLy)
        delta_edd_a_SLy = np.append(delta_edd_a_SLy,de_SLy)
        gamma_dev_per_a_SLy = np.append(gamma_dev_per_a_SLy,gamma_dev_SLy)
        delta_dev_per_a_SLy = np.append(delta_dev_per_a_SLy,delta_dev_SLy)
        pdot_a_SLy = np.append(pdot_a_SLy,pdot_SLy)
        mass_ADM_a_SLy = np.append(mass_ADM_a_SLy, mass_ADM_SLy)

    index_max_SLy = np.where(vsurc_a_SLy > 1/np.sqrt(3))[0][0] #imposing sound speed not to exceed c/sqrt(3)
    den_space_SLy = den_space_SLy[0:index_max_SLy]
    gamma_edd_a_SLy = gamma_edd_a_SLy[0:index_max_SLy]
    delta_edd_a_SLy = delta_edd_a_SLy[0:index_max_SLy]
    gamma_dev_per_a_SLy = gamma_dev_per_a_SLy[0:index_max_SLy]
    delta_dev_per_a_SLy = delta_dev_per_a_SLy[0:index_max_SLy]
    pdot_a_SLy = pdot_a_SLy[0:index_max_SLy]
    mass_ADM_a_SLy = mass_ADM_a_SLy[0:index_max_SLy]


    #imposing minimal value of 1 solar mass
    index_mass = np.where(mass_ADM_a > 1)[0][0]
    mass_ADM_a = mass_ADM_a[index_mass:-1]
    gamma_edd_a = gamma_edd_a[index_mass:-1]
    delta_edd_a = delta_edd_a[index_mass:-1]
    gamma_dev_per_a = gamma_dev_per_a[index_mass:-1]
    delta_dev_per_a = delta_dev_per_a[index_mass:-1]
    pdot_a = pdot_a[index_mass:-1]

    index_mass_SLy = np.where(mass_ADM_a_SLy > 1 )[0][0]
    mass_ADM_a_SLy = mass_ADM_a_SLy[index_mass_SLy:-1]
    gamma_edd_a_SLy = gamma_edd_a_SLy[index_mass_SLy:-1]
    delta_edd_a_SLy = delta_edd_a_SLy[index_mass_SLy:-1]
    gamma_dev_per_a_SLy = gamma_dev_per_a_SLy[index_mass_SLy:-1]
    delta_dev_per_a_SLy = delta_dev_per_a_SLy[index_mass_SLy:-1]
    pdot_a_SLy = pdot_a_SLy[index_mass_SLy:-1]

#Ploting pdot versus star's mass for both EoS
    fig,ax = plt.subplots()
    plt.xlabel('Star mass ($M_\odot$)')
    ax.plot(mass_ADM_a,pdot_a, label=f'$\dot P^D$ polytrop', color = 'tab:blue')
    ax.plot(mass_ADM_a_SLy,pdot_a_SLy, label=f'$\dot P^D$ SLy', color = 'tab:brown')
    ax.set_ylabel('$\dot{P}^D$', fontsize=18)
    plt.legend()
    plt.savefig(f'./pdot_versus_mass.png', dpi = 200,bbox_inches='tight')
    # plt.show()
    plt.close()

# #Ploting pdot versus star's mass for polytropic EoS
#     fig,ax = plt.subplots()
#     plt.xlabel('Star mass ($M_\odot$)')
#     ax.plot(mass_ADM_a,pdot_a, label=f'$\dot P^D$ polytrop', color = 'tab:blue')
#     # ax.plot(mass_ADM_a_SLy,pdot_a_SLy, label=f'$\dot P^D$ SLy', color = 'tab:brown')
#     ax.set_ylabel('$\dot{P}^D$', fontsize=18)
#     plt.legend()
#     plt.savefig(f'./pdot_versus_mass_only_polytrop.png', dpi = 200,bbox_inches='tight')
#     # plt.show()
#     plt.close()


# #Ploting pdot versus code density for both EoS
#     fig,ax = plt.subplots()
#     plt.xlabel(f'Core density $Mev/fm^3$')
#     ax.plot(den_space[0:len(pdot_a)],pdot_a, label=f'$\dot P^D$ polytrop', color = 'tab:blue')
#     ax.plot(den_space[0:len(pdot_a_SLy)],pdot_a_SLy, label=f'$\dot P^D$ SLy', color = 'tab:brown')
#     ax.set_ylabel('$\dot{P}^D$', fontsize=18)
#     plt.legend()
#     plt.savefig(f'./pdot_versus_density.png', dpi = 200,bbox_inches='tight')
#     # plt.show()
#     plt.close()

# defining minimal and maximal index values
    min_index = np.argmin(pdot_a)
    min_value = pdot_a[min_index]

    max_index = np.argmax(pdot_a)
    max_value = pdot_a[max_index]

# reminder, we took minimal when maximal because pdot is negative. Its not false but just to enhance the comprehension.
    print('\nFor a polytropic equation of state :')
    print('Maximal core density', den_space[-1],'\n')
    print(f"Maximal pdot = {min_value:.3e}")
    print(f'Mass associated to maximal pdot = {mass_ADM_a[min_index]:.3f} solar mass\n')


    print(f'Minimal pdot = {max_value:.3e}')
    print(f'Mass associated to minimal pdot = {mass_ADM_a[max_index]:.3f} solar mass \n')

    print('----------------------------------------------------------------------------\n')
    min_index_SLy = np.argmin(pdot_a_SLy)
    min_value_SLy = pdot_a_SLy[min_index_SLy]

    max_index_SLy = np.argmax(pdot_a_SLy)
    max_value_SLy = pdot_a_SLy[max_index_SLy]
    print('For a piece wise equation of state :')
    print('Maximal core density', den_space_SLy[-1],'\n')
    print(f"Maximal pdot = {min_value_SLy:.3e}")
    print(f'Mass associated to maximal pdot = {mass_ADM_a_SLy[min_index_SLy]:.3f} solar mass\n')


    print(f'Minimal pdot = {max_value_SLy:.3e}')
    print(f'Mass associated to minimal pdot = {mass_ADM_a_SLy[max_index_SLy]:.3f} solar mass')


evaluate_pdot(300)
