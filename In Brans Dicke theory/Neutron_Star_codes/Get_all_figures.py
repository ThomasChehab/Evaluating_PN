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

from matplotlib.colors import LogNorm
from matplotlib.ticker import FuncFormatter

from matplotlib.transforms import Bbox
import mplhep as hep
hep.style.use("ATLAS")
from tqdm import tqdm

# function that compute JNW parameter and hence gamma exact
def run(rho_cen, w):
    PhiInit = 1 #definition of initial values
    PsiInit = 0
    radiusMax_in = 40000 # maximal radius of the star
    radiusMax_out = 10000000 # maximal radius outside the star
    Npoint = 1000000 #number of points
    log_active = False #True = print star's structure values
    rhoInit = rho_cen*cst.eV*10**6/(cst.c**2*cst.fermi**3)
    tov = TOV(rhoInit , PsiInit, PhiInit, radiusMax_in, radiusMax_out, Npoint, log_active, w) # introducing tov class
    tov.ComputeTOV_normalization() # Launching TOV numerical integration with the normalization of phi at infinity
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
    SoS_c_max = np.max(tov.v_c) #defining the speed of sound (should not exceed c/sqrt(3))
    b_ = 2/(np.sqrt(3+2*w))
    C = f_b[-1]/f_phi[-1] #parameter that tend to infinity
    #recovering gamma by solving the Second degree equation obtain by analytically solving C
    a1 = 1
    a2 = b_*(C+1)
    a3 = -1
    a4 = a2*a2-4*a1*a3
    gamma = (-a2-np.sqrt(a4))/(2*a1)
    D = (1-gamma**2)/(1+gamma**2) #renaming JNW parameter
    g_bd = (1+w)/(2+w) #gamma post-Newtonian in Brans Dicke
    ge = ( D - np.sign(gamma)* np.sqrt(1-D**2) * 1/np.sqrt(3+2*w))/( D + np.sign(gamma)* np.sqrt(1-D**2) * 1/np.sqrt(3+2*w)) # gamma exact
    ge_theta = tov.Ge_theta # recovering gamma exact computed in TOV class
    gamma_dev_per = (ge- ge_theta)/ge_theta *100
    delta = 4/3 * ( ge**2 - 1/4 * (D + np.sign(gamma) * np.sqrt((1-D**2 )/(3+2*w)))**(-2) )
    delta_theta = tov.Delta_theta
    delta_bd = 4/3 * (g_bd)**2 + 4/3 * 1 - g_bd/6 - 3/2 #delta post-Newtonian in Brans Dicke
    delta_dev_per = (delta - delta_theta)/delta_theta * 100
    return (b_, D, rho_cen, gamma, g_bd, ge_theta, delta_theta, delta_bd, gamma_dev_per, delta_dev_per, SoS_c_max)
###################################################

#function to compute gamma exact for every density
def compute_gamma(n,w):
    nspace = n #number of w values

    # Saving repository
    save_dir = 'saved_matrices_and_plots'
    if not os.path.exists(save_dir): # Creating the folder if it does not exist
        os.makedirs(save_dir)

    all_a = os.path.join(save_dir, f'matrice_{n}.npy') # Matrices to store values

    # density
    den_space = np.linspace(100,2000,num=n)

    # creating the vectors that will contains useful parameters
    vsurc_vec = np.array([])
    gamma_edd_vec = np.array([])
    delta_edd_vec =np.array([])
    gamma_BD_vec = np.array([])
    delta_BD_vec = np.array([])
    gamma_dev_vec = np.array([])
    delta_dev_vec = np.array([])
    start_idx = len(all_a[0])

    for den in tqdm(den_space):
        b_,D , rho_cen, gamma, gamma_BD, gamma_edd, delta_edd, delta_BD, gamma_dev_per, delta_dev_per, vsurc  = run(den,w) # computing gamma and storing it in vectors for every value of density in den_space
        vsurc_vec = np.append(vsurc_vec,vsurc ) # Storing sound velocity
        gamma_edd_vec = np.append(gamma_edd_vec,gamma_edd ) # Storing gamma exact
        delta_edd_vec = np.append(delta_edd_vec, delta_edd)
        gamma_BD_vec = np.append(gamma_BD_vec, gamma_BD)# Storing gamma PN
        delta_BD_vec = np.append(delta_BD_vec, delta_BD)# Storing gamma PN
        gamma_dev_vec = np.append(gamma_dev_vec, gamma_dev_per)
        delta_dev_vec = np.append(delta_dev_vec, delta_dev_per)

        if den == den_space[-1]: # at the last value of density, rejecting values that exceed the desired speed of sound
            index_max = np.where(vsurc_vec > 1/np.sqrt(3))[0][0]
            gamma_edd_vec = gamma_edd_vec[0:index_max]
            delta_edd_vec = delta_edd_vec[0:index_max]
            gamma_BD_vec = gamma_BD_vec[0:index_max]
            delta_BD_vec = delta_BD_vec[0:index_max]
            gamma_dev_vec = gamma_dev_vec[0:index_max]
            delta_dev_vec = delta_dev_vec[0:index_max]
            den_space = den_space[0:index_max]

    return gamma_edd_vec, delta_edd_vec, gamma_BD_vec, delta_BD_vec, gamma_dev_vec, delta_dev_vec, den_space


#function that recovers the previously computed vectors to plot them
def plot_w_vs_rho(lowest_w, highest_w, n, count):

# if count = 0 some files exists
    if count==0:
        save_dir = 'saved_matrices_and_plots'
        os.makedirs(save_dir, exist_ok=True)

        w_values = np.linspace(lowest_w, highest_w, n) #setting the possible values of w
        w_values = np.exp(w_values)

        all_a = os.path.join(save_dir, f'matrice_{n}.npy')# recovering matrices that contain values. all_a containt values of gamma exact and gamma PN
        all_w = os.path.join(save_dir, f'matrice_{n}_w.npy') # all_w contains values of w
        all_rho = os.path.join(save_dir, f'matrice_{n}_rho.npy')# all_rho contains density values

        # Loading existing data
        if os.path.exists(all_a):
            gamma_edd_all, gamma_BD_all, gamma_dev_per_all, delta_edd_all, delta_BD_all, delta_dev_per_all = np.load(all_a, allow_pickle=True) #loading gamma values

            print(f"Founded incomplete files : {all_a}, {all_w}, {all_rho}")

            start_idx = len(gamma_edd_all) #last indexed stored is equal to the len of gamma_edd_all that contains gamma exact values
            print(f'Restarting at w = {start_idx}')
            gamma_edd_all = list(gamma_edd_all)
            delta_edd_all = list(delta_edd_all)
            gamma_BD_all = list(gamma_BD_all)
            delta_BD_all = list(delta_BD_all)
            gamma_dev_per_all = list(gamma_dev_per_all)
            delta_dev_per_all = list(delta_dev_per_all)
        else:
            gamma_edd_all = []
            delta_edd_all = []
            gamma_BD_all = []
            delta_BD_all = []
            gamma_dev_per_all = []
            delta_dev_per_all = []
            start_idx = 0

        remaining_w_values = w_values[start_idx:]
        counter = len(remaining_w_values)-1 # is useful to print how much values it remains to compute
        for value in remaining_w_values:
            gamma_edd_vec, delta_edd_vec, gamma_BD_vec, delta_BD_vec,gamma_dev_vec, delta_dev_vec, den_space = compute_gamma(n, value)
 # launching the function that computes gamma for every density, for every w values it remains

            print(f'Remaing {counter} iterations')
            gamma_edd_all.append(gamma_edd_vec) # adding computed values
            delta_edd_all.append(delta_edd_vec)
            gamma_BD_all.append(gamma_BD_vec)
            delta_BD_all.append(delta_BD_vec)
            gamma_dev_per_all.append(gamma_dev_vec)
            delta_dev_per_all.append(delta_dev_vec)

            np.save(all_a, [np.array(gamma_edd_all),np.array(gamma_BD_all), np.array(gamma_dev_per_all), np.array(delta_edd_all), np.array(delta_BD_all), np.array(delta_dev_per_all)]) #saving values at every entire w value computed
            np.save(all_w, [w_values])
            np.save(all_rho, [den_space])
            counter -= 1
        gamma_edd_all = np.array(gamma_edd_all)
        gamma_BD_all = np.array(gamma_BD_all)
        delta_edd_all = np.array(delta_edd_all)
        delta_BD_all = np.array(delta_BD_all)
        gamma_dev_per_all = np.array(gamma_dev_per_all)
        delta_dev_per_all = np.array(delta_dev_per_all)

        print(f'maximal deviation in percent between both gamma computation = {max(abs(gamma_dev_per_all[0])):.3f}%')
        print(f'maximal deviation in percent between both delta computation = {max(abs(delta_dev_per_all[0])):.3f}%\n')

        dev_gamma_unity =  max(1-gamma_edd_all[0])
        print(f'maximal deviation from unity of gamma exact {dev_gamma_unity:.3f}%')
        dev_delta_unity = max(1-delta_edd_all[0])
        print(f'maximal deviation from unity of delta exact {dev_delta_unity:.3f}%\n')

        rel_dev_gamma =  max(((gamma_edd_all[-1]-gamma_BD_all)[-1]/(1-gamma_BD_all[-1]) ) * 100)
        print(f"maximal relative deviation for gamma {rel_dev_gamma:.3f}%")
        rel_dev_delta = max((((delta_edd_all[-1]-delta_BD_all)[-1]/(1-delta_BD_all[-1]) ) * 100))
        print(f"maximal relative deviation for delta {rel_dev_delta:.3f}%")

#ploting the results

        plt.figure(figsize=(8,6))
        mesh = plt.pcolormesh(den_space, w_values, 1-delta_edd_all, shading='auto', cmap='viridis')
        cbar = plt.colorbar(mesh)
        cbar.set_label(r'$1-\delta_e$')
        # cbar.ax.set_yscale("log")
        plt.yscale("log")
        plt.xlabel(r'Density $\rho_c$ (MeV/fm$^3$)')
        plt.ylabel(r'$\omega$')
        plt.ylim(np.exp(lowest_w), np.exp(highest_w))
        plt.xlim(min(den_space),max(den_space))
        plt.savefig(f'./saved_matrices_and_plots/delta_exact_m_{n}.png', dpi=200, bbox_inches="tight")

        # 1-gamma_e
        plt.figure(figsize=(8,6))
        mesh = plt.pcolormesh(den_space, w_values, 1-gamma_edd_all, shading='auto', cmap='viridis')
        cbar = plt.colorbar(mesh)
        cbar.set_label(r'$1-\gamma_e$')
        # cbar.ax.set_yscale("log")
        plt.yscale("log")
        plt.xlabel(r'Density $\rho_c$ (MeV/fm$^3$)')
        plt.ylabel(r'$\omega$')
        plt.ylim(np.exp(lowest_w), np.exp(highest_w))
        plt.xlim(min(den_space),max(den_space))
        plt.savefig(f'./saved_matrices_and_plots/heatmap_gamma_exact_m_{n}.png', dpi=200, bbox_inches="tight")

        # gamma_e-gamma_BD/1-g_BD
        plt.figure(figsize=(8,6))
        mesh = plt.pcolormesh(den_space, w_values, ((gamma_edd_all-gamma_BD_all)/(1-gamma_BD_all) ) * 100, shading='auto', cmap='viridis')
        cbar = plt.colorbar(mesh)
        cbar.set_label(r'$(\gamma_e-\gamma_{BD})/(1-\gamma_{BD})\%$')
        # cbar.ax.set_yscale("log")
        plt.yscale("log")
        plt.ylim(np.exp(lowest_w), np.exp(highest_w))
        plt.xlim(min(den_space),max(den_space))
        plt.xlabel(r'Density $\rho_c$ (MeV/fm$^3$)')
        plt.ylabel(r'$\omega$')
        plt.savefig(f'./saved_matrices_and_plots/ge-gb_on_1-gb_{n}.png', dpi=200, bbox_inches="tight")

        plt.figure(figsize=(8,6))
        mesh = plt.pcolormesh(den_space, w_values, ((delta_edd_all-delta_BD_all)/(1-delta_BD_all) ) * 100, shading='auto', cmap='viridis')
        cbar = plt.colorbar(mesh)
        cbar.set_label(r'$(\delta_e-\delta_{BD})/(1-\delta_{BD})\%$')
        # cbar.ax.set_yscale("log")
        plt.yscale("log")
        plt.xlabel(r'Density $\rho_c$ (MeV/fm$^3$)')
        plt.ylabel(r'$\omega$')
        plt.ylim(np.exp(lowest_w), np.exp(highest_w))
        plt.xlim(min(den_space),max(den_space))
        plt.savefig(f'./saved_matrices_and_plots/delta_relative_{n}.png', dpi=200, bbox_inches="tight")

    else:

        save_dir = 'saved_matrices_and_plots'
        os.makedirs(save_dir, exist_ok=True)
        all_a = os.path.join(save_dir, f'matrice_{n}.npy')
        all_w = os.path.join(save_dir, f'matrice_{n}_w.npy')
        all_rho = os.path.join(save_dir, f'matrice_{n}_rho.npy')

        gamma_edd_all, gamma_BD_all, gamma_dev_per_all, delta_edd_all, delta_BD_all, delta_dev_per_all = np.load(all_a, allow_pickle=True)
        w_values = np.load(all_w, allow_pickle=True)[0]
        den_space = np.load(all_rho, allow_pickle=True)[0]

        print(f'maximal deviation in percent between both gamma computation = {max(abs(gamma_dev_per_all[0])):.3f}%')
        print(f'maximal deviation in percent between both delta computation = {max(abs(delta_dev_per_all[0])):.3f}%\n')

        dev_gamma_unity =  max(1-gamma_edd_all[0])
        print(f'maximal deviation from unity of gamma exact {dev_gamma_unity:.3f}%')
        dev_delta_unity = max(1-delta_edd_all[0])
        print(f'maximal deviation from unity of delta exact {dev_delta_unity:.3f}%\n')

        rel_dev_gamma =  max(((gamma_edd_all[-1]-gamma_BD_all)[-1]/(1-gamma_BD_all[-1]) ) * 100)
        print(f"maximal relative deviation for gamma {rel_dev_gamma:.3f}%")
        rel_dev_delta = max((((delta_edd_all[-1]-delta_BD_all)[-1]/(1-delta_BD_all[-1]) ) * 100))
        print(f"maximal relative deviation for delta {rel_dev_delta:.3f}%")

        plt.figure(figsize=(8,6))
        mesh = plt.pcolormesh(den_space, w_values, 1-delta_edd_all, shading='auto', cmap='viridis')
        cbar = plt.colorbar(mesh)
        cbar.set_label(r'$1-\delta_e$')
        # cbar.ax.set_yscale("log")
        plt.yscale("log")
        plt.xlabel(r'Density $\rho_c$ (MeV/fm$^3$)')
        plt.ylabel(r'$\omega$')
        plt.ylim(np.exp(lowest_w), np.exp(highest_w))
        plt.xlim(min(den_space),max(den_space))
        plt.savefig(f'./saved_matrices_and_plots/delta_exact_m_{n}.png', dpi=200, bbox_inches="tight")


        # Second plot 1-gamma_e
        plt.figure(figsize=(8,6))
        mesh = plt.pcolormesh(den_space, w_values, 1-gamma_edd_all, shading='auto', cmap='viridis')
        cbar = plt.colorbar(mesh)
        cbar.set_label(r'$1-\gamma_e$')
        # cbar.ax.set_yscale("log")
        plt.yscale("log")
        plt.ylim(np.exp(lowest_w), np.exp(highest_w))
        plt.xlim(min(den_space),max(den_space))
        plt.xlabel(r'Density $\rho_c$ (MeV/fm$^3$)')
        plt.ylabel(r'$\omega$')
        plt.savefig(f'./saved_matrices_and_plots/heatmap_gamma_exact_m_{n}.png', dpi=200, bbox_inches="tight")

        # gamma_e-gamma_BD/1-g_BD
        plt.figure(figsize=(8,6))
        mesh = plt.pcolormesh(den_space, w_values, ((gamma_edd_all-gamma_BD_all)/(1-gamma_BD_all) ) * 100, shading='auto', cmap='viridis')
        cbar = plt.colorbar(mesh)
        cbar.set_label(r'$(\gamma_e-\gamma_{BD})/(1-\gamma_{BD})\%$')
        # cbar.ax.set_yscale("log")
        plt.yscale("log")
        plt.ylim(np.exp(lowest_w), np.exp(highest_w))
        plt.xlim(min(den_space),max(den_space))
        plt.xlabel(r'Density $\rho_c$ (MeV/fm$^3$)')
        plt.ylabel(r'$\omega$')
        plt.savefig(f'./saved_matrices_and_plots/ge-gb_on_1-gb_{n}.png', dpi=200, bbox_inches="tight")

        plt.figure(figsize=(8,6))
        mesh = plt.pcolormesh(den_space, w_values, ((delta_edd_all-delta_BD_all)/(1-delta_BD_all) ) * 100, shading='auto', cmap='viridis')
        cbar = plt.colorbar(mesh)
        cbar.set_label(r'$(\delta_e-\delta_{BD})/(1-\delta_{BD})\%$')
        # cbar.ax.set_yscale("log")
        plt.yscale("log")
        plt.xlabel(r'Density $\rho_c$ (MeV/fm$^3$)')
        plt.ylabel(r'$\omega$')
        plt.ylim(np.exp(lowest_w), np.exp(highest_w))
        plt.xlim(min(den_space),max(den_space))
        plt.savefig(f'./saved_matrices_and_plots/delta_relative_{n}.png', dpi=200, bbox_inches="tight")

# function that recover and plot and if files does not exist, launch the other function to compute them
def recover_and_plot(n, lowest_w, highest_w):

    save_dir = 'saved_matrices_and_plots'# creating the directiories
    os.makedirs(save_dir, exist_ok=True)

    file_path = os.path.join(save_dir, f'matrice_{n}.npy')

    if os.path.exists(file_path):
        all_a = np.load(file_path)
        if len(all_a[0]) == n:# if len of the function that contains gamma exact is equal to the desired number of iteration, files are complete and count = 1 (code directly ploting the results)
            print('Complete files detected')
            count = 1
            plot_w_vs_rho(lowest_w, highest_w, n, count)
        else: # if all_a exist but is not complete
            print('Incomplete files')
            count=0
            plot_w_vs_rho(lowest_w, highest_w, n, count)

    else:
        print('Missing files') #if all_a does not exist
        count=0
        plot_w_vs_rho(lowest_w, highest_w, n, count)

recover_and_plot(n=150, lowest_w = np.log(1e-1),highest_w = np.log(1e6))

