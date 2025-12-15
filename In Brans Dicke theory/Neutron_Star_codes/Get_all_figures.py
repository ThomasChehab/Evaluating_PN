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


def run(rho_cen, w):
    PhiInit = 1
    PsiInit = 0
    radiusMax_in = 40000
    radiusMax_out = 10000000
    Npoint = 1000000
    log_active = False
    rhoInit = rho_cen*cst.eV*10**6/(cst.c**2*cst.fermi**3)
    tov = TOV(rhoInit , PsiInit, PhiInit, radiusMax_in, radiusMax_out, Npoint, log_active, w)
    tov.ComputeTOV_normalization()
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
    b_ = 1/(np.sqrt(3+2*w))
    C = f_b[-1]/f_phi[-1]
    a1 = 1
    a2 = b_*(C+1)
    a3 = -1
    a4 = a2*a2-4*a1*a3
    gamma = (-a2-np.sqrt(a4))/(2*a1)
###
    D = (1-gamma**2)/(1+gamma**2)
    g_bd = (1+w)/(2+w)
    ge = ( D - np.sign(gamma)* np.sqrt(1-D**2) * (1/(2 * np.sqrt(3+2*w))))/( D + np.sign(gamma)* np.sqrt(1-D**2) * (1/(2 * np.sqrt(3+2*w))))
    theta_inf = ((w+1-ge * ( w+2))/(ge*w+ge-w-2))
    ge_theta = tov.Ge_theta
    print(' écart pourcent theta', (ge- ge_theta)/ge_theta *100, '\n')
###
    return (b_, D, rho_cen, gamma, g_bd, ge_theta, SoS_c_max)

###################################################
def compute_gamma(n,w):
    nspace = n

    # Saving repository
    save_dir = 'saved_matrices_and_plots'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    all_a = os.path.join(save_dir, f'matrice_{n}.npy')

    #densité
    den_space = np.linspace(100,2000,num=n)

    # création des vecteurs qui vont contenir les paramèrtes utiles
    vsurc_vec = np.array([])
    gamma_edd_vec = np.array([])
    gamma_BD_vec = np.array([])

    start_idx = len(all_a[0])

    for den in tqdm(den_space):
        b_,D , rho_cen, gamma, gamma_BD, gamma_edd, vsurc  = run(den,w)

        vsurc_vec = np.append(vsurc_vec,vsurc )

        gamma_edd_vec = np.append(gamma_edd_vec,gamma_edd )
        # on récupère gamma BD
        gamma_BD_vec = np.append(gamma_BD_vec, gamma_BD)
    # On ne veut que les cas ou v < c/sqrt(3)

        if den == den_space[-1]: # si la valeur de densité est la dernière, tronquer
            index_max = np.where(vsurc_vec > 1/np.sqrt(3))[0][0]
            gamma_edd_vec = gamma_edd_vec[0:index_max]
            gamma_BD_vec = gamma_BD_vec[0:index_max]
            den_space = den_space[0:index_max]
            # print(den_space[-1])

    return gamma_edd_vec,gamma_BD_vec, den_space



def plot_w_vs_rho(lowest_w, highest_w, n, count):

    if count==0:

        save_dir = 'saved_matrices_and_plots'
        os.makedirs(save_dir, exist_ok=True)

        w_values = np.linspace(lowest_w, highest_w, n)
        w_values = np.exp(w_values)
        den_space = np.linspace(100,2000,num=n)

        all_a = os.path.join(save_dir, f'matrice_{n}.npy')
        all_w = os.path.join(save_dir, f'matrice_{n}_w.npy')
        all_rho = os.path.join(save_dir, f'matrice_{n}_rho.npy')

        # Loading existing data
        if os.path.exists(all_a):
            gamma_edd_all, gamma_BD_all = np.load(all_a, allow_pickle=True)

            print(f"Founded incomplete files : {all_a}, {all_w}, {all_rho}")

            start_idx = len(gamma_edd_all)
            print(f'Restarting at w = {start_idx}')
            gamma_edd_all = list(gamma_edd_all)
            gamma_BD_all = list(gamma_BD_all)
        else:
            gamma_edd_all = []
            gamma_BD_all = []
            start_idx = 0

        remaining_w_values = w_values[start_idx:]
        counter = len(remaining_w_values)-1
        for value in remaining_w_values:
            gamma_edd_vec, gamma_BD_vec, den_space = compute_gamma(n, value)

            print(f'Remaing {counter} iterations')
            gamma_edd_all.append(gamma_edd_vec)
            gamma_BD_all.append(gamma_BD_vec)

            np.save(all_a, [np.array(gamma_edd_all),np.array(gamma_BD_all)])
            np.save(all_w, [w_values])
            np.save(all_rho, [den_space])
            counter -= 1
        gamma_edd_all = np.array(gamma_edd_all)
        gamma_BD_all = np.array(gamma_BD_all)

        # 1-gamma_e
        plt.figure(figsize=(8,6))
        mesh = plt.pcolormesh(den_space, w_values, 1-gamma_edd_all, shading='auto', cmap='viridis')
        cbar = plt.colorbar(mesh)
        cbar.set_label(r'$1-\gamma_e$')
        # cbar.ax.set_yscale("log")
        plt.yscale("log")
        plt.xlabel(r'Density $\rho$ (MeV/fm$^3$)')
        plt.ylabel(r'$\omega$')
        plt.ylim(np.exp(lowest_w), np.exp(highest_w))
        plt.xlim(min(den_space),max(den_space))
        plt.savefig(f'./saved_matrices_and_plots/heatmap_gamma_exact_m_{n}.png', dpi=200, bbox_inches="tight")

        # gamma_e-gamma_BD
        plt.figure(figsize=(8,6))
        mesh = plt.pcolormesh(den_space, w_values, gamma_edd_all-gamma_BD_all, shading='auto', cmap='viridis')
        cbar = plt.colorbar(mesh)
        cbar.set_label(r'$\gamma_e - \gamma_{BD}$')
        # cbar.ax.set_yscale("log")
        plt.yscale("log")
        plt.ylim(np.exp(lowest_w), np.exp(highest_w))
        plt.xlim(min(den_space),max(den_space))
        plt.xlabel(r'Density $\rho$ (MeV/fm$^3$)')
        plt.ylabel(r'$\omega$')
        plt.savefig(f'./saved_matrices_and_plots/heatmap_gamma_e-gamma_BD_{n}.png', dpi=200, bbox_inches="tight")

        # gamma_e-gamma_BD/1-g_BD
        plt.figure(figsize=(8,6))
        mesh = plt.pcolormesh(den_space, w_values, ((gamma_edd_all-gamma_BD_all)/(1-gamma_BD_all) ) * 100, shading='auto', cmap='viridis')
        cbar = plt.colorbar(mesh)
        cbar.set_label(r'$(\gamma_e-\gamma_{BD})/(1-\gamma_{BD})\%$')
        # cbar.ax.set_yscale("log")
        plt.yscale("log")
        plt.ylim(np.exp(lowest_w), np.exp(highest_w))
        plt.xlim(min(den_space),max(den_space))
        plt.xlabel(r'Density $\rho$ (MeV/fm$^3$)')
        plt.ylabel(r'$\omega$')
        plt.savefig(f'./saved_matrices_and_plots/ge-gb_on_1-gb_{n}.png', dpi=200, bbox_inches="tight")

    else:

        save_dir = 'saved_matrices_and_plots'
        os.makedirs(save_dir, exist_ok=True)
        all_a = os.path.join(save_dir, f'matrice_{n}.npy')
        all_w = os.path.join(save_dir, f'matrice_{n}_w.npy')
        all_rho = os.path.join(save_dir, f'matrice_{n}_rho.npy')

        gamma_edd_all, gamma_BD_all = np.load(all_a, allow_pickle=True)
        w_values = np.load(all_w, allow_pickle=True)[0]
        den_space = np.load(all_rho, allow_pickle=True)[0]


        # Second plot 1-gamma_e
        plt.figure(figsize=(8,6))
        mesh = plt.pcolormesh(den_space, w_values, 1-gamma_edd_all, shading='auto', cmap='viridis')
        cbar = plt.colorbar(mesh)
        cbar.set_label(r'$1-\gamma_e$')
        # cbar.ax.set_yscale("log")
        plt.yscale("log")
        plt.ylim(np.exp(lowest_w), np.exp(highest_w))
        plt.xlim(min(den_space),max(den_space))
        plt.xlabel(r'Density $\rho$ (MeV/fm$^3$)')
        plt.ylabel(r'$\omega$')
        plt.savefig(f'./saved_matrices_and_plots/heatmap_gamma_exact_m_{n}.png', dpi=200, bbox_inches="tight")

        # gamma_e-gamma_BD
        plt.figure(figsize=(8,6))
        mesh = plt.pcolormesh(den_space, w_values, gamma_edd_all-gamma_BD_all, shading='auto', cmap='viridis')
        cbar = plt.colorbar(mesh)
        cbar.set_label(r'$\gamma_e - \gamma_{BD}$')
        # cbar.ax.set_yscale("log")
        plt.yscale("log")
        plt.ylim(np.exp(lowest_w), np.exp(highest_w))
        plt.xlim(min(den_space),max(den_space))
        plt.xlabel(r'Density $\rho$ (MeV/fm$^3$)')
        plt.ylabel(r'$\omega$')
        plt.savefig(f'./saved_matrices_and_plots/heatmap_gamma_e-gamma_BD_{n}.png', dpi=200, bbox_inches="tight")


        # gamma_e-gamma_BD/1-g_BD
        plt.figure(figsize=(8,6))
        mesh = plt.pcolormesh(den_space, w_values, ((gamma_edd_all-gamma_BD_all)/(1-gamma_BD_all) ) * 100, shading='auto', cmap='viridis')
        cbar = plt.colorbar(mesh)
        cbar.set_label(r'$(\gamma_e-\gamma_{BD})/(1-\gamma_{BD})\%$')
        # cbar.ax.set_yscale("log")
        plt.yscale("log")
        plt.ylim(np.exp(lowest_w), np.exp(highest_w))
        plt.xlim(min(den_space),max(den_space))
        plt.xlabel(r'Density $\rho$ (MeV/fm$^3$)')
        plt.ylabel(r'$\omega$')
        plt.savefig(f'./saved_matrices_and_plots/ge-gb_on_1-gb_{n}.png', dpi=200, bbox_inches="tight")


def recover_and_plot(n, lowest_w, highest_w):

    save_dir = 'saved_matrices_and_plots'
    os.makedirs(save_dir, exist_ok=True)

    file_path = os.path.join(save_dir, f'matrice_{n}.npy')

    if os.path.exists(file_path):
        all_a = np.load(file_path)
        if len(all_a[0]) == n:
            print('Complete files detected')
            count = 1
            plot_w_vs_rho(lowest_w, highest_w, n, count)
        else:
            print('Incomplete files')
            count=0
            plot_w_vs_rho(lowest_w, highest_w, n, count)

    else:
        print('Missing files')
        count=0
        plot_w_vs_rho(lowest_w, highest_w, n, count)

recover_and_plot(n=150, lowest_w = np.log(1e-1),highest_w = np.log(1e5))

