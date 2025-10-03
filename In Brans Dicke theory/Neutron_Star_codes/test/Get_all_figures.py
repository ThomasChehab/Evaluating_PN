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
from tqdm import tqdm

def run(opti,rho_cen,w):
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
    tov = TOV(rhoInit, PsiInit, PhiInit, radiusMax_in, radiusMax_out, Npoint, option, dilaton_active, log_active, retro, dependence, w)
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
#new
    SoS_c_max = np.max(tov.v_c)
    #
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
    # print('gamma', gamma)
    # print('r_m',r_m )
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

def make_gamma_beta_plots(n,opti,w):
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

    xi = 1/(2*np.sqrt(3))
    den_space = np.linspace(100,3000,num=n)
    gamma_vec = np.array([])
    beta_vec = np.array([])
    delta_PN_vec = np.array([])
    relative_gamma_vec = np.array([])
    vsurc_a = np.array([])
    gamma_edd_a = np.array([])
    gamma_edd_a_m = np.array([])
    delta_PN = np.array([])
    relative_gamma = np.array([])

    for den in den_space:
        r,a,b,phi,rho_u,a_u,phi_u,b_u,couleur,nom, comment,radiusStar,r_lim,descr,mass_ADM,rho_cen,gamma, vsurc  = run(option,den,w)

        gamma_edd_m = -(8 * gamma * xi)/(1-gamma**2 +4 *gamma * xi)#(1 - gamma**2 - 4 * gamma * xi)/(1 - gamma**2 + 4 * gamma * xi)
        gamma_edd = (1 - gamma**2 - 4 * gamma * xi)/(1 - gamma**2 + 4 * gamma * xi)
        vsurc_a = np.append(vsurc_a,vsurc )
        gamma_BD = (1+w)/(2+w)
        print("gamma_BD", gamma_BD)
        print("gamma_e", gamma_edd)
        print('diff',gamma_edd-gamma_BD )
        print('1-gamma_BD', 1-gamma_BD)
        print('1-gamma_e', 1-gamma_edd)
        print('gBD-gedd/1-gbd*100', (gamma_BD - gamma_edd)/(1-gamma_BD) * 100 )
        gamma_vec = np.append(gamma_vec,gamma)
        beta_vec = np.append(beta_vec,(1-gamma**2)/(1+gamma**2))
        relative_gamma_vec = np.append(relative_gamma_vec, (1- gamma_edd)/(1-gamma_BD ))

        gamma_edd_a_m = np.append(gamma_edd_a_m,gamma_edd_m )
        gamma_edd_a = np.append(gamma_edd_a,gamma_edd )

        delta = 4/3 * (( 1 - gamma**2 - 4 * gamma * xi)**2 - (1/4) * (1 + gamma**2)**2 ) * 1/(1 - gamma**2 + 4 * gamma * xi)**2
        delta_PN = np.append(delta_PN,delta)

    index_max = np.where(vsurc_a > 1/np.sqrt(3))[0][0]+1 # +1 pour correspondre avec hbar in the sky
    den_space = den_space[0:index_max]
    gamma_vec = gamma_vec[0:index_max]
    beta_vec = beta_vec[0:index_max]
    gamma_edd_a = gamma_edd_a[0:index_max]
    gamma_edd_a_m = gamma_edd_a_m[0:index_max]
    relative_gamma_a = relative_gamma_vec[0:index_max]
    delta_a = delta_PN[0:index_max]


    fig,ax = plt.subplots()
    plt.xlabel('Central density ($MeV/fm^3$)')
    ax.plot(den_space,gamma_edd_a_m, label=f'$\\gamma -1$ ({comment})', color = 'tab:blue') # m = minus
    ax.set_ylabel('$\\gamma - 1 $', fontsize=18)#, color = 'tab:blue')

    fig.legend( bbox_to_anchor=(0.9, 0.6))
    plt.savefig(f'./saved_matrices_and_plots/evaluating_gamma_with_alpha.png', dpi = 200,bbox_inches='tight')
    # plt.show()
    plt.close()

    fig,ax = plt.subplots()
    plt.xlabel('Central density ($MeV/fm^3$)')
    ax.plot(den_space,delta_a-1, label=f'$\\delta$ ({comment})', color = 'tab:blue')
    ax.set_ylabel('$\\delta-1$', fontsize=18)#, color = 'tab:blue')

    fig.legend( bbox_to_anchor=(0.9, 0.6))
    plt.savefig(f'./saved_matrices_and_plots/evaluating_delta_with_alpha.png', dpi = 200,bbox_inches='tight')
    # plt.show()
    plt.close()

    fig,ax = plt.subplots()
    plt.xlabel('Central density ($MeV/fm^3$)')
    ax.plot(den_space,relative_gamma_a, label=f'$\\gamma_r$ ({comment})', color = 'tab:blue')
    ax.set_ylabel('$\\gamma_r$', fontsize=18)#, color = 'tab:blue')

    fig.legend( bbox_to_anchor=(0.9, 0.6))
    plt.savefig(f'./saved_matrices_and_plots/evaluating_gamma_relative_with_alpha.png', dpi = 200,bbox_inches='tight')
    # plt.show()
    plt.close()


###################################################
def compute_gamma(n,opti,w):
    nspace = n
    option = opti

    # Saving repository
    save_dir = 'saved_matrices_and_plots'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    all_a = os.path.join(save_dir, f'matrice_{n}.npy')

    #facteur conforme
    xi = 1/(2*np.sqrt(3))
    #densité
    den_space = np.linspace(100,3000,num=n)
    # création des vecteurs qui vont contenir les paramèrtes utiles
    vsurc_vec = np.array([])
    gamma_edd_vec = np.array([])
    gamma_BD_vec = np.array([])
    gamma_diff_vec = np.array([])
    gamma_deviation_vec = np.array([])

    start_idx = len(all_a[0])

    for den in tqdm(den_space):
        r,a,b,phi,rho_u,a_u,phi_u,b_u,couleur,nom, comment,radiusStar,r_lim,descr,mass_ADM,rho_cen,gamma, vsurc  = run(option,den,w)

        vsurc_vec = np.append(vsurc_vec,vsurc )
        # print(gamma)
        # on récupère gamma_e
        gamma_edd = (1 - gamma**2 - 4 * gamma * xi)/(1 - gamma**2 + 4 * gamma * xi)
        # print(repr(gamma_edd))
        gamma_edd_vec = np.append(gamma_edd_vec,gamma_edd )
        # on récupère gamma
        gamma_BD = (1+w)/(2+w)
        gamma_BD_vec = np.append(gamma_BD_vec, gamma_BD)
        # On récupère gamma - gamma_exact
        gamma_diff = gamma_BD - gamma_edd
        gamma_diff_vec = np.append(gamma_diff_vec, gamma_diff)
        #on récupère gamma_e-1/gamma-1
        gamma_deviation = (1-gamma_edd)/(1-gamma_BD)
        gamma_deviation_vec = np.append(gamma_deviation_vec,gamma_deviation)
    # On ne veut que les cas ou v < c/sqrt(3)
        if den == den_space[-1]: # si la valeur de densité est la dernière, tronquer
            index_max = np.where(vsurc_vec > 1/np.sqrt(3))[0][0]
            gamma_edd_vec = gamma_edd_vec[0:index_max]
            gamma_diff_vec = gamma_diff_vec[0:index_max]
            gamma_BD_vec = gamma_BD_vec[0:index_max]
            gamma_deviation_vec = gamma_deviation_vec[0:index_max]
            den_space = den_space[0:index_max]

    return gamma_edd_vec,gamma_BD_vec, gamma_diff_vec,gamma_deviation_vec, den_space#, w_values



def plot_w_vs_rho(lowest_w, highest_w, n, count):

    if count==0:

        save_dir = 'saved_matrices_and_plots'
        os.makedirs(save_dir, exist_ok=True)

        w_values = np.linspace(lowest_w, highest_w, n)
        w_values = np.exp(w_values)
        den_space = np.linspace(100,3000,num=n)

        all_a = os.path.join(save_dir, f'matrice_{n}.npy')
        all_w = os.path.join(save_dir, f'matrice_{n}_w.npy')
        all_rho = os.path.join(save_dir, f'matrice_{n}_rho.npy')
        all_gamma = os.path.join(save_dir, f'matrice_{n}_gamma.npy')

        # Loading existing data
        if os.path.exists(all_a):
            gamma_edd_all, gamma_BD_all, gamma_diff_all, gamma_deviation_all = np.load(all_a, allow_pickle=True)
            print(f"Founded incomplete files : {all_a}, {all_w}, {all_rho}, {all_gamma}")
            start_idx = len(gamma_edd_all)
            print(f'Restarting at w = {start_idx}')
            gamma_edd_all = list(gamma_edd_all)
            gamma_BD_all = list(gamma_BD_all)
            gamma_diff_all = list(gamma_diff_all)
            gamma_deviation_all = list(gamma_deviation_all)
        else:
            gamma_edd_all = []
            gamma_BD_all = []
            gamma_diff_all = []
            gamma_deviation_all = []
            start_idx = 0

        remaining_w_values = w_values[start_idx:]
        counter = len(remaining_w_values)
        for value in remaining_w_values:
            gamma_edd_vec, gamma_BD_vec, gamma_diff_vec, gamma_deviation_vec, den_space = compute_gamma(n, 1, value)
            print(f'Remaing {counter} iterations')
            # print('edd_vec', gamma_edd_vec[-1])
            gamma_edd_all.append(gamma_edd_vec)
            # print('EDD',repr(gamma_edd_all[-1][-1]))
            gamma_BD_all.append(gamma_BD_vec)
            gamma_diff_all.append(gamma_diff_vec)
            gamma_deviation_all.append(gamma_deviation_vec)

            np.save(all_a, [np.array(gamma_edd_all),np.array(gamma_BD_all), np.array(gamma_diff_all), np.array(gamma_deviation_all)])
            np.save(all_w, [w_values])
            np.save(all_rho, [den_space])
            counter -= 1
        print('EDD', gamma_edd_all[-1])
        print('BD',gamma_BD_all[-1])
        gamma_edd_all = np.array(gamma_edd_all)
        gamma_deviation_all = np.array(gamma_deviation_all)
        gamma_BD_all = np.array(gamma_BD_all)
        # First plot only gamma_e
        plt.figure(figsize=(8,6))
        mesh = plt.pcolormesh(den_space, w_values, gamma_edd_all, shading='auto', cmap='viridis')
        cbar = plt.colorbar(mesh)
        cbar.set_label(r'$\gamma$')
        # cbar.ax.set_yscale("log")
        plt.yscale("log")
        plt.xlabel(r'Densité $\rho$ (MeV/fm$^3$)')
        plt.ylabel(r'$\omega$')
        plt.ylim(1e-1, 1e6)
        plt.xlim(min(den_space),max(den_space))
        plt.savefig(f'./saved_matrices_and_plots/heatmap_gamma_exact_{n}.png', dpi=200, bbox_inches="tight")

        # plt.show()
        # plt.close()
        # Second plot 1-gamma_e
        plt.figure(figsize=(8,6))
        mesh = plt.pcolormesh(den_space, w_values, 1-gamma_edd_all, shading='auto', cmap='viridis')
        cbar = plt.colorbar(mesh)
        cbar.set_label(r'$1-\gamma$')
        # cbar.ax.set_yscale("log")
        plt.yscale("log")
        plt.xlabel(r'Densité $\rho$ (MeV/fm$^3$)')
        plt.ylabel(r'$\omega$')
        plt.ylim(1e-1, 1e6)
        plt.xlim(min(den_space),max(den_space))
        plt.savefig(f'./saved_matrices_and_plots/heatmap_gamma_exact_m_{n}.png', dpi=200, bbox_inches="tight")

        # Third plot 1-gamma_e/1-gamma_BD
        plt.figure(figsize=(8,6))
        mesh = plt.pcolormesh(den_space, w_values, gamma_deviation_all, shading='auto', cmap='viridis')
        cbar = plt.colorbar(mesh)
        cbar.set_label(r'$(1-\gamma_e)/(1-\gamma)$')
        # cbar.ax.set_yscale("log")
        plt.yscale("log")
        plt.xlabel(r'Densité $\rho$ (MeV/fm$^3$)')
        plt.ylabel(r'$\omega$')
        plt.ylim(1e-1, 1e6)
        plt.xlim(min(den_space),max(den_space))
        plt.savefig(f'./saved_matrices_and_plots/heatmap_gamma_deviation_{n}.png', dpi=200, bbox_inches="tight")

        # Fourth plot gamma_BD
        plt.figure(figsize=(8,6))
        mesh = plt.pcolormesh(den_space, w_values, gamma_BD_all, shading='auto', cmap='viridis')
        cbar = plt.colorbar(mesh)
        cbar.set_label(r'$\gamma$')
        # cbar.ax.set_yscale("log")
        plt.yscale("log")
        plt.ylim(1e-1, 1e6)
        plt.xlim(min(den_space),max(den_space))
        plt.xlabel(r'Densité $\rho$ (MeV/fm$^3$)')
        plt.ylabel(r'$\omega$')
        plt.savefig(f'./saved_matrices_and_plots/heatmap_gamma_BD_{n}.png', dpi=200, bbox_inches="tight")

    else:

        save_dir = 'saved_matrices_and_plots'
        os.makedirs(save_dir, exist_ok=True)
        all_a = os.path.join(save_dir, f'matrice_{n}.npy')
        all_w = os.path.join(save_dir, f'matrice_{n}_w.npy')
        all_rho = os.path.join(save_dir, f'matrice_{n}_rho.npy')

        gamma_edd_all, gamma_BD_all, gamma_diff_all, gamma_deviation_all = np.load(all_a, allow_pickle=True)
        w_values = np.load(all_w, allow_pickle=True)[0]
        den_space = np.load(all_rho, allow_pickle=True)[0]
        # print('edd_a', gamma_edd_all[-1])
        # print('BD',gamma_BD_all[-1])
        plt.figure(figsize=(8,6))
        mesh = plt.pcolormesh(den_space, w_values, gamma_edd_all , shading='auto', cmap='viridis')
        cbar = plt.colorbar(mesh)
        cbar.set_label(r'$\gamma_e$')
        plt.yscale("log")
        # cbar.ax.set_yscale("log")
        plt.ylim(1e-1, 1e6)
        plt.xlim(min(den_space),max(den_space))
        plt.xlabel(r'Densité $\rho$ (MeV/fm$^3$)')
        plt.ylabel(r'$\omega$')
        # plt.hlines(y=w_values[1], xmin=min(den_space), xmax=max(den_space), color='b', linestyle='-')
        plt.savefig(f'./saved_matrices_and_plots/heatmap_{n}.png', dpi=200, bbox_inches="tight")
        # plt.show()

        # Second plot 1-gamma_e
        plt.figure(figsize=(8,6))
        mesh = plt.pcolormesh(den_space, w_values, 1-gamma_edd_all, shading='auto', cmap='viridis')
        cbar = plt.colorbar(mesh)
        cbar.set_label(r'$1-\gamma_e$')
        # cbar.ax.set_yscale("log")
        plt.yscale("log")
        plt.ylim(1e-1, 1e6)
        plt.xlim(min(den_space),max(den_space))
        plt.xlabel(r'Densité $\rho$ (MeV/fm$^3$)')
        plt.ylabel(r'$\omega$')

        plt.savefig(f'./saved_matrices_and_plots/heatmap_gamma_exact_m_{n}.png', dpi=200, bbox_inches="tight")

        # Third plot 1-gamma_e/1-gamma_BD
        plt.figure(figsize=(8,6))
        mesh = plt.pcolormesh(den_space, w_values, gamma_deviation_all, shading='auto', cmap='viridis')
        cbar = plt.colorbar(mesh)
        cbar.set_label(r'$(1-\gamma_e)/(1-\gamma)$')
        # cbar.ax.set_yscale("log")
        plt.yscale("log")
        plt.ylim(1e-1, 1e6)
        plt.xlim(min(den_space),max(den_space))
        plt.xlabel(r'Densité $\rho$ (MeV/fm$^3$)')
        plt.ylabel(r'$\omega$')
        plt.savefig(f'./saved_matrices_and_plots/heatmap_gamma_deviation_{n}.png', dpi=200, bbox_inches="tight")

        # Fourth plot gamma_BD
        plt.figure(figsize=(8,6))
        mesh = plt.pcolormesh(den_space, w_values, gamma_BD_all, shading='auto', cmap='viridis')
        cbar = plt.colorbar(mesh)
        cbar.set_label(r'$\gamma$')
        # cbar.ax.set_yscale("log")
        plt.yscale("log")
        plt.ylim(1e-1, 1e6)
        plt.xlim(min(den_space),max(den_space))
        plt.xlabel(r'Densité $\rho$ (MeV/fm$^3$)')
        plt.ylabel(r'$\omega$')
        plt.savefig(f'./saved_matrices_and_plots/heatmap_gamma_BD_{n}.png', dpi=200, bbox_inches="tight")

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

recover_and_plot(n=6, lowest_w = np.log(1e-1),highest_w = np.log(1e6))



