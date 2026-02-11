from TOV import *
import matplotlib
import matplotlib.pyplot as plt
import scipy.constants as cst
import numpy as np
import os
import tqdm

c2 = cst.c**2
n = 250
lowest_density = 1e9 # kg/m3
highest_density = 1e13
densities = np.linspace(np.log(lowest_density), np.log(highest_density), n)
densities = np.exp(densities)
# count = 0

def run_ER(rho_cen, count):
    PhiInit = 1
    PsiInit = 0
    option = 1
    radiusMax_in = 20000000
    radiusMax_out = 100000000
    Npoint = 50000
    log_active = False # Change for True for seeing star's data
    dilaton_active = True # Change for false for deactivating scalar field
    rhoInit = rho_cen#*cst.eV*10**6/(cst.c**2*cst.fermi**3)

    tov = TOV(rhoInit , PsiInit, PhiInit, radiusMax_in, radiusMax_out, Npoint, option, dilaton_active, log_active, count)
    tov.ComputeTOV()


    r = tov.radius #Recovering parameters from TOV code
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
    b_ = 1/(2*np.sqrt(3)) # Conformal factor
    C = f_b[-1]/f_phi[-1]#parameter that tend to infinity cf Eq.(43-45) of << On the numerical evaluation of the ‘exact’ Post-Newtonian parameters in Brans-Dickeand Entangled Relativity theories >>
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

    alpha_def_exact = np.sqrt(3) * (1-ge_theta)/(1+ge_theta)
    alpha_def_jnw = 2 * gamma /(1-gamma**2)

    if count == 0:
        print('===============================')
        print('ALPHA PARAMETER FOR WHITE DWARFS')
        print('===============================\n')
        print(f'Minimal alpha_c', alpha_def_exact,'\n')
    else:
        print(f'Maximal alpha_c', alpha_def_exact)
    # print('alpha_def_jnw', alpha_def_jnw)

run_ER(1e9, 0)
run_ER(1e13, 1)
