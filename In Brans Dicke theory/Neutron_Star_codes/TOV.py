#!/usr/bin/env python
import scipy.constants as cst
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import FormatStrFormatter
import numpy as np
from numpy import linalg as npla
from scipy.integrate import solve_ivp
from scipy.integrate import cumulative_trapezoid as integcum
from scipy.integrate import trapezoid as integ
import os
import csv
from scipy.integrate import simpson as simps

c2 = cst.c**2
kappa = 8*np.pi*cst.G/c2**2
k = 1.475*10**(-3)*(cst.fermi**3/(cst.eV*10**6))**(2/3)*c2**(5/3)
massSun = 1.989*10**30

#Equation of state
def PEQS(rho):
    return k*rho**(5/3)

#Inverted equation of state
def RhoEQS(P):
    return (P/k)**(3/5)

# def v_sound_c(rho):
def v_sound_c(Phi, P):
    return np.sqrt(5/3 * k * RhoEQS(P)**(2/3)) / cst.c


#Equation for b
def b(r, m):
    return (1-(c2*m*kappa/(4*np.pi*r)))**(-1)

#Equation for da/dr
def adota(r, P, m, Psi, Phi, w):
    A = (b(r, m)/r)
    B = (1-(1/b(r, m))+P*kappa*r**2*Phi**(-1)-2*r*Psi/(b(r,m)*Phi) + ( -H00(r,m,Psi, Phi, w) ))
    C = (1+r*Psi/(2*Phi))**(-1)
    return A*B*C

#Equation for D00
def D00(r, P, m, Psi, Phi, w):
    ADOTA = adota(r, P, m, Psi, Phi, w)
    rho = RhoEQS(P)
    T = -c2*rho + 3*P
    A = Psi*ADOTA/(2*Phi*b(r,m))
    B = -kappa*(T)/(Phi * (3+2*w))
    return A+B

def H00(r,m,Psi, Phi, w):
    A = - w * Psi**2/Phi**2 * 1/(2*b(r,m))
    return A

#Equation for db/dr
def bdotb(r, P, m, Psi, Phi, w):
    rho = RhoEQS(P)
    A = -b(r,m)/r
    B = 1/r
    C = b(r,m)*r*(-H00(r,m,Psi, Phi, w)-D00(r, P, m, Psi, Phi, w)+kappa*c2*rho*Phi**(-1))
    return A+B+C

#Equation for dP/dr
def f1(r, P, m, Psi, Phi, w):
    ADOTA = adota(r, P, m, Psi, Phi, w)
    rho = RhoEQS(P)
    return -(ADOTA/2)*(P+rho*c2)

#Equation for dm/dr
def f2(r, P, m, Psi, Phi, w):
    rho = RhoEQS(P)
    A = 4*np.pi*rho*(Phi**(-1))*r**2
    B = 4*np.pi*(-D00(r, P, m, Psi, Phi,w)/(kappa*c2))*r**2
    C = 4*np.pi*(-H00(r, m, Psi, Phi, w)/(kappa*c2))*r**2
    return A+B

#Equation for dPsi/dr
def f4(r, P, m, Psi, Phi, w):
    ADOTA = adota(r, P, m, Psi, Phi,w)
    BDOTB = bdotb(r, P, m, Psi, Phi,w)
    rho = RhoEQS(P)
    T = -c2*rho + 3*P
    A = (-Psi/2)*(ADOTA-BDOTB+4/r)
    B = b(r,m)*kappa*T/(3+2*w)
    return A+B

#Equation for dPhi/dr
def f3(r, P, m, Psi, Phi):
    return Psi


#Define for dy/dr
def dy_dr(r, y, w):
    P, M, Phi, Psi = y
    dy_dt = [f1(r, P, M, Psi, Phi, w), f2(r, P, M, Psi, Phi, w),f3(r, P, M, Psi, Phi),f4(r, P, M, Psi, Phi, w) ]
    return dy_dt

#Define for dy/dr out of the star
def dy_dr_out(r, y, P, w):
    M, Phi, Psi = y
    dy_dt = [f2(r, P, M, Psi, Phi, w),f3(r, P, M, Psi, Phi),f4(r, P, M, Psi, Phi, w) ]
    return dy_dt

class TOV():

    def __init__(self, initDensity, initPsi, initPhi, radiusMax_in, radiusMax_out, Npoint, log_active, w):
#Init value
        self.initDensity = initDensity
        self.initPressure = PEQS(initDensity)
        self.initPsi = initPsi
        self.initPhi = initPhi
        self.initMass = 0
        self.log_active = log_active
        self.w = w

#Computation variable
        self.radiusMax_in = radiusMax_in
        self.radiusMax_out = radiusMax_out
        self.Npoint = Npoint
#Star data
        self.Nstar = 0
        self.massStar = 0
        self.massADM = 0
        self.pressureStar = 0
        self.radiusStar = 0
        self.phiStar = 0
#Output data
        self.pressure = 0
        self.mass = 0
        self.Phi = 0
        self.Psi = 0
        self.radius = 0
        self.g_tt = 0
        self.g_rr = 0
        self.g_tt_ext = 0
        self.g_rr_ext = 0
        self.r_ext = 0
        self.phi_inf = 0
        self.R = 0

    def Compute(self):
        if self.log_active:
            print('===========================================================')
            print('SOLVER INSIDE THE STAR')
            print('===========================================================\n')
            print('Initial density: ', self.initDensity, ' MeV/fm^3')
            print('Initial pressure: ', self.initPressure/10**12, ' GPa')
            print('Initial mass: ', self.initMass/massSun, ' solar mass')
            print('Initial phi: ', self.initPhi)
            print('Initial psi: ', self.initPsi)
            print('Number of points: ', self.Npoint)
            print('Radius max: ', self.radiusMax_in/1000, ' km')
        y0 = [self.initPressure,self.initMass,self.initPhi,self.initPsi]
        if self.log_active:
            print('y0 = ', y0,'\n')
        r_min = 0.000000001
        r = np.linspace(r_min,self.radiusMax_in,self.Npoint)
        if self.log_active:
            print('radius min ',r_min)
            print('radius max ',self.radiusMax_in)
        sol = solve_ivp(dy_dr, [r_min, self.radiusMax_in], y0, method='RK45',t_eval=r ,args=(self.w,))
        if sol.t[-1]<self.radiusMax_in:
            self.pressure = sol.y[0][0:-2]
            self.mass = sol.y[1][0:-2]
            self.Phi = sol.y[2][0:-2]
            self.v_c = v_sound_c(self.Phi, self.pressure)
            self.Psi = sol.y[3][0:-2]
            self.radius = sol.t[0:-2]
            # Value at the radius of star
            self.massStar = sol.y[1][-1]
            self.radiusStar = sol.t[-1]
            self.pressureStar = sol.y[0][-1]
            self.phiStar = sol.y[2][-1]
            n_star = len(self.radius)
            if self.log_active:
                print('Star radius: ', self.radiusStar/1000, ' km')
                print('Star Mass: ', self.massStar/massSun, ' solar mass')
                print('Star Mass: ', self.massStar, ' kg')
                print('Star pressure: ', self.pressureStar, ' Pa\n')
                print('===========================================================')
                print('SOLVER OUTSIDE THE STAR')
                print('===========================================================\n')
            y0 = [self.massStar, sol.y[2][-1],sol.y[3][-1]]
            if self.log_active:
                print('y0 = ', y0,'\n')
            r = np.logspace(np.log(self.radiusStar)/np.log(10),np.log(self.radiusMax_out)/np.log(10),self.Npoint)
            if self.log_active:
                print('radius min ',self.radiusStar)
                print('radius max ',self.radiusMax_out)
            sol = solve_ivp(dy_dr_out, [r[0], self.radiusMax_out], y0,method='DOP853', t_eval=r, args=(0,self.w))
            self.pressure = np.concatenate([self.pressure, np.zeros(self.Npoint)])
            self.mass = np.concatenate([self.mass, sol.y[0]])
            self.Phi = np.concatenate([self.Phi, sol.y[1]])
            self.Psi = np.concatenate([self.Psi,  sol.y[2]])
            ##
            radiusetoile = self.radius
            ##
            self.radius = np.concatenate([self.radius, r])
            self.phi_inf = self.Phi[-1]
            if self.log_active:
                print('Phi at infinity ', self.phi_inf)
            # Compute metrics
            self.g_rr = b(self.radius, self.mass)
            a_dot_a = adota(self.radius, self.pressure, self.mass, self.Psi, self.Phi, self.w)
            b_dot_b = bdotb(self.radius, self.pressure, self.mass, self.Psi, self.Phi, self.w)
            self.g_tt = np.exp(np.concatenate([[0.0], integcum(a_dot_a,self.radius)])-integ(a_dot_a,self.radius))
            #compute Ricci scalar
            a_dot = a_dot_a*self.g_tt
            a_2dot = (a_dot[1:-1]-a_dot[0:-2])/(self.radius[1:-1]-self.radius[0:-2])
            A = self.g_tt[0:-2]
            B = self.g_rr[0:-2]
            r = self.radius[0:-2]
            a_dot_a = a_dot_a[0:-2]
            b_dot_b = b_dot_b[0:-2]
            self.massADM = self.mass[-1]
            self.g_tt_ext = np.array(self.g_tt[n_star:-1])
            self.g_rr_ext = np.array(self.g_rr[n_star:-1])
            self.r_ext = np.array(self.radius[n_star:-1])
            self.r_ext[0] = self.radiusStar
            if self.log_active:
                print('Star Mass ADM: ', self.massADM, ' kg')
                print('===========================================================')
                print('END')
                print('===========================================================\n')

####################

            E_int = 4 * cst.pi * simps(radiusetoile**2 * np.sqrt( self.g_tt[0:len(radiusetoile)] * self.g_rr[0:len(radiusetoile)] ) * (((self.pressure[0:len(radiusetoile)])/k)**(3/5) *c2), radiusetoile )
            P_int = 4 * cst.pi * simps(radiusetoile**2 * np.sqrt( self.g_tt[0:len(radiusetoile)] * self.g_rr[0:len(radiusetoile)] ) * self.pressure[0:len(radiusetoile)], radiusetoile)
            theta = (3 * P_int)/E_int
            # print( 'THETA', theta)
            Xi = 1/(np.sqrt(3+2*self.w)) * ((1-theta)/(1+theta))
            gamma_bd = (1+self.w) /(2+self.w)
            gamma_theta = (np.sqrt(3+2*self.w) - Xi)/(np.sqrt(3+2*self.w)+Xi)
            self.Ge_theta = gamma_theta

            delta_theta = 4/3 * (gamma_theta**2 - ((3+2*self.w)*(1+Xi**2))/(4 *( np.sqrt(3+2*self.w) + Xi)**2))

            self.Delta_theta = delta_theta

##################
        else:
            print('Pressure=0 not reached')

    def ComputeTOV_normalization(self):
        self.Compute()
        self.initPhi = self.initPhi/self.phi_inf
        self.Compute()


