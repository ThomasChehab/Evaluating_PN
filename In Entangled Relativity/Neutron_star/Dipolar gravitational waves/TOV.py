from eqs import *
import scipy.constants as cst
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as npla
from pynverse import inversefunc
from scipy import optimize
from scipy.optimize import fsolve
from scipy.optimize import root_scalar
from scipy.integrate import solve_ivp
from scipy.integrate import cumulative_trapezoid as integcum
from scipy.integrate import trapezoid as integ
import csv
from scipy.integrate import simpson as simps
import math
c2 = cst.c**2
kappa = 8*np.pi*cst.G/c2**2
massSun = 1.989*10**30

def stop_condition(t, y, a, b, c):
    return y[0]

#Lagrangian
def Lagrangian(rho, option, option_eqs):
    P, dP_drho = PEQS(rho, option_eqs)
    if option == 0:
        return -c2*rho+3*P
    elif option == 1:
        return -c2*rho
    elif option == 2:
        return P
    else:
        print('not a valid option')

#Speed of sound
def v_sound_c(option_eqs, rho):
    return np.sqrt(PEQS(rho, option_eqs)[1]) / cst.c

#Equation for b
def b(r, m):
    return (1-(c2*m*kappa/(4*np.pi*r)))**(-1) # Eq (29) in 2011.14629.pdf

#Equation for m
def mass(r, b):
    return r/(c2*kappa/(4*np.pi))/(1-1/b)

#Equation for (da/dr)/a
def adota(r, rho, m, Psi, Phi, option_eqs):
    P, dP_drho  = PEQS(rho, option_eqs)
    A = (b(r, m)/r)
    B = (1-(1/b(r, m))+P*kappa*r**2*Phi**(-1/2)-2*r*Psi/(b(r,m)*Phi))
    C = (1+r*Psi/(2*Phi))**(-1)
    return A*B*C # Eq (17) in 2011.14629.pdf

#Equation for D00
def D00(r, rho, m, Psi, Phi, option, option_eqs):
    ADOTA = adota(r, rho, m, Psi, Phi, option_eqs)
    P, dP_drho = PEQS(rho, option_eqs)
    Lm = Lagrangian(rho, option, option_eqs)
    T = -c2*rho + 3*P
    A = Psi*ADOTA/(2*Phi*b(r,m))
    B = kappa*(Lm-T)/(3*Phi**(1/2))
    return A+B # Eq (11) in 2011.14629.pdf

#Equation for (db/dr)/b
def bdotb(r, rho, m, Psi, Phi, option, option_eqs):
    P, dP_drho = PEQS(rho, option_eqs)
    A = -b(r,m)/r
    B = 1/r
    C = b(r,m)*r*(-D00(r, rho, m, Psi, Phi, option, option_eqs)+kappa*c2*rho*Phi**(-1/2))
    return A+B+C  # Eq (10) in 2011.14629.pdf

#Equation for drho/dr
def f1(r, rho, m, Psi, Phi, option, option_eqs):
    ADOTA = adota(r, rho, m, Psi, Phi, option_eqs)
    Lm = Lagrangian(rho, option, option_eqs)
    P, dP_drho = PEQS(rho, option_eqs)
    return (-(ADOTA/2)*(P+rho*c2)+(Psi/(2*Phi))*(Lm-P))/dP_drho

#Equation for dm/dr
def f2(r, rho, m, Psi, Phi, option, option_eqs):
    P, dP_drho = PEQS(rho, option_eqs)
    A = 4*np.pi*rho*(Phi**(-1/2))*r**2
    B = 4*np.pi*(-D00(r, rho, m, Psi, Phi, option, option_eqs)/(kappa*c2))*r**2
    return A+B  # Eq (23) in 2011.14629.pdf

#Equation for dPsi/dr
def f4(r, rho, m, Psi, Phi, option, dilaton_active, option_eqs):
    ADOTA = adota(r, rho, m, Psi, Phi, option_eqs)
    BDOTB = bdotb(r, rho, m, Psi, Phi, option, option_eqs)
    P, dP_drho = PEQS(rho, option_eqs)
    Lm = Lagrangian(rho, option, option_eqs)
    T = -c2*rho + 3*P
    A = (-Psi/2)*(ADOTA-BDOTB+4/r)
    B = b(r,m)*kappa*Phi**(1/2)*(T-Lm)/3
    if dilaton_active:
        return A+B  # Eq (21) in 2011.14629.pdf
    else:
        return 0

#Equation for dPhi/dr
def f3(r, rho, m, Psi, Phi, option, dilaton_active):
    if dilaton_active:
        return Psi # Eq (24) in 2011.14629.pdf
    else:
        return 0

#Define for dy/dr
def dy_dr(r, y, option, dilaton_active, option_eqs):
    rho, M, Phi, Psi = y
    dy_dt = [f1(r, rho, M, Psi, Phi, option, option_eqs), f2(r, rho, M, Psi, Phi, option, option_eqs),f3(r, rho, M, Psi, Phi, option, dilaton_active),f4(r, rho, M, Psi, Phi, option, dilaton_active, option_eqs) ]
    return dy_dt

#Define for dy/dr out of the star
def dy_dr_out(r, y, rho, option, dilaton_active, option_eqs):
    M, Phi, Psi = y
    dy_dt = [f2(r, rho, M, Psi, Phi, option, option_eqs),f3(r, rho, M, Psi, Phi, option, dilaton_active),f4(r, rho, M, Psi, Phi, option, dilaton_active, option_eqs) ]
    return dy_dt

#######################################""
#Equations in order to compute the second integral with P as the radial variable

#Equation for dr/drho
def drdrho(r, rho, m, Psi, Phi, option, option_eqs):
    ADOTA = adota(r, rho, m, Psi, Phi, option_eqs)
    Lm = Lagrangian(rho, option, option_eqs)
    P, dP_drho = PEQS(rho, option_eqs)
    return ((-(ADOTA/2)*(P+rho*c2)+(Psi/(2*Phi))*(Lm-P))/dP_drho)**(-1)


#Equation for dm/drho
def dmdrho(r, rho, m, Psi, Phi, option, option_eqs):
    P, dP_drho = PEQS(rho, option_eqs)
    A = 4*np.pi*rho*(Phi**(-1/2))*r**2
    Lm = Lagrangian(rho, option, option_eqs)
    B = 4*np.pi*(-D00(r, rho, m, Psi, Phi, option, option_eqs)/(kappa*c2))*r**2
    ADOTA = adota(r, rho, m, Psi, Phi, option_eqs)
    drhodr = (-(ADOTA/2)*(P+rho*c2)+(Psi/(2*Phi))*(Lm-P))/dP_drho
    return (A+B)/drhodr  # Eq (23) in 2011.14629.pdf


#Equation for dPhi/drho
def dphidrho(r, rho, m, Psi, Phi, option, dilaton_active, option_eqs):
    Lm = Lagrangian(rho, option, option_eqs)
    P, dP_drho = PEQS(rho, option_eqs)
    ADOTA = adota(r, rho, m, Psi, Phi, option_eqs)
    drhodr = (-(ADOTA/2)*(P+rho*c2)+(Psi/(2*Phi))*(Lm-P))/dP_drho
    if dilaton_active:
        return Psi/drhodr # Eq (24) in 2011.14629.pdf
    else:
        return 0

#Equation for dPsi/drho
def dPsidrho(r, rho, m, Psi, Phi, option, dilaton_active, option_eqs):
    ADOTA = adota(r, rho, m, Psi, Phi, option_eqs)
    BDOTB = bdotb(r, rho, m, Psi, Phi, option, option_eqs)
    P, dP_drho = PEQS(rho, option_eqs)
    Lm = Lagrangian(rho, option, option_eqs)
    T = -c2*rho + 3*P
    A = (-Psi/2)*(ADOTA-BDOTB+4/r)
    B = b(r,m)*kappa*Phi**(1/2)*(T-Lm)/3
    drhodr = (-(ADOTA/2)*(P+rho*c2)+(Psi/(2*Phi))*(Lm-P))/dP_drho
    if dilaton_active:
        return (A+B)/drhodr  # Eq (21) in 2011.14629.pdf
    else:
        return 0

#Define for dy/drho
def dy_drho(rho, y, option, dilaton_active, option_eqs):
    r, M, Phi, Psi = y
    dy_dt = [drdrho(r, rho, M, Psi, Phi, option, option_eqs), dmdrho(r, rho, M, Psi, Phi, option, option_eqs),dphidrho(r, rho, M, Psi, Phi, option, dilaton_active, option_eqs),dPsidrho(r, rho, M, Psi, Phi, option, dilaton_active, option_eqs) ]
    return dy_dt

#performing a change of variable such that x = ln(rho)
#easier to integrate
def drdx(x, r, m, Psi, Phi, option, option_eqs):
    rho = np.exp(x)
    return rho * drdrho(r, rho, m, Psi, Phi, option, option_eqs)

def dmdx(x, r, m, Psi, Phi, option, option_eqs):
    rho = np.exp(x)
    return rho * dmdrho(r, rho, m, Psi, Phi, option, option_eqs)

def dPhidx(x, r, m, Psi, Phi, option, dilaton_active, option_eqs):
    rho = np.exp(x)
    return rho * dphidrho(r, rho, m, Psi, Phi, option, dilaton_active, option_eqs)

def dPsidx(x, r, m, Psi, Phi, option, dilaton_active, option_eqs):
    rho = np.exp(x)
    return rho * dPsidrho(r, rho, m, Psi, Phi,option, dilaton_active, option_eqs)

def dy_dx(x, y, option, dilaton_active, option_eqs):
    r, M, Phi, Psi = y
    return [drdx(x, r, M, Psi, Phi, option, option_eqs), dmdx(x, r, M, Psi, Phi, option, option_eqs), dPhidx(x, r, M, Psi, Phi, option, dilaton_active, option_eqs), dPsidx(x, r, M, Psi, Phi, option, dilaton_active, option_eqs)]

class TOV():
    """
    * Initialization
        - initDensity: initial (meaning at star center) value of density [MeV/fm3]
        - initPsi: initial value for psi (usually to 1).
        - initPhi: initial value for the derivative of psi (usually taken to 0).
        - radiusMax_in: For star interior, the solver integrates until it reach radiusMax_in.
        - radiusMax_out: For star exterior, the solver integrates until it reach radiusMax_out.
        - Npoint: Number at which the solution is evaluated (t_span in solve_ivp).
        - option_lag: Select lagrangian.
            0 -> Lm=T
            1 -> Lm=-c²rho
            2 -> Lm=P
        - dilaton_active:
            True -> Solves for equation of ER.
            False -> Solves for equation of GR.
        - log_active: Consol outputs.
            True -> activates consol output
        - option_eqs:
            0 -> polytropic
            1 -> SLy
    * ComputeTOV
    * Compute
    * Plot
    * PlotMetric
    """
    def __init__(self, initDensity, initPsi, initPhi, radiusMax_in, radiusMax_out, Npoint, option_lag, dilaton_active, log_active, option_eqs):
#Init value
        self.initDensity = initDensity#*cst.eV*10**6/(cst.c**2*cst.fermi**3)
        self.initPressure, dP_drho = PEQS(self.initDensity, option_eqs)
        self.initPsi = initPsi
        self.initPhi = initPhi
        self.initMass = 0
        self.option = option_lag
        self.dilaton_active = dilaton_active
        self.log_active = log_active
        self.option_eqs = option_eqs

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
        self.density_in = 0
        self.density = 0
        self.mass = 0
        self.Phi = 0
        self.Psi = 0
        self.radius = 0
        self.g_tt = 0
        self.g_rr = 0
        self.g_tt_ext = 0
        self.g_rr_ext = 0
        self.r_ext = 0
        self.r_in = 0
        self.phi_inf = 0

    def finding_pressure_vanishes(self):

        #initial values for integration
        y0 = [self.Radius_Last, self.Mass_Last, self.Phi_Last, self.Psi_Last]
        #minimal density value
        density_min = 10**(-10)

        #initial x value
        x0 = np.log(self.density_Last)
        #minimal x value
        xmin = np.log(density_min)
        #linspace of values of x for integration
        x_eval = np.linspace(x0, xmin, self.Npoint)
        #integrating differential equation with variable x
        sol = solve_ivp(
            dy_dx,
            [x0, xmin],
            y0,
            t_eval=x_eval,
            method='RK45',
            args=(self.option, self.dilaton_active, self.option_eqs)
        )

        density = np.exp(sol.t) # recovering density from x = ln(rho)

        # recovering different parameter
        self.density = np.exp(sol.t[:-2]) # taking exp because code return x = ln(rho)
        self.radius = sol.y[0][:-2]
        self.mass = sol.y[1][:-2]
        self.Phi = sol.y[2][:-2]
        self.Psi = sol.y[3][:-2]
        self.pressure = k * (self.density)**(5/3)

        self.densityStar = sol.t[-1]
        self.radiusStar = sol.y[0][-1]
        self.massStar = sol.y[1][-1]
        self.PhiStar = sol.y[2][-1]
        self.PsiStar = sol.y[3][-1]
        self.pressureStar = k * self.densityStar**(5/3)

        return self.density, self.radius, self.mass, self.Phi, self.Psi, self.pressure, self.densityStar, self.radiusStar, self.massStar, self.PhiStar, self.PsiStar, self.pressureStar

    def Compute(self):
        if self.log_active:
            print('===========================================================')
            print('SOLVER INSIDE THE STAR')
            print('===========================================================\n')
            print('Initial density: ', self.initDensity/(cst.eV*10**6/(cst.c**2*cst.fermi**3)), ' MeV/fm^3')
            print('Initial pressure: ', self.initPressure/10**12, ' GPa')
            print('Initial mass: ', self.initMass/massSun, ' solar mass')
            print('Initial phi: ', self.initPhi)
            print('Initial psi: ', self.initPsi)
            print('Number of points: ', self.Npoint)
            print('Radius max: ', self.radiusMax_in/1000, ' km')
        y0 = [self.initDensity,self.initMass,self.initPhi,self.initPsi]
        if self.log_active:
            print('y0 = ', y0,'\n')
        r_min = 10**(-15) # if too low, bugs appears in the plots
        r = np.linspace(r_min,self.radiusMax_in,self.Npoint)
        if self.log_active:
            print('radius min ',r_min)
            print('radius max ',self.radiusMax_in)
        stop_condition.terminal = True
        stop_condition.direction = -1
        sol = solve_ivp(dy_dr, [r_min, self.radiusMax_in], y0, method='RK45', t_eval=r, events = stop_condition, args=(self.option,self.dilaton_active, self.option_eqs))



        self.Radius_frst = sol.t[:]
        self.density_frst = sol.y[0][:]
        self.pressure_frst = k * (self.density_frst)**(5/3)
        self.Mass_frst = sol.y[1][:]
        self.Phi_frst = sol.y[2][:]
        self.Psi_frst = sol.y[3][:]

        self.Radius_Last = sol.t[-1]
        self.density_Last = sol.y[0][-1]
        self.pressure_Last = k * (self.density_Last)**(5/3)
        self.Mass_Last = sol.y[1][-1]
        self.Phi_Last = sol.y[2][-1]
        self.Psi_Last = sol.y[3][-1]


        #Functions that compute the second integral in order to find lowest pressure
        self.density, self.radius, self.mass, self.Phi, self.Psi, self.presure, self.densityStar, self.radiusStar, self.massStar, self.PhiStar, self.PsiStar, self.pressureStar = self.finding_pressure_vanishes()

#############
#here we concatenate values from previous integration to new ones
        if self.radiusStar<self.radiusMax_in:
            self.radius = np.concatenate([self.Radius_frst, self.radius])
            self.density = np.concatenate([self.density_frst, self.density])
            self.mass = np.concatenate([self.Mass_frst, self.mass])
            self.Phi = np.concatenate([self.Phi_frst, self.Phi])
            self.Psi = np.concatenate([self.Psi_frst, self.Psi])
            self.pressure = np.concatenate([self.pressure_frst, self.pressure])
            self.v_c = v_sound_c(self.option_eqs, self.initDensity)
#############
            n_star = len(self.radius)
            if self.log_active:
                print('Star radius: ', self.radiusStar/1000, ' km')
                print('Star Mass: ', self.massStar/massSun, ' solar mass')
                print('Star Mass: ', self.massStar, ' kg')
                print('Star density: ', self.densityStar, ' Pa\n')# le densité
            if self.log_active:
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
            sol = solve_ivp(dy_dr_out, [r[0], self.radiusMax_out], y0,method='DOP853', t_eval=r, args=(0,self.option,self.dilaton_active, self.option_eqs))
            self.density = np.concatenate([self.density, np.zeros(self.Npoint)])
            self.mass = np.concatenate([self.mass, sol.y[0]])
            self.Phi = np.concatenate([self.Phi, sol.y[1]])
            self.Psi = np.concatenate([self.Psi,  sol.y[2]])
            radius_in_star = self.radius # contains radius in the star
            self.radius = np.concatenate([self.radius, r])
            # Compute metrics
            self.g_rr = b(self.radius, self.mass)
            a_dot_a = adota(self.radius, self.density, self.mass, self.Psi, self.Phi, self.option_eqs)
            self.g_tt = np.exp(np.concatenate([[0.0], integcum(a_dot_a,self.radius)])-integ(a_dot_a,self.radius))
            self.phi_inf = self.Phi[-1]
            if self.log_active:
                print('Phi at infinity ', self.phi_inf)
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

            #computation of exact parameter using TOV output data
            #Internal Energy density computation
            E_int = kappa/3 * simps(radius_in_star**2 * np.sqrt( self.g_tt[0:len(radius_in_star)] * self.g_rr[0:len(radius_in_star)] ) * (((self.density[0:len(radius_in_star)]))*c2) * np.sqrt(self.Phi[0:len(radius_in_star)]) , radius_in_star )
            #Internal pressure computation
            P_int = kappa/3 * simps(radius_in_star**2 * np.sqrt( self.g_tt[0:len(radius_in_star)] * self.g_rr[0:len(radius_in_star)] ) * self.density[0:len(radius_in_star)]**(5/3) * k * np.sqrt(self.Phi[0:len(radius_in_star)]), radius_in_star)
            #computing their ration
            theta = 3 * P_int/E_int
            #computing exact parameter according to Eq.() of
            gamma_theta = (1+ theta*(2) + 1/2)/(2 + theta*(1)-1/2)
            self.Ge_theta = gamma_theta
            #computing exact parameter according to Eq.() of
            delta_theta = 4/3 * (gamma_theta**2 - 1/4 * ((3+2*theta)/((9*(1+theta)**2) + 3 * theta**2)**(1/2) )**(-2))
            self.Delta_theta = delta_theta
        else:
            print('Pressure=0 not reached')


    def ComputeTOV(self):
        """
        ComputeTOV is the function to consider in order to compute "physical" quantities. It takes into account phi_inf->1 r->ininity
        """
        self.Compute()
        if self.dilaton_active:
            self.initPhi = self.initPhi/self.phi_inf
            self.Compute()
