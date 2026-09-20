"""
Geometric program for spiral inductors.

Taken from 'Optimization of Inductor Circuits via Geometric Programming' by
Hershenson et al.
"""

import gpkit as gp, numpy as np
import scipy.constants as sc
from . import mohan as m
from rapid_coil_synthesis.pdk import ProcessParams

"""
GP Problem class
"""
class InductanceTargetMaxQ:
  def __init__(self, pdk:ProcessParams, ind_nh:float, freq_hz:float, srf_min_hz:float=0, shape:str="square", pgs:bool=False):
    if shape not in m.SHAPES:
      raise ValueError(f"Shape must be one of {list(m.SHAPES)}")

    self.pdk: ProcessParams = pdk
    self.ind_nh: float = ind_nh
    self.freq_hz: float = freq_hz
    self.srf_min_hz: float = srf_min_hz
    self.shape: m.ShapeParams = m.SHAPES[shape]
    self.pgs = pgs

    self.create_variables()
    self.create_monomials()
    self.create_model()
  
  def create_variables(self):
    "-- SI Constants --"
    self.mu_0 = gp.Variable("\\mu_0", sc.mu_0, "H/m", "Magnetic permeability of free space", constant=True)
    self.e_0 = gp.Variable("\\epsilon_0", sc.epsilon_0, "F/m", "Electric permittivity of free space", constant=True)

    "-- Unit Constants --"
    self.um = gp.Variable("um", 1e-6, "m", "Micron unit, for Mohan's dimensionless length fit", constant=True)
    self.nh = gp.Variable("nH", 1e-9, "H", "Nanohenry unit, for Mohan's dimensionless inductance fit", constant=True)
    self.F_per_m2 = gp.Variable("F_per_m2", 1, "F/m^2", constant=True)
    self.ohm_m2   = gp.Variable("ohm_m2", 1, "ohm*m^2", constant=True)

    "-- Requirements --"
    self.omega = gp.Variable("\\omega", 2*np.pi*self.freq_hz, "rad/s", "Operating angular frequency", constant=True)
    self.L_req = gp.Variable("L_{req}", self.ind_nh, "nH", "Required inductance in nano Henries", constant=True)
    self.omega_sr_min = gp.Variable("\\omega_{sr,min}", 2*np.pi*self.srf_min_hz, "rad/s", "Minimum self-resonance frequency", constant=True)

    "-- Process Constants --"
    metal = self.pdk.top_metals[0]
    self.sigma_m = gp.Variable("\\sigma_m", metal.sigma, "S/m", "Top metal conductivity", constant=True)
    self.t_m = gp.Variable("t_m", metal.thickness, "m", "Top metal thickness", constant=True)
    self.w_min = gp.Variable("w_{min}", metal.w_min, "m", "Minimum top metal width", constant=True)
    self.s_min = gp.Variable("s_{min}", metal.s_min, "m", "Minimum top metal spacing", constant=True)

    self.eps_ox = gp.Variable("\\epsilon_{ox}", self.pdk.eps_r_ox * sc.epsilon_0, "F/m", "Oxide permittivity", constant=True)
    self.t_ox = gp.Variable("t_{ox}", self.pdk.t_ox, "m", "Oxide thickness", constant=True)

    via = self.pdk.top_vias[0]
    self.t_ox_tm1tm2 = gp.Variable("t_{ox,TM1-TM2}", via.thickness, "m", "Oxide thickness between TopMetal1 and TopMetal2", constant=True)

    self.eps_sub = gp.Variable("\\epsilon_{sub}", self.pdk.eps_r_sub * sc.epsilon_0, "F/m", "Substrate permittivity", constant=True)
    self.sigma_sub = gp.Variable("\\sigma_{sub}", self.pdk.sigma_sub, "S/m", "Substrate conductivity", constant=True)
    self.t_sub = gp.Variable("t_{sub}", self.pdk.t_sub, "m", "Substrate thickness", constant=True)

    "-- Problem Variables --"
    self.Q_min = gp.Variable("Q_{min}", "-", "Minimum quality factor", positive=True)
    self.d_out = gp.Variable("d_{out}", "m", "Outter diameter of the inductor", positive=True)
    self.d_avg = gp.Variable("d_{avg}", "m", "Average diameter of the inductor", positive=True)

    self.w = gp.Variable("w", "m", "Metal track width", positive=True)
    self.s = gp.Variable("s", "m", "Spacing between metal tracks", positive=True)
    self.n = gp.Variable("n", "-", "Number of turns", positive=True)
  
  def create_monomials(self):
    self.l = self.shape.sides * self.n * self.d_avg
    skin_depth_m = np.sqrt(2/(self.omega.value * self.mu_0.value * self.sigma_m.value))
    if (skin_depth_m < self.t_m.value):
      print(f"Skin depth is {skin_depth_m.to("um")}")
      k_1 = 1 / (self.sigma_m * skin_depth_m*(1 - np.exp(-self.t_m.value/skin_depth_m)))
    else:
      print("No skin effect")
      k_1 = 1 / (self.sigma_m * self.t_m)  # No skin effect occurs

    k_2 = (self.eps_ox )/(2 * self.t_ox)
    k_3 = (self.eps_ox)/(self.t_ox_tm1tm2)
    k_4 = self.eps_sub / (2 * self.t_sub)
    k_5 = 2 * self.t_sub / self.sigma_sub

    "-- Equivalent Circuit --"
    self.R_s  = (k_1 * self.l / self.w).to("ohm")
    self.C_ox = (k_2 * self.l * self.w).to("farad")
    self.C_s  = (k_3 * self.n * self.w**2).to("farad")
    self.C_si = (k_4 * self.l * self.w).to("farad")
    self.R_si = (k_5 / (self.l * self.w)).to("ohm")

    coeffs = self.shape.inductance
    self.L_s = (coeffs.beta
                * (self.d_out / self.um)**coeffs.a1
                * (self.w / self.um)**coeffs.a2
                * (self.d_avg / self.um)**coeffs.a3
                * self.n**coeffs.a4
                * (self.s / self.um)**coeffs.a5
                * self.nh).to("henry")

    "-- Shunt Legs --"
    # Here .value must be used because GPkit wont allow posynomial divisions.
    # The math is correct, just the way GPkit handles it must be sidestepped
    k2_v, k4_v, k5_v, omega_v = k_2.value, k_4.value, k_5.value, self.omega.value

    k_6 = (k5_v * (k2_v + k4_v)**2 / k2_v**2 + 1 / (k2_v**2 * k5_v * omega_v**2)) \
              .to("ohm*m^2").magnitude * self.ohm_m2
    k_7 = ((k2_v * (1 + k4_v * k5_v**2 * omega_v**2 * (k2_v + k4_v)))
               / (1 + k5_v**2 * omega_v**2 * (k2_v + k4_v)**2)) \
              .to("F/m^2").magnitude * self.F_per_m2

    self.R_p = (k_6 / (self.l * self.w)).to("ohm") if not self.pgs else gp.Variable("R_p", np.inf, "ohm", constant=True)
    self.C_p = (k_7 * self.l * self.w).to("farad") if not self.pgs else self.C_ox
  
  def create_model(self):
    objective = self.Q_min**-1
    # Normalized variables
    C_tot = self.C_p + 2*self.C_s
    rho = self.omega * self.L_s / self.R_s  # inductive reactance
    gamma = self.omega**2 * self.L_s * C_tot/2  # Capacitive loading
    delta = self.R_s**2 * C_tot/2 / self.L_s  # Q factor normalization

    k_sr = self.omega_sr_min / self.omega

    constraints = [
        self.L_s == self.L_req,  # Required inductance

        # normalized Q constraint
        self.Q_min * (2*self.R_p + (rho**2 + 1)*self.R_s) / (rho * 2*self.R_p) + delta + gamma <= 1,

        # self resonance
        delta + gamma <= 1,

        # minimum SRF
        k_sr**2 * gamma + delta <= 1,
        
        # PDK dimensional constraints
        self.w >= self.w_min,
        self.s >= self.s_min,
        #self.d_out**2 <= self.A_max,

        #Geometry constriants
        ## TODO: Warn when this constraint is not tight in the solution
        self.d_avg + self.n*self.s + self.n*self.w <= self.d_out,
    ]

    self.model = gp.Model(objective, constraints)

  def solve(self):
    self.solution = self.model.solve()

  def __str__(self):
    if not hasattr(self, "solution"):
      return "not solved yet"

    def v(expr):
      return expr.sub(self.solution["variables"]).value

    d_in = v(self.d_avg) - (v(self.n) - 1)*v(self.s) - v(self.n)*v(self.w)

    return f"""
L_s={v(self.L_s).to("nH")}
w={v(self.w).to("um")}
s={v(self.s).to("um")}
l={v(self.l).to("um")}
n={v(self.n)} turns
d_out={v(self.d_out).to("um")}
d_in={d_in.to("um")}
d_avg={v(self.d_avg).to("um")}
C_si={v(self.C_si).to("fF")}
C_ox={v(self.C_ox).to("fF")}
C_s={v(self.C_s).to("fF")}
C_p={v(self.C_p).to("fF")}
R_si={v(self.R_si).to("kohm")}
R_s={v(self.R_s).to("ohm")}
R_p={v(self.R_p).to("kohm")}
"""
