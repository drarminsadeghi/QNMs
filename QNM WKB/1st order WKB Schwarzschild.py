import numpy as np
from scipy.optimize import minimize_scalar

# Schwarzschild mass
M = 1
l = 2  # angular momentum number

def V_RW(r):
    """Regge-Wheeler potential for axial gravitational perturbations."""
    return (1 - 2*M/r) * (l*(l+1)/r**2 - 6*M/r**3)
# Function for minimization (negative of potential to find maximum)
def neg_V(r):
    return -V_RW(r)

res = minimize_scalar(neg_V, bounds=(2*M, 10), method='bounded')
r_max = res.x
V0 = V_RW(r_max)

# 2nd derivative at the maximum (numerical)
dr = 1e-5
Vpp = (V_RW(r_max + dr) - 2*V0 + V_RW(r_max - dr)) / dr**2
n = 0  # fundamental mode
omega_squared = V0 - 1j*(n + 0.5)*np.sqrt(-2 * Vpp)
omega = np.sqrt(omega_squared)
print("Fundamental QNM frequency (WKB 1st order):", omega)
