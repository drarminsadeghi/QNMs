import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# ============================================================
# Units and constants
# ============================================================

MBH = 1.0
rh  = 2.0 * MBH

# Parameters shown in the figure
M_over_MBH_list = [1.0, 100.0]        # rows
M_over_a0_list  = [0.1, 0.001]        # columns

# Profile definitions
profiles = {
    "Hernquist": dict(alpha=1.0, beta=4.0, gamma=1.0, linestyle="-"),
    "NFW":       dict(alpha=1.0, beta=3.0, gamma=1.0, linestyle="--"),
}

# Radial grid (close to horizon, wide outer range)
r = np.logspace(np.log10(2.0001 * MBH), 3.5, 3000)

# ============================================================
# Density profile
# ============================================================
# This is the “shape” of the halo: it tells you how the density changes with radius.
# For example, the Hernquist profile is steep near the center and falls off at large radii.

def rho_shape(r, a0, alpha, beta, gamma):
    x = r / a0
    return x**(-gamma) * (1.0 + x**alpha)**((gamma - beta) / alpha)

# ============================================================
# Exact normalization
# ============================================================

# rho0 his is the overall scaling factor. By multiplying rho.shape wth rho0.
# we ensure that the integrated density over all space gives exactly the total halo mass Mhalo
# rho actually fixes the rho otherwise it could be any number

def compute_rho0(Mhalo, a0, alpha, beta, gamma, is_nfw):

    def integrand(r):
        rho = rho_shape(r, a0, alpha, beta, gamma)
        rho *= (1.0 - rh / r)        # horizon truncation
        return r**2 * rho

    r_max = 5.0 * a0 if is_nfw else np.inf

    integral, _ = quad(
        integrand,
        rh,
        r_max,
        epsabs=0.0,
        # (absolute tolerance) This sets the absolute error allowed in the integral.
        # If equal to zero it means the absolute error is not limiting the integration accuracy;
        # the solver relies entirely on relative error.
        epsrel=1e-9,
        # (relative tolerance) This sets the relative error allowed in the integral.
        #means the algorithm tries to make the error smaller than one part in a billion relative to the integral value.
        limit=500,
        full_output=False)[:2]

    rho0 = Mhalo / (4.0 * np.pi * integral)
    return rho0

# ============================================================
# Plot
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)

for i, Mhalo in enumerate(M_over_MBH_list):
    for j, comp in enumerate(M_over_a0_list):

        ax = axes[i, j]
        a0 = Mhalo / comp
        rc = 5.0 * a0

        for name, p in profiles.items():
            alpha = p["alpha"]
            beta  = p["beta"]
            gamma = p["gamma"]
            style = p["linestyle"]

            is_nfw = (name == "NFW")

            # exact normalization
            rho0 = compute_rho0(
                Mhalo,
                a0,
                alpha,
                beta,
                gamma,
                is_nfw
            )

            # full density
            rho = rho0 * rho_shape(r, a0, alpha, beta, gamma)
            rho *= (1.0 - rh / r)
            rho[r <= rh] = 0.0

            if is_nfw:
                rho[r > rc] = 0.0

            ax.plot(
                np.log10(r / MBH),
                np.log10(rho * MBH**2),
                style,
                label=name
            )

        ax.set_title(rf"$M/a_0 = {comp}$")
        ax.grid(True)

# Add vertical labels for M/M_BH = 1 and 100
for i, Mhalo in enumerate(M_over_MBH_list):
    axes[i, -1].text(1.06, 0.5, rf"$M/M_{{\rm BH}} = {Mhalo}$",
                     va='center', ha='right', rotation=-90,
                     transform=axes[i, -1].transAxes, fontsize=12)


# Axis labels
axes[1, 0].set_xlabel(r"$\log_{10}(r/M_{\rm BH})$")
axes[1, 1].set_xlabel(r"$\log_{10}(r/M_{\rm BH})$")
axes[0, 0].set_ylabel(r"$\log_{10}(\rho M_{\rm BH}^2)$")
axes[1, 0].set_ylabel(r"$\log_{10}(\rho M_{\rm BH}^2)$")

axes[0, 0].legend()
plt.tight_layout()
plt.savefig("Plots/HernqvsNFW.pdf", format="pdf")
plt.show()

