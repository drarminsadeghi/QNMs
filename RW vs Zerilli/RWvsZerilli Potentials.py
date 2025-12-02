import numpy as np
import matplotlib.pyplot as plt

M = 0.5

def tortoise_r_to_x(r):
    return r + 2*M*np.log(r/(2*M) - 1)

def Lambda(r, l):
    return l*(l+1) - 2 + 6*M/r

def V(r, l):
    Lam = Lambda(r, l)
    term1 = (1 - 2*M/r)
    return term1 * (
        (72*M**3)/(r**5 * Lam**2)
        - (12*M)/(r**3 * Lam**2) * (l-1)*(l+2)*(1 - 3*M/r)
        + ((l-1)*l*(l+2)*(l+1))/(r**2 * Lam)
    )

def VRW(r, l):
    return (1 - 2*M/r) * (l*(l+1)/r**2 - 6*M/r**3)

r = np.linspace(1.00001, 20, 10000)
x_over_2M = tortoise_r_to_x(r)

plt.figure(figsize=(8, 5))
plt.plot(x_over_2M, V(r, 2), label=r"$V_{\text{Z}}\,\,(\ell=2)$")
plt.plot(x_over_2M, V(r, 3), label=r"$V_{\text{Z}}\,\,(\ell=3)$")
plt.plot(x_over_2M, VRW(r, 2), linestyle="--", label=r"$V_{\text{RW}}\,\,(\ell=2)$")
plt.plot(x_over_2M, VRW(r, 3), linestyle="--", label=r"$V_{\text{RW}}\,\,(\ell=3)$")

plt.xlabel("x")
plt.ylabel("Potential")
plt.legend()
plt.grid(True)
plt.savefig("Plots/RWvsZPot.eps", format="eps")
plt.show()
