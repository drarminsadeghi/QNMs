import qnm

# Download precomputed data (run once)
qnm.download_data()


# Example for gravitational perturbation (s=-2), l=2, m=2, n=0 (fundamental mode)
grav_220 = qnm.modes_cache(s=-2, l=2, m=2, n=0)

# For Schwarzschild, a=0 (dimensionless spin parameter)
omega, A, C = grav_220(a=0.0)

print("Quasinormal frequency (in units of 1/M):", omega)