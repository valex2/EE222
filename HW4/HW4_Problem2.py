import numpy as np
import matplotlib.pyplot as plt
from scipy.special import airy
from scipy.optimize import brentq
from tqdm import tqdm

# Constants (scaled down to avoid numerical issues)
m = 9.11e-31  # Electron mass (kg)
m = m * 0.041  # InGaAs effective mass (scaled down)
hbar = 1.054571817e-34  # Reduced Planck's constant (J.s)
e = 1.602176634e-19  # Elementary charge (C)
epsilon = 10e6  # Electric field (V/m)
Lz = 12e-9  # Length in z-direction (m)
E1_inf = (hbar**2 / (2 * m)) * ((np.pi**2) / (Lz**2))  # Baseline energy

# Function to calculate the zeta values
def zeta(E, Lz, m, e, epsilon, hbar):
    zeta_left = (((2 * m * e * epsilon) / (hbar**2)) ** (1 / 3)) * (-(Lz / 2) - (E / (e * epsilon)))
    zeta_right = (((2 * m * e * epsilon) / (hbar**2)) ** (1 / 3)) * ((Lz / 2) - (E / (e * epsilon)))
    return zeta_left, zeta_right

def boundary_condition(E):
    zeta_l, zeta_r = zeta(E, Lz, m, e, epsilon, hbar)
    Ai_l, _, Bi_l, _ = airy(zeta_l)
    Ai_r, _, Bi_r, _ = airy(zeta_r)

    print(f"zeta_l: {zeta_l}, zeta_r: {zeta_r}")
    print(f"Ai_l: {Ai_l}, Ai_r: {Ai_r}, Bi_l: {Bi_l}, Bi_r: {Bi_r}")

    return (Ai_l * Bi_r) - (Ai_r * Bi_l)

def is_distinct(new_solution, solutions, threshold=1e-21):
    return all(abs(new_solution - s) > threshold for s in solutions)

def find_multiple_solutions(bounds):
    solutions = []
    for i in tqdm(range(len(bounds) - 1)):
        try:
            sol = brentq(boundary_condition, bounds[i], bounds[i+1])
            # Add the solution if it is not too close to previous ones
            if is_distinct(sol, solutions):
                solutions.append(sol)
        except ValueError:
            # If no root found in this interval, continue
            pass
    return solutions

# Create bounds for E where we expect solutions, picking the EV range
E_min = 1e-21
E_max = 5e-17
bounds = np.linspace(E_min, E_max, 10000)  # 1000 intervals between E_min and E_max

energies = find_multiple_solutions(bounds)

def compute_coefficients(E):
    zeta_l, zeta_r = zeta(E, Lz, m, e, epsilon, hbar)
    Ai_l, _, Bi_l, _ = airy(zeta_l)
    Ai_r, _, Bi_r, _ = airy(zeta_r)

    b_div_a = - (Ai_l) / (Bi_l)
    return b_div_a

for i, E in enumerate(energies):
    print(f"Solution {i+1}: E = {E:.5e} J, or {E*6.242e18:.5f} eV, normalized E = {E / E1_inf}")

    print(f"Coefficients for E = {E:.5e} J: b/a = {compute_coefficients(E)}")

############################################################################################################################################################################
# prep for plotting
first_three_energies = energies[:3]
first_three_energies = np.array(first_three_energies)

def zeta_n(E_n, z):
    zeta_n = (((2 * m * e * epsilon) / (hbar**2)) ** (1 / 3)) * (z - (E_n / (e * epsilon)))
    return zeta_n

def psi_n(z, E_n, b_div_a):
    zeta_n_val = zeta_n(E_n, z)

    # let a = 1
    a = 1
    b = b_div_a * a

    Ai, _, Bi, _ = airy(zeta_n_val)
    psi_n = a * Ai + b * Bi
    return psi_n

def potential_eV(z):
    out = []
    for z_val in z:
        if z_val < -Lz / 2 or z_val > Lz / 2:
            out.append(0)
        else:
            out.append(epsilon * z_val)
    return np.array(out)

def normalize_psi(psi, z_values):
    norm_squared = np.trapz(np.abs(psi)**2, z_values)
    print(f"Normalization factor: {norm_squared}")
    return psi / np.sqrt(norm_squared), norm_squared # Normalize by dividing the wavefunction by the square root of the norm

# Create z values for plotting
z_values = np.linspace(-(Lz + 2e-9) / 2, (Lz + 2e-9) / 2, 1000)
############################################################################################################################################################################

############################################################################################################################################################################
# Plot the wavefunctions
plt.figure(figsize=(10, 6))
for idx, E in enumerate(first_three_energies):
    b_div_a = compute_coefficients(E)
    plt.plot(z_values*1e9, psi_n(z_values, E, b_div_a) + (E)*6.242e18, label=f'E{idx+1} = {E*6.242e18:.5f} eV')


plt.title('Wavefunctions for the first three energy levels')
plt.xlabel('z (nm)')
plt.ylabel('Energy (eV)')
plt.axhline(0, color='black', lw=0.5, ls='--')
plt.legend()
plt.grid()
############################################################################################################################################################################

############################################################################################################################################################################
# Plotting the normalized probability densities
plt.figure(figsize=(10, 6))
for idx, E in enumerate(first_three_energies):
    b_div_a = compute_coefficients(E)
    z_values = np.linspace(-(Lz) / 2, (Lz) / 2, 1000)
    psi = psi_n(z_values, E, b_div_a)
    normalized_psi, norm_squared = normalize_psi(psi, z_values)
    plt.plot(z_values*1e9, (np.abs(normalized_psi/np.sqrt(1e9))**2), label=f'|ψ{idx+1}(z)|^2')

plt.title('Normalized Wavefunctions and Probability Densities')
plt.xlabel('z (nm)')
plt.ylabel('Probability')
plt.axhline(0, color='black', lw=0.5, ls='--')
plt.legend()
plt.grid()
############################################################################################################################################################################

plt.show(block=False)
input("Press Enter to continue...")
plt.close()






