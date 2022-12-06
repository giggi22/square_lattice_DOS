import numpy as np
import functions_DOS as fD
import matplotlib.pyplot as plt


"""defining lattice parameters"""
lattice_constant = 1  # lattice constant
energy_0 = 0  # atomic energy level

"""defining the different values of beta"""
beta = np.linspace(1, 5, 9)  # hopping energy, from 1 to 10 with step of 0.5

"""defining plot parameters"""
points_energy_div_2 = 500  # number of half the energy points
interval_energy = 5  # interval over which the energy is computed, measured in beta

"""defining integration parameters"""
num_points_xi = 1000  # number of points over which the integral is computed
energy_border_parameter = 3  # number of points added to the integral tails, measured in num_points_xi

"""file name where the DOS is stored, keep it empty to not save it"""
file_name = ""

"""computing the DOS"""
E, DOS = [], []
for idx, val in enumerate(beta):
    E_buff, DOS_buff = fD.gross_dos(energy_0, val, lattice_constant, points_energy_div_2,
                                    num_points_xi, interval_energy, energy_border_parameter)
    E.append(E_buff)
    DOS.append(DOS_buff)

"""saving the file if file_name is not empty"""
if not file_name == "":
    np.save(file_name, [E, DOS])

"""plotting the DOS"""
fig, axs = plt.subplots(1)
for idx, val in enumerate(beta):
    axs.plot(E[idx], DOS[idx], linewidth=2, label=r"$\beta$ = {}".format(val))
axs.set_xlabel("Energy")
axs.set_ylabel("DOS")
axs.grid()
fig.tight_layout()

"""inserting the lattice parameters in the legend"""
axs.plot([], [], ' ', label=r"$E_0$ = {}".format(energy_0))
axs.plot([], [], ' ', label="a = {}".format(lattice_constant))
plt.legend()

"""extracting the values at the band edges"""
band_edge_values = np.zeros_like(beta)
for idx, val in enumerate(beta):
    index = next((i for i, x in enumerate(DOS[idx]) if x > 0 and not np.isnan(x)), None)
    band_edge_values[idx] = np.average(DOS[idx][int(index)])

"""plotting the DOS edge value as function of beta"""
fig, axs = plt.subplots(1)
axs.plot(beta, beta * band_edge_values, ".", linewidth=2)
axs.set_xlabel(r"$\beta$")
axs.set_ylabel(r"DOS at band edge * $\beta$ * $a^2$")
axs.grid()
fig.tight_layout()

plt.show()
