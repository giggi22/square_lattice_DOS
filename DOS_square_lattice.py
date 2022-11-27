import numpy as np
import functions_DOS as fD
import matplotlib.pyplot as plt
from scipy import integrate


"""defining lattice parameters"""
lattice_constant = 1  # lattice constant
energy_0 = 0  # atomic energy level
beta = 5  # hopping energy

"""defining plot parameters"""
points_energy_div_2 = 500  # number of half the energy points
interval_energy = 5  # interval over which the energy is computed, measured in beta

"""defining integration parameters"""
num_points_xi = 1000  # number of points over which the integral is computed
energy_border_parameter = 3  # number of points added to the integral tails, measured in num_points_xi

"""file name where the DOS is stored, keep it empty to not save it"""
file_name = ""

"""computing the DOS"""
E, DOS = fD.gross_dos(energy_0, beta, lattice_constant, points_energy_div_2,
                      num_points_xi, interval_energy, energy_border_parameter)

"""saving the file if file_name is not empty"""
if not file_name == "":
    np.save(file_name, [E, DOS])


"""plotting the DOS"""
fig, axs = plt.subplots(1)
axs.plot(E, DOS)
axs.set_xlabel("Energy")
axs.set_ylabel("DOS")
axs.grid()
fig.tight_layout()

"""defining goodness_parameter, the closer it is to 1, the better the DOS is"""
goodness_parameter = lattice_constant**2*integrate.trapz(DOS[~np.isnan(DOS)], E[~np.isnan(DOS)])/2

"""inserting the lattice parameters in the legend"""
axs.plot([], [], ' ', label=r"$E_0$ = {}".format(energy_0))
axs.plot([], [], ' ', label=r"$\beta$ = {}".format(beta))
axs.plot([], [], ' ', label="a = {}".format(lattice_constant))
axs.plot([], [], ' ', label=r"$\Theta$ = {:.5f}".format(goodness_parameter))
plt.legend()

plt.show()
