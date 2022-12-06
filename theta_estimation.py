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
interval_energy = 4  # interval over which the energy is computed, measured in beta

"""defining integration parameters"""
num_points_xi = 500  # number of points over which the integral is computed
energy_border_parameter = np.linspace(0, 4, 5, dtype=int)  # points added to integral tails, measured in num_points_xi

"""computing the DOSs with varying border parameter, the DOS are 'cleaned'"""
E = []
DOS = []
for bor_par in energy_border_parameter:
    E_buff, DOS_buff = fD.gross_dos(energy_0, beta, lattice_constant, points_energy_div_2,
                                    num_points_xi, interval_energy, bor_par)
    E_buff, DOS_buff = E_buff[~np.isnan(DOS_buff)], DOS_buff[~np.isnan(DOS_buff)]
    E.append(E_buff)
    DOS.append(DOS_buff)

"""computing the goodness parameters"""
goodness_parameters = []
for i in range(len(E)):
    const = lattice_constant**2/2
    goodness_parameters.append(const*integrate.trapz(DOS[i][~np.isnan(DOS[i])], E[i][~np.isnan(DOS[i])]))

"""plotting the DOS"""
fig, axs = plt.subplots(1)
for i in range(len(E)):
    axs.semilogy(E[i][::10], DOS[i][::10], ".", linewidth=2.5, label=r"$\Theta$ = {:.5f}".format(goodness_parameters[i]))
axs.set_xlabel("Energy")
axs.set_ylabel("DOS")
axs.grid()
fig.tight_layout()

"""inserting the lattice parameters in the legend"""
axs.plot([], [], ' ', label=r"$E_0$ = {}".format(energy_0))
axs.plot([], [], ' ', label=r"$\beta$ = {}".format(beta))
axs.plot([], [], ' ', label="a = {}".format(lattice_constant))
plt.legend()

plt.show()
