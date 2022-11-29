import numpy as np
import functions_DOS as fD
import matplotlib.pyplot as plt

"""defining lattice parameters"""
lattice_constant = 1  # lattice constant
energy_0 = 0  # atomic energy level
beta = 5  # hopping energy

"""defining parameters of the integrand function"""
num_points_xi = 40  # number of points of the integral function in the "overall" region
energy_border_parameter = 1  # number of points added to the integral tails, measured in num_points_xi

"""defining energy vector that will be used for plotting different curves"""
energy = np.linspace(-2, 2, 5)

"""plotting the integrand function for the different values of energy"""
fig, axs = plt.subplots(1)
for E in energy:
    k, f = fD.integrand_function(E, energy_0, beta, num_points_xi, energy_border_parameter)
    axs.plot(k, f, ".", label="Energy = {}".format(E))
axs.set_xlabel(r"$\xi$")
axs.set_ylabel("Integrand function")
plt.legend()
axs.grid()
fig.tight_layout()

plt.show()
