import numpy as np
from scipy import integrate
from tqdm import tqdm


def integral_lower_limit(
        energy: float,
        energy_0: float,
        beta: float
) -> float:
    """
    Parameters
    ----------
    energy (float): current energy
    energy_0 (float): energy of isolated atom
    beta (float): hopping energy

    Returns
    -------
    (float) lower limit for the DOS integral

    Notes
    -----
    This functions returns the lower limit for the DOS integral. In case of not valid energy, it would
    return 0 which correspond to the lower border of the first quarter of the FBZ.
    """

    if not energy_0 <= energy <= energy_0 + 4 * beta:
        return 0
    else:
        return np.arccos((energy_0 - energy) / (2 * beta) + 1)


def integral_upper_limit(
        energy: float,
        energy_0: float,
        beta: float
) -> float:
    """
    Parameters
    ----------
    energy (float): current energy
    energy_0 (float): energy of isolated atom
    beta (float): hopping energy

    Returns
    -------
    (float) lower limit for the DOS integral

    Notes
    -----
    This functions returns the upper limit for the DOS integral. In case of not valid energy, it would
    return pi which correspond to the upper border of the first quarter of the FBZ.
    """

    if not energy_0 - 4 * beta <= energy <= energy_0:
        return np.pi
    else:
        return np.arccos((energy_0 - energy) / (2 * beta) - 1)


def integrand_function(
        energy: float,
        energy_0: float,
        beta: float,
        num_points: int,
        bord_param: int = 0
) -> (np.ndarray, np.ndarray):

    """
    Parameters
    ----------
    energy (float): current energy
    energy_0 (float): energy of isolated atom
    beta (float): hopping energy
    num_points (int): array length of the outputs
    bord_param (int): defines the amount of points at the borders of xi-array

    Returns
    -------
    xi_array (np.ndarray): array containing the xi values for the specific value of energy
    y_array (np.ndarray): array containing the y values for the specific value of energy

    Notes
    -----
    This function returns two arrays, which correspond to the value of xi
    and the corresponding ones of the integrand function (y).
    """

    "defining the integration limits"
    lower_limit = integral_lower_limit(energy, energy_0, beta)
    upper_limit = integral_upper_limit(energy, energy_0, beta)

    "creating the output arrays"
    xi_array = np.linspace(lower_limit, upper_limit, num_points)
    step = upper_limit - lower_limit
    # xi_array is denser at the border, where singularities occur, based on bord_param
    for i in range(bord_param):
        xi_array = np.concatenate((
            np.linspace(lower_limit, lower_limit + step / num_points ** (i + 1), num_points),
            xi_array[1:-1],
            np.linspace(upper_limit - step / num_points ** (i + 1), upper_limit, num_points)
        ))
    y_array = np.zeros_like(xi_array)

    "computing the y-values for appropriate values of energy"
    if np.abs((energy_0 - energy) / (4 * beta)) <= 1:
        for idx, val in enumerate(xi_array):
            differ = ((energy_0 - energy) / (2 * beta) - np.cos(val)) ** 2
            "in order to avoid a divergence in the integrand function, the"
            "next value would be equal to the previous one"
            if differ < 1:
                y_array[idx] = 1 / np.sqrt(1 - differ)
            else:
                y_array[idx] = y_array[idx - 1]
    return xi_array, y_array


def gross_dos(
        energy_0: float,
        beta: float,
        lattice_constant: float,
        num_points_energy: int,
        num_points_xi: int,
        interval_energy: float = 4.2,
        bord_param: int = 0
) -> (np.ndarray, np.ndarray):

    """
    Parameters
    ----------
    energy_0 (float): energy of isolated atom
    beta (float): hopping energy
    lattice_constant (float): lattice constant length
    num_points_energy (int): array length of the outputs (energy and DOS)
    num_points_xi (int): array length of the xi-array for the integration step
    interval_energy (float): interval of energy respect to E0 measured in beta, suggested value is slightly above 4
    bord_param (int): defines the amount of points at the borders of xi-array

    Returns
    -------
    energy_array (np.ndarray): array containing the energy values
    dos_array (np.ndarray): dos array

    Notes
    -----
    This function returns two arrays, which correspond to the value of the energy
    and the corresponding ones of the density of states.
    """

    "creating energy and dos arrays; being energy_0 a singularity point, the two arrays are denser in its proximity."
    num_points = int((num_points_energy+2)/2)
    step = interval_energy * beta
    energy_array = np.concatenate((
        np.linspace(energy_0 - step, energy_0, num_points)[:-1],
        np.linspace(energy_0 - step / num_points, energy_0, num_points)[:-1],
        np.linspace(energy_0, energy_0 + step / num_points, num_points)[1:],
        np.linspace(energy_0, energy_0 + step, num_points)[1:]
    ))
    dos_array = np.zeros_like(energy_array)

    "tqdm allows to have the progress bar"
    for idx in tqdm(range(len(energy_array))):
        energy_val = energy_array[idx]
        x_data, y_data = integrand_function(energy_val, energy_0, beta, num_points_xi, bord_param)
        dos_array[idx] = 1 / (lattice_constant ** 2 * beta * np.pi ** 2) * integrate.simps(y_data, x_data)

    return energy_array, dos_array


def low_dos(
        energy: np.ndarray,
        dos: np.ndarray
) -> (np.ndarray, np.ndarray):

    """
    Parameters
    ----------
    energy (np.ndarray): energy array
    dos (np.ndarray): density of states array

    Returns
    -------
    energy_low (np.ndarray): array containing the new energy values
    dos_low (np.ndarray): dos array containing the dos values below the average value

    Notes
    -----
    This function returns two arrays, which correspond to the value of the energy
    and the corresponding ones of the density of states with the values below the average.
    """

    dos_low = dos[[n for n, i in enumerate(dos) if i < np.average(dos)]]
    energy_low = energy[[n for n, i in enumerate(dos) if i < np.average(dos)]]
    return energy_low, dos_low


def high_dos(
        energy: np.ndarray,
        dos: np.ndarray
) -> (np.ndarray, np.ndarray):

    """
    Parameters
    ----------
    energy (np.ndarray): energy array
    dos (np.ndarray): density of states array

    Returns
    -------
    energy_low (np.ndarray): array containing the new energy values
    dos_low (np.ndarray): dos array containing the dos values above the average value

    Notes
    -----
    This function returns two arrays, which correspond to the value of the energy
    and the corresponding ones of the density of states with the values above the average.
    """

    dos_high = dos[[n for n, i in enumerate(dos) if i > np.average(dos)]]
    energy_high = energy[[n for n, i in enumerate(dos) if i > np.average(dos)]]
    return energy_high, dos_high
