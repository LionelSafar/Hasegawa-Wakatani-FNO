import numpy as np
import jax
import jax.numpy as jnp

from typing import Tuple, Dict

# This file contains functions to compute physical quantities from the HW2D system.

def transpose_fields(
        tensor: jnp.ndarray, 
        inc_vorticity: bool = False
) -> Dict:
    """
    Transposes prediction and gt to match the shape required for the physical quantities functions 
    and returns the transposed fields in a dictionary.
    Transposes (x, y, t, channels) to {channel: (t, x, y) for channel in channels}.
    
    Args:
        tensor: (jnp.ndarray): shape (x, y, t, 2) or (x, y, t, 3)
        inc_vorticity (bool): whether the input tensor includes vorticity

    Returns:
        transposed (Dict): transposed fields

    """
    if inc_vorticity:
        transposed = {
            'phi': jnp.transpose(tensor[:, :, :, 0], (2, 0, 1)), 
            'n': jnp.transpose(tensor[:, :, :, 1], (2, 0, 1)),
            'omega': jnp.transpose(tensor[:, :, :, 2], (2, 0, 1))
            }
    else:
        transposed = {
            'phi': jnp.transpose(tensor[:, :, :, 0], (2, 0, 1)),
            'n': jnp.transpose(tensor[:, :, :, 1], (2, 0, 1))
            }
    return transposed


def get_physics(
        pred: jnp.ndarray, 
        y: jnp.ndarray, 
        dt:float, 
        dx: float, 
        c1:float, 
        nu: float, 
        kappa: float, 
        hyperdiff: int = 3,
        inc_vorticity: bool = True, 
        starttime: int = 0
) -> Tuple[Dict, Dict]:
    """
    This function takes the prediction and ground truth fields and computes the physical quantities
    and returns them in a dictionary.

    Args:
        pred (jnp.ndarray): shape (x, y, t, 2) or (x, y, t, 3), predicted field time-series
        y (jnp.ndarray): shape (x, y, t, 2) or (x, y, t, 3), ground truth field time-series
        dt (float): time increment between two consecutive frames in physical units c_s/ L_n
        dx (float): grid spacing
        c1 (float): adiabaticit parameter
        nu (float): viscosity coefficient
        kappa (float): background gradient drive
        hyperdiff (int): order of hyperdiffusion
        inc_vorticity (bool): whether the input tensor includes vorticity field or not
        starttime (int): truncates the time axis to start from this index

    Returns:
        gt_stats (Dict): ground truth physical quantities
        pred_stats (Dict): predicted physical quantities

    Each dictionary contains the following keys:
        time (jnp.ndarray): time axis
        energy (jnp.ndarray): energy time-series
        enstrophy (jnp.ndarray): enstrophy time-series
        gamma_n (jnp.ndarray): particle flux time-series
        gamma_c (jnp.ndarray): sink time-series
        DE (jnp.ndarray): viscous energy dissipation
        phi (jnp.ndarray): mean potential field
        n (jnp.ndarray): mean density field

        if inc_vorticity is True:
            omega (jnp.ndarray): mean vorticity field
            enstrophy_omega (jnp.ndarray): enstrophy from vorticity field
            DU (jnp.ndarray): viscous vorticity dissipation
    
    """
    # Transpose fields to correct format
    y = transpose_fields(y, inc_vorticity)
    pred = transpose_fields(pred, inc_vorticity)

    # Get all the physical quantities
    energy_pred = get_energy(pred['n'], pred['phi'], dx)
    enstrophy_pred = get_enstrophy_phi(pred['n'], pred['phi'], dx)
    if inc_vorticity:
        enstrophy_omega_pred = get_enstrophy(pred['n'], pred['omega'])
    
    energy_gt = get_energy(y['n'], y['phi'], dx)
    enstrophy_gt = get_enstrophy_phi(y['n'], y['phi'], dx)
    if inc_vorticity:
        enstrophy_omega_gt = get_enstrophy(y['n'], y['omega'])

    gamma_n_gt = get_gamma_n(y['n'], y['phi'], dx=dx, kappa=kappa)
    gamma_c_gt = get_gamma_c(y['n'], y['phi'], c1=c1)
    gamma_n_pred = get_gamma_n(pred['n'], pred['phi'], dx=dx, kappa=kappa)
    gamma_c_pred = get_gamma_c(pred['n'], pred['phi'], c1=c1)

    Dphi_pred = get_D(pred['phi'], nu=nu, N=hyperdiff, dx=dx)
    Dn_pred = get_D(pred['n'], nu=nu, N=hyperdiff, dx=dx)
    DE_pred = get_DE(pred['n'], pred['phi'], Dn_pred, Dphi_pred)

    Dphi_gt = get_D(y['phi'], nu=nu, N=hyperdiff, dx=dx)
    Dn_gt = get_D(y['n'], nu=nu, N=hyperdiff, dx=dx)
    DE_gt = get_DE(y['n'], y['phi'], Dn_gt, Dphi_gt)

    if inc_vorticity:
        DU_pred = get_DU(pred['n'], pred['omega'], Dn_pred, Dphi_pred)
        DU_gt = get_DU(y['n'], y['omega'], Dn_gt, Dphi_gt)

    N = pred['phi'].shape[0]
    time = jnp.linspace(starttime, starttime + dt*(N-1), N)

    phi_mean_gt = jnp.mean(y['phi'], axis=(-1, -2))
    phi_mean_pred = jnp.mean(pred['phi'], axis=(-1, -2))
    n_mean_gt = jnp.mean(y['n'], axis=(-1, -2))
    n_mean_pred = jnp.mean(pred['n'], axis=(-1, -2))

    # Create the dictionaries
    pred_stats = {
        'time': time,
        'energy': energy_pred,
        'enstrophy': enstrophy_pred,
        'gamma_n': gamma_n_pred,
        'gamma_c': gamma_c_pred,
        'DE': DE_pred,
        'phi': phi_mean_pred,
        'n': n_mean_pred,
    }
    gt_stats = {
        'time': time,
        'energy': energy_gt,
        'enstrophy': enstrophy_gt,
        'gamma_n': gamma_n_gt,
        'gamma_c': gamma_c_gt,
        'DE': DE_gt,
        'phi': phi_mean_gt,
        'n': n_mean_gt,
    }
    # Add vorticity related quantities if vorticity is included
    if inc_vorticity:
        omega_mean_gt = jnp.mean(y['omega'], axis=(-1, -2))
        omega_mean_pred = jnp.mean(pred['omega'], axis=(-1, -2))
        gt_stats['omega'] = omega_mean_gt
        pred_stats['omega'] = omega_mean_pred
        gt_stats['enstrophy_omega'] = enstrophy_omega_gt 
        pred_stats['enstrophy_omega'] = enstrophy_omega_pred
        gt_stats['DU'] = DU_gt
        pred_stats['DU'] = DU_pred

    return gt_stats, pred_stats

# Gradient functions

def periodic_gradient(
        input_field: jnp.ndarray, 
        dx: float, 
        axis: int = 0
) -> jnp.ndarray:
    """
    Compute the gradient of a 2D array using finite differences with periodic boundary conditions.

    Args:
        input_field (np.ndarray): Input 2D array.
        dx (float): The spacing between grid points.
        axis (int): Axis along which the gradient is tkaen

    Returns:
        tuple: Gradient in y-direction, gradient in x-direction with periodic boundary conditions.

    """
    if axis < 0:
        pad_size = [(0, 0) for _ in range(len(input_field.shape))]
        pad_size[-1] = (1, 1)
        pad_size[-2] = (1, 1)
    else:
        pad_size = 1
    padded = jnp.pad(input_field, pad_width=pad_size, mode="wrap")

    return gradient(padded, dx, axis=axis)


def gradient(
        padded: jnp.ndarray, 
        dx: float, 
        axis: int = 0
) -> jnp.ndarray:
    """
    Compute the gradient of a 2D array using 2nd order central finite differences.

    Args:
        padded (np.ndarray): 2D array with padding of size 1.
        dx (float): The spacing between grid points.
        axis (int): Axis along which the gradient is tkaen

    Returns:
        jnp.ndarray: Gradient in axis-direction.

    """
    if axis == 0:
        return (padded[2:, 1:-1] - padded[0:-2, 1:-1]) / (2 * dx)
    elif axis == 1:
        return (padded[1:-1, 2:] - padded[1:-1, 0:-2]) / (2 * dx)
    elif axis == -2:
        return (padded[..., 2:, 1:-1] - padded[..., 0:-2, 1:-1]) / (2 * dx)
    elif axis == -1:
        return (padded[..., 1:-1, 2:] - padded[..., 1:-1, 0:-2]) / (2 * dx)


def laplace(padded: jnp.ndarray, dx: float) -> jnp.ndarray:
    """
    Compute the Laplace of a 2D array using 2nd order central finite differences.

    Args:
        padded (np.ndarray): 2D array with padding of size 1.
        dx (float): The spacing between grid points.

    Returns:
        np.ndarray: The Laplace of the input array.

    """
    return (
        padded[..., 0:-2, 1:-1]  # above
        + padded[..., 1:-1, 0:-2]  # left
        - 4 * padded[..., 1:-1, 1:-1]  # center
        + padded[..., 1:-1, 2:]  # right
        + padded[..., 2:, 1:-1]  # below
    ) / dx**2


def periodic_laplace(arr: jnp.ndarray, dx: float) -> jnp.ndarray:
    """
    Compute the Laplace of a 2D array using 2nd order central finite differences 
    with periodic boundary conditions.

    Args:
        a (np.ndarray): Input 2D array.
        dx (float): The spacing between grid points.

    Returns:
        jnp.ndarray: The Laplace of the input array with periodic boundary conditions.

    """
    pad_size = 1
    if len(arr.shape) > 2:
        pad_size = [(0, 0) for _ in range(len(arr.shape))]
        pad_size[-1] = (1, 1)
        pad_size[-2] = (1, 1)

    return jnp.array(laplace(jnp.pad(arr, pad_size, "wrap"), dx))


def periodic_laplace_N(array: jnp.ndarray, dx: float, N: int) -> jnp.ndarray:
    """
    Compute the Laplace of a 2D array using 2nd order central finite differences N times 
    successively with periodic boundary conditions.

    Args:
        array (np.ndarray): Input 2D array.
        dx (float): The spacing between grid points.
        N (int): Number of iterations.

    Returns:
        np.ndarray: The Laplace of the input array with periodic boundary conditions.

    """
    for _ in range(N):
        array = periodic_laplace(array, dx)

    return array

# Flux quantities
    
def get_gamma_n(n: jnp.ndarray, p: jnp.ndarray, dx: float, kappa: float, dy_p=None) -> jnp.ndarray:
    """
    Compute the average particle flux $(\\Gamma_n)$ using the formula:
    $$
        \\Gamma_n = - \\int{\\mathrm{d^2} x \; \\tilde{n} \\frac{\partial \\tilde{\\phi}}{\\partial y}}
    $$

    Args:
        n (np.ndarray): Density (or similar field).
        p (np.ndarray): Potential (or similar field).
        dx (float): Grid spacing.
        dy_p (np.ndarray, optional): Gradient of potential in the y-direction.
            Computed from `p` if not provided.

    Returns:
        float: Computed average particle flux value.

    """
    if dy_p is None:
        dy_p = periodic_gradient(p, dx=dx, axis=-2)  # gradient in y
    gamma_n = -kappa * jnp.mean((n * dy_p), axis=(-1, -2))  # mean over y & x

    return gamma_n


def get_gamma_c(n: jnp.ndarray, p: jnp.ndarray, c1: float) -> float:
    """
    Compute the sink $\\Gamma_c$ using the formula:
    $$
        \\Gamma_c = c_1 \\int{\\mathrm{d^2} x \; (\\tilde{n} - \\tilde{\\phi})^2}
    $$

    Args:
        n (np.ndarray): Density (or similar field).
        p (np.ndarray): Potential (or similar field).
        c1 (float): Proportionality constant.
        dx (float): Grid spacing.

    Returns:
        float: Computed particle flux value.

    """
    gamma_c = c1 * jnp.mean(jnp.square((n - p)), axis=(-1, -2))

    return gamma_c

# Energy and enstrophy

def get_energy(n: jnp.ndarray, phi: jnp.ndarray, dx: float) -> jnp.ndarray:
    """Energy of the HW2D system, sum of thermal and kinetic energy
    $$
        E = \\frac{1}{2} \\int{
            \\mathrm {d^2} x \;
            \\left(n^2 + | \\nabla \\phi |^2 \\right)
        }
    $$

    Args:
        n (jnp.ndarray): Density field shape (t, x, y)
        phi (jnp.ndarray): Potential field (t, x, y)
        dx (float): Grid spacing

    Returns:
        jnp.ndarray: Energy of the HW2D system

    """
    squared_norm_grad_phi = jnp.square(periodic_gradient(phi, dx=dx, axis=-1)) + periodic_gradient(
        phi, dx=dx, axis=-2
    )**2
    # Integrate, then divide by 2
    integral = jnp.mean((jnp.square(n)) + squared_norm_grad_phi, axis=(-1, -2))

    return integral / 2


def get_enstrophy(n: jnp.ndarray, omega: jnp.ndarray) -> jnp.ndarray:
    """Enstrophy of the HW2D system
    $$
        \\mathbf{U = \\frac{1}{2} \\int{\\mathrm{d^2} x \; (n - \\Omega)^2}}
                   = \\frac{1}{2} \\int{\\mathrm{d^2} x \; (n^2 - \\nabla^2 \\phi)^2}
    $$

    Args:
        n (jnp.ndarray): Density field shape (t, x, y)
        omega (jnp.ndarray): Vorticity field (t, x, y)
        dx (float): Grid spacing

    Returns:
        jnp.ndarray: Enstrophy of the HW2D system

    """
    integral = jnp.mean((jnp.square(n - omega)), axis=(-1, -2))

    return integral / 2


def get_enstrophy_phi(n: jnp.ndarray, phi: jnp.ndarray, dx: float) -> jnp.ndarray:
    """Enstrophy of the HW2D system from phi
    $$
        \\mathbf{U = \\frac{1}{2} \\int{\\mathrm{d^2} x \; (n^2 - \\nabla^2 \\phi)^2}}
                   = \\frac{1}{2} \\int{\\mathrm{d^2} x \; (n - \\Omega)^2}
    $$

    Args:
        n (jnp.ndarray): Density field shape (t, x, y)
        phi (jnp.ndarray): Potential field (t, x, y)
        dx (float): Grid spacing

    Returns:
        jnp.ndarray: Enstrophy of the HW2D system

    """
    omega = periodic_laplace_N(phi, dx, N=1)
    omega -= jnp.mean(omega, axis=(-1, -2), keepdims=True)
    integral = jnp.mean(((n - omega) ** 2), axis=(-1, -2))

    return integral / 2

# Dissipation terms

def get_D(array: jnp.ndarray, nu: float, N: int, dx: float) -> jnp.ndarray:
    """Calculate the hyperdiffusion coefficient
    $$
        \\nu \; \\nabla F
    $$

    Args:
        arr (np.ndarray): Field to work on
        nu (float): hyperdiffusion coefficient
        N (int): order of hyperdiffusion
        dx (float): grid spacing

    Returns:
        np.ndarray: Hyperdiffusion term of the field

    """
    return nu * periodic_laplace_N(array, dx=dx, N=N)


def get_DE(n: jnp.ndarray, p: jnp.ndarray, Dn: jnp.ndarray, Dp: jnp.ndarray) -> jnp.ndarray:
    """
    $$
        DE = \\int{\\mathrm{d^2} x \; n \; D_n - \\phi \; D_p}
    $$

    Args:
        n (np.ndarray): density field $n$
        p (np.ndarray): potential field $\\phi$
        Dn (np.ndarray): hyperdiffusion of the density field $n$
        Dp (np.ndarray): hyperdiffusion of the potential field $\\phi$

    Returns:
        jnp.ndarray: Value of DE

    """
    DE = np.mean(n * Dn - p * Dp, axis=(-1, -2))

    return DE


def get_DU(n: jnp.ndarray, o: jnp.ndarray, Dn: jnp.ndarray, Dp: jnp.ndarray) -> jnp.ndarray:
    """
    $$
        DU = \\int{\\mathrm{d^2} x \; (n - \Omega)  (D_n - D_\\phi)}
    $$

    Args:
        n (np.ndarray): density field $n$
        o (np.ndarray): potential field $\\phi$
        Dn (np.ndarray): hyperdiffusion of the density field $n$
        Dp (np.ndarray): hyperdiffusion of the potential field $\\phi$

    Returns:
        jnp.ndarray: Value of D()

    """
    DU = -jnp.mean((n - o) * (Dn - Dp), axis=(-1, -2))
    return DU

# Additional testing functions

def HW_residue(
        n: jnp.ndarray, 
        phi: jnp.ndarray, 
        dx: float, 
        dt: float, 
        c1: float, 
        nu: float,
        hyperdiff: int, 
        kappa:float
) -> float:
    """
    $$
        d_n = c_1 (n - \\phi) - \\nabla^2 n - \\kappa \\partial_y \\phi 
            - \\nu \\nabla^N n

        d_\\omega = c_1 (n - \\phi) - \\nabla^2 \\omega - \\nu \\nabla^N \\omega 
    $$

    Calculate the residue of the HW2D system, using
     - 2nd order CFD for time and spatial derivatives
     - Arakawa scheme for the Poisson bracket

    Args:
        n (np.ndarray): Density field
        phi (np.ndarray): Potential field
        dx (float): Grid spacing
        dt (float): Time increment between two consecutive frames
        c1 (float): Adiabaticity parameter
        nu (float): Viscosity coefficient
        hyperdiff (int): Order of hyperdiffusion
        kappa (float): Background gradient drive

    Returns:
        float: Residue of the HW2D system

    NOTE: This function was used for testing, residue calculation does not work well due to 
    the large time step of the snapshots the time derivative is very inaccurate generally.

    """
    # Use vmap to vectorize the periodic laplace function over time axis
    omega = jax.vmap(periodic_laplace, in_axes=(0, None))(phi, dx)

    poisson_phi = periodic_arakawa_vec(omega[2], phi[2], dx)
    poisson_n = periodic_arakawa_vec(n[2], phi[2], dx)

    hyperdiff_n = get_D(n[2], nu, hyperdiff, dx)
    hyperdiff_phi = get_D(omega[2], nu, hyperdiff, dx)

    d_yphi = kappa*periodic_gradient(phi[2], dx=dx, axis=-2)
    c1_term = c1 * (phi[2] - n[2])

    d_n = c1_term - poisson_n - d_yphi - hyperdiff_n
    d_omega = c1_term - poisson_phi - hyperdiff_phi

    # Time derivatives from 2nd order CFD
    dndt = (n[3]-n[1])/(2*dt)
    domegadt = (omega[3]-omega[1])/(2*dt)

    # Residue
    residue_n = jnp.abs(dndt - d_n)
    residue_omega = jnp.abs(domegadt - d_omega)
    residue = residue_n + residue_omega

    return jnp.mean(residue)


def arakawa_vec(zeta: jnp.ndarray, psi: jnp.ndarray, dx: float) -> jnp.ndarray:
    """
    Compute the Poisson bracket (Jacobian) of vorticity and stream function
    using a vectorized version of the Arakawa scheme. This function is designed
    for a 2D periodic domain and requires a 1-cell padded input on each border.

    Args:
        zeta (jnp.ndarray): Vorticity field with padding.
        psi (jnp.ndarray): Stream function field with padding.
        dx (float): Grid spacing.

    Returns:
        jnp.ndarray: Discretized Poisson bracket (Jacobian) over the grid.

    """
    return (
        zeta[..., 1:-1, 2:] * (psi[..., 2:,1:-1] - psi[..., 0:-2, 1:-1] + psi[..., 2:, 2:] - psi[..., 0:-2, 2:])
        - zeta[..., 1:-1,0:-2]
        * (psi[..., 2:, 1:-1] - psi[..., 0:-2, 1:-1] + psi[..., 2:, 0:-2] - psi[..., 0:-2, 0:-2])
        - zeta[..., 2:, 1:-1]
        * (psi[..., 1:-1, 2:] - psi[..., 1:-1, 0:-2] + psi[..., 2:, 2:] - psi[..., 2:, 0:-2])
        + zeta[..., 0:-2, 1:-1]
        * (psi[..., 1:-1, 2:] - psi[..., 1:-1, 0:-2] + psi[..., 0:-2, 2:] - psi[..., 0:-2, 0:-2])
        + zeta[..., 0:-2, 2:] * (psi[..., 1:-1, 2:] - psi[..., 0:-2, 1:-1])
        + zeta[..., 2:, 2:] * (psi[..., 2:, 1:-1] - psi[..., 1:-1, 2:])
        - zeta[..., 2:, 0:-2] * (psi[..., 2:, 1:-1] - psi[..., 1:-1, 0:-2])
        - zeta[..., 0:-2, 0:-2] * ( psi[..., 1:-1, 0:-2] - psi[..., 0:-2, 1:-1])
    ) / (12 * dx**2)


def periodic_arakawa_vec(zeta: jnp.ndarray, psi: jnp.ndarray, dx: float):
    """
    Compute the Poisson bracket (Jacobian) of vorticity and stream function for a 2D periodic
    domain using a vectorized version of the Arakawa scheme. This function automatically
    handles the required padding.

    Args:
        zeta (jnp.ndarray): Vorticity field.
        psi (jnp.ndarray): Stream function field.
        dx (float): Grid spacing.

    Returns:
        jnp.ndarray: Discretized Poisson bracket (Jacobian) over the grid without padding.

    """
    if len(zeta.shape) > 2:
        # Apply padding if specified
        zeta_padded = jnp.pad(zeta, ((0, 0), (1, 1), (1, 1)), mode="wrap")
        psi_padded = jnp.pad(psi, ((0, 0), (1, 1), (1, 1)), mode="wrap")
    else:
        # Use arrays without padding if pad=False
        zeta_padded = jnp.pad(zeta, ((1, 1), (1, 1)), mode="wrap")
        psi_padded = jnp.pad(psi, ((1, 1), (1, 1)), mode="wrap")
    return arakawa_vec(zeta_padded, psi_padded, dx)
    


