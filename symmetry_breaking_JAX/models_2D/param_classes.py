from dataclasses import dataclass
import jax.numpy as jnp

@dataclass
class Params2D:
    # Diffusion
    D_N: float
    D_L: float
    # Kinetics
    sigma_N: float
    sigma_L: float
    mu_N: float
    mu_L: float
    # Hill / interactions
    n: int
    p: int
    alpha: float
    K_A: float
    K_NL: float
    K_I: float
    apply_to_L: bool = True

    # Geometry / numerics
    geometry: str = "disk"  # "disk" or "sphere"

    # Disk parameters
    Lx: float = 1500.0
    Ly: float = 1500.0
    dx: float = 10.0
    dy: float = 10.0
    bc: str = "neumann"  # For rectangular container

    # Sphere parameters
    R: float = 500.0  # sphere radius, microns
    n_theta: int = 128
    n_phi: int = 256

    # Impulse parameters
    tau_impulse: float = 1.0
    sigma_t_direct: float = 0.05



@dataclass
class BlipSet2D:
    """
    Pre-sampled events for 2D domains.
    Disk: positions given as (x, y).
    Sphere: positions given as (theta, phi) in radians.
    """
    # Nodal direct
    N_dir_times: jnp.ndarray
    N_dir_pos: jnp.ndarray  # shape (M, 2) → (x,y) or (theta,phi)
    N_dir_amp: jnp.ndarray
    N_dir_sig: jnp.ndarray  # spatial width (absolute for disk, arc length for sphere)

    # Nodal impulse
    N_imp_times: jnp.ndarray
    N_imp_pos: jnp.ndarray
    N_imp_amp: jnp.ndarray
    N_imp_sig: jnp.ndarray

    # Lefty direct
    L_dir_times: jnp.ndarray
    L_dir_pos: jnp.ndarray
    L_dir_amp: jnp.ndarray
    L_dir_sig: jnp.ndarray

    # Lefty impulse
    L_imp_times: jnp.ndarray
    L_imp_pos: jnp.ndarray
    L_imp_amp: jnp.ndarray
    L_imp_sig: jnp.ndarray

    @classmethod
    def empty(cls):
        """Empty blip set with 0 events for both Nodal and Lefty."""
        empty_arr = jnp.array([]).reshape((0,))
        empty_pos = jnp.array([]).reshape((0, 2))
        return cls(
            N_dir_times=empty_arr, N_dir_pos=empty_pos, N_dir_amp=empty_arr, N_dir_sig=empty_arr,
            N_imp_times=empty_arr, N_imp_pos=empty_pos, N_imp_amp=empty_arr, N_imp_sig=empty_arr,
            L_dir_times=empty_arr, L_dir_pos=empty_pos, L_dir_amp=empty_arr, L_dir_sig=empty_arr,
            L_imp_times=empty_arr, L_imp_pos=empty_pos, L_imp_amp=empty_arr, L_imp_sig=empty_arr,
        )
