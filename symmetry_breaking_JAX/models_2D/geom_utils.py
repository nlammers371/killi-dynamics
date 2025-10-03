import jax.numpy as jnp

def make_grid_disk(nx, ny, Lx, Ly):
    """Cartesian grid covering a disk of radius min(Lx,Ly)/2."""
    x = jnp.linspace(-Lx/2, Lx/2, nx)
    y = jnp.linspace(-Ly/2, Ly/2, ny)
    X, Y = jnp.meshgrid(x, y, indexing="ij")
    R = jnp.sqrt(X**2 + Y**2)
    return X, Y, R


def make_grid_sphere(n_theta, n_phi):
    """Latitude-longitude grid covering full sphere."""
    thetas = jnp.linspace(0.0, jnp.pi, n_theta)
    phis   = jnp.linspace(0.0, 2.0*jnp.pi, n_phi, endpoint=False)
    Theta, Phi = jnp.meshgrid(thetas, phis, indexing="ij")
    return Theta, Phi

import jax.numpy as jnp

def angular_distance(theta1, phi1, theta2, phi2):
    """Great-circle distance between two sets of points (broadcasted)."""
    cos_d = (
        jnp.sin(theta1) * jnp.sin(theta2) * jnp.cos(phi1 - phi2)
        + jnp.cos(theta1) * jnp.cos(theta2)
    )
    return jnp.arccos(jnp.clip(cos_d, -1.0, 1.0))


def space_kernel_disk(x: jnp.ndarray, y: jnp.ndarray,
                      x0: jnp.ndarray, y0: jnp.ndarray, sigma: float):
    """Gaussian kernel in flat 2D Cartesian coordinates."""
    dx = x[:, None] - x0[None, :]
    dy = y[:, None] - y0[None, :]
    r2 = dx**2 + dy**2
    return jnp.exp(-0.5 * r2 / (sigma**2))


def space_kernel_sphere(theta: jnp.ndarray, phi: jnp.ndarray,
                        theta0: jnp.ndarray, phi0: jnp.ndarray, sigma_ang: float):
    """Gaussian kernel on a sphere, width in radians."""
    dist = angular_distance(theta[:, None], phi[:, None],
                            theta0[None, :], phi0[None, :])
    return jnp.exp(-0.5 * (dist/sigma_ang)**2)

def space_kernel_sphere(theta, phi, theta0, phi0, sigma_abs, R=1.0):
    """Gaussian kernel on a sphere, width sigma_abs in same units as R."""
    dist_ang = angular_distance(theta[:, None], phi[:, None],
                                theta0[None, :], phi0[None, :])  # radians
    dist_abs = R * dist_ang
    return jnp.exp(-0.5 * (dist_abs / sigma_abs)**2)

def space_kernel_sphere(theta, phi, theta0, phi0, sigma_abs, R=1.0):
    """Gaussian kernel on a sphere, width sigma_abs in same units as R."""
    dist_ang = angular_distance(theta[:, None], phi[:, None],
                                theta0[None, :], phi0[None, :])  # radians
    dist_abs = R * dist_ang
    return jnp.exp(-0.5 * (dist_abs / sigma_abs)**2)



def laplace_rect(u: jnp.ndarray, dx: float, bc: str = "neumann") -> jnp.ndarray:
    """
    5-point stencil Laplacian on a flat 2D Cartesian grid with radius cutoff.
    u: (nx, ny)
    bc: 'neumann' (reflecting) or 'periodic'
    """
    if bc == "periodic":
        u_up    = jnp.roll(u, -1, axis=0)
        u_down  = jnp.roll(u, +1, axis=0)
        u_left  = jnp.roll(u, -1, axis=1)
        u_right = jnp.roll(u, +1, axis=1)
    elif bc == "neumann":
        u_up    = jnp.concatenate([u[:1, :], u[:-1, :]], axis=0)
        u_down  = jnp.concatenate([u[1:, :], u[-1:, :]], axis=0)
        u_left  = jnp.concatenate([u[:, :1], u[:, :-1]], axis=1)
        u_right = jnp.concatenate([u[:, 1:], u[:, -1:]], axis=1)
    else:
        raise ValueError("bc must be 'periodic' or 'neumann'")

    return (u_up + u_down + u_left + u_right - 4.0 * u) / (dx * dx)


def laplace_disk(u, dx, mask, bc_rect="neumann", bc_circle="neumann"):
    """5-point Laplacian on a circular mask embedded in a rectangular grid."""
    if bc_rect == "periodic":
        u_up    = jnp.roll(u, -1, axis=0)
        u_down  = jnp.roll(u, +1, axis=0)
        u_left  = jnp.roll(u, -1, axis=1)
        u_right = jnp.roll(u, +1, axis=1)

        m_up    = jnp.roll(mask, -1, axis=0)
        m_down  = jnp.roll(mask, +1, axis=0)
        m_left  = jnp.roll(mask, -1, axis=1)
        m_right = jnp.roll(mask, +1, axis=1)
    elif bc_rect == "neumann":
        u_up    = jnp.concatenate([u[:1, :],  u[:-1, :]], axis=0)
        u_down  = jnp.concatenate([u[1:, :],  u[-1:, :]], axis=0)
        u_left  = jnp.concatenate([u[:, :1],  u[:, :-1]], axis=1)
        u_right = jnp.concatenate([u[:, 1:],  u[:, -1:]], axis=1)

        m_up    = jnp.concatenate([mask[:1, :],  mask[:-1, :]], axis=0)
        m_down  = jnp.concatenate([mask[1:, :],  mask[-1:, :]], axis=0)
        m_left  = jnp.concatenate([mask[:, :1],  mask[:, :-1]], axis=1)
        m_right = jnp.concatenate([mask[:, 1:],  mask[:, :-1]], axis=1)
    else:
        raise ValueError("bc_rect must be 'periodic' or 'neumann'")

    if bc_circle == "neumann":
        u_up_eff    = jnp.where(m_up,    u_up,    u)
        u_down_eff  = jnp.where(m_down,  u_down,  u)
        u_left_eff  = jnp.where(m_left,  u_left,  u)
        u_right_eff = jnp.where(m_right, u_right, u)
    elif bc_circle == "dirichlet":
        u_up_eff    = jnp.where(m_up,    u_up,    0.0)
        u_down_eff  = jnp.where(m_down,  u_down,  0.0)
        u_left_eff  = jnp.where(m_left,  u_left,  0.0)
        u_right_eff = jnp.where(m_right, u_right, 0.0)
    else:
        raise ValueError("bc_circle must be 'neumann' or 'dirichlet'")

    count = (m_up + m_down + m_left + m_right).astype(u.dtype)
    sum_nb = u_up_eff + u_down_eff + u_left_eff + u_right_eff
    lap = (sum_nb - count * u) / (dx * dx)

    return jnp.where(mask, lap, 0.0)


def laplace_sphere(u: jnp.ndarray, dtheta: float, dphi: float) -> jnp.ndarray:
    """
    Spherical Laplacian on a latitude/longitude grid.
    u: (n_theta, n_phi) with theta = colatitude [0, pi], phi = longitude [0, 2pi)
    """
    n_theta, n_phi = u.shape

    # roll along phi (longitude) for periodicity
    u_phi_plus  = jnp.roll(u, -1, axis=1)
    u_phi_minus = jnp.roll(u, +1, axis=1)

    # finite differences in theta (latitude / colatitude)
    u_theta_plus  = jnp.concatenate([u[1:, :], u[-1:, :]], axis=0)
    u_theta_minus = jnp.concatenate([u[:1, :], u[:-1, :]], axis=0)

    thetas = jnp.linspace(0.0, jnp.pi, n_theta)
    sin_t = jnp.sin(thetas)[:, None]
    sin_t_safe = jnp.where(sin_t < 1e-8, 1e-8, sin_t)

    # second derivative in theta, with metric
    d2theta = (u_theta_plus - 2.0*u + u_theta_minus) / (dtheta**2)
    term_theta = d2theta + (jnp.cos(thetas)[:, None]/sin_t_safe) * (u_theta_plus - u_theta_minus)/(2*dtheta)

    # second derivative in phi
    term_phi = (u_phi_plus - 2.0*u + u_phi_minus) / (dphi**2)

    return term_theta + term_phi/(sin_t_safe**2)
