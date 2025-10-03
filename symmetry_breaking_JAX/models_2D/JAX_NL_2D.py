import jax.numpy as jnp
from .geom_utils import space_kernel_disk, space_kernel_sphere, laplace_disk, laplace_sphere, laplace_disk
from .param_classes import Params2D, BlipSet2D
from symmetry_breaking_JAX.models_1D.JAX_NL_1D import binding_free_N, hill01

def build_rhs_2d(params, blips, grid):
    """
    grid: (X,Y,mask) for disk, or (Theta,Phi) for sphere
    state layout: y = [N, L, rho, F_N, F_L] with each field flattened
    """
    geom = params.geometry

    if geom == "disk":
        X, Y, mask = grid
        nxy = X.size
    elif geom == "sphere":
        Theta, Phi = grid
        nxy = Theta.size
    else:
        raise ValueError("geometry must be 'disk' or 'sphere'")

    # Precompute space kernels for all blip sets
    def precompute_kernels(blip_pos, blip_sig):
        if geom == "disk":
            return space_kernel_disk(X, Y, blip_pos, blip_sig)
        elif geom == "sphere":
            return space_kernel_sphere(Theta, Phi, blip_pos, blip_sig, R=params.R)

    Sx_N_dir = precompute_kernels(blips.N_dir_pos, blips.N_dir_sig)
    Sx_N_imp = precompute_kernels(blips.N_imp_pos, blips.N_imp_sig)
    Sx_L_dir = precompute_kernels(blips.L_dir_pos, blips.L_dir_sig)
    Sx_L_imp = precompute_kernels(blips.L_imp_pos, blips.L_imp_sig)

    def time_source(t, times, amps, sigma_t):
        if times.size == 0:
            return jnp.zeros((0,))
        return amps * jnp.exp(-0.5*((t - times)/sigma_t)**2) / (jnp.sqrt(2*jnp.pi)*sigma_t)

    def rhs(t, y, args):
        # unpack
        N = y[0:nxy].reshape(-1)
        L = y[nxy:2*nxy].reshape(-1)
        rho = y[2*nxy:3*nxy].reshape(-1)
        F_N = y[3*nxy:4*nxy].reshape(-1)
        F_L = y[4*nxy:5*nxy].reshape(-1)

        # diffusion
        if geom == "disk":
            N_mat = N.reshape(X.shape)
            L_mat = L.reshape(X.shape)
            lapN = laplace_disk(N_mat, params.dx, mask, params.bc)
            lapL = laplace_disk(L_mat, params.dx, mask, params.bc)
            dN = params.D_N * lapN.reshape(-1)
            dL = params.D_L * lapL.reshape(-1)
        elif geom == "sphere":
            N_mat = N.reshape(Theta.shape)
            L_mat = L.reshape(Theta.shape)
            dtheta = jnp.pi/(params.n_theta-1)
            dphi   = 2*jnp.pi/params.n_phi
            lapN = laplace_sphere(N_mat, params.R, dtheta, dphi)
            lapL = laplace_sphere(L_mat, params.R, dtheta, dphi)
            dN = params.D_N * lapN.reshape(-1)
            dL = params.D_L * lapL.reshape(-1)

        drho = jnp.zeros_like(rho)

        # sources
        S_N_dir_t = time_source(t, blips.N_dir_times, blips.N_dir_amp, params.sigma_t_direct)
        S_L_dir_t = time_source(t, blips.L_dir_times, blips.L_dir_amp, params.sigma_t_direct)
        S_N_dir = (Sx_N_dir @ S_N_dir_t) if Sx_N_dir.shape[1] > 0 else jnp.zeros(nxy)
        S_L_dir = (Sx_L_dir @ S_L_dir_t) if Sx_L_dir.shape[1] > 0 else jnp.zeros(nxy)

        # impulse accumulators
        S_N_imp_t = time_source(t, blips.N_imp_times, blips.N_imp_amp, params.sigma_t_direct)
        S_L_imp_t = time_source(t, blips.L_imp_times, blips.L_imp_amp, params.sigma_t_direct)
        inp_N = (Sx_N_imp @ S_N_imp_t) if Sx_N_imp.shape[1] > 0 else jnp.zeros(nxy)
        inp_L = (Sx_L_imp @ S_L_imp_t) if Sx_L_imp.shape[1] > 0 else jnp.zeros(nxy)

        dF_N = inp_N - F_N/jnp.maximum(params.tau_impulse, 1e-12)
        dF_L = inp_L - F_L/jnp.maximum(params.tau_impulse, 1e-12)
        S_N_imp = F_N/jnp.maximum(params.tau_impulse, 1e-12)
        S_L_imp = F_L/jnp.maximum(params.tau_impulse, 1e-12)

        # binding + reactions
          # reuse your 1D helpers
        N_free = binding_free_N(N, L, params.K_I)
        N_eff = params.alpha * N_free

        N_act = params.sigma_N * hill01(N_eff/params.K_A, params.n)
        L_arg = jnp.where(params.apply_to_L, N_eff, N) / params.K_A
        L_prod = params.sigma_L * (jnp.power(jnp.clip(L_arg, 0, 1e6), params.p) /
                                  (jnp.power(params.K_NL/params.K_A, params.p) +
                                   jnp.power(jnp.clip(L_arg, 0, 1e6), params.p)))

        N_loss = params.mu_N * N
        L_loss = params.mu_L * L

        # total
        dN += N_act - N_loss + S_N_dir + S_N_imp
        dL += L_prod - L_loss + S_L_dir + S_L_imp

        dN = jnp.nan_to_num(dN, nan=0.0, posinf=0.0, neginf=0.0)
        dL = jnp.nan_to_num(dL, nan=0.0, posinf=0.0, neginf=0.0)

        dy = jnp.concatenate([dN, dL, drho, dF_N, dF_L])
        return dy

    return rhs