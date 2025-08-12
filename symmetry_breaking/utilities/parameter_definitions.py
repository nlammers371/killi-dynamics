import numpy as np

def porosity_linear(rho, rho_max=1.0, eps=0.05):
    """φ(ρ) = 1 - ρ/(ρ_max + eps*rho_max). rho in [0,1]."""
    denom = rho_max + eps * rho_max
    phi = 1.0 - rho / denom
    return np.clip(phi, eps * rho_max / denom, 1.0)

def da0_initial(N0, sigma0, rho0, sigma_N, mu_N, D0_N, K_A, n=2,
                phi_fn=porosity_linear, alpha_N=1.0, rho_max=1.0, eps=0.05):
    """
    Damköhler-like initial-growth test (t=0, L≈0):
    Da0 ≳ 1 ⇒ seed tends to grow; Da0 ≪ 1 ⇒ seed dies.
    Includes density-scaled production and D_eff at ρ0.
    """
    # production at t=0 (density-scaled)
    prod = (rho0 / rho_max) * sigma_N * (N0**n) / (K_A**n + N0**n)
    # effective diffusion at rho0
    phi0 = phi_fn(rho0, rho_max=rho_max, eps=eps)
    D_eff = D0_N * (phi0 ** alpha_N)
    leak = mu_N * N0 + (D_eff / (sigma0**2)) * N0  # decay + diffusive leak
    return prod / max(leak, 1e-30)

def diffusion_length_N(D0_N, mu_N, rho0=0.0, alpha_N=1.0,
                       phi_fn=porosity_linear, rho_max=1.0, eps=0.05):
    """
    Effective Nodal diffusion length at baseline density rho0:
    ℓ_N = sqrt(D_eff / μ_N), with D_eff = D0_N φ(ρ0)^α_N
    """
    phi0 = phi_fn(rho0, rho_max=rho_max, eps=eps)
    D_eff = D0_N * (phi0 ** alpha_N)
    return np.sqrt(D_eff / max(mu_N, 1e-30))

def dt_cfl(dx, Dmax, safety=0.5):
    """Explicit stability guideline: dt <= safety * dx^2 / (2*Dmax)."""
    return safety * (dx**2) / (2.0 * Dmax)