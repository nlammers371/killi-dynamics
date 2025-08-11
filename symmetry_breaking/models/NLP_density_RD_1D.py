from pde import ScalarField, VectorField, FieldCollection, PDEBase, CartesianGrid
import numpy as np


def make_1d_grid(length=3000, dx=10, periodic=True):
    N = int(length / dx)
    grid = CartesianGrid([[0, length]], shape=(N,), periodic=periodic)
    return grid


class NodalLeftyDensity1D(PDEBase):
    """
    1D Nodal–Lefty reaction–diffusion system coupled to a density field ρ(x,t).

    Equations (matching your LaTeX):
      ∂N/∂t = ρ σ_N * N^n/(K_A^n + N^n)  - N λ * L^m/(K_R^m + L^m)  - μ_N N
               + ∇·( D_N(ρ) ∇N )

      ∂L/∂t = ρ σ_L * N^p/(K_NL^p + N^p) - μ_L L
               + ∇·( D_L(ρ) ∇L )

      ∂ρ/∂t = ( P(N) - ρ ) / τ_ρ  -  D_ρ ∇²ρ
        with P(N) = ρ_max * N^q / (K_ρ^q + N^q)

    Density-dependent diffusion:
      D_X(ρ) = D0_X * φ(ρ)^α_X,
      φ(ρ)   = 1 - ρ / (ρ_max + ε), with ε = eps_rho
    """

    def __init__(self,
                 # Diffusion baselines (free diffusion) and density sensitivity
                 D0_N=1.85, D0_L=15.0,
                 alpha_N=1.14, alpha_L=0.46, # defaults derived from empirical measurements
                 # Production (σ) and decay (μ)
                 sigma_N=1.0, sigma_L=1e-2,
                 mu_N=1.11e-4, mu_L=0.61e-4,
                 # Interaction strengths and Hill params
                 lambda_=0.01,  # subtractive inhibition strength on N by L
                 n=2, m=2, p=2, q=2,
                 K_A=100.0,     # N auto-activation threshold
                 K_R=100.0,     # L inhibition threshold in N eq
                 K_NL=100.0,    # N→L activation threshold
                 K_rho=100.0,   # P(N) threshold
                 # Density dynamics
                 rho_max=1.0,
                 eps_rho=None,  # if None, set to 0.05 * rho_max
                 tau_rho=1e4,
                 D_rho=0.0,     # set >0 to allow smoothing of ρ
                 # Numerics / BCs
                 bc="auto_periodic_neumann",
                 # Initial conditions
                 N_init="gaussian", N_amp=10.0, N_sigma=10.0,
                 L_init="constant", L_value=0.0, L_noise_range=(0.2, 0.6),
                 rho_init="constant", rho_value=0.1, rho_noise_amp=0.0):
        super().__init__()

        # Transport
        self.D0_N = D0_N
        self.D0_L = D0_L
        self.alpha_N = alpha_N
        self.alpha_L = alpha_L

        # Kinetics
        self.sigma_N = sigma_N
        self.sigma_L = sigma_L
        self.mu_N = mu_N
        self.mu_L = mu_L
        self.lambda_ = lambda_

        # Hill parameters
        self.n = n
        self.m = m
        self.p = p
        self.q = q
        self.K_A = K_A
        self.K_R = K_R
        self.K_NL = K_NL
        self.K_rho = K_rho

        # Density model
        self.rho_max = rho_max
        self.eps_rho = eps_rho if eps_rho is not None else 0.05 * rho_max
        self.tau_rho = tau_rho
        self.D_rho = D_rho

        # Numerics
        self.bc = bc

        # ICs
        self.N_init = N_init
        self.N_amp = N_amp
        self.N_sigma = N_sigma

        self.L_init = L_init
        self.L_value = L_value
        self.L_noise_range = L_noise_range

        self.rho_init = rho_init
        self.rho_value = rho_value
        self.rho_noise_amp = rho_noise_amp

    # ---------- Helpers for porosity and variable diffusion ----------
    def porosity(self, rho: ScalarField) -> ScalarField:
        # φ(ρ) = 1 - ρ / (ρ_max + ε); clamp to (φ_min, 1]
        denom = self.rho_max + self.eps_rho
        phi = 1.0 - (rho / denom)
        # ensure strictly positive porosity to avoid numerical issues
        phi_min = float(self.eps_rho / denom)  # value at ρ = ρ_max
        # clamp
        phi.data = np.clip(phi.data, phi_min, 1.0)
        return phi

    def D_N(self, rho: ScalarField) -> ScalarField:
        phi = self.porosity(rho)
        return self.D0_N * (phi ** self.alpha_N)

    def D_L(self, rho: ScalarField) -> ScalarField:
        phi = self.porosity(rho)
        return self.D0_L * (phi ** self.alpha_L)

    # ---------- PDE core ----------
    def evolution_rate(self, state: FieldCollection, t: float = 0.0) -> FieldCollection:
        N, L, rho = state  # ScalarField, ScalarField, ScalarField

        # Production terms (density-scaled)
        N_prod = (rho / self.rho_max) * self.sigma_N * (N**self.n) / (self.K_A**self.n + N**self.n)
        L_prod = (rho / self.rho_max) * self.sigma_L * (N**self.p) / (self.K_NL**self.p + N**self.p)

        # Inhibition + decay
        N_inhib = self.lambda_ * N * (L**self.m) / (self.K_R**self.m + L**self.m)
        N_loss = self.mu_N * N
        L_loss = self.mu_L * L

        # Variable-diffusion terms: ∇·( D(ρ) ∇C )
        # Use divergence form directly to include the ∇D · ∇C cross term
        grad_N = N.gradient(self.bc)            # VectorField
        grad_L = L.gradient(self.bc)            # VectorField
        DN = self.D_N(rho)                      # ScalarField
        DL = self.D_L(rho)                      # ScalarField
        diff_N = (DN * grad_N).divergence(self.bc)
        diff_L = (DL * grad_L).divergence(self.bc)

        # ρ dynamics: relaxation to P(N) plus plain diffusion
        P_N = self.rho_max * (N**self.q) / (self.K_rho**self.q + N**self.q)
        rho_relax = (P_N - rho) / self.tau_rho
        rho_diff = self.D_rho * rho.laplace(self.bc)

        # Assemble RHS
        dN_dt = N_prod - N_inhib - N_loss + diff_N
        dL_dt = L_prod - L_loss + diff_L
        drho_dt = rho_relax - rho_diff  # note the minus sign: ∂ρ/∂t = ... - D_ρ ∇²ρ

        return FieldCollection([dN_dt, dL_dt, drho_dt])

    # ---------- Initial conditions ----------
    def get_state(self, grid: CartesianGrid) -> FieldCollection:
        N = self._init_N(grid).copy(label="Nodal")
        L = self._init_L(grid).copy(label="Lefty")
        rho = self._init_rho(grid).copy(label="Density")
        return FieldCollection([N, L, rho])

    def _init_N(self, grid: CartesianGrid) -> ScalarField:
        if self.N_init == "gaussian":
            field = ScalarField(grid, data=0.0)
            x = grid.axes_coords[0]
            x0 = 0.5 * (x[0] + x[-1])
            profile = self.N_amp * np.exp(-((x - x0) ** 2) / (2 * self.N_sigma ** 2))
            field.data = profile
            return field
        elif self.N_init == "constant":
            return ScalarField(grid, data=self.N_amp)
        elif self.N_init == "random":
            return ScalarField(grid, data=np.random.rand(*grid.shape))
        else:
            raise ValueError(f"Unrecognized N_init: {self.N_init}")

    def _init_L(self, grid: CartesianGrid) -> ScalarField:
        if self.L_init == "constant":
            return ScalarField(grid, data=self.L_value)
        elif self.L_init == "random":
            noise = np.random.uniform(*self.L_noise_range, size=grid.shape)
            return ScalarField(grid, data=noise)
        elif self.L_init == "gaussian":
            # mirror N settings but with L parameters if desired later
            field = ScalarField(grid, data=0.0)
            x = grid.axes_coords[0]
            x0 = 0.5 * (x[0] + x[-1])
            profile = self.L_value + self.N_amp * np.exp(-((x - x0) ** 2) / (2 * self.N_sigma ** 2))
            field.data = profile
            return field
        else:
            raise ValueError(f"Unrecognized L_init: {self.L_init}")

    def _init_rho(self, grid: CartesianGrid) -> ScalarField:
        if self.rho_init == "constant":
            return ScalarField(grid, data=self.rho_value)
        elif self.rho_init == "random":
            base = self.rho_value
            noise = self.rho_noise_amp * (np.random.rand(*grid.shape) - 0.5)
            return ScalarField(grid, data=np.clip(base + noise, 0.0, self.rho_max))
        elif self.rho_init == "hotspot":
            field = ScalarField(grid, data=self.rho_value)
            x = grid.axes_coords[0]
            x0 = 0.5 * (x[0] + x[-1])
            hotspot = 0.2 * np.exp(-((x - x0) ** 2) / (2 * (self.N_sigma ** 2)))
            field.data = np.clip(field.data + hotspot, 0.0, self.rho_max)
            return field
        else:
            raise ValueError(f"Unrecognized rho_init: {self.rho_init}")