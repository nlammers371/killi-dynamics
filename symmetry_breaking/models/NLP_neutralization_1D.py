from pde import ScalarField, FieldCollection, PDEBase, CartesianGrid
import numpy as np


def make_1d_grid(length=3000, dx=10, periodic=True):
    N = int(length / dx)
    grid = CartesianGrid([[0, length]], shape=(N,), periodic=periodic)
    return grid


class NodalLeftyNeutralization1D(PDEBase):
    """
    1D Nodal–Lefty RD system with a density field ρ(x,t) and Lefty-mediated neutralization.

    Equations (aligned with your LaTeX, using neutralization):
      N_free = N / (1 + (L/K_I)^m)

      ∂N/∂t = ρ σ_N * (N_free^n)/(K_A^n + N_free^n)  - μ_N N
               + ∇·( D_N(ρ) ∇N )

      ∂L/∂t = ρ σ_L * (X^p)/(K_NL^p + X^p)  - μ_L L
               + ∇·( D_L(ρ) ∇L )
         where X = N_free if apply_to_L else N

      ∂ρ/∂t = ( P(N) - ρ ) / τ_ρ  -  D_ρ ∇²ρ
         with P(N) = ρ_max * N^q / (K_ρ^q + N^q)

    Density-dependent diffusion:
      D_X(ρ) = D0_X * φ(ρ)^α_X,   φ(ρ) = 1 - ρ / (ρ_max + ε)
    """

    def __init__(self,
                 # Free diffusion baselines and density sensitivity
                 D0_N=60.0, D0_L=60.0,
                 alpha_N=1.14, alpha_L=0.46,
                 # Production (σ) and decay (μ)
                 sigma_N=1.0, sigma_L=1e-2,
                 mu_ratio=1.11e-4/0.61e-4,  # ratio of mu_N to mu_L
                 mu_L=0.61e-4,
                 no_density_dependence=False,  # if True, set alpha_N=alpha_L=0
                 # Hill parameters and thresholds
                 n=2, m=1, p=2, q=2,
                 K_A=100.0,      # N auto-activation threshold
                 # K_R=100.0,      # kept for API parity; unused in neutralization
                 K_NL=100.0,     # N -> L activation threshold
                 K_rho=100.0,    # P(N) threshold
                 K_P=None,       # optional alias: if provided, overrides K_NL
                 # Neutralization (Lefty-Nodal) scale
                 K_I=100.0,      # inhibition/neutralization scale
                 apply_to_L=True,  # if True, gate N→L production by N_free as well
                 lock_rho_K=True,
                 # Density dynamics
                 rho_max=1.0,
                 eps_rho=None,     # if None, defaults to 0.05 * rho_max
                 tau_rho=1e4,
                 D_rho=0.0,        # set >0 to add smoothing of ρ (note sign per LaTeX)
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
        self.alpha_N = alpha_N if not no_density_dependence else 0.0
        self.alpha_L = alpha_L if not no_density_dependence else 0.0

        # Kinetics
        self.sigma_N = sigma_N
        self.sigma_L = sigma_L
        self.mu_N = mu_ratio * mu_L
        self.mu_L = mu_L

        # Hill params
        self.n = n
        self.m = m
        self.p = p
        self.q = q
        self.K_A = K_A
        self.K_R = K_R
        self.K_NL = K_P if K_P is not None else K_NL
        self.K_rho = K_A if lock_rho_K else K_rho

        # Neutralization
        self.K_I = K_I
        self.apply_to_L = apply_to_L

        # Density model
        self.no_density_dependence = no_density_dependence
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

    # ---------- Porosity & variable diffusion ----------
    def porosity(self, rho: ScalarField) -> ScalarField:
        # φ(ρ) = 1 - ρ / (ρ_max + ε); clamp to [φ_min, 1]
        denom = self.rho_max + self.eps_rho
        phi = 1.0 - (rho / denom)
        phi_min = float(self.eps_rho / denom)  # φ at ρ = ρ_max
        phi.data = np.clip(phi.data, phi_min, 1.0)
        return phi

    def D_N(self, rho):
        if self.no_density_dependence:
            return ScalarField(rho.grid, data=self.D0_N)
        return self.D0_N * (self.porosity(rho) ** self.alpha_N)

    def D_L(self, rho):
        if self.no_density_dependence:
            return ScalarField(rho.grid, data=self.D0_L)
        return self.D0_L * (self.porosity(rho) ** self.alpha_L)

    # ---------- PDE core ----------
    def evolution_rate(self, state: FieldCollection, t: float = 0.0) -> FieldCollection:
        N, L, rho = state  # ScalarFields

        rho_scale = 1.0 if self.no_density_dependence else (rho / self.rho_max)
        # Lefty neutralization: fast binding reduces signaling-competent N
        # N_free used ONLY in activation terms; diffusion/decay act on total N
        # I = 1.0 + (L / self.K_I) ** self.m
        # avoid division warnings for tiny I (shouldn't happen with K_I>0)
        # N_free = N / I
        N_free = self._free_nodal_from_binding(N, L)

        # Production terms (density-scaled)
        N_act = rho_scale * self.sigma_N * (N_free ** self.n) / (self.K_A ** self.n + N_free ** self.n)
        L_arg = N_free if self.apply_to_L else N
        L_prod = rho_scale * self.sigma_L * (L_arg ** self.p) / (self.K_NL ** self.p + L_arg ** self.p)

        # Physical loss (on total fields)
        N_loss = self.mu_N * N
        L_loss = self.mu_L * L

        # Variable diffusion in divergence form: ∇·( D(ρ) ∇C )
        DN = self.D_N(rho)
        DL = self.D_L(rho)
        diff_N = self._div_D_grad_1d(N, DN)
        diff_L = self._div_D_grad_1d(L, DL)
        # DN = self.D_N(rho)
        # DL = self.D_L(rho)
        # diff_N = (DN * N.gradient(self.bc)).divergence(self.bc)
        # diff_L = (DL * L.gradient(self.bc)).divergence(self.bc)

        # Density dynamics
        P_N = self.rho_max * (N ** self.q) / (self.K_rho ** self.q + N ** self.q)
        rho_relax = (P_N - rho) / self.tau_rho
        rho_diff = self.D_rho * rho.laplace(self.bc)  # note: ∂ρ/∂t = ... - D_ρ ∇²ρ per LaTeX
        drho_dt = rho_relax + rho_diff

        # Assemble RHS
        dN_dt = N_act - N_loss + diff_N
        dL_dt = L_prod - L_loss + diff_L

        return FieldCollection([dN_dt, dL_dt, drho_dt])

    # ---------- Initial conditions ----------
    def get_state(self, grid: CartesianGrid) -> FieldCollection:
        N = self._init_N(grid).copy(label="Nodal_total")
        L = self._init_L(grid).copy(label="Lefty")
        rho = self._init_rho(grid).copy(label="Density")
        return FieldCollection([N, L, rho])

    def _init_N(self, grid: CartesianGrid) -> ScalarField:
        if self.N_init == "gaussian":
            field = ScalarField(grid, data=0.0)
            x = grid.axes_coords[0]
            x0 = 0.5 * (x[0] + x[-1])
            field.data = self.N_amp * np.exp(-((x - x0) ** 2) / (2 * self.N_sigma ** 2))
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
            field = ScalarField(grid, data=0.0)
            x = grid.axes_coords[0]
            x0 = 0.5 * (x[0] + x[-1])
            field.data = self.L_value + self.N_amp * np.exp(-((x - x0) ** 2) / (2 * self.N_sigma ** 2))
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

    def _free_nodal_from_binding(self, N, L):
        # Fast 1:1 binding: (N-C)(L-C) = Kd * C
        Kd = self.K_I  # interpret K_I as Kd
        Nt = N.data
        Lt = L.data
        S = Nt + Lt + Kd
        disc = S * S - 4.0 * Nt * Lt
        # numerical guard
        np.maximum(disc, 0.0, out=disc)
        C = 0.5 * (S - np.sqrt(disc))
        N_free = Nt - C
        # clamp to [0, Nt]
        np.clip(N_free, 0.0, Nt, out=N_free)
        return ScalarField(N.grid, data=N_free)

    def _div_D_grad_1d(self, C: ScalarField, D: ScalarField) -> ScalarField:
        """Compute ∂x [ D(x) ∂x C ] with face-centered D and harmonic averaging."""
        grid = C.grid
        x = grid.axes_coords[0]
        dx = float(x[1] - x[0])

        c = C.data
        d = D.data

        # D at faces (i+1/2): harmonic mean to be monotone on jumps
        d_ip1 = np.roll(d, -1)
        D_face = 2.0 * d * d_ip1 / (d + d_ip1 + 1e-14)

        # gradient at faces: (C_{i+1} - C_i)/dx
        c_ip1 = np.roll(c, -1)
        grad_face = (c_ip1 - c) / dx

        # flux at faces: D_face * grad_face  (this is +∂x-term; no extra minus)
        flux_face = D_face * grad_face

        # divergence back to cell centers: (flux_{i} - flux_{i-1})/dx
        div = (flux_face - np.roll(flux_face, 1)) / dx

        # Non-periodic fallback: simple Neumann copy at ends
        if not grid.periodic[0]:
            # left boundary: use forward diff and D at left face
            D_left = d
            grad_left = (c[1] - c[0]) / dx
            flux_left = D_left * grad_left
            div[0] = (flux_face[0] - flux_left) / dx
            # right boundary
            D_right = d
            grad_right = (c[-1] - c[-2]) / dx
            flux_right = D_right * grad_right
            div[-1] = (flux_right - flux_face[-1]) / dx

        return ScalarField(grid, data=div)