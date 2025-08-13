from pde import ScalarField, FieldCollection, PDEBase, CartesianGrid
import numpy as np


class NodalLeftyField1D(PDEBase):
    """
    1D Nodal–Lefty RD system with Lefty-mediated neutralization and passive density field.

    rho evolves but has no causal effect on N or L.
    Deterministic and stochastic triggers can be applied to N and/or L channels.
    Each trigger type (deterministic, stochastic) can use "direct" or "impulse" mode.
    """

    def __init__(self,
                 # Diffusion (constant coefficients)
                 D_N=1.85, D_L=15.0,
                 # Production / decay / interactions
                 sigma_N=1.0, sigma_L=1e-2,
                 mu_ratio=1.11e-4/0.61e-4, mu_L=0.61e-4,
                 # Hill / neutralization parameters
                 n=2, m=1, p=2, q=2,
                 K_A=100.0, K_NL=100.0,
                 K_I=100.0, apply_to_L=True,
                 # Density params (passive)
                 rho_max=1.0, eps_rho=None, tau_rho=1e4, D_rho=0.0,
                 # Numerics
                 bc="auto_periodic_neumann", rng_seed=None,
                 # ICs
                 N_init="gaussian", N_amp=10.0, N_sigma=10.0,
                 L_init="constant", L_value=0.0, L_noise_range=(0.2, 0.6),
                 rho_init="constant", rho_value=0.1, rho_noise_amp=0.0,
                 # ---- Deterministic triggers ----
                 N_t_init=None, N_t_amp=None, N_t_sigma=None, det_N_mode="direct",
                 L_t_init=None, L_t_amp=None, L_t_sigma=None, det_L_mode="direct",
                 # ---- Stochastic triggers ----
                 stoch_to_N=False, rate_N=0.0, amp_median_N=1.0, amp_sigma_N=0.5,
                 stoch_N_mode="direct",
                 stoch_to_L=False, rate_L=0.0, amp_median_L=1.0, amp_sigma_L=0.5,
                 stoch_L_mode="direct",
                 sigma_x=10.0, tau_impulse=1.0):
        super().__init__()

        # Diffusion
        self.D_N = D_N
        self.D_L = D_L

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
        self.K_NL = K_NL

        # Neutralization
        self.K_I = K_I
        self.apply_to_L = apply_to_L

        # Passive rho params
        self.rho_max = rho_max
        self.eps_rho = eps_rho if eps_rho is not None else 0.05 * rho_max
        self.tau_rho = tau_rho
        self.D_rho = D_rho

        # Numerics
        self.bc = bc
        self.rng = np.random.default_rng(rng_seed)

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

        # Trigger config
        self.N_t_init, self.N_t_amp, self.N_t_sigma, self.det_N_mode = N_t_init, N_t_amp, N_t_sigma, det_N_mode
        self.L_t_init, self.L_t_amp, self.L_t_sigma, self.det_L_mode = L_t_init, L_t_amp, L_t_sigma, det_L_mode

        self.stoch_to_N, self.rate_N, self.amp_median_N, self.amp_sigma_N, self.stoch_N_mode = \
            stoch_to_N, rate_N, amp_median_N, amp_sigma_N, stoch_N_mode
        self.stoch_to_L, self.rate_L, self.amp_median_L, self.amp_sigma_L, self.stoch_L_mode = \
            stoch_to_L, rate_L, amp_median_L, amp_sigma_L, stoch_L_mode

        self.sigma_x = sigma_x
        self.tau_impulse = tau_impulse

        # Internal trigger state
        self._t_prev = None
        self._det_N_fired = False
        self._det_L_fired = False
        self._F_N = None
        self._F_L = None

    def evolution_rate(self, state: FieldCollection, t: float = 0.0) -> FieldCollection:
        N, L, rho = state

        # Init accumulators for impulse mode
        if self._F_N is None:
            self._F_N = ScalarField(N.grid, 0.0)
        if self._F_L is None:
            self._F_L = ScalarField(L.grid, 0.0)

        # Time step length
        dt = 0.0 if self._t_prev is None or t <= self._t_prev else t - self._t_prev

        # Decay accumulators
        if dt > 0.0 and self.tau_impulse > 0.0:
            decay = np.exp(-dt / self.tau_impulse)
            self._F_N.data *= decay
            self._F_L.data *= decay

        # ---- Deterministic triggers ----
        if self.N_t_init is not None and not self._det_N_fired and self._t_prev is not None and self._t_prev < self.N_t_init <= t:
            self._apply_trigger(N, self._F_N, mode=self.det_N_mode,
                                amp=self.N_t_amp or self.N_amp,
                                sigma=self.N_t_sigma or self.N_sigma)
            self._det_N_fired = True

        if self.L_t_init is not None and not self._det_L_fired and self._t_prev is not None and self._t_prev < self.L_t_init <= t:
            self._apply_trigger(L, self._F_L, mode=self.det_L_mode,
                                amp=self.L_t_amp or self.L_value,
                                sigma=self.L_t_sigma or self.N_sigma)
            self._det_L_fired = True

        # ---- Stochastic triggers ----
        if dt > 0.0:
            length = float(N.grid.axes_bounds[0][1] - N.grid.axes_bounds[0][0])
            if self.stoch_to_N and self.rate_N > 0.0:
                self._sample_spikes(N, self._F_N, dt, length,
                                    rate=self.rate_N,
                                    amp_median=self.amp_median_N,
                                    amp_sigma=self.amp_sigma_N,
                                    sigma_x=self.sigma_x,
                                    mode=self.stoch_N_mode)
            if self.stoch_to_L and self.rate_L > 0.0:
                self._sample_spikes(L, self._F_L, dt, length,
                                    rate=self.rate_L,
                                    amp_median=self.amp_median_L,
                                    amp_sigma=self.amp_sigma_L,
                                    sigma_x=self.sigma_x,
                                    mode=self.stoch_L_mode)

        # ---- Core PDE dynamics ----
        N_free = self._free_nodal_from_binding(N, L)

        N_act = self.sigma_N * (N_free ** self.n) / (self.K_A ** self.n + N_free ** self.n)
        L_arg = N_free if self.apply_to_L else N
        L_prod = self.sigma_L * (L_arg ** self.p) / (self.K_NL ** self.p + L_arg ** self.p)

        N_loss = self.mu_N * N
        L_loss = self.mu_L * L

        diff_N = self.D_N * N.laplace(self.bc)
        diff_L = self.D_L * L.laplace(self.bc)

        S_N = (self._F_N / self.tau_impulse) if self.stoch_N_mode == "impulse" or self.det_N_mode == "impulse" else 0.0
        S_L = (self._F_L / self.tau_impulse) if self.stoch_L_mode == "impulse" or self.det_L_mode == "impulse" else 0.0

        dN_dt = N_act - N_loss + diff_N + S_N
        dL_dt = L_prod - L_loss + diff_L + S_L

        # Passive rho evolution
        rho_relax = (rho - rho) / self.tau_rho  # zero; placeholder
        rho_diff = self.D_rho * rho.laplace(self.bc)
        drho_dt = rho_relax + rho_diff

        self._t_prev = t
        return FieldCollection([dN_dt, dL_dt, drho_dt])

    def get_state(self, grid: CartesianGrid) -> FieldCollection:
        N = self._init_N(grid).copy(label="Nodal_total")
        L = self._init_L(grid).copy(label="Lefty")
        rho = self._init_rho(grid).copy(label="Density")
        self._t_prev = None
        self._det_N_fired = False
        self._det_L_fired = False
        self._F_N = ScalarField(grid, 0.0)
        self._F_L = ScalarField(grid, 0.0)
        return FieldCollection([N, L, rho])

    def _init_N(self, grid):
        if self.N_init == "gaussian":
            x = grid.axes_coords[0]
            x0 = 0.5 * (x[0] + x[-1])
            return ScalarField(grid, self.N_amp * np.exp(-((x - x0) ** 2) / (2 * self.N_sigma ** 2)))
        elif self.N_init == "constant":
            return ScalarField(grid, self.N_amp)
        elif self.N_init == "random":
            return ScalarField(grid, self.rng.random(grid.shape))
        return ScalarField(grid, 0.0)

    def _init_L(self, grid):
        if self.L_init == "constant":
            return ScalarField(grid, self.L_value)
        elif self.L_init == "random":
            lo, hi = self.L_noise_range
            return ScalarField(grid, self.rng.uniform(lo, hi, size=grid.shape))
        elif self.L_init == "gaussian":
            x = grid.axes_coords[0]
            x0 = 0.5 * (x[0] + x[-1])
            return ScalarField(grid, self.L_value + self.N_amp * np.exp(-((x - x0) ** 2) / (2 * self.N_sigma ** 2)))
        return ScalarField(grid, 0.0)

    def _init_rho(self, grid):
        if self.rho_init == "constant":
            return ScalarField(grid, self.rho_value)
        elif self.rho_init == "random":
            base = self.rho_value
            noise = self.rho_noise_amp * (self.rng.random(grid.shape) - 0.5)
            return ScalarField(grid, np.clip(base + noise, 0.0, self.rho_max))
        return ScalarField(grid, self.rho_value)

    def _free_nodal_from_binding(self, N, L):
        Kd = self.K_I
        Nt = N.data
        Lt = L.data
        S = Nt + Lt + Kd
        disc = S * S - 4.0 * Nt * Lt
        np.maximum(disc, 0.0, out=disc)
        C = 0.5 * (S - np.sqrt(disc))
        N_free = Nt - C
        np.clip(N_free, 0.0, Nt, out=N_free)
        return ScalarField(N.grid, N_free)

    def _apply_trigger(self, field, F_accum, mode, amp, sigma):
        if mode == "direct":
            self._add_gaussian_bump(field, amp, sigma)
        elif mode == "impulse":
            self._add_gaussian_bump(F_accum, amp, sigma)

    def _sample_spikes(self, field, F_accum, dt, length, rate, amp_median, amp_sigma, sigma_x, mode):
        N_events = self.rng.poisson(rate * length * dt)
        if N_events == 0:
            return
        xlo, xhi = field.grid.axes_bounds[0]
        xs = self.rng.uniform(xlo, xhi, size=N_events)
        amps = self.rng.lognormal(mean=np.log(amp_median), sigma=amp_sigma, size=N_events)
        for xi, Ai in zip(xs, amps):
            if mode == "direct":
                self._add_gaussian_bump(field, Ai, sigma_x, center=xi)
            elif mode == "impulse":
                self._add_gaussian_bump(F_accum, Ai, sigma_x, center=xi)

    def _add_gaussian_bump(self, target_field, amp, sigma, center="center"):
        x = target_field.grid.axes_coords[0]
        if center == "center":
            x0 = 0.5 * (x[0] + x[-1])
        else:
            x0 = float(center)
        kernel = np.exp(-((x - x0) ** 2) / (2 * sigma ** 2))
        norm = np.trapz(kernel, x=x)
        if norm > 0:
            target_field.data += (amp / norm) * kernel
