from pde import ScalarField, FieldCollection, PDEBase, CartesianGrid
import numpy as np


class NodalLeftyField1D_ND(PDEBase):
    """
    Non-dimensional 1D activator–inhibitor (Nodal–Lefty-like) RD system.

    Variables:
        a(ξ, τ) := N / K_A
        r(ξ, τ) := L / K_A
        ρ(ξ, τ) := passive density (unchanged logic; evolves but decoupled)

    Scales:
        ξ = x / sqrt(D_N / μ_N)
        τ = μ_N * t

    Dimensionless parameters:
        beta_a  = σ_N / (μ_N K_A)
        beta_r  = σ_L / (μ_N K_A)
        rho_mu  = μ_L / μ_N
        delta   = D_L / D_N
        kappa_I = K_I / K_A
        kappa_NL= K_NL / K_A
        n, m, p, q = Hill exponents (unchanged)
        alpha ∈ [0,1] = inhibitor efficacy (unchanged)

    Logic is preserved:
        - Lefty-mediated neutralization via fast 1:1 binding.
        - Deterministic & stochastic triggers with "direct" (per-step) and "impulse" (leaky) modes.
        - Direct triggers accumulate to per-step jumps J and are applied as sources S_dir = J/dt.
        - Impulse triggers accumulate in F and leak out with time constant tau_impulse.
        - Passive ρ field with same (decoupled) dynamics.

    Notes on sources and ICs:
        - All amplitudes/values here are dimensionless (divide your old N, L by K_A).
        - Spatial sigmas are in ξ-units (divide physical σ_x by sqrt(D_N/μ_N)).
        - Times are in τ-units (multiply physical times by μ_N).
    """

    def __init__(self,
                 # --- Dimensionless kinetics & transport ---
                 beta_a=1.0, beta_r=1e-2,
                 rho_mu=0.55,          # = μ_L / μ_N
                 delta=8.1,            # = D_L / D_N
                 # Hill / neutralization
                 n=2, m=1, p=2, q=2,
                 alpha=1.0,
                 kappa_NL=1.0,         # = K_NL / K_A
                 kappa_I=1.0,          # = K_I  / K_A
                 apply_to_r=True,      # same logic as apply_to_L
                 # Passive density (still dimensionless time)
                 rho_max=1.0, eps_rho=None, tau_rho=1e4, D_rho=0.0,
                 # Numerics
                 bc="auto_periodic_neumann", rng_seed=None,
                 # --- Initial conditions (dimensionless) ---
                 a_init="gaussian", a_amp=0.1, a_sigma=10.0,
                 r_init="constant", r_value=0.0, r_noise_range=(0.2, 0.6),
                 rho_init="constant", rho_value=0.1, rho_noise_amp=0.0,
                 # ---- Deterministic triggers (dimensionless) ----
                 a_t_init=None, a_t_amp=None, a_t_sigma=None, det_a_mode="direct",
                 r_t_init=None, r_t_amp=None, r_t_sigma=None, det_r_mode="direct",
                 # ---- Stochastic triggers (dimensionless) ----
                 stoch_to_a=False, rate_a=0.0, amp_median_a=1.0, amp_sigma_a=0.5,
                 stoch_a_mode="direct",
                 stoch_to_r=False, rate_r=0.0, amp_median_r=1.0, amp_sigma_r=0.5,
                 stoch_r_mode="direct",
                 sigma_x=30.0, tau_impulse=1.0,
                 blip_logger=None):
        super().__init__()

        # Kinetics (dimensionless)
        self.beta_a = float(beta_a)
        self.beta_r = float(beta_r)
        self.rho_mu = float(rho_mu)
        self.delta = float(delta)

        # Hill / neutralization
        self.n = int(n)
        self.m = int(m)
        self.p = int(p)
        self.q = int(q)
        self.kappa_NL = float(kappa_NL)
        self.kappa_I = float(kappa_I)
        self.alpha = float(np.clip(alpha, 0.0, 1.0))
        self.apply_to_r = bool(apply_to_r)

        # Passive rho (same logic)
        self.rho_max = float(rho_max)
        self.eps_rho = float(eps_rho if eps_rho is not None else 0.05 * rho_max)
        self.tau_rho = float(tau_rho)
        self.D_rho = float(D_rho)

        # Numerics
        self.bc = bc
        self.rng = np.random.default_rng(rng_seed)

        # ICs (dimensionless)
        self.a_init = a_init
        self.a_amp = float(a_amp)
        self.a_sigma = float(a_sigma)
        self.r_init = r_init
        self.r_value = float(r_value)
        self.r_noise_range = tuple(r_noise_range)
        self.rho_init = rho_init
        self.rho_value = float(rho_value)
        self.rho_noise_amp = float(rho_noise_amp)

        # Trigger config (dimensionless)
        self.a_t_init, self.a_t_amp, self.a_t_sigma, self.det_a_mode = a_t_init, a_t_amp, a_t_sigma, det_a_mode
        self.r_t_init, self.r_t_amp, self.r_t_sigma, self.det_r_mode = r_t_init, r_t_amp, r_t_sigma, det_r_mode

        self.stoch_to_a, self.rate_a, self.amp_median_a, self.amp_sigma_a, self.stoch_a_mode = \
            stoch_to_a, float(rate_a), float(amp_median_a), float(amp_sigma_a), stoch_a_mode
        self.stoch_to_r, self.rate_r, self.amp_median_r, self.amp_sigma_r, self.stoch_r_mode = \
            stoch_to_r, float(rate_r), float(amp_median_r), float(amp_sigma_r), stoch_r_mode

        self.sigma_x = float(sigma_x)          # already in ξ-units
        self.tau_impulse = float(tau_impulse)  # already in τ-units

        # Pulse logging
        self.blip_logger = blip_logger

        # Internal trigger state (unchanged logic)
        self._t_prev = None
        self._det_a_fired = False
        self._det_r_fired = False
        self._F_a = None   # accumulator for impulse-mode a
        self._F_r = None   # accumulator for impulse-mode r
        self._J_a = None   # per-step pending direct jump for a
        self._J_r = None   # per-step pending direct jump for r

    # ---------------- Core PDE (ND) ----------------
    def evolution_rate(self, state: FieldCollection, t: float = 0.0) -> FieldCollection:
        a, r, rho = state  # ScalarFields in ND units

        # Init accumulators for impulse & per-step direct jumps
        if self._F_a is None:
            self._F_a = ScalarField(a.grid, 0.0)
        if self._F_r is None:
            self._F_r = ScalarField(r.grid, 0.0)
        if self._J_a is None:
            self._J_a = ScalarField(a.grid, 0.0)
        if self._J_r is None:
            self._J_r = ScalarField(r.grid, 0.0)

        # Time step (solver-provided; already ND)
        dt = 0.0 if self._t_prev is None or t <= self._t_prev else t - self._t_prev

        # Decay impulse accumulators (impulse mode leakage)
        if dt > 0.0 and self.tau_impulse > 0.0:
            decay = np.exp(-dt / self.tau_impulse)
            self._F_a.data *= decay
            self._F_r.data *= decay

        # ---------------- Deterministic triggers ----------------
        if self.a_t_init is not None and not self._det_a_fired and self._t_prev is not None and self._t_prev < self.a_t_init <= t:
            self._apply_trigger(field=a, F_accum=self._F_a, mode=self.det_a_mode, t=t,
                                amp=self.a_t_amp if self.a_t_amp is not None else self.a_amp,
                                sigma=self.a_t_sigma if self.a_t_sigma is not None else self.a_sigma)
            self._det_a_fired = True

        if self.r_t_init is not None and not self._det_r_fired and self._t_prev is not None and self._t_prev < self.r_t_init <= t:
            self._apply_trigger(field=r, F_accum=self._F_r, mode=self.det_r_mode, t=t,
                                amp=self.r_t_amp if self.r_t_amp is not None else self.r_value,
                                sigma=self.r_t_sigma if self.r_t_sigma is not None else self.a_sigma)
            self._det_r_fired = True

        # ---------------- Stochastic triggers ----------------
        if dt > 0.0:
            length = float(a.grid.axes_bounds[0][1] - a.grid.axes_bounds[0][0])  # ND length

            if self.stoch_to_a and self.rate_a > 0.0:
                self._sample_spikes(field=a, F_accum=self._F_a, dt=dt, length=length,
                                    rate=self.rate_a, amp_median=self.amp_median_a, amp_sigma=self.amp_sigma_a,
                                    sigma_x=self.sigma_x, mode=self.stoch_a_mode, t=t)

            if self.stoch_to_r and self.rate_r > 0.0:
                self._sample_spikes(field=r, F_accum=self._F_r, dt=dt, length=length,
                                    rate=self.rate_r, amp_median=self.amp_median_r, amp_sigma=self.amp_sigma_r,
                                    sigma_x=self.sigma_x, mode=self.stoch_r_mode, t=t)

        # ---------------- Reaction terms (ND) ----------------
        a_free = self._free_activator_from_binding(a, r)  # Lefty repression via binding
        a_eff = a_free * self.alpha                       # inhibitor efficacy

        # Activator production: beta_a * a_eff^n / (1 + a_eff^n)
        a_prod = self.beta_a * (a_eff ** self.n) / (1.0 + a_eff ** self.n)
        # Repressor production: beta_r * (arg)^p / (kappa_NL^p + (arg)^p)
        a_arg = a_eff if self.apply_to_r else a
        r_prod = self.beta_r * (a_arg ** self.p) / (self.kappa_NL ** self.p + a_arg ** self.p)

        # Linear losses (ND): -a, -rho_mu * r
        a_loss = a
        r_loss = self.rho_mu * r

        # ---------------- Diffusion (ND) ----------------
        diff_a = a.laplace(self.bc)          # coefficient = 1 in ND units
        diff_r = self.delta * r.laplace(self.bc)

        # ---------------- Sources (ND) ----------------
        # Impulse-mode source (continuous)
        S_imp_a = (self._F_a / self.tau_impulse) if (self.stoch_a_mode == "impulse" or self.det_a_mode == "impulse") else 0.0
        S_imp_r = (self._F_r / self.tau_impulse) if (self.stoch_r_mode == "impulse" or self.det_r_mode == "impulse") else 0.0

        # Direct-mode source (per-step)
        if dt > 0.0:
            S_dir_a = ScalarField(a.grid, self._J_a.data / dt)
            S_dir_r = ScalarField(r.grid, self._J_r.data / dt)
            # clear after use
            self._J_a.data.fill(0.0)
            self._J_r.data.fill(0.0)
        else:
            S_dir_a = ScalarField(a.grid, 0.0)
            S_dir_r = ScalarField(r.grid, 0.0)

        S_a = S_imp_a + S_dir_a
        S_r = S_imp_r + S_dir_r

        # ---------------- Final RHS (ND) ----------------
        da_dt = a_prod - a_loss + diff_a + S_a
        dr_dt = r_prod - r_loss + diff_r + S_r

        # Passive ρ evolution (same logic, ND time)
        rho_relax = (rho - rho) / self.tau_rho  # zero
        rho_diff = self.D_rho * rho.laplace(self.bc)
        drho_dt = rho_relax + rho_diff

        self._t_prev = t
        return FieldCollection([da_dt, dr_dt, drho_dt])

    # ---------------- State init ----------------
    def get_state(self, grid: CartesianGrid) -> FieldCollection:
        a = self._init_a(grid).copy(label="a")     # activator
        r = self._init_r(grid).copy(label="r")     # repressor
        rho = self._init_rho(grid).copy(label="rho")

        # reset internal trigger state
        self._t_prev = None
        self._det_a_fired = False
        self._det_r_fired = False
        self._F_a = ScalarField(grid, 0.0)
        self._F_r = ScalarField(grid, 0.0)
        self._J_a = ScalarField(grid, 0.0)
        self._J_r = ScalarField(grid, 0.0)

        return FieldCollection([a, r, rho])

    # ---------------- Helpers ----------------
    def _init_a(self, grid):
        if self.a_init == "gaussian":
            x = grid.axes_coords[0]
            x0 = 0.5 * (x[0] + x[-1])
            return ScalarField(grid, self.a_amp * np.exp(-((x - x0) ** 2) / (2 * self.a_sigma ** 2)))
        elif self.a_init == "constant":
            return ScalarField(grid, self.a_amp)
        elif self.a_init == "random":
            return ScalarField(grid, self.rng.random(grid.shape))
        return ScalarField(grid, 0.0)

    def _init_r(self, grid):
        if self.r_init == "constant":
            return ScalarField(grid, self.r_value)
        elif self.r_init == "random":
            lo, hi = self.r_noise_range
            return ScalarField(grid, self.rng.uniform(lo, hi, size=grid.shape))
        elif self.r_init == "gaussian":
            x = grid.axes_coords[0]
            x0 = 0.5 * (x[0] + x[-1])
            return ScalarField(grid, self.r_value + self.a_amp * np.exp(-((x - x0) ** 2) / (2 * self.a_sigma ** 2)))
        return ScalarField(grid, 0.0)

    def _init_rho(self, grid):
        if self.rho_init == "constant":
            return ScalarField(grid, self.rho_value)
        elif self.rho_init == "random":
            base = self.rho_value
            noise = self.rho_noise_amp * (self.rng.random(grid.shape) - 0.5)
            return ScalarField(grid, np.clip(base + noise, 0.0, self.rho_max))
        return ScalarField(grid, self.rho_value)

    def _free_activator_from_binding(self, a, r):
        """
        Fast 1:1 binding in ND units:
            (a - c)(r - c) = κ_I * c,  with κ_I = K_I / K_A
        """
        kI = self.kappa_I
        at = a.data
        rt = r.data
        S = at + rt + kI
        disc = S * S - 4.0 * at * rt
        np.maximum(disc, 0.0, out=disc)
        C = 0.5 * (S - np.sqrt(disc))
        a_free = at - C
        np.clip(a_free, 0.0, at, out=a_free)
        return ScalarField(a.grid, a_free)

    def _apply_trigger(self, field, F_accum, mode, amp, sigma, t):
        """Route triggers to either impulse accumulator or per-step direct jump (ND units)."""
        label = getattr(field, "label", "")
        is_activator = (label == "a")
        channel = "Activator" if is_activator else "Repressor"

        if mode == "direct":
            target = self._J_a if is_activator else self._J_r
            self._add_gaussian_bump(target, amp, sigma, t=t, channel=channel)
        elif mode == "impulse":
            self._add_gaussian_bump(F_accum, amp, sigma, t=t, channel=channel)
        else:
            raise ValueError(f"Unrecognized trigger mode: {mode}")

    def _sample_spikes(self, field, F_accum, dt, length, rate, amp_median, amp_sigma, sigma_x, mode, t):
        """Draw Poisson spikes and deposit them as direct jumps (J) or impulse (F) in ND units."""
        N_events = self.rng.poisson(rate * length * dt)
        if N_events == 0:
            return
        xlo, xhi = field.grid.axes_bounds[0]
        xs = self.rng.uniform(xlo, xhi, size=N_events)
        amps = self.rng.lognormal(mean=np.log(amp_median), sigma=amp_sigma, size=N_events)
        for xi, Ai in zip(xs, amps):
            label = getattr(field, "label", "")
            is_activator = (label == "a")
            channel = "Activator" if is_activator else "Repressor"
            if mode == "direct":
                target = self._J_a if is_activator else self._J_r
                self._add_gaussian_bump(target, Ai, sigma_x, center=xi, channel=channel, t=t)
            elif mode == "impulse":
                self._add_gaussian_bump(F_accum, Ai, sigma_x, center=xi, channel=channel, t=t)
            else:
                raise ValueError(f"Unrecognized stochastic mode: {mode}")

    def _add_gaussian_bump(self, target_field, amp, sigma, t, center="center", channel="Activator"):
        """Add a Gaussian bump with PEAK = amp (ND units; no area normalization)."""
        x = target_field.grid.axes_coords[0]
        x0 = 0.5 * (x[0] + x[-1]) if center == "center" else float(center)
        kernel = np.exp(-((x - x0) ** 2) / (2 * float(sigma) ** 2))
        target_field.data += float(amp) * kernel

        # Log the event
        if self.blip_logger is not None:
            self.blip_logger.append({
                "time": float(t),
                "channel": channel,
                "x": float(x0),
                "amp": float(amp),
                "sigma": float(sigma) if sigma is not None else None
            })


# ---------- Helper: convert dimensional → ND parameters ----------
def nd_params_from_dimensional(*,
                               D_N, D_L, sigma_N, sigma_L, mu_N, mu_L,
                               K_A, K_NL, K_I,
                               n=2, m=1, p=2, q=2, alpha=1.0,
                               apply_to_L=True,
                               # ICs & triggers (dimensional)
                               N_amp=0.1, N_sigma=10.0, L_value=0.0,
                               N_t_amp=None, N_t_sigma=None, L_t_amp=None, L_t_sigma=None,
                               sigma_x=30.0, tau_impulse=1.0):
    """
    Map your dimensional parameters to the ND ones used by NodalLeftyField1D_ND.

    Returns a dict you can unpack into the ND class.
    """
    beta_a = sigma_N / (mu_N * K_A)
    beta_r = sigma_L / (mu_N * K_A)
    rho_mu = mu_L / mu_N
    delta = D_L / D_N
    kappa_I = K_I / K_A
    kappa_NL = K_NL / K_A

    # Space/time conversion (for IC/trigger widths & times, if you wish to carry them over):
    # ξ = x / sqrt(D_N/μ_N),  τ = μ_N t
    xi_sigma = N_sigma / np.sqrt(D_N / mu_N)
    xi_sigma_x = sigma_x / np.sqrt(D_N / mu_N)
    tau_imp = tau_impulse * mu_N

    out = dict(
        beta_a=beta_a, beta_r=beta_r, rho_mu=rho_mu, delta=delta,
        n=n, m=m, p=p, q=q, alpha=alpha,
        kappa_NL=kappa_NL, kappa_I=kappa_I,
        apply_to_r=apply_to_L,
        a_amp=N_amp / K_A,
        a_sigma=xi_sigma,
        r_value=L_value / K_A,
        a_t_amp=None if N_t_amp is None else N_t_amp / K_A,
        a_t_sigma=None if N_t_sigma is None else (N_t_sigma / np.sqrt(D_N / mu_N)),
        r_t_amp=None if L_t_amp is None else L_t_amp / K_A,
        r_t_sigma=None if L_t_sigma is None else (L_t_sigma / np.sqrt(D_N / mu_N)),
        sigma_x=xi_sigma_x,
        tau_impulse=tau_imp
    )
    return out
