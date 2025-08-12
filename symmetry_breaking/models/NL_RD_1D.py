from pde import ScalarField, FieldCollection, PDEBase, CartesianGrid
import numpy as np


def make_1d_grid(length=3000, dx=10, periodic=True):
    N = int(length / dx)
    grid = CartesianGrid([[0, length]], shape=(N,), periodic=periodic)
    return grid


class TuringPDE1D(PDEBase):
    def __init__(self,
                 da=1.85,
                 dr=15,
                 k_a=1.11e-4,
                 k_r=0.61e-4,
                 V_A=1.0,
                 V_R=1e-2,
                 lambda_=0.01,  # Lefty inhibition rate
                 n=2, m=2, p=2,
                 K_A=100.0,
                 K_R=100.0,
                 K_P=100.0,
                 bc="auto_periodic_neumann",
                 amplitude=10,
                 radius=10.0,
                 r_init="constant",
                 r_value=0.0,
                 r_noise_range=(0.2, 0.6)):
        """
        1D activator–repressor PDE system with Hill-like activation and subtractive inhibition.

        Parameters
        ----------
        da : float
            Diffusion coefficient of the activator A.
        lambda_ : float
            Scaling factor for subtractive repression by R.
        k_a : float
            Decay rate of A.
        k_r : float
            Decay rate of R.
        n, m, p : int
            Hill exponents for A activation, R inhibition of A, and A activation of R.
        K_A, K_R, K_P : float
            Hill thresholds for activation and inhibition functions.
        bc : str
            Boundary condition.
        amplitude : float
            Peak value of the initial A hotspot.
        radius : float
            Std dev of Gaussian for initial A hotspot.
        r_init : str
            Initial state of R: "constant" or "random".
        r_value : float
            Constant value of R if r_init="constant".
        r_noise_range : tuple
            Min/max range for uniform noise if r_init="random".
        """
        super().__init__()
        self.da = da
        self.dr = dr
        self.lambda_ = lambda_
        self.V_A = V_A
        self.V_R = V_R
        self.k_a = k_a
        self.k_r = k_r
        self.n = n
        self.m = m
        self.p = p
        self.K_A = K_A
        self.K_R = K_R
        self.K_P = K_P
        self.bc = bc
        self.amplitude = amplitude
        self.radius = radius
        self.r_init = r_init
        self.r_value = r_value
        self.r_noise_range = r_noise_range

    def evolution_rate(self, state, t=0):
        A, R = state

        # Hill-like terms
        A_act = self.V_A * (A ** self.n) / (self.K_A ** self.n + A ** self.n)
        A_rep = self.lambda_ * A * (R ** self.m) / (self.K_R ** self.m + R ** self.m)
        R_act = self.V_R * (A ** self.p) / (self.K_P ** self.p + A ** self.p)

        # PDEs
        f_A = A_act - self.k_a * A - A_rep
        f_R = R_act - self.k_r * R

        dA_dt = self.da * A.laplace(self.bc) + f_A
        dR_dt = self.dr * R.laplace(self.bc) + f_R

        return FieldCollection([dA_dt, dR_dt])

    def get_state(self, grid):
        A = self.make_center_hotspot(grid).copy(label="Activator")
        R = self.initialize_repressor(grid).copy(label="Repressor")
        return FieldCollection([A, R])

    def make_center_hotspot(self, grid):
        field = ScalarField(grid, data=0.0)
        x = grid.axes_coords[0]
        x0 = 0.5 * (x[0] + x[-1])
        profile = self.amplitude * np.exp(-((x - x0) ** 2) / (2 * self.radius ** 2))
        field.data = profile
        return field

    def initialize_repressor(self, grid):
        if self.r_init == "constant":
            return ScalarField(grid, data=self.r_value)
        elif self.r_init == "random":
            noise = np.random.uniform(*self.r_noise_range, size=grid.shape)
            return ScalarField(grid, data=noise)
        else:
            raise ValueError(f"Unrecognized r_init method: {self.r_init}")