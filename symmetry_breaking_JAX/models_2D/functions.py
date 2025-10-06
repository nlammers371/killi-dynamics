import jax.numpy as jnp
import jax

def hill01(x: jnp.ndarray, k,  n: int) -> jnp.ndarray:
    x = jnp.clip(x, 0.0, 1e6)
    k = jnp.clip(k, 0.0, 1e6)
    xn = jnp.power(x, n)
    kn = jnp.power(k, n)
    return xn / (kn + xn)

def binding_free_N(N, L, K_I, KI_tol=1e-6):
    KI = jnp.asarray(K_I, dtype=N.dtype)

    def strong_binding(_):
        return jnp.maximum(N - L, 0.0)

    def quadratic(_):
        S = N + L + KI
        S_safe = jnp.maximum(S, 1e-12)
        x = 4.0 * (N / S_safe) * (L / S_safe)
        x = jnp.clip(x, 0.0, 1.0)
        sqrt_term = jnp.sqrt(1.0 - x)
        C = (2.0 * (N / S_safe) * L) / (1.0 + sqrt_term)
        return jnp.clip(N - C, 0.0, N)

    return jax.lax.cond(KI <= KI_tol, strong_binding, quadratic, operand=None)