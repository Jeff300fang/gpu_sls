import jax.numpy as jnp

def combine_constraints(*funcs):
    """
    Combine multiple g_i(x,u,t) functions into one by concatenation.
    Each func must return a 1D array.
    """
    def constraints(x, u, t):
        parts = [f(x, u, t) for f in funcs]
        return jnp.concatenate(parts, axis=0)
    return constraints