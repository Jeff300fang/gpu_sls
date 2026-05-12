#!/usr/bin/env python3
from __future__ import annotations

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"

for p in (REPO_ROOT, SRC_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import jax
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_debug_nans", True)

import jax.numpy as jnp
from trajax.optimizers import linearize

from simulation.environments import RopeEnv


def linearize_dynamics_once(
    env: RopeEnv,
    X: jnp.ndarray,
    U: jnp.ndarray,
):
    """
    Same Jacobian computation path as SQP:
        A_pad, B_pad = linearize(dynamics)(X, U_pad, t)
    """

    U_pad = jnp.pad(U, ((0, 1), (0, 0)))
    t = jnp.arange(X.shape[0])

    def dynamics(x, u, t):
        return env.step(x, u)

    A_pad, B_pad = linearize(dynamics)(X, U_pad, t)

    return A_pad[:-1], B_pad[:-1]


def main():
    print("backend:", jax.default_backend())
    print("devices:", jax.devices())

    import viser

    server = viser.ViserServer()
    server.scene.add_grid(name="/ground")

    env = RopeEnv(
        time_step=0.02,
        num_segments=10,
        rope_length=0.2,
        rope_diameter=0.004,
        youngs_modulus=1e5,
        mass_density=300,
        weld_ends=True,
        weld_stiffness=0.3,
    )

    # ---------------------------------------------------------
    # Initial rope configuration
    # ---------------------------------------------------------

    R = 0.06

    theta = jnp.linspace(
        -jnp.pi / 2,
        jnp.pi / 2,
        env.num_segments + 1,
    )

    # Clean U shape in x-z plane.
    # Ends high, middle low.
    xo = jnp.stack(
        (
            R * jnp.sin(theta),
            jnp.zeros_like(theta),
            0.12 - R * jnp.cos(theta),
        ),
        axis=1,
    )

    # Center in x.
    xo = xo.at[:, 0].set(xo[:, 0] - jnp.mean(xo[:, 0]))

    x0 = env.state(xo=xo)

    # Endpoint velocity control.
    u0 = jnp.zeros(
        env.control().shape,
        dtype=x0.dtype,
    )

    # ---------------------------------------------------------
    # Visualize rope
    # ---------------------------------------------------------
    env.visualize(server, x0, u0)

    # ---------------------------------------------------------
    # Build nominal trajectory
    # ---------------------------------------------------------

    N = 100

    X = jnp.tile(
        x0[None, :],
        (N + 1, 1),
    )

    U = jnp.tile(
        u0[None, :],
        (N, 1),
    )

    # ---------------------------------------------------------
    # Linearize
    # ---------------------------------------------------------

    start = time.time()

    A, B = linearize_dynamics_once(
        env,
        X,
        U,
    )

    elapsed = time.time() - start

    # ---------------------------------------------------------
    # Diagnostics
    # ---------------------------------------------------------

    print()
    print("=== SQP-style Dynamics Jacobians ===")
    print("x0 shape:", x0.shape)
    print("u0 shape:", u0.shape)

    print()
    print("X shape:", X.shape)
    print("U shape:", U.shape)

    print()
    print("A shape:", A.shape)
    print("B shape:", B.shape)

    print()
    print("finite A:", bool(jnp.all(jnp.isfinite(A))))
    print("finite B:", bool(jnp.all(jnp.isfinite(B))))

    print()
    print("A min/max:",
          float(jnp.min(A)),
          float(jnp.max(A)))

    print("B min/max:",
          float(jnp.min(B)),
          float(jnp.max(B)))

    print()
    print(f"linearization took {elapsed * 1e3:.2f} ms")

    print()
    print("Viser running...")
    print("Open the viewer URL shown above.")



if __name__ == "__main__":
    main()