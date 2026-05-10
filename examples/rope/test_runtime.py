from __future__ import annotations
import os

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.0"

import time
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"

for path in (REPO_ROOT, SRC_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import jax
import jax.numpy as jnp
from jax import config

config.update("jax_enable_x64", False)

from simulation.environments import RopeEnv


def block_until_ready_tree(x):
    jax.tree_util.tree_map(lambda y: y.block_until_ready(), x)


def time_fn(name, fn, num_runs):
    out = fn()
    block_until_ready_tree(out)

    start = time.perf_counter()

    for _ in range(num_runs):
        out = fn()

    block_until_ready_tree(out)

    end = time.perf_counter()

    print(f"{name}: {(end - start) * 1000 / num_runs:.3f} ms")


def main():
    print(f"JAX backend: {jax.default_backend()}")
    print(f"JAX devices: {jax.devices()}")

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

    # ------------------------------------------------------------
    # Initial rope state
    # ------------------------------------------------------------
    R = 0.06

    theta = jnp.arange(env.num_segments + 1) * (
        env.params.segment_length / R
    )

    xo = jnp.vstack(
        (
            R * jnp.sin(theta),
            R * jnp.cos(theta),
            jnp.linspace(0.05, 0.15, env.num_segments + 1),
        )
    )

    xo = xo.at[0:2].set(
        xo[0:2] - jnp.mean(xo[0:2], axis=1, keepdims=True)
    ).T

    x0 = env.state(xo=xo)

    u0 = jnp.zeros(
        env.control().shape,
        dtype=x0.dtype,
    )

    print("x0 shape:", x0.shape)
    print("u0 shape:", u0.shape)

    # ------------------------------------------------------------
    # Dynamics
    # ------------------------------------------------------------
    def dynamics(x, u):
        return env.step(x, u)

    # ------------------------------------------------------------
    # Scalar cost for Hessian test
    # ------------------------------------------------------------
    def scalar_cost(x, u):
        xn = dynamics(x, u)
        return jnp.sum(xn**2)

    # ------------------------------------------------------------
    # JIT functions
    # ------------------------------------------------------------
    step_jit = jax.jit(dynamics)

    jac_fn = jax.jit(
        jax.jacrev(
            dynamics,
            argnums=(0, 1),
        )
    )

    grad_x = jax.grad(scalar_cost, argnums=0)
    grad_u = jax.grad(scalar_cost, argnums=1)

    # Reverse-over-reverse Hessian blocks.
    # Avoids jax.hessian because env.step uses custom_vjp.
    hxx_fn = jax.jit(jax.jacrev(grad_x, argnums=0))
    huu_fn = jax.jit(jax.jacrev(grad_u, argnums=1))
    hxu_fn = jax.jit(jax.jacrev(grad_x, argnums=1))

    # ------------------------------------------------------------
    # Raw env.step timing
    # ------------------------------------------------------------
    print("\n========== env.step ==========")

    time_fn(
        "raw env.step",
        lambda: dynamics(x0, u0),
        num_runs=100,
    )

    print("\nCompiling jitted env.step...")

    start = time.perf_counter()

    x_next = step_jit(x0, u0)
    x_next.block_until_ready()

    end = time.perf_counter()

    print(f"jitted env.step compile + first run: {(end - start) * 1000:.3f} ms")

    time_fn(
        "jitted env.step runtime",
        lambda: step_jit(x0, u0),
        num_runs=1000,
    )

    # ------------------------------------------------------------
    # Jacobian timing
    # ------------------------------------------------------------
    print("\n========== Jacobian ==========")

    print("Compiling Jacobian...")

    start = time.perf_counter()

    A, B = jac_fn(x0, u0)

    A.block_until_ready()
    B.block_until_ready()

    end = time.perf_counter()

    print(f"Jacobian compile + first run: {(end - start) * 1000:.3f} ms")
    print("A shape:", A.shape)
    print("B shape:", B.shape)

    time_fn(
        "jitted Jacobian runtime",
        lambda: jac_fn(x0, u0),
        num_runs=100,
    )

    # ------------------------------------------------------------
    # Hessian timing
    # ------------------------------------------------------------
    print("\n========== Hessian blocks ==========")

    print("Compiling Hessian blocks...")

    start = time.perf_counter()

    Hxx = hxx_fn(x0, u0)
    Huu = huu_fn(x0, u0)
    Hxu = hxu_fn(x0, u0)

    Hxx.block_until_ready()
    Huu.block_until_ready()
    Hxu.block_until_ready()

    end = time.perf_counter()

    print(f"Hessian compile + first run: {(end - start) * 1000:.3f} ms")
    print("Hxx shape:", Hxx.shape)
    print("Huu shape:", Huu.shape)
    print("Hxu shape:", Hxu.shape)

    time_fn(
        "jitted Hxx runtime",
        lambda: hxx_fn(x0, u0),
        num_runs=20,
    )

    time_fn(
        "jitted Huu runtime",
        lambda: huu_fn(x0, u0),
        num_runs=20,
    )

    time_fn(
        "jitted Hxu runtime",
        lambda: hxu_fn(x0, u0),
        num_runs=20,
    )

    time_fn(
        "jitted Hessian blocks runtime",
        lambda: (
            hxx_fn(x0, u0),
            huu_fn(x0, u0),
            hxu_fn(x0, u0),
        ),
        num_runs=20,
    )


if __name__ == "__main__":
    main()