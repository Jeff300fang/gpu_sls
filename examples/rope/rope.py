from __future__ import annotations

import time
import argparse
import sys
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
for path in (REPO_ROOT, SRC_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import jax
import jax.numpy as jnp
from jax import config
import numpy as np
from simulation.environments import RopeEnv

from gpu_sls.gpu_admm import ADMMConfig
from gpu_sls.gpu_sls import SLSConfig
from gpu_sls.gpu_sqp import SQPConfig
from gpu_sls.generic_mpc import GenericMPC, MPCConfig

config.update("jax_enable_x64", False)


def render_cylinder_obstacle(
    server,
    *,
    center_xy,
    radius: float,
    z_min: float,
    z_max: float,
    n_sides: int = 48,
):
    theta = np.linspace(0.0, 2.0 * np.pi, n_sides, endpoint=False)

    bottom = np.stack(
        [
            center_xy[0] + radius * np.cos(theta),
            center_xy[1] + radius * np.sin(theta),
            np.full_like(theta, z_min),
        ],
        axis=1,
    )

    top = bottom.copy()
    top[:, 2] = z_max

    vertices = np.vstack([bottom, top])

    faces = []
    for k in range(n_sides):
        k_next = (k + 1) % n_sides
        faces.append([k, k_next, n_sides + k])
        faces.append([k_next, n_sides + k_next, n_sides + k])

    faces = np.array(faces, dtype=np.int32)

    server.scene.add_mesh_simple(
        name="/obstacles/cylinder",
        vertices=vertices,
        faces=faces,
        color=(1.0, 0.4, 0.0),
        opacity=0.35,
    )


def make_control_box_constraints(
    u_min: jnp.ndarray,
    u_max: jnp.ndarray,
) -> Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    def constraints(x: jnp.ndarray, u: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        return jnp.concatenate([u - u_max, u_min - u], axis=0)

    return constraints


def make_constant_disturbance(n: int, alpha: float):
    def disturbance(X_prefix: jnp.ndarray) -> jnp.ndarray:
        T = X_prefix.shape[0]
        E0 = alpha * jnp.eye(n, dtype=X_prefix.dtype)
        return jnp.broadcast_to(E0, (T, n, n))

    return disturbance

def make_control_and_cylinder_constraints(
    u_min: jnp.ndarray,
    u_max: jnp.ndarray,
    *,
    num_nodes: int,
    cylinder_center_xy: jnp.ndarray,
    cylinder_radius: float,
    cylinder_z_min: float,
    cylinder_z_max: float,
) -> Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    def constraints(x: jnp.ndarray, u: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        control_constraints = jnp.concatenate([u - u_max, u_min - u], axis=0)

        # First 3*num_nodes entries are rope node xyz positions.
        rope_nodes = x[: 3 * num_nodes].reshape((num_nodes, 3))

        node_xy = rope_nodes[:, 0:2]
        node_z = rope_nodes[:, 2]

        radial_dist = jnp.linalg.norm(node_xy - cylinder_center_xy[None, :], axis=1)

        # Constraint convention: values must be <= 0.
        # Positive means the node is inside the cylinder radius.
        cylinder_violation = cylinder_radius - radial_dist

        # Only activate obstacle constraint for nodes inside the cylinder height.
        within_height = (node_z >= cylinder_z_min) & (node_z <= cylinder_z_max)
        cylinder_constraints = jnp.where(
            within_height,
            cylinder_violation,
            -1.0,
        )

        return jnp.concatenate([control_constraints, cylinder_constraints], axis=0)

    return constraints


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-viz", action="store_true", help="Run without the viser visualizer.")
    parser.add_argument("--steps", type=int, default=150, help="Number of MPC steps to run.")
    args = parser.parse_args()

    print(f"JAX backend: {jax.default_backend()}")
    print(f"JAX devices: {jax.devices()}")

    server = None
    if not args.no_viz:
        try:
            import viser
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "The rope visualizer requires `viser`. Install it or run with `--no-viz`."
            ) from exc
        server = viser.ViserServer()
        _ = server.scene.add_grid(name="ground")

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

    # Initial rope shape
    R = 0.06
    theta = jnp.arange(env.num_segments + 1) * (env.params.segment_length / R)

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

    # R = 0.06

    # # Parameter along the U arc.
    # theta = jnp.linspace(-jnp.pi / 2, jnp.pi / 2, env.num_segments + 1)

    # # U shape in x-z.
    # # x varies left/right
    # # z forms the U curvature
    # # y stays constant so the rope lies in the x-z plane
    # xo = jnp.stack(
    #     (
    #         R * jnp.sin(theta),                 # x
    #         jnp.zeros_like(theta),             # y
    #         0.10 - R * jnp.cos(theta),         # z
    #     ),
    #     axis=1,
    # )

    # Center the rope in x.
    # xo = xo.at[:, 0].set(
    #     xo[:, 0] - jnp.mean(xo[:, 0])
    # )
    state = env.state(xo=xo)

    # NEW:
    # Control is now endpoint velocity, not endpoint position.
    # u = [v_left_x, v_left_y, v_left_z, v_right_x, v_right_y, v_right_z]
    control0 = jnp.zeros(env.control().shape, dtype=state.dtype)

    n = state.shape[0]
    nu = control0.shape[0]

    N = 20
    dt = env.params.dt

    # Target rope shape
    num_nodes = env.num_segments + 1
    segment_length = env.params.segment_length

    x_coords = jnp.arange(num_nodes) * segment_length
    x_coords = x_coords - jnp.mean(x_coords)

    forward_offset = 0.5

    goal_nodes = jnp.stack(
        (
            x_coords,
            jnp.full((num_nodes,), forward_offset),
            jnp.full((num_nodes,), 0.10),
        ),
        axis=1,
    )

    # Because env.state now includes weld target state xw,
    # this also sets target weld positions to the target rope endpoints.
    x_goal = env.state(xo=goal_nodes)

    reference = jnp.tile(x_goal[None, :], (N + 1, 1))

    # Cost weights.
    # q_state: rope + weld-target state tracking
    # r_control: velocity magnitude penalty
    # r_delta: left/right endpoint velocity mismatch penalty
    W = jnp.array([1.0, 1.0, 1.0], dtype=state.dtype)

    cfg = MPCConfig(
        n=n,
        nu=nu,
        N=N,
        W=W,
        u_ref=control0,
        dt=dt,
    )

    def dynamics(
        x: jnp.ndarray,
        u: jnp.ndarray,
        t: jnp.ndarray,
        *,
        parameter: Any,
    ) -> jnp.ndarray:
        return env.step(x, u)

    def cost(W, reference, x, u, t):
        q_state, r_control, r_delta = W

        x_ref = reference[t]
        u_ref = control0

        state_err = x - x_ref
        control_err = u - u_ref

        return (
            q_state * jnp.sum(state_err**2)
            + r_control * jnp.sum(control_err**2)
            + r_delta * jnp.sum((u[3:6] - u[0:3])**2)
        )

    # NEW:
    # Control limits are now velocity limits in m/s.
    vmax = 0.5
    u_min = -vmax * jnp.ones((nu,), dtype=state.dtype)
    u_max = vmax * jnp.ones((nu,), dtype=state.dtype)

    cylinder_center_xy = jnp.array([0.05, 0.25], dtype=state.dtype)
    cylinder_radius = 0.10
    cylinder_z_min = 0.0
    cylinder_z_max = 0.30

    constraints_all = make_control_and_cylinder_constraints(
        u_min,
        u_max,
        num_nodes=num_nodes,
        cylinder_center_xy=cylinder_center_xy,
        cylinder_radius=cylinder_radius,
        cylinder_z_min=cylinder_z_min,
        cylinder_z_max=cylinder_z_max,
    )

    # obstacles = jnp.array(
    #     [[cylinder_center_xy[0], cylinder_center_xy[1], cylinder_radius]],
    #     dtype=state.dtype,
    # )

    obstacles = jnp.zeros((0, 3), dtype=state.dtype)
    E_mag = 0.03
    alpha_sim = E_mag * dt
    nc = 2 * nu + num_nodes
    disturbance = make_constant_disturbance(n=n, alpha=alpha_sim)

    admm_cfg = ADMMConfig(
        eps_abs=1e-2,
        eps_rel=1e-2,
        rho_max=1e3,
        max_iterations=200,
        rho_update_frequency=25,
        initial_rho=10.0,
    )

    sls_cfg = SLSConfig(
        max_sls_iterations=2,
        sls_primal_tol=1e-2,
        enable_fastsls=False,
        initialize_nominal=True,
        max_initial_sqp_iterations=1,
        warm_start=False,
        rti=False,
    )

    sqp_cfg = SQPConfig(
        max_sqp_iterations=0,
        warm_start=False,
        feas_tol=1e-2,
        step_tol=1e-4,
        line_search=True,
    )

    controller = GenericMPC(
        sls_cfg,
        sqp_cfg,
        admm_cfg,
        config=cfg,
        dynamics=dynamics,
        constraints=constraints_all,
        obstacles=obstacles,
        cost=cost,
        num_constraints=nc,
        disturbance=disturbance,
        shift=1,
        X_in=jnp.tile(state[None, :], (N + 1, 1)),
        U_in=jnp.tile(control0[None, :], (N, 1)),
    )

    if server is not None:
        render_cylinder_obstacle(
            server,
            center_xy=cylinder_center_xy,
            radius=0.10,
            z_min=0.0,
            z_max=0.30,
        )
        env.visualize(server, state, control0)

    for i in range(args.steps):
        start = time.time()

        u0, X_pred, U_pred, V_pred, backoffs, Phi_x, Phi_u = controller.run(
            x0=state,
            reference=reference,
            parameter=None,
        )

        state = env.step(state, u0)

        elapsed = time.time() - start
        print(f"\rMPC step took {elapsed * 1e3:.2f} ms", end="")

        if jnp.isnan(state).any():
            raise RuntimeError("NaN occurred in rope state")

        if server is not None:
            env.visualize(server, state, u0)

        wait = dt - elapsed
        if wait > 0:
            time.sleep(wait)


if __name__ == "__main__":
    main()
