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
jax.config.update("jax_enable_x64", False)

import jax.numpy as jnp
import numpy as np
from simulation.environments import RopeEnv

from gpu_sls.gpu_admm import ADMMConfig
from gpu_sls.gpu_sls import SLSConfig
from gpu_sls.gpu_sqp import SQPConfig
from gpu_sls.generic_mpc import GenericMPC, MPCConfig


def render_cylinder_obstacle(
    server,
    *,
    name: str,
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
        name=name,
        vertices=vertices,
        faces=faces,
        color=(1.0, 0.4, 0.0),
        opacity=0.35,
    )


def make_control_and_cone_constraints(
    u_min: jnp.ndarray,
    u_max: jnp.ndarray,
    *,
    num_nodes: int,
    cone_centers_xy: jnp.ndarray,
    cone_radius: float,
    cone_z_top: float,
    clearance: float = 0.02,
    cone_extra_height: float = 0.10,
):
    slope = cone_extra_height / cone_radius

    def constraints(x, u, t):
        control_constraints = jnp.concatenate([u - u_max, u_min - u], axis=0)

        rope_nodes = x[: 3 * num_nodes].reshape((num_nodes, 3))
        node_xy = rope_nodes[:, 0:2]
        node_z = rope_nodes[:, 2]

        all_constraints = []

        for center_xy in cone_centers_xy:
            radial_dist = jnp.linalg.norm(node_xy - center_xy[None, :], axis=1)

            z_required = (
                cone_z_top
                + clearance
                + slope * jnp.maximum(cone_radius - radial_dist, 0.0)
            )

            cone_constraints = z_required - node_z

            cone_constraints = jnp.where(
                radial_dist <= cone_radius,
                cone_constraints,
                -1.0,
            )

            all_constraints.append(cone_constraints)

        obstacle_constraints = jnp.concatenate(all_constraints, axis=0)

        left_end_x = rope_nodes[0, 0]
        right_end_x = rope_nodes[-1, 0]

        endpoint_constraints = jnp.array([
            left_end_x,
            -right_end_x,
        ])

        return jnp.concatenate(
            [control_constraints, obstacle_constraints, endpoint_constraints],
            axis=0,
        )

    return constraints

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

def make_control_and_ellipsoid_constraints(
    u_min: jnp.ndarray,
    u_max: jnp.ndarray,
    *,
    num_nodes: int,
    ellipsoid_centers_xyz: jnp.ndarray,
    ellipsoid_radii_xyz: jnp.ndarray,
) -> Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    def constraints(x: jnp.ndarray, u: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        control_constraints = jnp.concatenate([u - u_max, u_min - u], axis=0)

        rope_nodes = x[: 3 * num_nodes].reshape((num_nodes, 3))

        all_constraints = []

        for center_xyz, radii_xyz in zip(ellipsoid_centers_xyz, ellipsoid_radii_xyz):
            q = (rope_nodes - center_xyz[None, :]) / radii_xyz[None, :]
            ellipsoid_constraints = 1.0 - jnp.sum(q**2, axis=1)
            all_constraints.append(ellipsoid_constraints)

        obstacle_constraints = jnp.concatenate(all_constraints, axis=0)

        left_end_x = rope_nodes[0, 0]
        right_end_x = rope_nodes[-1, 0]

        endpoint_constraints = jnp.array([
            left_end_x,      # left end <= 0
            -right_end_x,    # right end >= 0
        ])

        return jnp.concatenate(
            [
                control_constraints,
                obstacle_constraints,
                endpoint_constraints,
            ],
            axis=0,
        )

    return constraints

def make_control_and_cylinder_constraints(
    u_min: jnp.ndarray,
    u_max: jnp.ndarray,
    *,
    num_nodes: int,
    cylinder_centers_xy: jnp.ndarray,
    cylinder_radius: float,
    cylinder_z_min: float,
    cylinder_z_max: float,
) -> Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    def constraints(x: jnp.ndarray, u: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        control_constraints = jnp.concatenate([u - u_max, u_min - u], axis=0)

        rope_nodes = x[: 3 * num_nodes].reshape((num_nodes, 3))

        node_xy = rope_nodes[:, 0:2]
        node_z = rope_nodes[:, 2]

        all_constraints = []

        for center_xy in cylinder_centers_xy:
            radial_dist = jnp.linalg.norm(
                node_xy - center_xy[None, :],
                axis=1,
            )

            cylinder_violation = cylinder_radius - radial_dist

            within_height = (
                (node_z >= cylinder_z_min)
                & (node_z <= cylinder_z_max)
            )

            cylinder_constraints = jnp.where(
                within_height,
                cylinder_violation,
                -1.0,
            )

            all_constraints.append(cylinder_constraints)

        obstacle_constraints = jnp.concatenate(all_constraints, axis=0)

        # return jnp.concatenate(
        #     [control_constraints, obstacle_constraints],
        #     axis=0,
        # )
        left_end_x = rope_nodes[0, 0]
        right_end_x = rope_nodes[-1, 0]

        endpoint_constraints = jnp.array([
            left_end_x,      # left end <= 0
            -right_end_x,    # right end >= 0
        ])

        return jnp.concatenate(
            [
                control_constraints,
                obstacle_constraints,
                endpoint_constraints,
            ],
            axis=0,
        )


    return constraints

def make_projected_over_cylinder_X_in(
    x0: jnp.ndarray,
    x_goal: jnp.ndarray,
    *,
    N: int,
    num_nodes: int,
    cylinder_centers_xy: jnp.ndarray,
    cylinder_radius: float,
    cylinder_z_max: float,
    clearance: float = 0.03,
) -> jnp.ndarray:
    """
    Straight-line state trajectory from x0 to x_goal.

    For rope nodes whose xy position lies inside any cylinder radius,
    project their z position to the top of the cylinder.
    """

    alphas = jnp.linspace(0.0, 1.0, N + 1)

    X = (1.0 - alphas[:, None]) * x0[None, :] + alphas[:, None] * x_goal[None, :]

    rope_flat = X[:, : 3 * num_nodes]
    rope_nodes = rope_flat.reshape((N + 1, num_nodes, 3))

    node_xy = rope_nodes[:, :, 0:2]

    # shape: (N + 1, num_nodes, num_cylinders)
    d_xy = jnp.linalg.norm(
        node_xy[:, :, None, :] - cylinder_centers_xy[None, None, :, :],
        axis=-1,
    )

    inside_any_cylinder = jnp.any(d_xy <= cylinder_radius, axis=-1)

    z_projected = cylinder_z_max + clearance

    rope_nodes = rope_nodes.at[:, :, 2].set(
        jnp.where(
            inside_any_cylinder,
            jnp.maximum(rope_nodes[:, :, 2], z_projected),
            rope_nodes[:, :, 2],
        )
    )

    X = X.at[:, : 3 * num_nodes].set(rope_nodes.reshape((N + 1, 3 * num_nodes)))

    return X


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-viz", action="store_true", help="Run without the viser visualizer.")
    parser.add_argument("--steps", type=int, default=200, help="Number of MPC steps to run.")
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
    # R = 0.06
    # theta = jnp.arange(env.num_segments + 1) * (env.params.segment_length / R)

    # xo = jnp.vstack(
    #     (
    #         R * jnp.sin(theta),
    #         R * jnp.cos(theta),
    #         jnp.linspace(0.05, 0.15, env.num_segments + 1),
    #     )
    # )

    # xo = xo.at[0:2].set(
    #     xo[0:2] - jnp.mean(xo[0:2], axis=1, keepdims=True)
    # ).T

    R = 0.06

    theta = jnp.linspace(-jnp.pi / 2, jnp.pi / 2, env.num_segments + 1)
    s = theta / (jnp.pi / 2)

    theta_skew = theta + 0.08 * s**2

    xo = jnp.stack(
        (
            R * jnp.sin(theta_skew),
            0.004 * s,                      # small out-of-plane asymmetry
            0.10 - R * jnp.cos(theta_skew), # use same skewed theta
        ),
        axis=1,
    )

    xo = xo.at[:, 0].add(0.006 * s**3)
    xo = xo.at[:, 2].add(0.02)

    xo = xo.at[:, 0].set(xo[:, 0] - jnp.mean(xo[:, 0]))
    state = env.state(xo=xo)

    # NEW:
    # Control is now endpoint velocity, not endpoint position.
    # u = [v_left_x, v_left_y, v_left_z, v_right_x, v_right_y, v_right_z]
    control0 = jnp.zeros(env.control().shape, dtype=state.dtype)

    n = state.shape[0]
    nu = control0.shape[0]

    N = 100
    dt = env.params.dt

    # Target rope shape
    num_nodes = env.num_segments + 1
    segment_length = env.params.segment_length

    x_coords = jnp.arange(num_nodes) * segment_length
    x_coords = x_coords - jnp.mean(x_coords)

    forward_offset = 0.5

    cylinder_z_max = 0.2

    goal_nodes = jnp.stack(
        (
            x_coords,
            jnp.full((num_nodes,), forward_offset),
            jnp.full((num_nodes,), 0.1),
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
    W = jnp.array([1.0, 1.0, 5.0], dtype=state.dtype)

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

    cylinder_radius = 0.10
    cylinder_z_min = 0.0
    cylinder_z_max = 0.2

    # Original obstacle
    cylinder_center_xy = jnp.array([0.00, 0.25], dtype=state.dtype)

    # Second obstacle 0.3m to the left (negative x)
    cylinder_center_xy_2 = cylinder_center_xy + jnp.array(
        [-0.5, 0.0],
        dtype=state.dtype,
    )

    cylinder_centers_xy = jnp.stack(
        [
            cylinder_center_xy,
            cylinder_center_xy_2,
        ],
        axis=0,
    )

    # constraints_all = make_control_and_cylinder_constraints(
    #     u_min,
    #     u_max,
    #     num_nodes=num_nodes,
    #     cylinder_centers_xy=cylinder_centers_xy,
    #     cylinder_radius=cylinder_radius,
    #     cylinder_z_min=cylinder_z_min,
    #     cylinder_z_max=cylinder_z_max,
    # )
    ellipsoid_centers_xyz = jnp.array(
    [
        [cylinder_center_xy[0], cylinder_center_xy[1], cylinder_z_max / 2.0],
        [cylinder_center_xy_2[0], cylinder_center_xy_2[1], cylinder_z_max / 2.0],
    ],
    dtype=state.dtype,
)

    ellipsoid_radii_xyz = jnp.array(
        [
            [cylinder_radius, cylinder_radius, 0.16],
            [cylinder_radius, cylinder_radius, 0.16],
        ],
        dtype=state.dtype,
    )

    # constraints_all = make_control_and_ellipsoid_constraints(
    #     u_min,
    #     u_max,
    #     num_nodes=num_nodes,
    #     ellipsoid_centers_xyz=ellipsoid_centers_xyz,
    #     ellipsoid_radii_xyz=ellipsoid_radii_xyz,
    # )
    constraints_all = make_control_and_cone_constraints(
        u_min,
        u_max,
        num_nodes=num_nodes,
        cone_centers_xy=cylinder_centers_xy,
        cone_radius=cylinder_radius,
        cone_z_top=cylinder_z_max,
        clearance=0.03,
        cone_extra_height=0.10,
    )

    nc = 2 * nu + cylinder_centers_xy.shape[0] * num_nodes + 2

    # obstacles = jnp.array(
    #     [[cylinder_center_xy[0], cylinder_center_xy[1], cylinder_radius]],
    #     dtype=state.dtype,
    # )

    obstacles = jnp.zeros((0, 3), dtype=state.dtype)
    E_mag = 0.03
    alpha_sim = E_mag * dt
    # nc = 2 * nu + 2 * num_nodes + 2
    # nc = 2 * nu + ellipsoid_centers_xyz.shape[0] * num_nodes + 2
    disturbance = make_constant_disturbance(n=n, alpha=alpha_sim)

    admm_cfg = ADMMConfig(
        eps_abs=5e-2,
        eps_rel=1e-2,
        rho_max=1e3,
        max_iterations=1000,
        rho_update_frequency=25,
        initial_rho=10.0,
    )

    sls_cfg = SLSConfig(
        max_sls_iterations=2,
        sls_primal_tol=1e-2,
        enable_fastsls=False,
        initialize_nominal=True,
        max_initial_sqp_iterations=0,
        warm_start=False,
        rti=False,
    )

    sqp_cfg = SQPConfig(
        max_sqp_iterations=1,
        warm_start=False,
        feas_tol=1e-2,
        step_tol=1e-4,
        line_search=True,
    )

    X_in = jnp.tile(state[None, :], (N + 1, 1))

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
        # X_in=jnp.tile(state[None, :], (N + 1, 1)),
        X_in=X_in,
        U_in=jnp.tile(control0[None, :], (N, 1)),
    )

    if server is not None:
        render_cylinder_obstacle(
            server,
            name="/obstacles/cylinder_0",
            center_xy=cylinder_center_xy,
            radius=0.10,
            z_min=0.0,
            z_max=cylinder_z_max,
        )

        render_cylinder_obstacle(
            server,
            name="/obstacles/cylinder_1",
            center_xy=cylinder_center_xy_2,
            radius=0.10,
            z_min=0.0,
            z_max=cylinder_z_max,
        )

        env.visualize(server, state, control0)

    for i in range(args.steps):
        start = time.time()

        u0, X_pred, U_pred, V_pred, backoffs, Phi_x, Phi_u = controller.run(
            x0=state,
            reference=reference,
            parameter=None,
        )

        elapsed = time.time() - start

        state = env.step(state, u0)

        print(f"\rMPC step took {elapsed * 1e3:.2f} ms")

        if jnp.isnan(state).any():
            raise RuntimeError("NaN occurred in rope state")

        if server is not None:
            env.visualize(server, state, u0)

        wait = dt - elapsed
        if wait > 0:
            time.sleep(wait)


if __name__ == "__main__":
    main()
