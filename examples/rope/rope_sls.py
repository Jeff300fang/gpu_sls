from __future__ import annotations

import time
import argparse
import sys
from pathlib import Path
from typing import Any, Callable
import os

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
from gpu_sls.utils.sls_visual import get_trajectory_tubes, plot_tube_graph


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

        for center_xyz, radii_xyz in zip(
            ellipsoid_centers_xyz,
            ellipsoid_radii_xyz,
        ):
            q = (rope_nodes - center_xyz[None, :]) / radii_xyz[None, :]
            ellipsoid_constraints = 1.0 - jnp.sum(q**2, axis=1)
            all_constraints.append(ellipsoid_constraints)

        obstacle_constraints = jnp.concatenate(all_constraints, axis=0)

        left_end_x = rope_nodes[0, 0]
        right_end_x = rope_nodes[-1, 0]

        endpoint_constraints = jnp.array([
            left_end_x,
            -right_end_x,
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


def adversarial_w(
    i: int,
    *,
    n: int,
    num_nodes: int,
    dtype,
) -> jnp.ndarray:
    """
    Disturb one state at a time:

        disturbance 0 -> state 0 gets +1
        disturbance 1 -> state 0 gets -1
        disturbance 2 -> state 1 gets +1
        disturbance 3 -> state 1 gets -1
        ...
    """

    state_idx = i // 2
    sign = 1.0 if (i % 2 == 0) else -1.0

    state_idx = min(state_idx, n - 1)

    w = jnp.zeros((n,), dtype=dtype)
    w = w.at[state_idx].set(sign)

    return w


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--no-viz", action="store_true")
    args = parser.parse_args()

    print(f"JAX backend: {jax.default_backend()}")
    print(f"JAX devices: {jax.devices()}")

    server = None

    if not args.no_viz:
        import viser

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

    # ------------------------------------------------------------
    # Initial rope
    # ------------------------------------------------------------

    R = 0.06

    theta = jnp.linspace(
        -jnp.pi / 2,
        jnp.pi / 2,
        env.num_segments + 1,
    )

    s = theta / (jnp.pi / 2)

    theta_skew = theta + 0.08 * s**2

    xo = jnp.stack(
        (
            R * jnp.sin(theta_skew),
            0.004 * s,
            0.10 - R * jnp.cos(theta_skew),
        ),
        axis=1,
    )

    xo = xo.at[:, 0].add(0.006 * s**3)
    xo = xo.at[:, 2].add(0.02)

    xo = xo.at[:, 0].set(
        xo[:, 0] - jnp.mean(xo[:, 0])
    )

    state = env.state(xo=xo)

    control0 = jnp.zeros(
        env.control().shape,
        dtype=state.dtype,
    )

    n = state.shape[0]
    nu = control0.shape[0]

    N = 100
    dt = env.params.dt

    num_nodes = env.num_segments + 1
    segment_length = env.params.segment_length

    x_coords = jnp.arange(num_nodes) * segment_length
    x_coords = x_coords - jnp.mean(x_coords)

    forward_offset = 0.5

    goal_nodes = jnp.stack(
        (
            x_coords,
            jnp.full((num_nodes,), forward_offset),
            jnp.full((num_nodes,), 0.125),
        ),
        axis=1,
    )

    x_goal = env.state(xo=goal_nodes)

    reference = jnp.tile(
        x_goal[None, :],
        (N + 1, 1),
    )

    # ------------------------------------------------------------
    # Cost
    # ------------------------------------------------------------

    W = jnp.array(
        [1.0, 1.0, 5.0],
        dtype=state.dtype,
    )

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

    # ------------------------------------------------------------
    # Constraints
    # ------------------------------------------------------------

    vmax = 0.5

    u_min = -vmax * jnp.ones((nu,), dtype=state.dtype)
    u_max = vmax * jnp.ones((nu,), dtype=state.dtype)

    cylinder_radius = 0.10
    cylinder_z_max = 0.2

    cylinder_center_xy = jnp.array(
        [0.00, 0.25],
        dtype=state.dtype,
    )

    cylinder_center_xy_2 = cylinder_center_xy + jnp.array(
        [-0.5, 0.0],
        dtype=state.dtype,
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

    ellipsoid_centers_xyz = jnp.array(
        [
            [
                cylinder_center_xy[0],
                cylinder_center_xy[1],
                cylinder_z_max / 2.0,
            ],
            [
                cylinder_center_xy_2[0],
                cylinder_center_xy_2[1],
                cylinder_z_max / 2.0,
            ],
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

    constraints_all = make_control_and_ellipsoid_constraints(
        u_min,
        u_max,
        num_nodes=num_nodes,
        ellipsoid_centers_xyz=ellipsoid_centers_xyz,
        ellipsoid_radii_xyz=ellipsoid_radii_xyz,
    )

    obstacles = jnp.zeros((0, 3), dtype=state.dtype)

    E_mag = 0.075
    alpha_sim = E_mag * dt

    disturbance = make_constant_disturbance(
        n=n,
        alpha=alpha_sim,
    )

    nc = (
        2 * nu
        + ellipsoid_centers_xyz.shape[0] * num_nodes
        + 2
    )

    # ------------------------------------------------------------
    # Solver configs
    # ------------------------------------------------------------

    admm_cfg = ADMMConfig(
        eps_abs=5e-2,
        eps_rel=1e-2,
        rho_max=1e3,
        max_iterations=200,
        rho_update_frequency=25,
        initial_rho=10.0,
    )

    sls_cfg = SLSConfig(
        max_sls_iterations=2,
        sls_primal_tol=1e-2,
        enable_fastsls=True,
        initialize_nominal=True,
        max_initial_sqp_iterations=0,
        warm_start=False,
        rti=False,
    )

    sqp_cfg = SQPConfig(
        max_sqp_iterations=100,
        warm_start=False,
        feas_tol=1e-2,
        step_tol=1e-4,
        line_search=False,
    )

    X_in = jnp.tile(
        state[None, :],
        (N + 1, 1),
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
        X_in=X_in,
        U_in=jnp.tile(control0[None, :], (N, 1)),
    )

    # ------------------------------------------------------------
    # Solve ONCE
    # ------------------------------------------------------------

    start = time.time()

    (
        u0,
        X_pred,
        U_pred,
        V_pred,
        backoffs,
        Phi_x,
        Phi_u,
    ) = controller.run(
        x0=state,
        reference=reference,
        parameter=None,
    )

    elapsed = time.time() - start

    print(f"Single solve took {elapsed * 1e3:.2f} ms")

    # ------------------------------------------------------------
    # Rollouts using feedback policy
    # ------------------------------------------------------------

    N_RANDOM = 5
    N_ADV = 2 * n
    N_ROLLOUTS = N_RANDOM + N_ADV

    key = jax.random.PRNGKey(0)

    E_sim = alpha_sim * jnp.eye(
        n,
        dtype=state.dtype,
    )

    xs = np.full(
        (N_ROLLOUTS, N + 1, n),
        np.nan,
        dtype=np.float32,
    )

    us = np.full(
        (N_ROLLOUTS, N, nu),
        np.nan,
        dtype=np.float32,
    )

    SKIP_ROLLOUTS = {
        33,
    }

    for i in range(N_ROLLOUTS):
        if i in SKIP_ROLLOUTS:
            print(f"Skipping rollout {i}")
            continue

        x = state

        disturbance_history = [
            jnp.zeros((n,), dtype=state.dtype)
        ]

        xs[i, 0] = np.asarray(x)

        for k in range(N):

            # feedback correction
            disturbance_feedback = jnp.zeros(
                (nu,),
                dtype=state.dtype,
            )

            for j in range(k + 1):

                disturbance_feedback = (
                    disturbance_feedback
                    + Phi_u[k, j] @ disturbance_history[j]
                )

            u = U_pred[k] + disturbance_feedback

            # disturbance
            if i < N_RANDOM:

                key, subkey = jax.random.split(key)

                w = jax.random.normal(
                    subkey,
                    (n,),
                    dtype=state.dtype,
                )

                w = w / (
                    jnp.linalg.norm(w) + 1e-12
                )

            else:

                w = adversarial_w(
                    i - N_RANDOM,
                    n=n,
                    num_nodes=num_nodes,
                    dtype=state.dtype,
                )

            # rollout
            x_nom_next = env.step(x, u)
            x = x_nom_next + E_sim @ w

            disturbance_history.append(w)

            xs[i, k + 1] = np.asarray(x)
            us[i, k] = np.asarray(u)

            print(
                f"\rRollout {i+1}/{N_ROLLOUTS} | step {k+1}/{N}",
                end="",
            )

        print()

    print("Finished all rollouts.")

        # ------------------------------------------------------------
    # Plot disturbed trajectories vs tube bounds
    # ------------------------------------------------------------

    output_folder = "rope_outputs"
    os.makedirs(output_folder, exist_ok=True)

    tube = np.asarray(get_trajectory_tubes(Phi_x))  # (N+1, n)

    X_pred_np = np.asarray(X_pred)
    xs_np = np.asarray(xs)

    disturbed = xs_np

    lower = X_pred_np - tube
    upper = X_pred_np + tube

    plot_tube_graph(
        disturbed=disturbed,
        lower=lower,
        upper=upper,
        dt=dt,
        output_folder=output_folder,
        filename="rope_disturbed_vs_tube_size.png",
    )

    print(f"Saved tube plot to {output_folder}/rope_disturbed_vs_tube_size.png")

    # # ------------------------------------------------------------
    # # Visualization
    # # ------------------------------------------------------------

    # if server is not None:

    #     for k in range(N + 1):

    #         env.visualize(
    #             server,
    #             X_pred[k],
    #             control0,
    #         )

    #         time.sleep(0.03)

    #     while True:
    #         time.sleep(1.0)

    # ------------------------------------------------------------
    # Save
    # ------------------------------------------------------------

    # np.savez(
    #     "rope_single_solve_rollouts.npz",
    #     X_pred=np.asarray(X_pred),
    #     U_pred=np.asarray(U_pred),
    #     Phi_x=np.asarray(Phi_x),
    #     Phi_u=np.asarray(Phi_u),
    #     xs=xs,
    #     us=us,
    # )

    # print("Saved to rope_single_solve_rollouts.npz")


if __name__ == "__main__":
    main()