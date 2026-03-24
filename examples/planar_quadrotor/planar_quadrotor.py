from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Callable

import jax
import jax.numpy as jnp
from jax import config

import numpy as np

from gpu_sls.gpu_admm import ADMMConfig
from gpu_sls.gpu_sls import SLSConfig
from gpu_sls.gpu_sqp import SQPConfig
from gpu_sls.generic_mpc import GenericMPC, MPCConfig
from gpu_sls.utils.constraint_utils import (
    combine_constraints,
    make_control_box_constraints,
    make_state_box_constraints,
    make_constant_disturbance,
)
from gpu_sls.utils.sls_visual import get_trajectory_tubes
from visualize_experiment import plot_rollouts_tubes_centers, plot_tube_graph_quadrotor

config.update("jax_enable_x64", False)
config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
config.update("jax_persistent_cache_min_compile_time_secs", 0)
config.update("jax_persistent_cache_min_entry_size_bytes", -1)
config.update(
    "jax_persistent_cache_enable_xla_caches",
    "xla_gpu_per_fusion_autotune_cache_dir",
)

# -----------------------------
# Goal stopping config
# -----------------------------
GOAL_TOL = 0.25  # meters in px, py

# -----------------------------
# Planar quadrotor parameters
# x = [px, py, phi, vx, vy, vphi]
# u = [u1, u2]  (individual rotor thrusts)
# -----------------------------
GRAVITY = 9.81
MASS_NOM = 2.0576       # nominal mass [kg]
ARM_LENGTH = 0.25       # half-distance between rotors [m]
INERTIA = 0.01          # moment of inertia [kg·m²]

NUM_RANDOM = 5
NUM_ADV = 26


def wrap_to_pi(a: jnp.ndarray) -> jnp.ndarray:
    return (a + jnp.pi) % (2.0 * jnp.pi) - jnp.pi


def reached_goal_xy(x: jnp.ndarray, x_goal: jnp.ndarray, tol: float = GOAL_TOL) -> jnp.bool_:
    dpos = x[:2] - x_goal[:2]
    return (dpos @ dpos) <= (tol * tol)


# -----------------------------
# Dynamics
# -----------------------------
def dynamics_planar_quad(
    x: jnp.ndarray,
    u: jnp.ndarray,
    t: jnp.ndarray,
    *,
    parameter: float,   # dt — mass uncertainty folded into MASS_NOM here
) -> jnp.ndarray:
    """
    Planar quadrotor Euler-discretised dynamics.

    State:  x = [px, py, phi, vx, vy, vphi]   (6,)
    Input:  u = [u1, u2]  — individual rotor thrusts   (2,)
    Param:  parameter = dt

    Continuous-time EOMs:
        px_dot   = vx
        py_dot   = vy
        phi_dot  = vphi
        vx_dot   = -(u1 + u2) * sin(phi) / m
        vy_dot   =  (u1 + u2) * cos(phi) / m  -  g
        vphi_dot =  (u2 - u1) * L / J
    """
    dt = parameter
    mass = MASS_NOM

    px, py, phi, vx, vy, vphi = x
    u1, u2 = u[0], u[1]

    total_thrust = u1 + u2
    torque = (u2 - u1) * ARM_LENGTH

    cphi = jnp.cos(phi)
    sphi = jnp.sin(phi)

    dpx   = vx
    dpy   = vy
    dphi  = vphi
    dvx   = -total_thrust * sphi / mass
    dvy   =  total_thrust * cphi / mass - GRAVITY
    dvphi =  torque / INERTIA

    x_next = x + dt * jnp.stack([dpx, dpy, dphi, dvx, dvy, dvphi])
    x_next = x_next.at[2].set(wrap_to_pi(x_next[2]))  # wrap phi
    return x_next


def planar_quad_step_with_disturbance(
    key: jax.Array,
    x: jnp.ndarray,      # (6,)
    u: jnp.ndarray,      # (2,)
    E: jnp.ndarray,      # (6,6)
    dt: float,
    i: int,
) -> tuple[jax.Array, jnp.ndarray, jnp.ndarray]:
    """
    x_{k+1} = f(x_k, u_k) + E w,  ||w||_2 <= 1
    Mirrors the adversarial disturbance schedule from the 3D version.
    """
    x_nom = dynamics_planar_quad(x, u, None, parameter=dt)

    # Random disturbance in unit ball
    key, key_dir, key_rad = jax.random.split(key, 3)

    nx = x.shape[0]  # 6
    z = jax.random.normal(key_dir, (nx,), dtype=x.dtype)
    z = z / (jnp.linalg.norm(z) + jnp.asarray(1e-12, dtype=x.dtype))

    n_f = jnp.asarray(nx, dtype=x.dtype)
    uu = jax.random.uniform(key_rad, (), dtype=x.dtype)
    r = uu ** (1.0 / n_f)
    w = r * z

    # Deterministic adversarial directions (adapted for 6-dim state)
    start = i - NUM_RANDOM + 5
    if 5 <= start <= 10:           # +e_i  (6 dims)
        idx = start - 5
        w = jnp.zeros((nx,), dtype=x.dtype).at[idx].set(1.0)
    if 11 <= start <= 16:          # -e_i  (6 dims)
        idx = start - 11
        w = jnp.zeros((nx,), dtype=x.dtype).at[idx].set(-1.0)
    if start == 17:
        w = jnp.ones((nx,), dtype=x.dtype) / jnp.sqrt(jnp.asarray(float(nx), dtype=x.dtype))
    if start == 18:
        w = -jnp.ones((nx,), dtype=x.dtype) / jnp.sqrt(jnp.asarray(float(nx), dtype=x.dtype))
    # w = jnp.zeros_like(w)
    x_next = x_nom + E @ w
    x_next = x_next.at[2].set(wrap_to_pi(x_next[2]))  # wrap phi
    return key, x_next, w


# -----------------------------
# Cost
# -----------------------------
def cost(W, reference, x, u, t):
    """
    W = [wpx, wpy, wphi, wvx, wvy, wvphi, wu1, wu2]
    """
    wpx, wpy, wphi, wvx, wvy, wvphi, wu1, wu2 = W

    xref = reference[t]

    dpos  = x[:2] - xref[:2]
    dphi  = wrap_to_pi(x[2] - xref[2])
    dvel  = x[3:5] - xref[3:5]
    dvphi = x[5] - xref[5]

    # Nominal hover thrust split evenly
    u_hover = MASS_NOM * GRAVITY / 2.0
    du = u - jnp.array([u_hover, u_hover], dtype=x.dtype)

    return (
        wpx  * dpos[0] ** 2
        + wpy  * dpos[1] ** 2
        + wphi * (1.0 - jnp.cos(dphi))   # smooth angle cost
        + wvx  * dvel[0] ** 2
        + wvy  * dvel[1] ** 2
        + wvphi * dvphi ** 2
        + wu1  * du[0] ** 2
        + wu2  * du[1] ** 2
    )


# -----------------------------
# Reference
# -----------------------------
def build_piecewise_reference(
    x0: jnp.ndarray, x_goal: jnp.ndarray, N: int, dt: float
) -> jnp.ndarray:
    """
    Straight-line reference from x0 to x_goal.
    Interpolates px, py, phi; zeros for velocities.
    """
    t = jnp.linspace(0.0, 1.0, N + 1)

    pos  = (1.0 - t[:, None]) * x0[:2] + t[:, None] * x_goal[:2]  # (N+1, 2)
    dphi = wrap_to_pi(x_goal[2] - x0[2])
    phi  = wrap_to_pi(x0[2] + t * dphi)                            # (N+1,)

    X_ref = jnp.zeros((N + 1, 6), dtype=jnp.float64)
    X_ref = X_ref.at[:, :2].set(pos)
    X_ref = X_ref.at[:, 2].set(phi)

    # Feed-forward velocity along the line
    vel = (x_goal[:2] - x0[:2]) / (N * dt)
    X_ref = X_ref.at[:, 3:5].set(vel)

    return X_ref


def save_tube_to_numpy(tube: jnp.ndarray, filepath: str) -> None:
    """
    Save tube sizes (N+1, n) to a .npy file.

    Args:
        tube: jnp.ndarray of shape (N+1, n)
        filepath: output path (e.g., 'tube.npy')
    """
    # Ensure host array (JAX → NumPy)
    tube_np = np.asarray(jax.device_get(tube))

    # Create directory if needed
    os.makedirs(os.path.dirname(filepath), exist_ok=True) if os.path.dirname(filepath) else None

    # Save
    np.save(filepath, tube_np)

    print(f"[INFO] Saved tube to {filepath}, shape={tube_np.shape}")

# -----------------------------
# Main
# -----------------------------
def main():
    # Dimensions
    n  = 6   # [px, py, phi, vx, vy, vphi]
    nu = 2   # [u1, u2]

    # Horizon and dt
    N  = 40
    dt = 0.15

    # Cost weights: [wpx, wpy, wphi, wvx, wvy, wvphi, wu1, wu2]
    W = jnp.array([
        15.0, 15.0,   # position
        2.0,          # roll (phi)
        0.5, 0.5,     # translational velocities
        0.05,         # roll rate
        0.01, 0.01,   # control effort
    ], dtype=jnp.float64)

    u_hover = MASS_NOM * GRAVITY / 2.0
    cfg = MPCConfig(
        n=n,
        nu=nu,
        N=N,
        W=W,
        u_ref=jnp.array([u_hover, u_hover], dtype=jnp.float64),
        dt=dt,
    )

    parameter = dt  # passed through to dynamics as `parameter`

    # Control limits
    u_max_val = 2.0 * MASS_NOM * GRAVITY   # generous upper bound per rotor
    u_min = jnp.array([-1.0, -1.0], dtype=jnp.float64)
    u_max = jnp.array([u_max_val, u_max_val], dtype=jnp.float64)
    constraints_u = make_control_box_constraints(u_min, u_max)

    # State limits
    x_max = jnp.array([
        15.0, 15.0,         # px, py
        jnp.pi / 2.0,       # phi (tilt limit)
        5.0, 5.0,           # vx, vy
        8.0,                # vphi
    ], dtype=jnp.float64)
    x_min = -x_max

    constraints_x   = make_state_box_constraints(x_min, x_max)
    constraints_all = combine_constraints(constraints_x, constraints_u)

    # Obstacles: [cx, cy, radius]  (2-D circles)
    obstacles = jnp.array([
        [-1.5, -0.9, 0.16],
        [-1.0, -0.5, 0.16],
        [-0.7, -1.2, 0.16],
        [0.3, -1.0, 0.16],
    ], dtype=jnp.float64)
    # obstacles = jnp.array([[1.0, 1.0, 0.01]], dtype=jnp.float64)

    n_obs = obstacles.shape[0]
    nc = 2 * nu + 2 * n + n_obs

    # Disturbance model
    E_mag     = 0.071
    alpha_sim = E_mag * dt
    disturbance = make_constant_disturbance(n=n, alpha=alpha_sim)

    # Initial / goal states
    x0 = jnp.array([
        -2.0, -2.0,   # px, py
        0.0,           # phi
        0.0, 0.0,      # vx, vy
        0.0,           # vphi
    ], dtype=jnp.float64)

    x_goal = jnp.array([
        0.0, 0.0,      # px, py
        0.0,           # phi
        0.0, 0.0,      # vx, vy
        0.0,           # vphi
    ], dtype=jnp.float64)

    X_ref     = build_piecewise_reference(x0, x_goal, N, dt)
    reference = X_ref
    T_steps   = N

    key   = jax.random.PRNGKey(0)
    E_sim = alpha_sim * jnp.eye(n, dtype=jnp.float64)
    # Solver configs
    admm_cfg = ADMMConfig(
        eps_abs=1e-2,
        eps_rel=0,
        rho_max=1e3,
        max_iterations=400,
        rho_update_frequency=20,
        initial_rho=1.0,
    )

    sls_cfg = SLSConfig(
        max_sls_iterations=2,
        sls_primal_tol=1e-2,
        enable_fastsls=True,
        initialize_nominal=True,
        max_initial_sqp_iterations=100,
        warm_start=False,
        rti=False,
    )

    sqp_cfg = SQPConfig(
        max_sqp_iterations=10,
        warm_start=False,
        feas_tol=0.01,
        step_tol=0.0001,
        line_search=True,
    )

    controller = GenericMPC(
        sls_cfg,
        sqp_cfg,
        admm_cfg,
        config=cfg,
        dynamics=dynamics_planar_quad,
        constraints=constraints_all,
        obstacles=obstacles,
        cost=cost,
        num_constraints=nc,
        disturbance=disturbance,
        shift=1,
        X_in=X_ref,
        U_in=jnp.zeros((cfg.N, cfg.nu), dtype=jnp.float64).at[:, :].set(u_hover),
    )

    # Robust plan
    N_ROLLOUTS = NUM_RANDOM + NUM_ADV
    u0, X_pred, U_pred, V_pred, backoffs, Phi_x, Phi_u, EN = controller.run(
        x0=x0, reference=reference, parameter=parameter
    )
    print("e_sim:", EN)
    # Rollout simulations
    xs       = np.full((N_ROLLOUTS, T_steps, n), np.nan, dtype=np.float64)
    disturbed = np.full((N_ROLLOUTS, T_steps, n), np.nan, dtype=np.float64)
    stop_steps = np.full((N_ROLLOUTS,), T_steps, dtype=np.int32)
    jax.debug.print("{} {}", EN.shape, E_sim.shape)
    for i in range(N_ROLLOUTS):
        disturbance_history = [jnp.zeros((n,), dtype=jnp.float64)]
        x = x0
        jax.debug.print("Rolling out iteration {}", i)

        for k in range(T_steps):
            disturbance_feedback = jnp.zeros((nu,), dtype=jnp.float64)
            for j in range(k + 1):
                disturbance_feedback = disturbance_feedback + Phi_u[k, j] @ disturbance_history[j]

            u = U_pred[k] + disturbance_feedback
            u = jnp.clip(u, u_min, u_max)

            key, x, w = planar_quad_step_with_disturbance(key, x, u, EN[0], dt, i)

            err = np.abs(np.asarray(X_pred[k + 1] - x))
            err[2] = np.abs(np.asarray(wrap_to_pi(X_pred[k + 1, 2] - x[2])))  # phi wrap

            disturbed[i, k, :] = err
            disturbance_history.append(w)
            xs[i, k] = np.asarray(x)


    # Tube visualization (2D: px vs py)
    plans_xy = []
    lowers_xy = []
    uppers_xy = []

    tube = get_trajectory_tubes(Phi_x)   # (N+1, n)

    save_tube_to_numpy(tube, "tube.npy")



    plan_xy = X_pred[:, :2]              # (px, py)
    lower   = plan_xy - tube[:, :2]
    upper   = plan_xy + tube[:, :2]

    plans_xy.append(plan_xy)
    lowers_xy.append(lower)
    uppers_xy.append(upper)

    plot_rollouts_tubes_centers(
        xs=xs,
        centers=np.asarray(obstacles[:, :2]),
        radii=np.asarray(obstacles[:, 2]),
        plans_xy=np.asarray(plans_xy),
        lowers_xy=np.asarray(lowers_xy),
        uppers_xy=np.asarray(uppers_xy),
        step_idx=0,
        tube_stride=1,
        filename="planar_quad_rollouts_xy.png",
        show_plan=False,
        tube_alpha=0.1,
        margin=0.2,
        rollout_alpha=0.5,
        x_label="x [m]",
        y_label="y [m]",
        title="Planar Quadrotor: XY Rollouts + Tube",
    )

    plot_tube_graph_quadrotor(
        disturbed=disturbed[:, :, :3],   # px, py, phi only
        tube=tube[:, :3],
        dt=dt,
        filename="planar_quad_disturbance_vs_tube_pose.png",
    )


if __name__ == "__main__":
    main()