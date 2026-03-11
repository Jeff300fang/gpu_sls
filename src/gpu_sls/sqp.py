from jax import jit, lax, vmap
import jax
import jax.numpy as jnp

from functools import partial

from trajax.optimizers import linearize, quadratize,vectorize
from gpu_sls.gpu_sls import SLSConfig, sls_solve_gpu
from gpu_sls.gpu_admm import ADMMConfig, constrained_solve
from jax.tree_util import register_pytree_node_class
from dataclasses import dataclass

@register_pytree_node_class
@dataclass(frozen=True)
class SQPConfig:
    max_sqp_iterations: int = 1
    feas_tol: float = 1e-2
    step_tol: float = 1e-4
    warm_start: bool = True
    line_search: bool = True

    def tree_flatten(self):
        children = (self.max_sqp_iterations, self.feas_tol, self.step_tol, self.warm_start, self.line_search)
        return children, None

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children)

# TODO: Add constraints to this?
def lagrangian(cost, dynamics, x0):
    def fun(x, u, t, v, v_prev):
        c1 = cost(x, u, t)
        c2 = jnp.dot(v, dynamics(x, u, t))
        c3 = jnp.dot(v_prev, lax.select(t == 0, x0 - x, -x))
        return c1 + c2 + c3

    return fun

@jax.jit
def add_obstacle_constraints(C: jnp.ndarray, D: jnp.ndarray, f: jnp.ndarray,
                             obstacles: jnp.ndarray, x_curr: jnp.ndarray, eps=1e-5):
    if obstacles.shape[0] == 0:
        return C, D, f

    Tp1, _, nx = C.shape
    _,  _, nu = D.shape

    centers = obstacles[:, :2]
    radii   = obstacles[:, 2]
    pos = x_curr[:, :2]
    diff = pos[:, None, :] - centers[None, :, :]
    dist = jnp.linalg.norm(diff, axis=-1) + eps
    n = diff / dist[..., None]
    coeffs = -n

    C_obstacle = jnp.zeros((Tp1, centers.shape[0], nx), dtype=C.dtype)
    D_obstacle = jnp.zeros((Tp1, centers.shape[0], nu), dtype=D.dtype)

    C_obstacle = C_obstacle.at[..., 0:2].set(coeffs)

    f_obstacle = (dist - radii[None, :]).astype(f.dtype)

    C_all = jnp.concatenate([C, C_obstacle], axis=1)
    D_all = jnp.concatenate([D, D_obstacle], axis=1)
    f_all = jnp.concatenate([f, f_obstacle], axis=1)
    
    return C_all, D_all, f_all

@partial(jit, static_argnums=(0, 1, 2, 3, 4, 5, 6, 7))
def compute_search_direction(
    sls_config: SLSConfig, admm_config: ADMMConfig,
    cost, dynamics, hessian_approx,
    constraints, disturbance,
    obstacles,
    x0, X, U, V, c,
    w, y, rho,
    h_ct_ws, beta_ws, mu_ws, Phi_x_ws, Phi_u_ws, E_prev
):
    T = U.shape[0]
    nc = w.shape[1]
    pad = lambda A: jnp.pad(A, [[0, 1], [0, 0]])

    if hessian_approx is None:
        quadratizer = quadratize(cost)
        Q, R_pad, M_pad = quadratizer(X, pad(U), jnp.arange(T + 1))
    else:
        Q, R_pad, M_pad = jax.vmap(hessian_approx)(X, pad(U), jnp.arange(T + 1))

    R = R_pad[:-1]
    M = M_pad[:-1]

    linearizer = linearize(lagrangian(cost, dynamics, x0),argnums = 5)
    dynamics_linearizer = linearize(dynamics)
    q, r_pad = linearizer(X, pad(U), jnp.arange(T + 1), pad(V[1:]), V)
    r = r_pad[:-1]
    A_pad, B_pad = dynamics_linearizer(X, pad(U), jnp.arange(T + 1))
    A = A_pad[:-1]
    B = B_pad[:-1]
    nx = A.shape[1]
    nu = B.shape[2]
    pad = lambda A: jnp.pad(A, ((0, 1), (0, 0)))
    U_pad = pad(U)

    t = jnp.arange(X.shape[0])
    g = vectorize(constraints)(X, U_pad, t)
    f = -g
    C, D = linearize(constraints)(X, U_pad, t)
    C_all, D_all, f_all = add_obstacle_constraints(C, D, f, obstacles, X)
    
    E = disturbance(X)

    # TODO: Correctly set Q_bar and R_bar?
    Q_bar = jnp.broadcast_to(jnp.eye(Q.shape[1]), Q.shape)
    R_bar = jnp.broadcast_to(jnp.eye(R.shape[1]), R.shape)
    if sls_config.enable_fastsls:
        dX, dU, dV, w, y, rho, converged, converged_admm, backoffs, Phi_x, Phi_u, K_kjN, betaN, muN = sls_solve_gpu(
            admm_config,
            Q, q, R, r, M, A, B, c,
            C_all, D_all, f_all, w, y, rho, sls_config,
            E, Q_bar, R_bar, obstacles, X, h_ct_ws, beta_ws, mu_ws, Phi_x_ws, Phi_u_ws, X, U,
        )
    else:
        dX, dU, dV, w, y, rho, _, converged_admm = constrained_solve(
            admm_config, Q, q, R, r, M, A, B, c, C_all, D_all, f_all, w, y, rho
        )
        backoffs = jnp.zeros((T + 1, nc - obstacles.shape[0]))
        Phi_x = jnp.zeros((T + 1, T + 1, nx, nx))
        Phi_u = jnp.zeros((T, T + 1, nu, nx))
        betaN = jnp.ones((T + 1, T + 1, nc - obstacles.shape[0])) * 1e-10
        muN = jnp.zeros((T + 1, nc))
        K_kjN = jnp.zeros((T, T + 1, nu, nx))

    return dX, dU, dV, q, r, w, y, rho, backoffs, Phi_x, Phi_u, K_kjN, betaN, muN

@jit
def merit_rho(c, dV):
    """Determines the merit function penalty parameter to be used.

    Args:
      c:             [T+1, n]  numpy array.
      dV:            [T+1, n]  numpy array.

    Returns:
        rho: the penalty parameter.
    """
    c2 = jnp.sum(c * c)
    dV2 = jnp.sum(dV * dV)
    return lax.select(c2 > 1e-12, 2.0 * jnp.sqrt(dV2 / c2), 1e-2)

@partial(jit, static_argnums=(0, 1))
def model_evaluator_helper(cost, dynamics,x0, X, U):
    """Evaluates the costs and constraints based on the provided primal variables.

    Args:
      cost:            cost function with signature cost(x, u, t).
      dynamics:        dynamics function with signature dynamics(x, u, t).
      x0:              [n]           numpy array.
      X:               [T+1, n]      numpy array.
      U:               [T, m]        numpy array.

    Returns:
      g: the cost value (a scalar).
      c: the constraint values (a [T+1, n] numpy array).
    """
    T = U.shape[0]
    costs = vmap(cost)(X, jnp.pad(U, [[0, 1], [0, 0]]), jnp.arange(T + 1))
    g = jnp.sum(costs)

    residual_fn = lambda t: dynamics(X[t], U[t], t) - X[t + 1]
    c = jnp.vstack([x0 - X[0], vmap(residual_fn)(jnp.arange(T))])

    return g, c

def merit_function_factory(rho_merit):
    def merit_fn(V, g, c):
        return g + jnp.sum(V * c) + 0.5 * rho_merit * jnp.sum(c * c)
    return merit_fn

@partial(jit, static_argnums=(0, 1))
def line_search(
    merit_function, model_evaluator,
    X_in, U_in, V_in,
    dX, dU, dV,
    current_merit, current_g, current_c,
    merit_slope, armijo_factor,
    alpha_0, alpha_mult, alpha_min,
):
    """Performs a primal-dual line search on an augmented Lagrangian merit function.

    Args:
      merit_function:  merit function mapping V, g, c to the merit scalar.
      X_in:            [T+1, n]      numpy array.
      U_in:            [T, m]        numpy array.
      V_in:            [T+1, n]      numpy array.
      dX:              [T+1, n]      numpy array.
      dU:              [T, m]        numpy array.
      dV:              [T+1, n]      numpy array.
      current_merit:   the merit function value at X, U, V.
      current_g:       the cost value at X, U, V.
      current_c:       the constraint values at X, U, V.
      merit_slope:     the directional derivative of the merit function.
      armijo_factor:   the Armijo parameter to be used in the line search.
      alpha_0:         initial line search value.
      alpha_mult:      a constant in (0, 1) that gets multiplied to alpha to update it.
      alpha_min:       minimum line search value.

    Returns:
      X: [T+1, n]     numpy array, representing the optimal state trajectory.
      U: [T, m]       numpy array, representing the optimal control trajectory.
      V: [T+1, n]     numpy array, representing the optimal multiplier trajectory.
      new_g:          the cost value at the new X, U, V.
      new_c:          the constraint values at the new X, U, V.
      no_errors:       whether no error occurred during the line search.
    """

    def continuation_criterion(inputs):
        _, _, _, _, _, new_merit, alpha = inputs
        # debug.print(f"{new_merit=}, {current_merit=}, {alpha=}, {merit_slope=}")\
        return jnp.logical_and(
            new_merit > current_merit + alpha * armijo_factor * merit_slope,
            alpha > alpha_min,
        )

    def body(inputs):
        _, _, _, _, _, _, alpha = inputs
        alpha *= alpha_mult
        X_new = X_in + alpha * dX
        U_new = U_in + alpha * dU
        V_new = V_in + alpha * dV
        new_g, new_c = model_evaluator(X_new, U_new)
        new_merit = merit_function(V_new, new_g, new_c)
        new_merit = jnp.where(jnp.isnan(new_merit), current_merit, new_merit)
        return X_new, U_new, V_new, new_g, new_c, new_merit, alpha

    X, U, V, new_g, new_c, new_merit, alpha = lax.while_loop(
        continuation_criterion,
        body,
        (X_in, U_in, V_in, current_g, current_c, jnp.inf, alpha_0 / alpha_mult),
    )
    no_errors = alpha > alpha_min
    return X, U, V, new_g, new_c, no_errors

@jit
def slope(dX, dU, dV, c, q, r, rho):
    """Determines the directional derivative of the merit function.

    Args:
      dX: [T+1, n] numpy array.
      dU: [T, m]   numpy array.
      dV: [T+1, n] numpy array.
      c:  [T+1, n] numpy array.
      q:  [T+1, n] numpy array.
      r:  [T, m] numpy array.
      rho: the penalty parameter of the merit function.

    Returns:
        dir_derivative: the directional derivative.
    """
    return jnp.sum(q * dX) + jnp.sum(r * dU) + 2*jnp.sum(dV * c) - rho * jnp.sum(c * c)

@partial(jit, static_argnums=(0,1,2,3,4,5,6,7))
def sqp(
    sls_config: SLSConfig, sqp_config: SQPConfig, admm_config: ADMMConfig,
    cost, dynamics, hessian_approx,
    constraints, disturbance,
    reference, parameter,
    W,
    x0, X_in, U_in, V_in,
    w, y, rho,
    obstacles,
    h_ct_ws, beta_ws, mu_ws, Phi_x_ws, Phi_u_ws,
):
    _cost = partial(cost, W, reference)
    if hessian_approx is not None:
        _hessian_approx = partial(hessian_approx, W, reference)
    else:
        _hessian_approx = None

    _dynamics = partial(dynamics, parameter=parameter)
    model_evaluator = partial(model_evaluator_helper, _cost, _dynamics, x0)

    def body(i, carry):
        i, X_curr, U_curr, V_curr, w, y, rho, converged, backoffs, Phi_x, Phi_u, _, beta_ws, mu_w = carry

        def do_nothing(_):
            return carry

        def do_iter(_):
            g, c = model_evaluator(X_curr, U_curr)
            feas = jnp.max(jnp.abs(c))
            warm_flag = jnp.array(bool(sqp_config.warm_start))

            w0   = lax.select(warm_flag, w, jnp.zeros_like(w))
            y0   = lax.select(warm_flag, y, jnp.zeros_like(y))
            # TODO: Make the defualt rho a parameter
            rho0 = lax.select(warm_flag, rho, jnp.asarray(10.0, dtype=rho.dtype))
            h_ct_ws = backoffs
            dX, dU, dV, q, r, w1, y1, rho1, backoffs1, Phi_x1, Phi_u1, K_kjN, betaN, muN = compute_search_direction(
                sls_config, admm_config,
                _cost, _dynamics, _hessian_approx,
                constraints, disturbance,
                obstacles,
                x0, X_curr, U_curr, V_curr, c,
                w0, y0, rho0,
                h_ct_ws, beta_ws, mu_ws, Phi_x_ws, Phi_u_ws,
            )

            step = jnp.maximum(
                jnp.max(jnp.abs(dX)),
                jnp.max(jnp.abs(dU))
            )
            z_norm = jnp.maximum(
                jnp.max(jnp.abs(X_curr)),
                jnp.max(jnp.abs(U_curr))
            )

            feas_ok = feas <= sqp_config.feas_tol
            step_ok = step <= sqp_config.step_tol * (1.0 + z_norm)
            # jax.debug.print("SQP Iteration {} Feas {} (<= {}) Step {} (<= {})", i, feas, sqp_config.feas_tol, step, sqp_config.step_tol)
            converged1 = jnp.logical_and(feas_ok, step_ok)
            X_next = lax.select(converged1, X_curr, X_curr + dX)
            U_next = lax.select(converged1, U_curr, U_curr + dU)
            V_next = lax.select(converged1, V_curr, V_curr + dV)

            g, c = model_evaluator(X_curr, U_curr)

            rho_merit = merit_rho(c, dV)
            merit_fn  = merit_function_factory(rho_merit)
            current_merit = merit_fn(V_curr, g, c)
            merit_slope = slope(dX, dU, dV, c, q, r, rho_merit)
            last_iter = (i == (sqp_config.max_sqp_iterations - 1))
            do_ls = jnp.logical_and(jnp.array(bool(sqp_config.line_search)), jnp.logical_not(last_iter))

            def ls_branch(_):
                Xn, Un, Vn, g_new, c_new, ok = line_search(
                    merit_fn, model_evaluator,
                    X_curr, U_curr, V_curr,
                    dX, dU, dV,
                    current_merit, g, c,
                    merit_slope,
                    armijo_factor=1e-4,
                    alpha_0=1.0,
                    alpha_mult=0.5,
                    alpha_min=1e-6,
                )
                return Xn, Un, Vn

            def fullstep_branch(_):
                return (X_curr + dX, U_curr + dU, V_curr + dV)

            X_next, U_next, V_next = lax.cond(do_ls, ls_branch, fullstep_branch, operand=None)

            w_next = lax.select(converged1, w, w1)
            y_next = lax.select(converged1, y, y1)
            rho_next = lax.select(converged1, rho, rho1)
            # rho_next = jnp.minimum(rho, 10.0)
            # y_next = rho / rho_next * y_next
            backoffs_next = lax.select(converged1, backoffs, backoffs1)
            Phi_x_next = lax.select(converged1, Phi_x, Phi_x1)
            Phi_u_next = lax.select(converged1, Phi_u, Phi_u1)

            return (i + 1, X_next, U_next, V_next, w_next, y_next, rho_next,
                    jnp.logical_or(converged, converged1),
                    backoffs_next, Phi_x_next, Phi_u_next, K_kjN, betaN, muN, EN)

        return lax.cond(converged, do_nothing, do_iter, operand=None)

    backoffs0 = h_ct_ws
    T, Tp1, nu, nw = Phi_u_ws.shape
    nx = Phi_x_ws.shape[2]
    K_0 = jnp.zeros((T, Tp1, nu, nx))
    carry0 = (0, X_in, U_in, V_in, w, y, rho, jnp.array(False), backoffs0, Phi_x_ws, Phi_u_ws, K_0, beta_ws, mu_ws)
    total_iterations, X_out, U_out, V_out, w_out, y_out, rho_out, converged, backoffs, Phi_x, Phi_u, K_kjN, betaN, muN = lax.fori_loop(
        0, sqp_config.max_sqp_iterations, body, carry0
    )
    return X_out, U_out, V_out, w_out, y_out, rho_out, backoffs, Phi_x, Phi_u, K_kjN, betaN, muN
