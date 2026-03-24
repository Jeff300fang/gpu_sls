import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle


# --- Styling palette ---
PALETTE = {
    "plan": "#1f77b4",
    "rollout": "#ff7f0e",
    "tube_face": "#2ca02c",
    "tube_edge": "#1b7f1b",
    "obs_face": "#7f7f7f",
    "obs_edge": "#4d4d4d",
}


mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": [
        "Times New Roman",
        "Times",
        "Nimbus Roman",
    ],
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
})


def _normalize_plan_array(arr, name: str):
    """
    Normalize plan/tube arrays to shape (n_steps, horizon_len, 2).
    Accepts either:
      - (horizon_len, 2)
      - (n_steps, horizon_len, 2)
    """
    if arr is None:
        return None
    arr = np.asarray(arr)
    if arr.ndim == 2:
        arr = arr[None, :, :]
    if arr.ndim != 3 or arr.shape[-1] != 2:
        raise ValueError(
            f"{name} has shape {arr.shape}. Expected (n_steps, N+1, 2) or (N+1, 2)."
        )
    return arr


def plot_rollouts_tubes_centers(
    xs,
    centers=None,
    radii=None,
    plans_xy=None,
    lowers_xy=None,
    uppers_xy=None,
    step_idx: int | None = 0,
    tube_stride: int = 1,
    tube_alpha: float = 0.15,
    rollout_alpha: float = 0.35,
    show_plan: bool = True,
    margin: float = 0.5,
    filename: str | None = "planar_quadrotor_rollouts_xy_projection.png",
    dpi: int = 300,
    x_label: str = "x",
    y_label: str = "y",
    title: str = "Planar Quadrotor: XY Rollouts + Robust Tube",
):
    """
    Plot planar quadrotor rollouts in the XY plane.

    Expected shapes:
      xs:        (n_rollouts, T, n) or (T, n), where n >= 2
      plans_xy:  (n_steps, N+1, 2) or (N+1, 2)
      lowers_xy: (n_steps, N+1, 2) or (N+1, 2)
      uppers_xy: (n_steps, N+1, 2) or (N+1, 2)
      centers:   (K, 2)
      radii:     (K,)

    Notes:
      - xs is assumed to be full state, and the XY projection uses state indices 0 and 1.
      - plans_xy / lowers_xy / uppers_xy are assumed already projected to XY.
    """
    xs = np.asarray(xs)

    if xs.ndim == 2:
        if xs.shape[1] < 2:
            raise ValueError(f"xs has shape {xs.shape}. Expected last dimension >= 2.")
        xs = xs[None, :, :]
    elif xs.ndim == 3:
        if xs.shape[2] < 2:
            raise ValueError(f"xs has shape {xs.shape}. Expected last dimension >= 2.")
    else:
        raise ValueError(f"xs has shape {xs.shape}. Expected 2D or 3D array.")

    n_rollouts, T, nx = xs.shape

    plans_xy = _normalize_plan_array(plans_xy, "plans_xy")
    lowers_xy = _normalize_plan_array(lowers_xy, "lowers_xy")
    uppers_xy = _normalize_plan_array(uppers_xy, "uppers_xy")

    if centers is not None:
        centers = np.asarray(centers)
        if centers.ndim == 1:
            centers = centers[None, :]
        if centers.shape[-1] != 2:
            raise ValueError(f"centers has shape {centers.shape}. Expected (K, 2).")

    if radii is not None:
        radii = np.asarray(radii).reshape(-1)

    if lowers_xy is not None and uppers_xy is not None:
        step_idx = int(step_idx if step_idx is not None else 0)
        step_idx = max(0, min(step_idx, lowers_xy.shape[0] - 1))
        lo = lowers_xy[step_idx]
        up = uppers_xy[step_idx]
    else:
        lo = up = None

    all_x = [xs[:, :, 0].ravel()]
    all_y = [xs[:, :, 1].ravel()]

    if plans_xy is not None:
        all_x.append(plans_xy[:, :, 0].ravel())
        all_y.append(plans_xy[:, :, 1].ravel())

    if lo is not None and up is not None:
        all_x.append(lo[:, 0].ravel())
        all_x.append(up[:, 0].ravel())
        all_y.append(lo[:, 1].ravel())
        all_y.append(up[:, 1].ravel())

    if centers is not None and centers.size:
        all_x.append(centers[:, 0].ravel())
        all_y.append(centers[:, 1].ravel())

    all_x = np.concatenate(all_x) if len(all_x) else np.array([0.0])
    all_y = np.concatenate(all_y) if len(all_y) else np.array([0.0])

    xmin, xmax = float(np.nanmin(all_x) - margin), float(np.nanmax(all_x) + margin)
    ymin, ymax = float(np.nanmin(all_y) - margin), float(np.nanmax(all_y) + margin)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True)

    # Obstacles
    if centers is not None and centers.size and radii is not None and radii.size == centers.shape[0]:
        for c, r in zip(centers, radii):
            circ = Circle(
                (float(c[0]), float(c[1])),
                float(r),
                facecolor=PALETTE["obs_face"],
                edgecolor=PALETTE["obs_edge"],
                alpha=0.5,
                linewidth=1.5,
            )
            ax.add_patch(circ)

    # Tube rectangles
    if lo is not None and up is not None:
        for k in range(0, lo.shape[0], max(1, int(tube_stride))):
            w = up[k, 0] - lo[k, 0]
            h = up[k, 1] - lo[k, 1]
            if not np.isfinite(w) or not np.isfinite(h) or w < 0.0 or h < 0.0:
                continue

            rect = Rectangle(
                (lo[k, 0], lo[k, 1]),
                w,
                h,
                facecolor=PALETTE["tube_face"],
                edgecolor=PALETTE["tube_edge"],
                alpha=tube_alpha,
            )
            ax.add_patch(rect)

        ax.plot([], [], color=PALETTE["tube_face"], alpha=tube_alpha, label=f"Tube boxes (step {step_idx})")

    # Planned trajectory
    if show_plan and plans_xy is not None:
        step_idx = int(step_idx if step_idx is not None else 0)
        step_idx = max(0, min(step_idx, plans_xy.shape[0] - 1))
        ax.plot(
            plans_xy[step_idx, :, 0],
            plans_xy[step_idx, :, 1],
            linestyle="--",
            linewidth=2,
            color=PALETTE["plan"],
            label="Planned trajectory",
        )

    # Rollouts
    for i in range(n_rollouts):
        ax.plot(
            xs[i, :, 0],
            xs[i, :, 1],
            alpha=rollout_alpha,
            color=PALETTE["rollout"],
        )
    ax.plot([], [], alpha=rollout_alpha, color=PALETTE["rollout"], label=f"Rollouts (n={n_rollouts})")

    ax.set_title(title)
    ax.legend(loc="best", framealpha=0.9)

    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_tube_graph_quadrotor(
    disturbed,
    tube,
    dt,
    filename: str = "planar_quadrotor_disturbance_vs_tube_size.png",
    state_labels: list | None = None,
):
    """
    Plot state deviation vs tube size for the planar quadrotor.

    disturbed: (n_rollouts, T, n_states)
    tube:      (T+1, n_states)

    For planar quadrotor:
      state = [px, py, phi, vx, vy, vphi]
    """
    disturbed = np.asarray(disturbed)
    tube = np.asarray(tube)

    if disturbed.ndim != 3:
        raise ValueError(
            f"disturbed has shape {disturbed.shape}. Expected (n_rollouts, T, n_states)."
        )

    n_states = disturbed.shape[2]

    if tube.ndim != 2 or tube.shape[1] != n_states:
        raise ValueError(
            f"tube has shape {tube.shape}. Expected (T+1, {n_states})."
        )

    if state_labels is None:
        if n_states == 6:
            state_labels = [
                ("px", "m"),
                ("py", "m"),
                ("phi", "rad"),
                ("vx", "m/s"),
                ("vy", "m/s"),
                ("vphi", "rad/s"),
            ]
        else:
            state_labels = [(f"x{i}", "units") for i in range(n_states)]

    T = disturbed.shape[1]
    tube_trim = tube[1:, :]

    if tube_trim.shape[0] != T:
        raise ValueError(
            f"tube[1:] has length {tube_trim.shape[0]}, but disturbed time dimension is {T}."
        )

    t = np.arange(T) * dt

    fig, axes = plt.subplots(n_states, 1, figsize=(10, 2 * n_states + 2), sharex=True)
    if n_states == 1:
        axes = [axes]

    for idx, ax in enumerate(axes):
        tube_i = tube_trim[:, idx]
        dev_all = disturbed[:, :, idx]

        ax.plot(t, tube_i, label=f"tube size ({state_labels[idx][0]})", linewidth=3)

        for r_idx, dev in enumerate(dev_all):
            m = np.isfinite(dev)
            ax.plot(
                t[m],
                dev[m],
                label=f"|{state_labels[idx][0]} - nominal|" if r_idx == 0 else None,
                alpha=0.8,
            )

        ax.set_ylabel(state_labels[idx][1])
        ax.set_title(f"{state_labels[idx][0]}: Deviation vs Tube Size")
        ax.grid(True)
        ax.legend(loc="best")

    axes[-1].set_xlabel("time (s)")
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)