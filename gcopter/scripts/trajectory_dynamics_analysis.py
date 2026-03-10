#!/usr/bin/env python3

import argparse
import csv
import json
import math
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("KMP_USE_SHM", "0")
os.environ.setdefault("MKL_THREADING_LAYER", "GNU")
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_COEFF_FILE = SCRIPT_DIR / "latest_trajectory_coefficients.json"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "trajectory_dynamics_output"

PLOTTING_AVAILABLE = False
PLOTTING_IMPORT_ERROR = ""
plt = None
if os.environ.get("GCOPTER_ENABLE_MATPLOTLIB", "1") == "1":
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D

        PLOTTING_AVAILABLE = True
    except Exception as exc:  # pragma: no cover - runtime environment dependent
        PLOTTING_AVAILABLE = False
        PLOTTING_IMPORT_ERROR = str(exc)
else:
    PLOTTING_IMPORT_ERROR = "matplotlib disabled by GCOPTER_ENABLE_MATPLOTLIB=0."


def derivative_factor(power: int, order: int) -> float:
    if order > power:
        return 0.0
    result = 1.0
    for value in range(power - order + 1, power + 1):
        result *= value
    return result


class PiecewisePolynomial3D:
    def __init__(self, breakpoints, coefficients):
        self.breakpoints = np.asarray(breakpoints, dtype=float)
        self.coefficients = np.asarray(coefficients, dtype=float)
        self.num_segments = len(self.breakpoints) - 1
        self.num_coeffs = self.coefficients.shape[0] // self.num_segments

    def _segment_index(self, t: float) -> int:
        if t <= self.breakpoints[0]:
            return 0
        if t >= self.breakpoints[-1]:
            return self.num_segments - 1
        return int(np.searchsorted(self.breakpoints, t, side="right") - 1)

    def evaluate(self, t: float, derivative_order: int) -> np.ndarray:
        idx = self._segment_index(t)
        local_t = t - self.breakpoints[idx]
        coeff_block = self.coefficients[idx * self.num_coeffs:(idx + 1) * self.num_coeffs]
        order_d = self.num_coeffs - derivative_order
        if order_d <= 0:
            return np.zeros(3, dtype=float)

        derived = np.zeros((order_d, 3), dtype=float)
        for row in range(order_d):
            source_power = row + derivative_order
            derived[row] = derivative_factor(source_power, derivative_order) * coeff_block[source_power]

        result = derived[-1].copy()
        for row in range(order_d - 2, -1, -1):
            result = result * local_t + derived[row]
        return result


class FlatnessMap:
    def __init__(self, params):
        self.mass = float(params["VehicleMass"])
        self.grav = float(params["GravAcc"])
        self.dh = float(params["HorizDrag"])
        self.dv = float(params["VertDrag"])
        self.cp = float(params["ParasDrag"])
        self.veps = float(params["SpeedEps"])

    def forward(self, vel, acc, jer, psi, dpsi):
        v0, v1, v2 = vel
        a0, a1, a2 = acc
        j0, j1, j2 = jer

        cp_term = math.sqrt(v0 * v0 + v1 * v1 + v2 * v2 + self.veps)
        w_term = 1.0 + self.cp * cp_term
        w0 = w_term * v0
        w1 = w_term * v1
        w2 = w_term * v2

        dh_over_m = self.dh / self.mass
        zu0 = a0 + dh_over_m * w0
        zu1 = a1 + dh_over_m * w1
        zu2 = a2 + dh_over_m * w2 + self.grav

        zu_sqr0 = zu0 * zu0
        zu_sqr1 = zu1 * zu1
        zu_sqr2 = zu2 * zu2
        zu01 = zu0 * zu1
        zu12 = zu1 * zu2
        zu02 = zu0 * zu2

        zu_sqr_norm = zu_sqr0 + zu_sqr1 + zu_sqr2
        zu_norm = math.sqrt(max(zu_sqr_norm, 1.0e-12))
        z0 = zu0 / zu_norm
        z1 = zu1 / zu_norm
        z2 = zu2 / zu_norm

        ng_den = zu_sqr_norm * zu_norm
        ng00 = (zu_sqr1 + zu_sqr2) / ng_den
        ng01 = -zu01 / ng_den
        ng02 = -zu02 / ng_den
        ng11 = (zu_sqr0 + zu_sqr2) / ng_den
        ng12 = -zu12 / ng_den
        ng22 = (zu_sqr0 + zu_sqr1) / ng_den

        v_dot_a = v0 * a0 + v1 * a1 + v2 * a2
        dw_term = self.cp * v_dot_a / cp_term
        dw0 = w_term * a0 + dw_term * v0
        dw1 = w_term * a1 + dw_term * v1
        dw2 = w_term * a2 + dw_term * v2

        dz_term0 = j0 + dh_over_m * dw0
        dz_term1 = j1 + dh_over_m * dw1
        dz_term2 = j2 + dh_over_m * dw2
        dz0 = ng00 * dz_term0 + ng01 * dz_term1 + ng02 * dz_term2
        dz1 = ng01 * dz_term0 + ng11 * dz_term1 + ng12 * dz_term2
        dz2 = ng02 * dz_term0 + ng12 * dz_term1 + ng22 * dz_term2

        f_term0 = self.mass * a0 + self.dv * w0
        f_term1 = self.mass * a1 + self.dv * w1
        f_term2 = self.mass * (a2 + self.grav) + self.dv * w2
        thrust = z0 * f_term0 + z1 * f_term1 + z2 * f_term2

        tilt_den = math.sqrt(max(2.0 * (1.0 + z2), 1.0e-12))
        tilt0 = 0.5 * tilt_den
        tilt1 = -z1 / tilt_den
        tilt2 = z0 / tilt_den
        c_half_psi = math.cos(0.5 * psi)
        s_half_psi = math.sin(0.5 * psi)
        quat = np.array([
            tilt0 * c_half_psi,
            tilt1 * c_half_psi + tilt2 * s_half_psi,
            tilt2 * c_half_psi - tilt1 * s_half_psi,
            tilt0 * s_half_psi,
        ])

        c_psi = math.cos(psi)
        s_psi = math.sin(psi)
        omg_den = max(z2 + 1.0, 1.0e-6)
        omg_term = dz2 / omg_den
        omg = np.array([
            dz0 * s_psi - dz1 * c_psi - (z0 * s_psi - z1 * c_psi) * omg_term,
            dz0 * c_psi + dz1 * s_psi - (z0 * c_psi + z1 * s_psi) * omg_term,
            (z1 * dz0 - z0 * dz1) / omg_den + dpsi,
        ])

        return thrust, quat, omg


def sample_times(breakpoints, dt):
    times = [float(breakpoints[0])]
    current = float(breakpoints[0])
    end = float(breakpoints[-1])
    dt = max(float(dt), 1.0e-4)
    while current + dt < end:
        current += dt
        times.append(current)
    if times[-1] != end:
        times.append(end)
    return np.asarray(times, dtype=float)


def write_csv(output_path: Path, series):
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "time",
            "pos_x", "pos_y", "pos_z",
            "vel_x", "vel_y", "vel_z",
            "acc_x", "acc_y", "acc_z",
            "jer_x", "jer_y", "jer_z",
            "thrust",
            "quat_w", "quat_x", "quat_y", "quat_z",
            "omega_x", "omega_y", "omega_z",
            "speed", "body_rate_mag", "tilt_angle",
        ])
        for idx, time_value in enumerate(series["time"]):
            writer.writerow([
                time_value,
                *series["pos"][idx],
                *series["vel"][idx],
                *series["acc"][idx],
                *series["jer"][idx],
                series["thrust"][idx],
                *series["quat"][idx],
                *series["omega"][idx],
                series["speed"][idx],
                series["body_rate_mag"][idx],
                series["tilt_angle"][idx],
            ])


def save_png_and_svg(fig, output_base: Path):
    fig.savefig(output_base.with_suffix(".png"), dpi=160)
    fig.savefig(output_base.with_suffix(".svg"), transparent=True)


def save_png_and_svg_opaque(fig, output_base: Path):
    fig.savefig(output_base.with_suffix(".png"), dpi=160, transparent=False)
    fig.savefig(output_base.with_suffix(".svg"), transparent=False)


def save_legend_svg(handles, labels, output_path: Path):
    fig = plt.figure(figsize=(max(14.0, 2.0 * len(labels)), 1.3))
    fig.legend(
        handles,
        labels,
        loc="center",
        ncol=len(labels),
        frameon=False,
        handlelength=3.0,
        columnspacing=1.6,
        fontsize=11,
    )
    fig.patch.set_alpha(0.0)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.08, transparent=True)
    plt.close(fig)


def apply_axis_headroom(ax, fraction=0.12):
    y_min, y_max = ax.get_ylim()
    span = max(y_max - y_min, 1.0e-6)
    ax.set_ylim(y_min, y_max + fraction * span)


def build_constraint_handles_and_labels(colors):
    legend_handles = [
        Line2D([0], [0], color=colors["speed"], linewidth=2.2),
        Line2D([0], [0], color=colors["speed"], linewidth=1.8, linestyle="--", alpha=0.6),
        Line2D([0], [0], color=colors["body_rate_mag"], linewidth=2.2),
        Line2D([0], [0], color=colors["body_rate_mag"], linewidth=1.8, linestyle="--", alpha=0.6),
        Line2D([0], [0], color=colors["tilt_angle"], linewidth=2.2),
        Line2D([0], [0], color=colors["tilt_angle"], linewidth=1.8, linestyle="--", alpha=0.6),
        Line2D([0], [0], color=colors["thrust"], linewidth=2.4),
        Line2D([0], [0], color=colors["thrust"], linewidth=1.9, linestyle="--", alpha=0.6),
        Line2D([0], [0], color=colors["thrust"], linewidth=2.2, linestyle=":", alpha=0.6),
    ]
    legend_labels = [
        "Speed [m/s]", "Speed Limit [m/s]",
        "Body Rate [rad/s]", "Body Rate Limit [rad/s]",
        "Tilt Angle [rad]", "Tilt Limit [rad]",
        "Thrust [N]", "Max Thrust [N]", "Min Thrust [N]",
    ]
    return legend_handles, legend_labels


def plot_constraints(output_dir: Path, series, constraints):
    fig, ax = plt.subplots(figsize=(15, 7))
    ax_thrust = ax.twinx()
    t = series["time"]
    colors = {
        "speed": "#127a6b",
        "body_rate_mag": "#2f59d9",
        "tilt_angle": "#8b5cf6",
        "thrust": "#dd6b20",
    }
    limit_alpha = 0.6

    legend_handles, legend_labels = build_constraint_handles_and_labels(colors)

    curve_specs = [
        ("Speed", series["speed"], constraints["MaxVelMag"], colors["speed"], "--"),
        ("Body Rate", series["body_rate_mag"], constraints["MaxBdrMag"], colors["body_rate_mag"], "--"),
        ("Tilt Angle", series["tilt_angle"], constraints["MaxTiltAngle"], colors["tilt_angle"], "--"),
    ]

    for label, values, limit_value, color, limit_style in curve_specs:
        ax.plot(t, values, color=color, linewidth=2.2)
        ax.axhline(limit_value, color=color, linestyle=limit_style, linewidth=1.8, alpha=limit_alpha)

    ax_thrust.plot(t, series["thrust"], color=colors["thrust"], linewidth=2.4)
    ax_thrust.axhline(constraints["MaxThrust"], color=colors["thrust"], linestyle="--", linewidth=1.9, alpha=limit_alpha)
    ax_thrust.axhline(constraints["MinThrust"], color=colors["thrust"], linestyle=":", linewidth=2.2, alpha=limit_alpha)

    ax.set_xlabel("time [s]")
    ax.set_ylabel("speed / body rate / tilt")
    ax_thrust.set_ylabel("thrust", color=colors["thrust"])
    ax_thrust.tick_params(axis="y", colors=colors["thrust"])
    ax_thrust.spines["right"].set_color(colors["thrust"])
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    save_png_and_svg(fig, output_dir / "constraint_overview")
    plt.close(fig)
    save_legend_svg(legend_handles, legend_labels, output_dir / "constraint_legend.svg")

    fig, ax = plt.subplots(figsize=(15, 7))
    ax_thrust = ax.twinx()
    for _, values, limit_value, color, limit_style in curve_specs:
        ax.plot(t, values, color=color, linewidth=2.2)
        ax.axhline(limit_value, color=color, linestyle=limit_style, linewidth=1.8, alpha=limit_alpha)
    ax_thrust.plot(t, series["thrust"], color=colors["thrust"], linewidth=2.4)
    ax_thrust.axhline(constraints["MaxThrust"], color=colors["thrust"], linestyle="--", linewidth=1.9, alpha=limit_alpha)
    ax_thrust.axhline(constraints["MinThrust"], color=colors["thrust"], linestyle=":", linewidth=2.2, alpha=limit_alpha)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("speed / body rate / tilt")
    ax_thrust.set_ylabel("thrust", color=colors["thrust"])
    ax_thrust.tick_params(axis="y", colors=colors["thrust"])
    ax_thrust.spines["right"].set_color(colors["thrust"])
    ax.grid(True, alpha=0.25)
    apply_axis_headroom(ax, 0.22)
    apply_axis_headroom(ax_thrust, 0.12)
    ax.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=5,
        frameon=False,
        fontsize=10,
        handlelength=2.8,
        columnspacing=1.4,
        borderaxespad=0.2,
    )
    fig.tight_layout()
    save_png_and_svg_opaque(fig, output_dir / "constraint_overview_with_legend")
    plt.close(fig)


def plot_states(output_dir: Path, series):
    fig, axes = plt.subplots(3, 2, figsize=(13, 11), sharex=True)
    t = series["time"]
    labels = ("x", "y", "z")

    for axis_idx in range(3):
        axes[0, 0].plot(t, series["pos"][:, axis_idx], label=labels[axis_idx])
        axes[0, 1].plot(t, series["vel"][:, axis_idx], label=labels[axis_idx])
        axes[1, 0].plot(t, series["acc"][:, axis_idx], label=labels[axis_idx])
        axes[1, 1].plot(t, series["jer"][:, axis_idx], label=labels[axis_idx])
        axes[2, 0].plot(t, series["omega"][:, axis_idx], label=labels[axis_idx])

    quat_labels = ("w", "x", "y", "z")
    for axis_idx in range(4):
        axes[2, 1].plot(t, series["quat"][:, axis_idx], label=quat_labels[axis_idx])

    axes[0, 0].set_title("Position")
    axes[0, 1].set_title("Velocity")
    axes[1, 0].set_title("Acceleration")
    axes[1, 1].set_title("Jerk")
    axes[2, 0].set_title("Body rates")
    axes[2, 1].set_title("Quaternion")

    for row in axes:
        for axis in row:
            axis.grid(True, alpha=0.25)
            axis.legend(loc="upper right")

    axes[2, 0].set_xlabel("time [s]")
    axes[2, 1].set_xlabel("time [s]")
    fig.tight_layout()
    save_png_and_svg(fig, output_dir / "flatness_state_curves")
    plt.close(fig)

def generate_plots(output_dir: Path, series, constraints):
    plotting_errors = []
    if PLOTTING_AVAILABLE:
        try:
            plot_constraints(output_dir, series, constraints)
            plot_states(output_dir, series)
        except Exception as exc:  # pragma: no cover - runtime environment dependent
            plotting_errors.append(f"matplotlib rendering failed: {exc}")
    else:
        plotting_errors.append(f"matplotlib import failed: {PLOTTING_IMPORT_ERROR}")

    if plotting_errors:
        (output_dir / "plotting_error.txt").write_text(
            "\n".join(plotting_errors) + "\n",
            encoding="utf-8",
        )
    return plotting_errors


def summarize(series, constraints):
    return {
        "speed_max": float(np.max(series["speed"])),
        "body_rate_max": float(np.max(series["body_rate_mag"])),
        "tilt_max": float(np.max(series["tilt_angle"])),
        "thrust_min": float(np.min(series["thrust"])),
        "thrust_max": float(np.max(series["thrust"])),
        "violations": {
            "MaxVelMag": bool(np.max(series["speed"]) > constraints["MaxVelMag"]),
            "MaxBdrMag": bool(np.max(series["body_rate_mag"]) > constraints["MaxBdrMag"]),
            "MaxTiltAngle": bool(np.max(series["tilt_angle"]) > constraints["MaxTiltAngle"]),
            "MinThrust": bool(np.min(series["thrust"]) < constraints["MinThrust"]),
            "MaxThrust": bool(np.max(series["thrust"]) > constraints["MaxThrust"]),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze GCOPTER trajectory dynamics from exported coefficients.")
    parser.add_argument("--coeff-file", default=str(DEFAULT_COEFF_FILE), help="Path to the exported trajectory coefficient JSON.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for CSV/PNG/SVG outputs.")
    args = parser.parse_args()

    coeff_path = Path(args.coeff_file).expanduser().resolve()
    with coeff_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    trajectory = PiecewisePolynomial3D(data["breakpoints"], data["coefficients"])
    flatness = FlatnessMap(data["physical_params"])
    constraints = data["constraints"]
    heading = data.get("heading", {"psi": 0.0, "dpsi": 0.0})
    psi = float(heading.get("psi", 0.0))
    dpsi = float(heading.get("dpsi", 0.0))
    dt = float(data.get("sampling", {}).get("dt", 0.01))
    times = sample_times(data["breakpoints"], dt)

    pos = np.stack([trajectory.evaluate(time_value, 0) for time_value in times])
    vel = np.stack([trajectory.evaluate(time_value, 1) for time_value in times])
    acc = np.stack([trajectory.evaluate(time_value, 2) for time_value in times])
    jer = np.stack([trajectory.evaluate(time_value, 3) for time_value in times])

    thrust_values = []
    quat_values = []
    omega_values = []
    tilt_values = []
    for vel_value, acc_value, jer_value in zip(vel, acc, jer):
        thrust, quat, omega = flatness.forward(vel_value, acc_value, jer_value, psi, dpsi)
        thrust_values.append(thrust)
        quat_values.append(quat)
        omega_values.append(omega)
        tilt_cos = 1.0 - 2.0 * (quat[1] * quat[1] + quat[2] * quat[2])
        tilt_values.append(math.acos(min(1.0, max(-1.0, tilt_cos))))

    series = {
        "time": times,
        "pos": pos,
        "vel": vel,
        "acc": acc,
        "jer": jer,
        "thrust": np.asarray(thrust_values),
        "quat": np.asarray(quat_values),
        "omega": np.asarray(omega_values),
        "speed": np.linalg.norm(vel, axis=1),
        "body_rate_mag": np.linalg.norm(np.asarray(omega_values), axis=1),
        "tilt_angle": np.asarray(tilt_values),
    }

    write_csv(output_dir / "dynamics_samples.csv", series)
    plotting_errors = generate_plots(output_dir, series, constraints)

    summary = summarize(series, constraints)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Coefficient file: {coeff_path}")
    print(f"Output directory: {output_dir}")
    if plotting_errors:
        print("Plotting notes:")
        for message in plotting_errors:
            print(f"  - {message}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
