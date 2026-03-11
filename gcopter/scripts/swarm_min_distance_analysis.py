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

    def evaluate(self, t: float, derivative_order: int = 0) -> np.ndarray:
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


class TrajectoryRecord:
    def __init__(self, drone_id: int, duration: float, breakpoints, coefficients):
        self.drone_id = int(drone_id)
        self.duration = float(duration)
        self.ppoly = PiecewisePolynomial3D(breakpoints, coefficients)

    def position(self, t: float) -> np.ndarray:
        clamped_t = min(max(t, 0.0), self.duration)
        return self.ppoly.evaluate(clamped_t, 0)


def sample_times(end_time: float, dt: float) -> np.ndarray:
    dt = max(float(dt), 1.0e-4)
    times = [0.0]
    current = 0.0
    while current + dt < end_time:
        current += dt
        times.append(current)
    if not math.isclose(times[-1], end_time, rel_tol=0.0, abs_tol=1.0e-9):
        times.append(end_time)
    return np.asarray(times, dtype=float)


def derive_safety_distance(payload, override):
    if override is not None:
        return float(override), "user_override"

    ellipsoid = (
        payload.get("dynamic_obstacle", {}).get("relative_ellipsoid", [])
    )
    if len(ellipsoid) >= 3:
        axes = np.asarray(ellipsoid[:3], dtype=float)
        if np.allclose(axes, axes[0], rtol=1.0e-6, atol=1.0e-6):
            return float(axes[0]), "dynamic_obstacle_ellipsoid_axis"
        return float(np.min(axes)), "min_dynamic_obstacle_ellipsoid_axis"

    return 1.0, "fallback_default"


def compute_min_distances(trajectories, times):
    drone_ids = [traj.drone_id for traj in trajectories]
    pair_indices = np.triu_indices(len(trajectories), k=1)

    min_distances = np.zeros(times.shape[0], dtype=float)
    min_pair_ids = np.zeros((times.shape[0], 2), dtype=int)

    for idx, time_value in enumerate(times):
        positions = np.stack([traj.position(time_value) for traj in trajectories], axis=0)
        deltas = positions[:, None, :] - positions[None, :, :]
        dist_sq = np.sum(deltas * deltas, axis=2)
        pair_dist_sq = dist_sq[pair_indices]
        min_pair_idx = int(np.argmin(pair_dist_sq))
        min_distances[idx] = float(np.sqrt(pair_dist_sq[min_pair_idx]))
        first = pair_indices[0][min_pair_idx]
        second = pair_indices[1][min_pair_idx]
        min_pair_ids[idx, 0] = drone_ids[first]
        min_pair_ids[idx, 1] = drone_ids[second]

    return min_distances, min_pair_ids


def write_csv(output_path: Path, times, min_distances, min_pair_ids):
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["time", "min_distance", "drone_id_a", "drone_id_b"])
        for idx, time_value in enumerate(times):
            writer.writerow([
                f"{time_value:.9f}",
                f"{min_distances[idx]:.9f}",
                int(min_pair_ids[idx, 0]),
                int(min_pair_ids[idx, 1]),
            ])


def save_plot(output_base: Path, times, min_distances, safety_distance):
    fig, ax = plt.subplots(figsize=(10.8, 4.8))
    ax.plot(times, min_distances, color="#1f77b4", linewidth=2.2, label="Minimum Inter-Drone Distance")
    ax.axhline(
        safety_distance,
        color="#d62728",
        linewidth=1.9,
        linestyle="--",
        label=f"Safety Distance = {safety_distance:.2f} m",
    )
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Distance [m]")
    # ax.set_title("Swarm Minimum Inter-Drone Distance During Execution")
    ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.6)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_base.with_suffix(".png"), dpi=160)
    fig.savefig(output_base.with_suffix(".svg"))
    plt.close(fig)


def build_summary(times, min_distances, min_pair_ids, safety_distance, safety_source, coeff_file):
    global_min_idx = int(np.argmin(min_distances))
    violations = min_distances < safety_distance
    violation_count = int(np.count_nonzero(violations))
    first_violation_time = None
    if violation_count > 0:
        first_violation_time = float(times[int(np.argmax(violations))])

    return {
        "coeff_file": str(coeff_file),
        "sample_count": int(times.shape[0]),
        "safety_distance": float(safety_distance),
        "safety_distance_source": safety_source,
        "global_min_distance": float(min_distances[global_min_idx]),
        "global_min_time": float(times[global_min_idx]),
        "global_min_pair": [
            int(min_pair_ids[global_min_idx, 0]),
            int(min_pair_ids[global_min_idx, 1]),
        ],
        "violation_count": violation_count,
        "violation_fraction": float(violation_count / max(1, times.shape[0])),
        "first_violation_time": first_violation_time,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze the minimum inter-drone distance over the full swarm execution."
    )
    parser.add_argument(
        "--coeff-file",
        type=Path,
        default=DEFAULT_COEFF_FILE,
        help="Path to the exported trajectory coefficient JSON.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for CSV/plot/summary outputs.",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=None,
        help="Sampling interval in seconds. Defaults to the exported sampling.dt.",
    )
    parser.add_argument(
        "--safety-distance",
        type=float,
        default=None,
        help="Override the plotted safety distance in meters.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    coeff_file = args.coeff_file.resolve()
    if not coeff_file.is_file():
        raise FileNotFoundError(f"Coefficient file not found: {coeff_file}")

    with coeff_file.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    traj_payloads = payload.get("trajectories", [])
    if len(traj_payloads) < 2:
        raise ValueError("At least two trajectories are required to analyze inter-drone distance.")

    sample_dt = float(args.dt if args.dt is not None else payload.get("sampling", {}).get("dt", 0.01))
    safety_distance, safety_source = derive_safety_distance(payload, args.safety_distance)

    trajectories = [
        TrajectoryRecord(
            item["drone_id"],
            item["duration"],
            item["breakpoints"],
            item["coefficients"],
        )
        for item in traj_payloads
    ]
    end_time = max(traj.duration for traj in trajectories)
    times = sample_times(end_time, sample_dt)
    min_distances, min_pair_ids = compute_min_distances(trajectories, times)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_base = output_dir / "swarm_min_distance"

    write_csv(output_base.with_suffix(".csv"), times, min_distances, min_pair_ids)

    summary = build_summary(times, min_distances, min_pair_ids, safety_distance, safety_source, coeff_file)
    with output_base.with_name(f"{output_base.name}_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")

    if PLOTTING_AVAILABLE:
        save_plot(output_base, times, min_distances, safety_distance)
    else:
        print(f"[warn] Plot skipped: {PLOTTING_IMPORT_ERROR}")

    print(
        "Swarm minimum-distance analysis complete. "
        f"global_min={summary['global_min_distance']:.3f} m at t={summary['global_min_time']:.3f} s, "
        f"pair={summary['global_min_pair']}, safety={summary['safety_distance']:.3f} m, "
        f"violations={summary['violation_count']}."
    )
    print(f"Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
