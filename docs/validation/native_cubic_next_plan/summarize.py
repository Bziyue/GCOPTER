#!/usr/bin/env python3
"""Reproduce the planning figures and evidence; no corridor benchmark is run."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import tempfile

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def limits():
    resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-dir", type=Path,
                        default=Path("/home/zdp/ForCodex/native_cubic_review_experiments"))
    args = parser.parse_args()
    out = Path(__file__).resolve().parent
    repo = out.parents[2]
    review = args.review_dir.resolve()
    manifest = json.loads((review / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        if digest(review / name) != expected:
            raise ValueError(f"Source hash mismatch: {name}")
    hot = json.loads((review / "hot_results.json").read_text())
    old = json.loads((review / "results.json").read_text())
    current_path = repo / "docs/validation/native_splines_round5/results.json"
    current = json.loads(current_path.read_text())
    cases = []
    for case, result in enumerate(current["corridor_generation"]["pruned_predict3_volume1"]):
        sample = result["sample"]
        volume = current["corridor_volume"][case]["C3-selected"]
        cases.append(dict(case=case, median_ms=result["median_ms"],
                          p95_ms=result["p95_ms"], h5_ms=result["h5_ms"],
                          stages=result["stages"], volume=volume,
                          quality={k: sample[k] for k in
                                   ["max_curvature", "shape_energy", "min_axis",
                                    "normalized_jacobian", "active_constraints"]}))

    # This is only the supplied standalone mathematical test, not a GCOPTER build.
    with tempfile.TemporaryDirectory(prefix="c3-plan-") as scratch:
        binary = str(Path(scratch) / "exact_group_reach")
        subprocess.run(["g++", "-std=c++17", "-O2",
                        str(review / "test_exact_group_reach.cpp"), "-o", binary],
                       check=True, preexec_fn=limits(), timeout=60)
        cpp = json.loads(subprocess.check_output([binary], text=True,
                                                preexec_fn=limits(), timeout=60))

    t = sp.symbols("t", real=True)
    third = sp.Rational(1, 3)
    macro = [sp.Rational(9, 2) * t**3,
             sp.Rational(9, 2) * (t**3 - 3 * (t - third)**3),
             sp.Rational(9, 2) * (t**3 - 3 * (t - third)**3 + 3 * (t - 2*third)**3)]
    breaks = [sp.S(0), third, 2*third, sp.S(1)]
    controls = []
    for k, p in enumerate(macro):
        a, b = breaks[k:k+2]
        h = b - a
        q = [p.subs(t, a), (p+h*sp.diff(p,t)/3).subs(t,a),
             (p-h*sp.diff(p,t)/3).subs(t,b), p.subs(t,b)]
        q = list(map(sp.simplify, q))
        assert all(q[j+1] >= q[j] for j in range(3))
        controls.append([str(v) for v in q])
    for k in range(2):
        for derivative in range(3):
            assert sp.simplify(sp.diff(macro[k]-macro[k+1],t,derivative).subs(t,breaks[k+1])) == 0
    assert macro[0].subs(t,0) == 0 and sp.simplify(macro[-1].subs(t,1)) == 1
    for derivative in [1,2]:
        assert sp.diff(macro[0],t,derivative).subs(t,0) == 0
        assert sp.simplify(sp.diff(macro[-1],t,derivative).subs(t,1)) == 0
    quintic = 10*t**3 - 15*t**4 + 6*t**5
    energies = {}
    for derivative in [2,3]:
        energies[f"E{derivative}"] = dict(
            macro_cubic=str(sp.simplify(sum(sp.integrate(sp.diff(p,t,derivative)**2,
                                                        (t,breaks[k],breaks[k+1]))
                                          for k,p in enumerate(macro)))),
            quintic=str(sp.integrate(sp.diff(quintic,t,derivative)**2,(t,0,1))))

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": "white"})
    methods = [("bezier0", "Cubic hull"), ("elevate5", "Elevated to degree 5"),
               ("bezier2", "Uniform depth 2"), ("dual_refine3", "Selective 25%, depth 3"),
               ("exchange_hot", "Hot exchange")]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    for key, label in methods:
        n = np.array([c["spans"] for c in hot["cases"]])
        axes[0].plot(n, [c["summary"][key]["total_ms"] for c in hot["cases"]], "o-", label=label)
        axes[1].plot(n, [100*(c["summary"][key]["volume"]/c["summary"]["bezier0"]["volume"]-1)
                         for c in hot["cases"]], "o-", label=label)
    axes[0].set(ylabel="Solve + continuous audit [ms]", xlabel="Cubic spans")
    axes[1].set(ylabel="Exact union volume gain over cubic hull [%]", xlabel="Cubic spans")
    for ax in axes:
        ax.grid(alpha=.18)
        ax.set_xticks([32,64,96,128])
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Independent straight-sweep LP: lower conservatism has a measurable cost")
    fig.tight_layout()
    fig.savefig(out / "synthetic-tradeoff.png", dpi=170)
    plt.close(fig)

    x = np.linspace(0,1,601)
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.7))
    for derivative, ax in enumerate(axes):
        y = np.empty_like(x)
        for k,p in enumerate(macro):
            mask = (x>=float(breaks[k])) & (x<=float(breaks[k+1]))
            y[mask] = sp.lambdify(t,sp.diff(p,t,derivative),"numpy")(x[mask])
        ax.plot(x, y, color="#c94744", label="Three-piece cubic C2")
        ax.plot(x, sp.lambdify(t,sp.diff(quintic,t,derivative),"numpy")(x),
                color="#276ea6", linestyle="--", label="Quintic smoothstep")
        for knot in [1/3,2/3]:
            ax.axvline(knot, color="gray", linewidth=.7, alpha=.3)
        ax.set(xlabel="Normalized macro-interval t", ylabel=["w(t)","w'(t)","w''(t)"][derivative])
        ax.grid(alpha=.18)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Two endpoint variables can give cubic C2, but not lower bending by default")
    fig.tight_layout()
    fig.savefig(out / "macro-cubic-transition.png", dpi=170)
    plt.close(fig)

    evidence = dict(
        date="2026-11-04", reviewed_repository_commit=manifest["repository_commit"],
        implementation_commit="0990161a2d5241292278396a9639076ac41dbe08",
        review_directory=str(review), source_hashes={f.name:digest(f) for f in review.iterdir() if f.is_file()},
        manifest_verified_files=len(manifest["files"]),
        current_measurement_source=str(current_path.relative_to(repo)),
        current_measurement_sha256=digest(current_path),
        scope="Existing measurements summarized; standalone bound test rerun; symbolic macro-cubic identities checked. No new map benchmark.",
        cpp_test_rerun=cpp, compiler=subprocess.check_output(["g++","--version"],text=True).splitlines()[0],
        symbolic=dict(c2=True, endpoint_first_and_second_derivatives_zero=True,
                      monotone=True, bezier_weights=controls, energies=energies),
        synthetic_bounds=old["bounds"], synthetic_backend=hot["backend"],
        synthetic_cases=[dict(spans=c["spans"],seed=c["seed"],summary=c["summary"]) for c in hot["cases"]],
        current_cases=cases)
    (out / "evidence.json").write_text(json.dumps(evidence,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps(dict(cpp=cpp, macro_energies=energies, cases=len(cases),
                          verified_source_files=len(manifest["files"])),ensure_ascii=False))


if __name__ == "__main__":
    main()
