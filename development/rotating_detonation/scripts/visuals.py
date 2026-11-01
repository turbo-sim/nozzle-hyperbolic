"""Plasma duct contours, periodicity assessment, and startup/periodic GIFs."""

import os

os.environ.setdefault("MPLBACKEND", "Agg")
from hashlib import sha256
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from PIL import Image

STUDY = Path(__file__).resolve().parents[1]


def periodicity(v, config, tolerance=1e-3):
    frames = config["forcing"]["frames_per_cycle"]
    z = np.asarray(v["states"])[:, 1:-1, :]
    # Fixed baseline primitive scales: pressure, a, r*a, a*a.
    gamma, R = 1.4016904909149628, 287.04739368889705
    inlet = config["inlet"]
    M = inlet["mach"]
    T = inlet["stagnation_temperature_K"] / (1 + (gamma - 1) * M * M / 2)
    p = inlet["stagnation_pressure_Pa"] * (T / inlet["stagnation_temperature_K"]) ** (
        gamma / (gamma - 1)
    )
    a = np.sqrt(gamma * R * T)
    radius = (
        config["geometry"]["tip_radius_m"]
        * (1 + config["geometry"]["inlet_hub_tip"])
        / 2
    )
    scales = np.array([p, a, radius * a, a * a])
    rows = []
    offset = round(v["times"][0] * config["forcing"]["frequency_Hz"] - 1 / frames)
    for cycle in range(1, len(z) // frames):
        diff = (
            z[cycle * frames : (cycle + 1) * frames]
            - z[(cycle - 1) * frames : cycle * frames]
        ) / scales
        rms = np.sqrt(np.mean(diff * diff, axis=(0, 1)))
        rows.append(
            dict(
                cycle=offset + cycle + 1,
                per_variable_RMS=rms.tolist(),
                maximum=float(max(rms)),
                passed=bool(max(rms) <= tolerance),
            )
        )
    verified = len(rows) >= 2 and all(r["passed"] for r in rows[-2:])
    return dict(
        verified=verified,
        tolerance=tolerance,
        criterion="Two consecutive cycle-to-cycle comparisons: max over four primitives of RMS across all physical cells and 24 phase-aligned outputs, divided by fixed baseline primitive scales.",
        primitive_scales=scales.tolist(),
        cycles=rows,
        end_time_s=float(v["times"][-1]),
    )


def continuation_solver(d, context, seed):
    if "solve" in context:
        return context["solve"]
    import jax

    jax.config.update("jax_enable_x64", True)
    from unittest.mock import patch
    from nozzle_hyperbolic.solvers.fvm import Operator, Scheme, solve_transient, drivers
    from .problem import geometry, model

    cfg = d["config"]
    geom = geometry(cfg)
    physical = model(
        cfg, geom, cfg["numerics"]["transient_cells"], driven=True, initial=seed
    )
    scheme = Scheme(**d["transient"]["rk3"]["scheme"])
    operator = Operator(physical, scheme)

    def solve(initial, cfl):
        frames = cfg["forcing"]["frames_per_cycle"]
        frequency = cfg["forcing"]["frequency_Hz"]
        times = np.arange(1, 4 * frames + 1) / (frequency * frames)
        with patch.object(drivers, "Operator", return_value=operator):
            snapshots = solve_transient(
                physical,
                times,
                scheme,
                cfl=cfl,
                initial_state=initial,
                max_steps=cfg["numerics"]["max_transient_steps"],
            )
        if len(snapshots) != len(times) or not all(item.success for item in snapshots):
            raise RuntimeError(
                "Periodic continuation failed; retained data cannot be called periodic"
            )
        return snapshots

    context.update(solve=solve, model=physical)
    return solve


def extend_if_needed(d, context):
    from .run import save, hashes

    digest = sha256((STUDY / "outputs/results.json").read_bytes()).hexdigest()
    retained = STUDY / "outputs/periodic_extension.json"
    if retained.exists():
        cached = json.loads(retained.read_text())
        if cached.get("results_sha256") == digest and cached["assessment"]["verified"]:
            return cached["record"], cached["assessment"]
    original = d["transient"]["rk3"]
    assessment = periodicity(original, d["config"])
    if assessment["verified"]:
        return original, assessment
    v = {k: original[k] for k in ("times", "fields", "states", "x")}
    v = {
        k: (
            list(val)
            if k in ("times", "states")
            else (
                {name: list(field) for name, field in val.items()}
                if k == "fields"
                else val
            )
        )
        for k, val in v.items()
    }
    before = hashes()
    solve = continuation_solver(d, context, np.asarray(v["states"][-1]))
    for _ in range(14):  # whole-period chunks, at most 64 total cycles
        start = v["times"][-1]
        snapshots = solve(
            np.asarray(v["states"][-1]), d["config"]["numerics"]["explicit_cfl"]
        )
        v["times"] += [start + item.time for item in snapshots]
        v["states"] += [item.state.tolist() for item in snapshots]
        samples = [context["model"].fields(item.state) for item in snapshots]
        for field in v["fields"]:
            v["fields"][field] += [
                np.asarray(sample[field]).tolist() for sample in samples
            ]
        assessment = periodicity(v, d["config"])
        save(
            retained,
            dict(
                results_sha256=digest,
                record=v,
                assessment=assessment,
                source_unchanged=hashes() == before,
            ),
        )
        print(
            "Periodic extension:",
            assessment["end_time_s"],
            assessment["cycles"][-1]["maximum"],
            flush=True,
        )
        if assessment["verified"]:
            break
    return v, assessment


def refined_periodic(d, v, context):
    from .run import save, hashes

    digest = sha256((STUDY / "outputs/results.json").read_bytes()).hexdigest()
    path = STUDY / "outputs/periodic_refinement.json"
    if path.exists():
        retained = json.loads(path.read_text())
        if (
            retained.get("results_sha256") == digest
            and retained["assessment"]["verified"]
        ):
            return retained
    before = hashes()
    solve = continuation_solver(d, context, np.asarray(v["states"][-1]))
    seed = np.asarray(v["states"][-1])
    start = float(v["times"][-1])
    refined = dict(
        times=[],
        states=[],
        fields={name: [] for name in ("p", "M", "alpha_deg")},
        x=v["x"],
    )
    for _ in range(8):
        offset = refined["times"][-1] if refined["times"] else start
        snapshots = solve(seed, d["config"]["numerics"]["explicit_cfl"] / 2)
        refined["times"] += [offset + item.time for item in snapshots]
        refined["states"] += [item.state.tolist() for item in snapshots]
        samples = [context["model"].fields(item.state) for item in snapshots]
        for field in refined["fields"]:
            refined["fields"][field] += [
                np.asarray(sample[field]).tolist() for sample in samples
            ]
        seed = snapshots[-1].state
        assessment = periodicity(refined, d["config"])
        retained = dict(
            results_sha256=digest,
            record=refined,
            assessment=assessment,
            cfl=d["config"]["numerics"]["explicit_cfl"] / 2,
            source_unchanged=hashes() == before,
        )
        save(path, retained)
        print(
            "Periodic CFL/2 refinement:",
            assessment["end_time_s"],
            assessment["cycles"][-1]["maximum"],
            flush=True,
        )
        if assessment["verified"]:
            break
    return retained


def coordinates(g, cells):
    m = np.linspace(0, g["m"][-1], cells + 1)
    x, r, b, tx, tr = [
        np.interp(m, g["m"], g[k]) for k in ("x", "r", "b", "tip_x", "tip_r")
    ]
    rp = 2 * (x - tx) / b
    xp = 2 * (tr - r) / b
    eta = np.linspace(-0.5, 0.5, 9)[:, None]
    return x[None, :] - eta * b * rp, r[None, :] + eta * b * xp


def draw(ax, X, Y, values, norm, label):
    mesh = ax.pcolormesh(
        X,
        Y,
        np.tile(values, (X.shape[0] - 1, 1)),
        cmap="plasma",
        norm=norm,
        shading="flat",
        rasterized=True,
    )
    ax.plot(X[0], Y[0], color="#30253d", lw=1.2)
    ax.plot(X[-1], Y[-1], color="#30253d", lw=1.2)
    ax.set(xlabel="Axial coordinate x [m]", ylabel="Radius [m]")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(float(X.min()) - 0.002, float(X.max()) + 0.002)
    ax.set_ylim(float(Y.min()) - 0.004, float(Y.max()) + 0.004)
    ax.set_title(label, loc="left", fontsize=12, fontweight="bold")
    return mesh


def gif(v, indices, path, title, X, Y, limits, fps=16):
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 4.7), layout="constrained")
    fields = [("M", "Total Mach number", 1), ("p", "Static pressure [bar]", 1e-5)]
    meshes = []
    for ax, (field, label, scale) in zip(axes, fields):
        values = np.asarray(v["fields"][field])[indices[0]] * scale
        mesh = draw(ax, X, Y, values, Normalize(*limits[field]), label)
        fig.colorbar(mesh, ax=ax, pad=0.015, shrink=0.85, aspect=16)
        meshes.append(mesh)
    note = fig.suptitle(title, fontsize=13, fontweight="bold")
    frames = []
    for index in indices:
        for mesh, (field, _, scale) in zip(meshes, fields):
            mesh.set_array(
                np.tile(
                    np.asarray(v["fields"][field])[index] * scale, (X.shape[0] - 1, 1)
                ).ravel()
            )
        note.set_text(title + f"  |  t = {v['times'][index]*1e3:.4f} ms")
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        frames.append(
            Image.fromarray(rgba)
            .convert("RGB")
            .quantize(colors=256, method=Image.Quantize.MEDIANCUT)
        )
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=round(1000 / fps),
        loop=0,
        optimize=False,
        disposal=2,
    )
    plt.close(fig)


def mean_comparison(d, v, assessment, baseline, refinement):
    phases = d["config"]["forcing"]["frames_per_cycle"]
    count = 2 * phases + 1
    times = np.asarray(v["times"][-count:])
    duration = times[-1] - times[0]
    means = {}
    envelope = {}
    summary = {}
    samples = {}
    for field in (
        "p",
        "M",
        "T",
        "alpha_deg",
        "mdot",
        "rho",
        "v_m",
        "v_theta",
        "h0",
        "s",
    ):
        values = np.asarray(v["fields"][field])[-count:]
        samples[field] = values.tolist()
        mean = np.trapezoid(values, times, axis=0) / duration
        steady = np.asarray(baseline["fields"][field])
        delta = mean - steady
        means[field] = mean.tolist()
        envelope[field] = dict(
            min=np.min(values, axis=0).tolist(), max=np.max(values, axis=0).tolist()
        )
        coarse_mean = np.trapezoid(values[::2], times[::2], axis=0) / duration
        summary[field] = dict(
            averaging_quadrature_RMS_change=float(
                np.sqrt(np.mean((mean - coarse_mean) ** 2))
            ),
            steady_outlet=float(steady[-1]),
            mean_outlet=float(mean[-1]),
            outlet_difference=float(delta[-1]),
            maximum_absolute_difference=float(max(abs(delta))),
            RMS_difference=float(np.sqrt(np.mean(delta * delta))),
        )
        if field != "alpha_deg":
            relative = delta / np.maximum(np.abs(steady), 1e-14)
            summary[field].update(
                outlet_percent=float(relative[-1] * 100),
                maximum_percent=float(max(abs(relative)) * 100),
                RMS_percent=float(np.sqrt(np.mean(relative * relative)) * 100),
            )
    sensitivity = {}
    if refinement["assessment"]["verified"]:
        refined = refinement["record"]
        rt = np.asarray(refined["times"][-count:])
        for field in ("p", "M", "alpha_deg"):
            fine = np.trapezoid(
                np.asarray(refined["fields"][field])[-count:], rt, axis=0
            ) / (rt[-1] - rt[0])
            diff = np.asarray(means[field]) - fine
            sensitivity[field] = dict(
                RMS_difference=float(np.sqrt(np.mean(diff * diff))),
                maximum_absolute_difference=float(max(abs(diff))),
                outlet_difference=float(diff[-1]),
                refined_mean=fine.tolist(),
            )
    data = dict(
        results_sha256=sha256(
            (STUDY / "outputs/results.json").read_bytes()
        ).hexdigest(),
        verified_periodic=assessment["verified"],
        averaging_samples=dict(times=times.tolist(), fields=samples),
        refined_periodicity=refinement["assessment"],
        window_start_s=float(times[0]),
        window_end_s=float(times[-1]),
        complete_cycles=2,
        quadrature="Trapezoidal integration over 49 phase-aligned outputs, including both endpoints of two complete periods. Nonlinear diagnostic fields are averaged directly, rather than evaluated from the averaged primitive state.",
        x=v["x"],
        steady=baseline["fields"],
        mean=means,
        envelope=envelope,
        summary=summary,
        temporal_mean_sensitivity=sensitivity,
        temporal_sensitivity_window="Final two verified periods at CFL 0.4 versus the final two verified periods of a separate CFL 0.2 continuation initialized from the periodic CFL 0.4 state. Both windows span complete periods at aligned phases.",
    )
    (STUDY / "outputs/mean_comparison.json").write_text(json.dumps(data, indent=2))


def main():
    d = json.loads((STUDY / "outputs/results.json").read_text())
    g = json.loads((STUDY / "outputs/geometry.json").read_text())
    if "rk3" not in d["transient"]:
        raise ValueError("RK3 transient required for animations")
    context = {}
    v, assessment = extend_if_needed(d, context)
    refinement = refined_periodic(d, v, context)
    periodics = {
        name: periodicity(value, d["config"]) for name, value in d["transient"].items()
    }
    periodics["animation_rk3"] = assessment
    periodics["rk3_periodic_CFL_half"] = refinement["assessment"]
    (STUDY / "outputs/periodicity.json").write_text(json.dumps(periodics, indent=2))
    baseline = d["steady"].get("implicit", next(iter(d["steady"].values())))[
        "representative"
    ][-1]
    mean_comparison(d, v, assessment, baseline, refinement)
    X, Y = coordinates(g, baseline["cells"])
    fig, axes = plt.subplots(3, 1, figsize=(11.5, 6.5), layout="constrained")
    for ax, (field, label, scale) in zip(
        axes,
        [
            ("M", "Total Mach number", 1),
            ("p", "Static pressure [bar]", 1e-5),
            ("alpha_deg", "Flow angle [deg]", 1),
        ],
    ):
        values = np.asarray(baseline["fields"][field]) * scale
        mesh = draw(
            ax, X, Y, values, Normalize(float(min(values)), float(max(values))), label
        )
        fig.colorbar(mesh, ax=ax, pad=0.015, shrink=0.85, aspect=16)
    fig.suptitle("Steady flow in the annular transition duct", fontweight="bold")
    fig.savefig(STUDY / "outputs/steady_contours.png", dpi=180)
    fig.savefig(STUDY / "outputs/steady_contours.pdf")
    plt.close(fig)
    # Add the exact steady t=0 physical field to the startup visualization.
    frame_count = d["config"]["forcing"]["frames_per_cycle"]
    startup = dict(
        times=[0] + v["times"],
        fields={
            field: [baseline["fields"][field]] + list(v["fields"][field])
            for field in ("M", "p")
        },
    )
    limits = {
        field: (
            float(min(np.min(v["fields"][field]), np.min(baseline["fields"][field])))
            * scale,
            float(max(np.max(v["fields"][field]), np.max(baseline["fields"][field])))
            * scale,
        )
        for field, scale in [("M", 1), ("p", 1e-5)]
    }
    gif(
        startup,
        list(range(min(3 * frame_count + 1, len(startup["times"])))),
        STUDY / "outputs/startup.gif",
        "Startup from steady inlet conditions",
        X,
        Y,
        limits,
    )
    late_name = "periodic.gif" if assessment["verified"] else "late_time_unverified.gif"
    gif(
        v,
        list(range(max(0, len(v["times"]) - 2 * frame_count - 1), len(v["times"]))),
        STUDY / "outputs" / late_name,
        (
            "Verified periodic response"
            if assessment["verified"]
            else "Late-time response: periodicity not established"
        ),
        X,
        Y,
        limits,
    )
    metadata = dict(
        results_sha256=sha256(
            (STUDY / "outputs/results.json").read_bytes()
        ).hexdigest(),
        periodic_verified=assessment["verified"],
        end_time_s=assessment["end_time_s"],
        cycles=round(assessment["end_time_s"] * d["config"]["forcing"]["frequency_Hz"]),
        startup_file="outputs/startup.gif",
        late_file="outputs/" + late_name,
        startup_cycles=3,
        late_cycles=2,
        frames_per_cycle=frame_count,
        playback_fps=16,
        steady_file="outputs/steady_contours.png",
        description="Quasi-1D values extended uniformly across the channel normal height; no transverse variation is resolved. Fixed plasma limits shared by both GIFs. Time shown is physical; playback is slowed. Startup loops with a deliberate restart at t=0; periodic animation covers two complete final periods.",
    )
    (STUDY / "outputs/visuals.json").write_text(json.dumps(metadata, indent=2))
    print(metadata, flush=True)
    from .report import build

    build()


if __name__ == "__main__":
    main()
