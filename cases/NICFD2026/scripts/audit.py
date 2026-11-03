"""Audit saved campaign evidence, input hashes and configured EOS coverage.

Run after the numerical campaign so property diagnostics do not compete with
solver timings. This script never modifies a solver state or property table.
"""

import hashlib
import inspect
import json
import os
from pathlib import Path
import platform
import sys

import jax
import jax.numpy as jnp
import jaxprop as jxp
import numpy as np

from nozzle_hyperbolic.models.two_fluid.thermodynamics import table_stencil_mask
from .case import (
    ROOT,
    FAMILIES,
    REGIMES,
    prepare_fluids,
    reference,
    specification,
    reservoir_states,
)
from .metrics import agreement
from .run import write_json


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def property_diagnostic(family, regime, fluids):
    data = reference(family, regime)
    config = specification(family, regime)
    bc = config["bc"]
    reservoirs = reservoir_states(family, bc["p0"], bc["T_g0"], bc["T_l0"], *fluids)
    reservoir_check = {
        phase: dict(
            computed_total_enthalpy=state["h"],
            published_inlet_total_enthalpy=float(
                data["h_" + phase][0] + 0.5 * data["u_" + phase][0] ** 2
            ),
            entropy=state["s"],
            density=state["rho"],
        )
        for phase, state in zip(("g", "l"), reservoirs)
    }
    phases = {}
    for phase, fluid in zip(("g", "l"), fluids):
        h, p = data["h_" + phase], data["p"]
        mask = np.asarray(table_stencil_mask(fluid, True))
        i = np.clip(
            np.floor((h - fluid.h_min) / fluid.delta_h).astype(int), 0, fluid.N_h - 2
        )
        j = np.clip(
            np.floor((np.log(p) - np.log(fluid.p_min)) / fluid.delta_logP).astype(int),
            0,
            fluid.N_p - 2,
        )
        good = (
            mask[i, j]
            & (h >= fluid.h_min)
            & (h <= fluid.h_max)
            & (p >= fluid.p_min)
            & (p <= fluid.p_max)
        )
        state = fluid.get_state(jxp.HmassP_INPUTS, jnp.asarray(h), jnp.asarray(p))
        table_path = Path(fluid.table_dir) / (fluid.table_name + ".pkl")
        phases[phase] = dict(
            table=table_path.name,
            table_sha256=fingerprint(table_path),
            reference_rows=len(h),
            invalid_stencil_rows=int(np.sum(~good)),
            invalid_positions=data["x"][~good],
            temperature_range_K=[float(np.min(state["T"])), float(np.max(state["T"]))],
            reference_temperature_range_K=[
                float(np.min(data["T_" + phase])),
                float(np.max(data["T_" + phase])),
            ],
            temperature_rmse_K=float(
                np.sqrt(
                    np.mean(
                        (np.asarray(state["T"])[1:-1] - data["T_" + phase][1:-1]) ** 2
                    )
                )
            ),
            temperature_max_error_K=float(
                np.max(abs(np.asarray(state["T"]) - data["T_" + phase]))
            ),
            density_relative_max_error=float(
                np.max(abs(np.asarray(state["rho"]) / data["rho_" + phase] - 1))
            ),
            density_relative_rmse=float(
                np.sqrt(
                    np.mean(
                        (
                            (
                                np.asarray(state["rho"])[1:-1]
                                - data["rho_" + phase][1:-1]
                            )
                            / data["rho_" + phase][1:-1]
                        )
                        ** 2
                    )
                )
            ),
        )
    failed = any(item["invalid_stencil_rows"] for item in phases.values())
    return dict(
        usable_on_published_profile=not failed,
        phases=phases,
        reservoirs=reservoir_check,
        explanation=(
            "The installed metastable liquid-table generator freezes rows after a bulk-modulus-ratio test. "
            "The generated table returns approximately room-temperature liquid at the published hot-liquid enthalpies. "
            "The current model rejects these copied stencils. No clipping, replacement EOS or weakened admissibility test was used. "
            "A valid flashing-water reproduction requires a validated table configuration covering the required phase states."
            if failed and family == "single_component"
            else (
                "The liquid table uses the supported 250 kJ/kg lower enthalpy bound to avoid the cold-water spinodal-ratio false trigger. Every published state remains inside the configured domain and passes the unchanged stencil checks. JAXProp source, HEOS, metastable phase, pressure range and table resolution are unchanged."
                if family == "single_component"
                else "The configured property stencils cover the published profile; this does not establish convergence of a current solve."
            )
        ),
        scope="EOS evaluated at published primitives; this is a property diagnostic, not a new CFD solution",
    )


def main():
    checks = []
    outcomes = []
    provenance = json.loads((ROOT / "provenance.json").read_text(encoding="utf-8"))
    for name, item in provenance["files"].items():
        raw = (ROOT / name).read_bytes()
        checks.append(
            dict(
                name="input:" + name,
                passed=fingerprint(ROOT / name) == item["sha256"]
                and hashlib.sha1(
                    b"blob " + str(len(raw)).encode() + b"\0" + raw
                ).hexdigest()
                == item["git_blob"],
            )
        )
    dependency = Path(inspect.getfile(jxp.FluidBicubic))
    dependencies = dict(
        bicubic_source=str(dependency), bicubic_source_sha256=fingerprint(dependency)
    )
    host = dict(
        platform=platform.platform(),
        machine=platform.machine(),
        processor=platform.processor(),
        logical_cpus=os.cpu_count(),
        backend=jax.default_backend(),
        devices=[str(device) for device in jax.devices()],
        x64=jax.config.jax_enable_x64,
        recorded="post-campaign evidence audit on the same host",
    )
    for family in FAMILIES:
        fluids, _ = prepare_fluids(specification(family, "subsonic"), family)
        for regime in REGIMES:
            directory = ROOT / family / regime / "outputs"
            data = json.loads((directory / "result.json").read_text(encoding="utf-8"))
            diagnostic = property_diagnostic(family, regime, fluids)
            diagnostic["dependency"] = dependencies
            diagnostic["host"] = host
            if family == "single_component":
                generation = json.loads(
                    (ROOT / family / "outputs/table_generation.json").read_text(
                        encoding="utf-8"
                    )
                )
                diagnostic["generation"] = generation
                checks.append(
                    dict(
                        name=f"{family}/{regime}:unmodified_jaxprop_since_table_generation",
                        passed=generation["dependency_source_unchanged"]
                        and generation["dependency_source_sha256"]
                        == dependencies["bicubic_source_sha256"],
                    )
                )
                checks.append(
                    dict(
                        name=f"{family}/{regime}:validated_liquid_cache",
                        passed=diagnostic["usable_on_published_profile"]
                        and diagnostic["phases"]["l"]["table_sha256"]
                        == generation["table_sha256"],
                    )
                )
            changed = [
                name
                for name, digest in data["source_before"].items()
                if data["source_after"].get(name) != digest
            ]
            # The broad snapshot includes every src file. These two unrelated
            # files were edited concurrently; neither belongs to this client's
            # import graph. Keep that fact rather than claiming the broad
            # working tree stayed unchanged or rerunning unchanged equations.
            unused = {
                "src/nozzle_hyperbolic/solvers/compilation.py": "nozzle_hyperbolic.solvers.compilation",
                "src/nozzle_hyperbolic/verification/problems.py": "nozzle_hyperbolic.verification.problems",
            }
            excluded = [
                name
                for name in changed
                if name in unused and unused[name] not in sys.modules
            ]
            source_audit = dict(
                broad_snapshot_unchanged=data["source_unchanged"],
                changed_files=changed,
                changed_files_outside_solver_import_graph=excluded,
                solver_sources_unchanged=all(name in excluded for name in changed),
                imported_modules=sorted(
                    name for name in sys.modules if name.startswith("nozzle_hyperbolic")
                ),
            )
            diagnostic["source_audit"] = source_audit
            checks.append(
                dict(
                    name=f"{family}/{regime}:reservoir_enthalpies_match_published_inlet",
                    passed=all(
                        np.isclose(
                            item["computed_total_enthalpy"],
                            item["published_inlet_total_enthalpy"],
                            rtol=1e-10,
                            atol=1e-5,
                        )
                        for item in diagnostic["reservoirs"].values()
                    ),
                )
            )
            write_json(directory / "property_audit.json", diagnostic)
            checks.append(
                dict(
                    name=f"{family}/{regime}:solver_sources_unchanged_during_execution",
                    passed=source_audit["solver_sources_unchanged"],
                )
            )
            checks.append(
                dict(
                    name=f"{family}/{regime}:execution_modes_accounted_for",
                    passed=[run["mode"] for run in data["runs"]] == ["cold", "warm"]
                    or (
                        bool(data.get("execution_failure"))
                        and [run["mode"] for run in data["runs"]] == ["cold"]
                    ),
                )
            )
            for run in data["runs"]:
                checks.append(
                    dict(
                        name=f"{family}/{regime}/{run['mode']}:requested_meshes_retained",
                        passed=[stage["cells"] for stage in run["stages"]]
                        == data["protocol"]["meshes"],
                    )
                )
                checks.append(
                    dict(
                        name=f"{family}/{regime}/{run['mode']}:timing_accounting",
                        passed=np.isclose(
                            run["total_s"],
                            sum(stage["total_s"] for stage in run["stages"]),
                        ),
                    )
                )
                for stage in run["stages"]:
                    if "fields_file" not in stage:
                        continue
                    with np.load(
                        directory / stage["fields_file"], allow_pickle=False
                    ) as saved:
                        identity = hashlib.sha256(
                            np.asarray(saved["state"], dtype=np.float64).tobytes()
                        ).hexdigest()
                        checks.append(
                            dict(
                                name=f"{family}/{regime}/{run['mode']}/{stage['cells']}:state_identity",
                                passed=identity == stage["final_state_sha256"],
                            )
                        )
                        checks.append(
                            dict(
                                name=f"{family}/{regime}/{run['mode']}/{stage['cells']}:field_dimensions",
                                passed=len(saved["x"]) == stage["cells"] + 2
                                and saved["state"].shape == (stage["cells"] + 2, 8),
                            )
                        )
                    checks.append(
                        dict(
                            name=f"{family}/{regime}/{run['mode']}/{stage['cells']}:honest_case_status",
                            passed=stage["case_success"]
                            == bool(
                                stage["success"]
                                and stage["audit"]["passed"]
                                and stage["regime_audit"]["passed"]
                            ),
                        )
                    )
            calculated = agreement(
                data["runs"][-1]["stages"][-1], data["protocol"]["agreement"]
            )
            final = data["runs"][-1]["stages"][-1]
            outcomes.append(
                dict(
                    family=family,
                    regime=regime,
                    valid_property_coverage=diagnostic["usable_on_published_profile"],
                    converged=bool(
                        final.get("success") and final.get("audit", {}).get("passed")
                    ),
                    verdict=calculated["status"],
                )
            )
            checks.append(
                dict(
                    name=f"{family}/{regime}:verdict_recomputed",
                    passed=calculated["status"] == data["verdict"]["status"],
                )
            )
    result = dict(
        passed=all(item["passed"] for item in checks),
        checks=checks,
        dependency=dependencies,
        scope="Evidence consistency and property coverage; passing this audit does not mean every CFD case converges or matches.",
        scientific_outcomes=outcomes,
    )
    write_json(ROOT / "outputs" / "audit.json", result)
    print(
        f"Evidence audit: {sum(c['passed'] for c in checks)}/{len(checks)} checks pass"
    )
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
