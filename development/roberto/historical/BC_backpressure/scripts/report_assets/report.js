/* One offline document: shared libraries, independent result panels and theory. */
(() => {
  "use strict";
  const bundles = JSON.parse(document.getElementById("report-data").textContent);
  const themeKey = "nozzle-backpressure-bc.report.theme";
  const names = {source: "Current/source BC", pressure_wave: "New pressure-wave BC"};
  const format = (x, digits = 3) => Number.isFinite(x) ? x.toExponential(digits) : "undefined";
  const escape = x => String(x ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
  const controllers = {};
  let active = "uniform";
  let ready = Promise.resolve();
  let mathResolve;
  const mathReady = new Promise(resolve => { mathResolve = resolve; });
  window.addEventListener("backpressure-theory-ready", () => {
    document.getElementById("theory-theory-status").textContent = "Equations rendered locally; no network connection required";
    mathResolve();
  });

  function palette() {
    const css = getComputedStyle(document.documentElement);
    const value = name => css.getPropertyValue(name).trim();
    return {bg: value("--panel"), ink: value("--ink"), muted: value("--muted"),
      axis: value("--line"), grid: value("--grid"), source: value("--source"),
      pressure_wave: value("--working"), reference: value("--reference"), initial: value("--initial")};
  }
  function fail(error) {
    document.documentElement.dataset.reportError = error.message;
    console.error(error);
  }

  function initializeResults(initialization, BUNDLE) {
    const root = document.getElementById(initialization);
    const $ = id => document.getElementById(`${initialization}-${id}`);
    const DATA = BUNDLE.report;
    const plots = new Map();
    let revision = 0;
    let queue = Promise.resolve();
    let pending = 0;
  const uniform = DATA.metadata.selected_initialization === "uniform";
  const cases = DATA.cases;
  const runs = cases.flatMap(c => c.runs.map(r => ({...r, case: c})));
  const controller = DATA.metadata.controller;
  const tolerance = controller.spatial_rms_tolerance;
  const config = name => ({responsive: true, displaylogo: false, scrollZoom: false,
    doubleClick: "reset", modeBarButtonsToRemove: ["lasso2d", "select2d"],
    toImageButtonOptions: {format: "svg", filename: name, width: 1400, height: 800, scale: 1}});

  function axis(title, extra = {}) {
    const p = palette();
    return {title: {text: title, standoff: 10, font: {size: 12}}, ticks: "inside",
      ticklen: 5, tickfont: {size: 11}, tickcolor: p.axis, color: p.ink,
      showline: true, mirror: true, linecolor: p.axis, linewidth: 1,
      showgrid: true, gridcolor: p.grid, zeroline: false, automargin: true,
      exponentformat: "power", ...extra};
  }

  function frame(caseId, kind, xTitle, yTitle) {
    const p = palette();
    return {paper_bgcolor: p.bg, plot_bgcolor: p.bg,
      font: {family: '"Segoe UI", Arial, sans-serif', size: 12, color: p.ink},
      margin: {l: 65, r: 22, t: 25, b: 90}, height: 390,
      xaxis: axis(xTitle), yaxis: axis(yTitle), hovermode: "closest", dragmode: "zoom",
      legend: {orientation: "h", x: 0, y: -.24, font: {size: 10}, bgcolor: "rgba(0,0,0,0)"},
      hoverlabel: {bgcolor: p.bg, bordercolor: p.axis, font: {family: '"Segoe UI", Arial, sans-serif', color: p.ink, size: 12}},
      uirevision: `${caseId}:${kind}:${revision}`};
  }

  function trace(x, y, name, color, extra = {}) {
    return {type: "scatter", mode: "lines", x, y, name,
      line: {color, width: 1.9}, connectgaps: false,
      hovertemplate: "%{x:.7g}<br>%{y:.7g}<extra>%{fullData.name}</extra>", ...extra};
  }

  function physical(c, values) {
    const x = c.x_m;
    return values.length === x.length ? {x: x.slice(1, -1), y: values.slice(1, -1)}
      : {x: x.slice(1, 1 + values.length), y: values};
  }

  async function plot(id, traces, layout) {
    await Plotly.react($(id), traces, layout, config(id));
    plots.set(id, {traces, layout});
  }

  async function drawFigures() {
    const p = palette();
    const work = [];
    cases.forEach((c, index) => {
      const historyTraces = c.runs.map(run => {
        const h = run.history;
        return trace(h.map(v => v.accepted_index), h.map(v => Math.max(v.residual_rms, 1e-18)),
          names[run.variant], p[run.variant], {mode: h.length === 1 ? "markers" : "lines+markers",
            marker: {size: h.length === 1 ? 8 : 2.7, color: p[run.variant],
              symbol: run.variant === "source" ? "circle-open" : "circle", line: {width: 1.4}}});
      });
      let layout = frame(c.id, "residual", "Accepted implicit update index", "Original spatial residual RMS");
      layout.yaxis.type = "log";
      layout.shapes = [{type: "line", xref: "paper", x0: 0, x1: 1, yref: "y",
        y0: tolerance, y1: tolerance, line: {color: p.muted, width: 1, dash: "dashdot"}}];
      work.push(plot(`residual-${index}`, historyTraces, layout));

      const cflTraces = [];
      c.runs.forEach(run => {
        const h = run.history.filter(v => v.accepted_index > 0 && v.effective_cfl != null);
        if (h.length) cflTraces.push(trace(h.map(v => v.accepted_index), h.map(v => v.effective_cfl),
          names[run.variant], p[run.variant], {mode: "lines+markers", marker: {size: 2.5, color: p[run.variant]}}));
      });
      layout = frame(c.id, "cfl", "Accepted implicit update index", "Actual accepted CFL");
      layout.yaxis.type = "log";
      layout.shapes = [{type: "line", xref: "paper", x0: 0, x1: 1, yref: "y",
        y0: controller.nominal_cfl, y1: controller.nominal_cfl,
        line: {color: p.muted, width: 1, dash: "dot"}}];
      layout.annotations = c.runs.some(r => r.variant === "source" && r.iterations === 0) ? [{xref: "paper", yref: "paper", x: .02, y: .98,
        xanchor: "left", yanchor: "top", showarrow: false,
        text: "Current/source BC: 0 accepted updates<br>Actual CFL is undefined",
        font: {size: 10, color: p.muted}, bgcolor: p.bg, borderpad: 4}] : [];
      work.push(plot(`cfl-${index}`, cflTraces, layout));

      for (const field of ["p", "M"]) {
        const fieldTraces = [];
        if ($("show-analytical").checked) {
          const a = physical(c, c.reference[field]);
          fieldTraces.push(trace(a.x, a.y, "Analytical shocked profile", p.reference,
            {line: {color: p.reference, width: 1.8, dash: "dash"}}));
        }
        if ($("show-initial").checked) {
          const a = physical(c, c.runs[0].initial_diagnostics.fields[field]);
          fieldTraces.push(trace(a.x, a.y, uniform ? "Uniform initial field" : "Adapted initial field", p.initial,
            {line: {color: p.initial, width: 1.6, dash: "dot"}}));
        }
        c.runs.forEach(run => {
          const a = physical(c, run.diagnostics.fields[field]);
          fieldTraces.push(trace(a.x, a.y, names[run.variant], p[run.variant],
            {line: {color: p[run.variant], width: 2, dash: run.variant === "source" ? "dashdot" : "solid"}}));
        });
        if (field === "p") fieldTraces.forEach(t => { t.y = t.y.map(v => v / 1000); });
        layout = frame(c.id, field, "Axial position [m]", field === "p" ? "Static pressure [kPa]" : "Mach number");
        if (field === "p" && $("pressure-range").checked) layout.yaxis.range = [0, 1.15 * Math.max(...c.reference.p) / 1000];
        if (field === "M") layout.shapes = [{type: "line", xref: "paper", x0: 0, x1: 1, yref: "y", y0: 1, y1: 1,
          line: {color: p.muted, width: 1, dash: "dot"}}];
        work.push(plot(`${field === "p" ? "pressure" : "mach"}-${index}`, fieldTraces, layout));
      }
    });
    await Promise.all(work);
  }

  function scheduleFigures() {
    pending += 1;
    root.dataset.figuresReady = "false";
    queue = queue.then(drawFigures).then(() => {
      pending -= 1;
      if (!pending) {
        root.dataset.figuresReady = "true";
        $("report-status").textContent = `${cases.length} target cases · ${runs.length} saved implicit runs · all figures use fixed numerical data`;
      }
    }).catch(error => { fail(error); throw error; });
    return queue;
  }

  function tables() {
    $("case-body").innerHTML = cases.map(c => `<tr><th>${escape(c.label)}</th><td class="numeric">${c.p_back_ratio.toFixed(6)}</td><td class="numeric">${(c.p_back_ratio * 200).toFixed(3)}</td><td class="numeric">${c.id === "shock35" ? "135" : "170"}</td><td>${DATA.metadata.mesh.n_points} points including two ghosts</td></tr>`).join("");
    $("settings-body").innerHTML = runs.map(run => {
      const s = {...DATA.metadata.settings, ...run.solver_settings};
      return `<tr><th>${escape(run.case.id)}</th><td>${names[run.variant]}</td><td class="numeric">${s.CFL}</td><td class="numeric">Fixed growth</td><td class="numeric">${format(s.ser_cfl_max, 0)}</td><td class="numeric">${s.dt_growth_factor}</td><td class="numeric">${controller.max_halvings}</td><td class="numeric">${s.newton_iter}</td><td class="numeric">${format(s.newton_tol, 0)}</td></tr>`;
    }).join("");
    $("controller-description").textContent = `Maximum ${controller.max_trials_per_update} timestep trials per update. Increment limits over ${controller.increment_check_scope.toLowerCase()}: |Δp|/p ≤ ${controller.relative_increment_limits.p}, |Δh|/|h| ≤ ${controller.relative_increment_limits.h}, and |Δu|/(|u| + a) ≤ ${controller.relative_increment_limits.u_scaled}. Maximum ${controller.max_accepted_updates} accepted updates.`;
    $("cfl-body").innerHTML = runs.map(run => {
      const values = run.history.filter(h => h.accepted_index > 0 && Number.isFinite(h.effective_cfl)).map(h => h.effective_cfl);
      const rejected = run.backend.rejected_trials ?? run.history.at(-1)?.rejected_trials;
      return `<tr><th>${escape(run.case.id)}</th><td>${names[run.variant]}</td><td class="numeric">${values.length}</td><td class="numeric">${values.length ? format(Math.min(...values)) : "undefined"}</td><td class="numeric">${values.length ? format(Math.max(...values)) : "undefined"}</td><td class="numeric">${rejected ?? "not recorded"}</td></tr>`;
    }).join("");
    $("outcome-body").innerHTML = runs.map(run => {
      const d = run.diagnostics;
      const status = run.physical_success ? '<span class="badge pass">Verified shock</span>'
        : run.root_success && d.exit_M > 1 ? '<span class="badge other">Supersonic branch</span>' : '<span class="badge other">Not verified</span>';
      return `<tr><th>${escape(run.case.id)}<span class="subtext">${names[run.variant]}</span></th><td class="numeric">${run.iterations}</td><td class="numeric">${format(d.residual_rms)}</td><td>${status}</td><td class="numeric">${d.exit_M.toFixed(4)}</td><td class="numeric">${format(d.boundary_pressure_relative_error, 2)}</td><td class="numeric">${format(d.mass_imbalance, 2)}</td><td class="numeric">${d.shock_location_m == null ? "none" : (d.shock_location_m * 1000).toFixed(2)}</td><td class="numeric">${format(Math.max(...Object.values(d.normalized_L1)), 2)}</td><td class="numeric">${run.wall_s.toFixed(3)}</td></tr>`;
    }).join("");
    const working = runs.filter(r => r.variant === "pressure_wave");
    const source = runs.filter(r => r.variant === "source");
    $("main-finding").innerHTML = `The new pressure-wave outlet reached the verified internal-shock solution in <strong>${working.filter(r => r.physical_success).length}/${working.length}</strong> cases. The current/source outlet reached that branch in <strong>${source.filter(r => r.physical_success).length}/${source.length}</strong> cases from the same ${uniform ? "uniform stagnation" : "adapted"} initialization. Both residual convergence and physical branch selection are reported below.`;
    $("seed-description").textContent = uniform ? "Both variants use the same interior p = 200 kPa, T = 300 K and u = 0 m/s. Their algebraic ghosts follow their respective boundary laws; neither analytical profiles nor adapted fields initialize these runs." : `The common adapted seed was computed by source implicit Euler: ${DATA.adapted.iterations} accepted updates, final spatial RMS ${format(DATA.adapted.diagnostics.residual_rms)}, recorded solve duration ${DATA.adapted.wall_s.toFixed(4)} s (includes checked-step first-call compilation; not a benchmark). Its interior SHA256 is ${DATA.adapted.initial_point_sha256}.`;
    const verification = BUNDLE.verification;
    $("audit-summary").textContent = `Recorded local boundary checks: ${verification.boundary.local_cases.length}; ${verification.boundary.passed ? "passed" : "not passed"}. Independent saved-state audit: ${verification.saved_state.passed && verification.saved_state_matches_summary ? "passed and matches this exact summary" : "not verified for this summary"}. Live production source unchanged: ${verification.saved_state.live_source_unchanged ? "yes" : "not established"}.`;
    $("settings-detail").textContent = JSON.stringify({metadata: DATA.metadata,
      per_run: runs.map(r => ({case: r.case.id, variant: r.variant, settings: r.solver_settings, backend: r.backend})),
      provenance: BUNDLE.provenance}, null, 2);
    $("audit-detail").textContent = JSON.stringify(verification, null, 2);
  }

  $("reset-plots").addEventListener("click", () => { revision += 1; scheduleFigures(); });
  for (const id of ["show-analytical", "show-initial", "pressure-range"]) $(id).addEventListener("change", scheduleFigures);
  $("download-csv").addEventListener("click", () => {
    const url = URL.createObjectURL(new Blob([BUNDLE.csv], {type: "text/csv;charset=utf-8"}));
    const a = document.createElement("a"); a.href = url; a.download = `BC_backpressure_${DATA.metadata.selected_initialization}_outcomes.csv`; a.click(); URL.revokeObjectURL(url);
  });

    tables();
    return {bundle: BUNDLE, plots, ready: () => queue, render: scheduleFigures};
  }
  for (const initialization of ["uniform", "adapted"]) {
    controllers[initialization] = initializeResults(initialization, bundles[initialization]);
  }

  function activate(tab, target = tab) {
    if (!["uniform", "adapted", "theory"].includes(tab)) tab = "uniform";
    active = tab;
    document.body.dataset.page = tab;
    document.documentElement.dataset.reportReady = "false";
    document.querySelectorAll("[data-report-page]").forEach(panel => { panel.hidden = panel.dataset.reportPage !== tab; });
    document.querySelectorAll("[data-report-nav]").forEach(nav => { nav.hidden = nav.dataset.reportNav !== tab; });
    document.querySelectorAll("[data-report-tab]").forEach(link => {
      if (link.dataset.reportTab === tab) link.setAttribute("aria-current", "page");
      else link.removeAttribute("aria-current");
    });
    document.getElementById("sidebar-label").textContent = {
      uniform: "Uniform start", adapted: "Adapted start", theory: "Theory & equations"
    }[tab];
    ready = Promise.all([mathReady, controllers[tab]?.render()]).then(() => {
      if (active !== tab) return;
      document.documentElement.dataset.reportReady = "true";
      if (target === tab) window.scrollTo({top: 0, left: 0, behavior: "instant"});
      else document.getElementById(target)?.scrollIntoView({behavior: "instant"});
    }).catch(fail);
    return ready;
  }
  function navigate() {
    const target = decodeURIComponent(location.hash.slice(1)) || "uniform";
    return activate(target.split("-")[0], target);
  }
  document.addEventListener("click", event => {
    const link = event.target.closest('a[href^="#"]');
    if (!link) return;
    const target = link.getAttribute("href").slice(1);
    const tab = target.split("-")[0];
    if (!["uniform", "adapted", "theory"].includes(tab)) return;
    event.preventDefault();
    history.pushState(null, "", `#${target}`);
    activate(tab, target);
  });
  window.addEventListener("hashchange", navigate);
  function setTheme(theme) {
    document.documentElement.dataset.theme = theme;
    document.querySelectorAll("[data-theme-choice]").forEach(button => {
      button.setAttribute("aria-pressed", String(button.dataset.themeChoice === theme));
    });
    try { localStorage.setItem(themeKey, theme); } catch {}
    activate(active, active);
  }
  document.querySelectorAll("[data-theme-choice]").forEach(button => {
    button.addEventListener("click", () => setTheme(button.dataset.themeChoice));
    button.setAttribute("aria-pressed", String(button.dataset.themeChoice === document.documentElement.dataset.theme));
  });
  document.getElementById("print-report").addEventListener("click", async () => {
    document.body.classList.add("printing");
    await Promise.all([mathReady, ...Object.values(controllers).map(c => c.render())]);
    window.print();
    document.body.classList.remove("printing");
    activate(active);
  });
  document.getElementById("license-button").addEventListener("click", () => {
    const details = document.getElementById("license-details");
    details.hidden = false;
    details.open = !details.open;
    details.scrollIntoView();
  });
  window.backpressureBCReport = {bundles, controllers, ready: () => ready,
    get active() { return active; }, selectTab: activate};
  navigate();
})();
