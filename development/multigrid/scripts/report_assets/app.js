/* Saved-data report application. No solver, filesystem or network calls. */
(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const COLORS = ['#6a00a8', '#fca636', '#cc4778', '#a62098', '#e97158'];
  const LABELS = {sequence: 'Grid sequencing', fas_fmg: 'FAS with coarse-to-fine initialization'};
  const CASES = {subsonic: 'Subsonic nozzle', shock_30: 'Internal shock · 30%', shock_70: 'Internal shock · 70%', adapted: 'Adapted expansion'};
  const RECON = {constant: 'Constant', muscl_van_leer: 'Linear MUSCL · Van Leer'};
  const state = {data: null, activePage: 'overview', enabled: new Set(), methodIds: [], theme: 'auto'};
  const escape = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const finite = value => typeof value === 'number' && Number.isFinite(value);
  const arr = value => Array.isArray(value) ? value : [];
  const last = values => values.length ? values[values.length - 1] : null;
  const maximum = value => {
    if (finite(value)) return value;
    const values = arr(value).filter(finite);
    return values.length ? Math.max(...values) : null;
  };
  const path = (value, key) => key.split('.').reduce((item, part) => item == null ? undefined : item[part], value);
  const first = (value, keys) => {
    for (const key of keys) { const answer = path(value, key); if (answer !== undefined && answer !== null) return answer; }
    return null;
  };
  const format = (value, digits = 3) => finite(value) ? (value === 0 ? '0' : Math.abs(value) < 1e-3 || Math.abs(value) >= 1e4 ? value.toExponential(2) : value.toLocaleString('en', {maximumFractionDigits: digits, minimumFractionDigits: 0})) : '—';
  const mean = values => values.length ? values.reduce((a, b) => a + b, 0) / values.length : null;
  const methodName = method => method.label || LABELS[method.method_id] || method.method_id;
  const caseName = group => group.label || CASES[group.case_id.split('/').pop()] || group.case_id;
  const color = method => COLORS[Math.max(0, state.methodIds.indexOf(method.method_id)) % COLORS.length];
  const samples = method => arr(method.samples);
  const elapsed = record => first(record, ['elapsed_s', 'solve_elapsed_s', 'wall_s', 'timing.elapsed_s']);
  const times = method => samples(method).map(elapsed).filter(finite);
  const requested = (method, group) => first(method, ['requested_repeats']) ?? first(group, ['requested_repeats']) ?? first(state.data.configuration, ['repeats', 'requested_repeats', 'measured_repeats']) ?? 10;
  const representative = method => {
    if (method.representative) return method.representative;
    const records = samples(method).filter(r => r.fields || r.history).sort((a, b) => (elapsed(a) || 0) - (elapsed(b) || 0));
    return records.length ? records[Math.floor(records.length / 2)] : method.warmup || {};
  };
  const flag = value => typeof value === 'boolean' ? value : typeof value?.passed === 'boolean' ? value.passed : null;
  const original = record => {
    const explicit = first(record, ['assessment.original_passed', 'assessment.original_checks_passed', 'assessment.converged', 'assessment.residual_passed', 'audit.original_passed', 'audit.original_checks_passed']);
    if (flag(explicit) !== null) return flag(explicit);
    const metrics = record.metrics || {};
    const names = ['finite', 'admissible', 'all_equations_passed', 'boundary_passed', 'constraints_passed', 'global_passed'];
    if (names.every(name => typeof metrics[name] === 'boolean')) return names.every(name => metrics[name]) && record.success !== false;
    return typeof record.success === 'boolean' ? record.success : null;
  };
  const analytical = record => flag(first(record, ['assessment.analytical_passed', 'assessment.accuracy_passed', 'assessment.accuracy.passed', 'audit.analytical_passed', 'analytical_passed']));
  const branch = record => flag(first(record, ['assessment.branch_passed', 'audit.branch_passed', 'branch_passed']));
  const agreement = method => flag(first(method, ['field_agreement', 'comparison', 'comparison.passed', 'same_root', 'assessment.field_agreement', 'assessment.same_root_passed']));
  const badge = (value, yes = 'Pass', no = 'Fail') => `<span class="badge ${value === true ? 'pass' : value === false ? 'fail' : 'pending'}">${value === true ? yes : value === false ? no : 'Unverified'}</span>`;
  const methodDot = method => `<span class="dot" style="background:${color(method)}"></span> ${escape(methodName(method))}`;

  function fields(record) {
    const f = record?.fields || record?.assessment?.fields || {};
    const values = f.values || f;
    return {...values, coordinate: f.coordinate || f.x || values.coordinate || values.x || record?.coordinate || record?.x || []};
  }

  function history(record) {
    if (Array.isArray(record.fine_history)) return record.fine_history;
    if (Array.isArray(record.history)) return record.history;
    const stages = arr(record.stages || record.representative_stages);
    return last(stages)?.history || [];
  }

  // Do not count the accepted=true initial-state observation as a pseudo-step.
  const acceptedStep = row => row.phase === 'accepted' || (row.accepted === true && row.phase !== 'initial' && row.phase !== 'rejected' && first(row, ['step', 'iteration']) > 0);

  function sequenceStages(record) {
    let rows = arr(record.stage_history);
    if (!rows.length) rows = arr(record.stages).flatMap(stage => arr(stage.history).map(row => ({...row, cells: row.cells ?? stage.cells})));
    const stages = [];
    for (const row of rows) {
      let stage = last(stages);
      if (!stage || stage.cells !== row.cells) {
        stage = {cells: row.cells, rows: [], accepted_steps: 0, rejected_steps: 0};
        stages.push(stage);
      }
      stage.rows.push(row);
      if (acceptedStep(row)) stage.accepted_steps++;
      if (row.phase === 'rejected' || row.accepted === false) stage.rejected_steps++;
    }
    let offset = 0;
    for (const stage of stages) {
      let accepted = 0;
      stage.start = offset;
      stage.points = stage.rows.filter(row => row.phase !== 'rejected' && row.accepted !== false).map(row => {
        if (acceptedStep(row)) accepted++;
        return {row, local_step: accepted, cumulative_step: offset + accepted};
      });
      stage.end = offset + stage.accepted_steps;
      const work = workRows(record).find(row => row.cells === stage.cells) || {};
      const progress = arr(record.work?.stage_progress).find(row => row.cells === stage.cells) || {};
      stage.elapsed_s = first(work, ['elapsed_s', 'solve_elapsed_s']);
      stage.completed_elapsed_s = progress.elapsed_s;
      offset = stage.end;
    }
    return stages;
  }

  function stepSummary(method) {
    const record = representative(method), stages = sequenceStages(record);
    if (method.method_id === 'sequence' && stages.length) return `${stages.reduce((sum, stage) => sum + stage.accepted_steps, 0)} total (${stages.map(stage => `N=${stage.cells}: ${stage.accepted_steps}`).join('; ')})`;
    const endpoint = last(history(record)) || {};
    const count = first(record, ['cycles', 'work.cycles']) ?? first(endpoint, ['cycle','step','iteration']);
    const early = first(record, ['work.early_converged_cycles']);
    return count === null ? '—' : `${count}${early > 0 ? ` (${early} finished after pre-smoothing)` : ''}`;
  }

  function initializationSummary(record) {
    const stages = [];
    for (const row of arr(record.initialization_history || record.work?.initialization_history)) {
      const previous = last(stages);
      if (previous && previous.cells === row.cells && previous.stage === row.stage) {
        stages[stages.length - 1] = {...row, observations: previous.observations + 1};
      } else stages.push({...row, observations: 1});
    }
    return stages;
  }

  function residualValue(row, mode) {
    if (mode === 'absolute') return maximum(first(row, ['equation_norm', 'metrics.equation_norm', 'residual_norm', 'original_residual']));
    if (mode === 'relative') return maximum(first(row, ['overall_relative', 'equation_relative', 'metrics.overall_relative']));
    if (mode === 'boundary') return maximum(first(row, ['boundary_max', 'metrics.boundary_max']));
    return maximum(first(row, ['global_defect', 'metrics.global_defect']));
  }

  function themeValues() {
    const css = getComputedStyle(document.documentElement);
    return Object.fromEntries(['paper','ink','muted','line','reference'].map(key => [key, css.getPropertyValue('--' + key).trim()]));
  }

  function layout(title, xTitle, yTitle, extra = {}) {
    const t = themeValues();
    return {
      title: {text: title, x: 0.055, y: .95, xanchor: 'left', font: {size: 13, color: t.ink}},
      paper_bgcolor: t.paper, plot_bgcolor: t.paper,
      font: {family: 'Segoe UI, Arial, sans-serif', size: 10, color: t.muted},
      margin: {l: 62, r: 20, t: 52, b: 58},
      xaxis: {title: {text: xTitle, standoff: 10}, gridcolor: t.line, zeroline: false, tickfont: {size: 9}, automargin: true},
      yaxis: {title: {text: yTitle, standoff: 10}, gridcolor: t.line, zeroline: false, tickfont: {size: 9}, automargin: true},
      legend: {orientation: 'h', x: 0, y: -.25, font: {size: 9}},
      hovermode: 'closest', hoverlabel: {font: {family: 'Segoe UI, Arial, sans-serif', size: 11}},
      ...extra,
    };
  }

  function plot(id, traces, figureLayout) {
    if (!traces.length) {
      if ($(id).classList.contains('js-plotly-plot')) Plotly.purge($(id));
      $(id).innerHTML = '<div class="empty-chart">No saved observations for this view.<br>Missing data have not been estimated.</div>';
      return;
    }
    if ($(id).querySelector('.empty-chart')) $(id).innerHTML = '';
    Plotly.react($(id), traces, figureLayout, {responsive: true, displaylogo: false, modeBarButtonsToRemove: ['lasso2d','select2d'], toImageButtonOptions: {format: 'svg', filename: 'nozzle_multigrid'}});
  }

  function timingStats(method) {
    const values = times(method);
    return {count: values.length, mean: mean(values), min: values.length ? Math.min(...values) : null, max: values.length ? Math.max(...values) : null};
  }

  function timingEligible(method, group) {
    return method.timing_eligible !== false && samples(method).length === requested(method, group) && samples(method).length > 0 && samples(method).every(r => original(r) === true && r.success !== false);
  }

  function startupEvents(record) {
    const contexts = [record.shared_compilation, record.setup_compilation, record.first_solve?.compilation]
      .filter(value => value && typeof value === 'object').map(value => value.stats || value);
    const sum = key => {
      const values = contexts.map(value => value[key]).filter(finite);
      return values.length ? values.reduce((total, value) => total + value, 0) : null;
    };
    return {duration: sum('compile_or_load_s'), requests: sum('backend_requests'), hits: sum('persistent_cache_hits')};
  }

  function renderStartup() {
    const probes = state.data.startup_probes || {};
    const records = arr(probes.records);
    $('startup-section').hidden = !records.length;
    const phases = ['empty_cache', 'populated_cache'];
    const phaseNames = {empty_cache: 'Empty executable cache', populated_cache: 'Populated executable cache'};
    const methodMap = new Map(state.data.groups.flatMap(group => group.methods).map(method => [method.method_id, method]));
    const name = record => methodName(methodMap.get(record.method_id) || record);
    const diagnostic = records.filter(record => record.first_solve?.success !== true || record.environment_verified !== true).length;
    $('startup-notice').hidden = false;
    $('startup-notice').textContent = records.length
      ? `${records.length} independent fresh-process observations${probes.complete === false ? '; requested probe coverage is incomplete' : ''}.${diagnostic ? ` ${diagnostic} have an unsuccessful first solve or unverified environment and remain diagnostic only.` : ' Every recorded first solve succeeded and its execution environment was verified.'} Cache-phase observations are single probes, not repeated warm timing samples.`
      : 'No independent startup probes are linked to this campaign. Excluded warm-campaign solves below are not substitutes for isolated cold-start measurements.';
    const process = [], compile = [];
    for (const phase of phases) {
      const selected = records.filter(record => record.phase === phase);
      if (!selected.length) continue;
      const x = selected.map(name);
      const colors = selected.map(record => color(methodMap.get(record.method_id) || record));
      const statuses = selected.map(record => `${record.case_id || 'case unrecorded'}; ${RECON[record.reconstruction] || record.reconstruction || 'reconstruction unrecorded'}; first solve ${record.first_solve?.success === true ? 'succeeded' : record.first_solve?.success === false ? 'failed' : 'unverified'}; environment ${record.environment_verified === true ? 'verified' : 'unverified'}`);
      const marker = {color: colors, opacity: phase === 'empty_cache' ? .9 : .55, pattern: {shape: phase === 'empty_cache' ? '' : '/'}};
      if (selected.some(record => finite(record.process_wall_s))) process.push({type: 'bar', name: phaseNames[phase], x, y: selected.map(record => record.process_wall_s), marker, customdata: statuses, hovertemplate: '%{y:.6g} s<br>%{customdata}<extra>%{fullData.name}</extra>'});
      const durations = selected.map(record => startupEvents(record).duration);
      if (durations.some(finite)) compile.push({type: 'bar', name: phaseNames[phase], x, y: durations, marker, customdata: statuses, hovertemplate: '%{y:.6g} s<br>%{customdata}<extra>%{fullData.name}</extra>'});
    }
    plot('startup-process-chart', process, layout('Independent process wall time', '', 'Process wall time [s]', {barmode: 'group'}));
    plot('startup-compile-chart', compile, layout('Observed backend compile / load events', '', 'Backend event duration [s]', {barmode: 'group'}));
    $('startup-table').innerHTML = records.map(record => {
      const observed = startupEvents(record);
      return `<tr><td>${escape(name(record))}</td><td>${escape(phaseNames[record.phase] || record.phase)}</td><td>${escape(record.case_id || '—')}</td><td>${escape(RECON[record.reconstruction] || record.reconstruction || '—')}</td><td>${format(record.process_wall_s, 4)}</td><td>${format(elapsed(record.first_solve || {}), 4)}</td><td>${format(observed.duration, 4)}</td><td>${format(observed.requests, 0)} / ${format(observed.hits, 0)}</td><td>${badge(record.first_solve?.success, 'Succeeded', 'Failed')}<span class="subline">${escape(record.first_solve?.reason || '')}</span></td><td>${badge(record.environment_verified, 'Verified', 'Unverified')}</td></tr>`;
    }).join('');
  }

  function createFindings() {
    const groups = state.data.groups;
    const pairs = [];
    let missing = 0, ineligible = 0;
    for (const group of groups) {
      const baseline = group.methods.find(m => m.method_id === 'sequence');
      for (const method of group.methods) {
        if (samples(method).length !== requested(method, group)) missing++;
        else if (!timingEligible(method, group)) ineligible++;
        if (baseline && method !== baseline && timingEligible(method, group) && timingEligible(baseline, group) && agreement(method) === true) {
          pairs.push({group, method, ratio: timingStats(method).mean / timingStats(baseline).mean});
        }
      }
    }
    const faster = pairs.filter(p => p.ratio < 1);
    const ratioText = pairs.length ? `Among ${pairs.length} fully timed comparisons with verified fine-field agreement, ${faster.length} have a lower mean than sequencing. Observed FAS / sequence ratios span ${format(Math.min(...pairs.map(p => p.ratio)))}–${format(Math.max(...pairs.map(p => p.ratio)))}.` : 'No fully timed, field-verified pair is available yet. The table retains descriptive timings and all unsuccessful outcomes.';
    const coverage = `${missing ? `${missing} configurations have incomplete or absent measurements. ` : ''}${ineligible ? `${ineligible} additional complete timing sets fail eligibility gates. ` : ''}`;
    const config = state.data.configuration || {}, sequenceCFL = config.steady || {};
    const fasCFL = {...(config.fas || {}), ...(config.method_settings?.fas_fmg || {})};
    const controllerText = `Sequencing uses initial CFL ${format(sequenceCFL.cfl)}, growth ${format(sequenceCFL.cfl_growth)} and cap ${format(sequenceCFL.cfl_max)}. Initialized FAS uses initial CFL ${format(fasCFL.cfl)}, growth ${format(fasCFL.cfl_growth)} and active-grid cap ${format(fasCFL.cfl_max)}${finite(fasCFL.coarse_cfl_max) ? `, with recursive forced coarse steps capped at ${format(fasCFL.coarse_cfl_max)}` : ''}. These results compare the complete strategies, including their controller choices. A matched-CFL diagnostic, when recorded separately below, tests whether the controller explains an advantage.`;
    let accuracyNotice = '';
    const shock30 = groups.find(group => group.case_id.endsWith('/shock_30') && group.reconstruction === 'constant');
    if (shock30) {
      const complete = shock30.methods.filter(method => samples(method).length === requested(method, shock30));
      const failedAccuracy = complete.filter(method => original(representative(method)) === true && analytical(representative(method)) === false);
      if (failedAccuracy.length) {
        const assessment = representative(failedAccuracy[0]).assessment || {};
        const error = first(assessment, ['shock.error_cells']);
        const target = first(assessment, ['accuracy_targets.shock_position_cells']);
        accuracyNotice = `<article class="finding wide accuracy-limitation"><strong>Constant reconstruction · 30% shock: retained accuracy miss</strong><p>The original equations converge, but the computed shock is ${format(error)} cells from the analytical location${finite(target) ? ` against a ${format(target)}-cell target` : ''}. ${failedAccuracy.length} complete timing sets remain visible as descriptive measurements and are excluded from validated timing comparisons. The fine-field agreement and accuracy flags remain separate in the ledger.</p></article>`;
      }
    }
    $('findings').innerHTML = `<article class="finding"><strong>Initialized FAS versus grid sequencing</strong><p>${escape(ratioText)}</p></article><article class="finding"><strong>Controller choices are part of this comparison</strong><p>${escape(controllerText)}</p></article><article class="finding"><strong>Initialization and rejected work are charged</strong><p>The measured FAS total begins before its original coarse solve and nested initialization. Coarse visits, transfer recovery, implicit corrections and rejected attempts remain included. A lower cycle count alone does not establish a faster solve.</p></article><article class="finding"><strong>First-use cost is a separate result</strong><p>${coverage}Excluded first solves retain compile/load observations where recorded. Their order and cache state matter; a warm runtime improvement does not establish shorter compilation.</p></article>${accuracyNotice}`;
  }

  function renderSummary() {
    const groups = state.data.groups;
    const all = groups.flatMap(group => group.methods.map(method => ({group, method})));
    const pass = all.filter(({method}) => original(representative(method)) === true).length;
    const count = all.reduce((sum, {method}) => sum + samples(method).length, 0);
    const warmups = all.filter(({method}) => method.warmup).length;
    const comparisons = all.filter(({method}) => method.method_id !== 'sequence' && samples(method).length > 0);
    const agrees = comparisons.filter(({method}) => agreement(method) === true).length;
    const complete = all.filter(({group, method}) => samples(method).length === requested(method, group) && samples(method).length > 0);
    const eligible = complete.filter(({group, method}) => timingEligible(method, group)).length;
    $('hero-count').textContent = `${pass} / ${all.length}`;
    $('hero-context').textContent = `${count} measured complete solves · ${warmups} excluded first solves. Failed and incomplete configurations remain visible.`;
    $('kpi-cases').textContent = new Set(groups.map(g => g.case_id)).size;
    const uniqueCases = [...new Map(groups.map(group => [group.case_id, group])).values()];
    $('kpi-case-labels').textContent = uniqueCases.map(caseName).join(' · ');
    const sizes = [...new Set(groups.flatMap(group => group.grids?.length ? [last(group.grids)] : [group.cells || group.finest_cells]).filter(finite))];
    const flux = first(state.data.configuration, ['scheme.flux']) || 'HLL';
    $('hero-tags').innerHTML = `<span>${uniqueCases.length} operating points</span><span>${new Set(groups.map(g=>g.reconstruction)).size} reconstructions</span><span>${escape(String(flux).toUpperCase())} flux</span><span>${sizes.length ? sizes.join(' / ') : '400'}-cell target</span>`;
    $('kpi-passed').textContent = `${pass} / ${all.length}`;
    $('kpi-samples').textContent = count;
    $('kpi-samples-sub').textContent = `${warmups} excluded first solves; ${all.reduce((n, {group, method}) => n + (Number(requested(method, group)) || 0), 0)} requested measurements`;
    $('kpi-eligible').textContent = `${eligible} / ${complete.length}`;
    $('kpi-eligible-sub').textContent = `${all.length - complete.length} incomplete sets; ${agrees}/${comparisons.length} measured FAS pairs agree with sequencing`;
    createFindings();
  }

  function renderTuning() {
    const tuning = state.data.tuning || {}, allRecords = arr(tuning.records);
    const records = allRecords.filter(record => record.kind !== 'matched_cfl_diagnostic');
    const diagnostics = allRecords.filter(record => record.kind === 'matched_cfl_diagnostic');
    $('tuning-panel').hidden = !records.length;
    $('tuning-protocol').textContent = typeof tuning.protocol === 'string' ? tuning.protocol : 'Retained FAS parameter pilots are shown separately from the final timing population.';
    $('tuning-selection').textContent = JSON.stringify(tuning.selection || {note: 'See the final campaign configuration for the selected settings.'}, null, 2);
    $('tuning-record').textContent = JSON.stringify(tuning, null, 2);
    $('tuning-table').innerHTML = records.map(record => {
      const duration = first(record, ['mean_s', 'elapsed_s', 'timing.mean_s']);
      const count = first(record, ['count', 'timing.count']) ?? (finite(record.elapsed_s) ? 1 : null);
      const name = record.label || record.method_id || record.variant || 'Unrecorded';
      const id = String(record.case_id || '').split('/').pop();
      const checks = [first(record, ['original_passed', 'assessment.original_checks_passed']), first(record, ['branch_passed', 'assessment.branch_passed']), first(record, ['analytical_passed', 'assessment.analytical_passed'])];
      return `<tr><td>${escape(CASES[id] || record.case_id || 'Unrecorded')}</td><td>${escape(RECON[record.reconstruction] || record.reconstruction || 'Unrecorded')}</td><td>${escape(name)}${record.phase ? `<span class="subline">${escape(record.phase)}</span>` : ''}</td><td>${format(count, 0)}</td><td>${format(duration, 5)}</td><td>${badge(flag(record.success), 'Succeeded', 'Failed')}</td><td>${checks.map(value => badge(flag(value))).join(' / ')}</td><td>${escape(record.reason || '')}</td><td><details><summary>Recorded settings</summary><pre>${escape(JSON.stringify(record.settings || {}, null, 2))}</pre></details></td></tr>`;
    }).join('');
    $('matched-control').hidden = !diagnostics.length;
    const diagnosticMean = record => first(record, ['mean_s', 'elapsed_s', 'timing.mean_s']);
    const eligible = record => typeof record.timing_eligible === 'boolean' ? record.timing_eligible : (
      record.success === true && record.original_passed === true && record.branch_passed === true && record.analytical_passed === true && flag(record.field_agreement) === true
      && (record.timing_environment_verified === true || record.environment_verified === true)
    );
    const reference = record => diagnostics.find(other => other.case_id === record.case_id && other.reconstruction === record.reconstruction && other.phase === record.phase && String(other.method_id).startsWith('sequence'));
    const pairs = diagnostics.filter(record => !String(record.method_id).startsWith('sequence')).flatMap(record => {
      const baseline = reference(record), value = diagnosticMean(record), base = baseline ? diagnosticMean(baseline) : null;
      return baseline && finite(value) && finite(base) && base > 0 && eligible(record) && eligible(baseline) ? [{ratio: value / base}] : [];
    });
    $('matched-summary').textContent = pairs.length
      ? `With matched shared-CFL parameters, ${pairs.filter(pair => pair.ratio < 1).length} / ${pairs.length} validated comparisons have a lower FAS mean. FAS / matched-sequence mean ratios span ${format(Math.min(...pairs.map(pair => pair.ratio)))}–${format(Math.max(...pairs.map(pair => pair.ratio)))}. These observations test the controller sensitivity of the primary ranking; they are not pooled with its samples.`
      : 'Recorded matched shared-CFL observations appear below. No fully validated timing pair is available for a speed comparison; values remain descriptive.';
    $('matched-table').innerHTML = diagnostics.map(record => {
      const baseline = reference(record), value = diagnosticMean(record), base = baseline ? diagnosticMean(baseline) : null;
      const ratio = finite(value) && finite(base) && base > 0 ? value / base : null;
      const descriptive = !eligible(record) || !baseline || !eligible(baseline);
      const id = String(record.case_id || '').split('/').pop();
      const quality = [record.original_passed, record.branch_passed, record.analytical_passed].map(value => badge(flag(value))).join(' / ');
      const range = `${format(first(record, ['min_s', 'timing.min_s']), 5)}–${format(first(record, ['max_s', 'timing.max_s']), 5)}`;
      return `<tr><td>${escape(CASES[id] || record.case_id || 'Unrecorded')}</td><td>${escape(RECON[record.reconstruction] || record.reconstruction || 'Unrecorded')}</td><td>${escape(record.label || LABELS[record.method_id] || record.method_id)}</td><td>${format(first(record, ['count', 'timing.count']), 0)}</td><td>${format(value, 5)}</td><td>${escape(range)}</td><td>${String(record.method_id).startsWith('sequence') ? 'Reference' : `${format(ratio)}×${descriptive ? '<span class="subline">descriptive</span>' : ''}`}</td><td>${quality}</td><td>${badge(eligible(record), 'Validated', 'Descriptive')}</td></tr>`;
    }).join('');
  }

  function renderOverview() {
    renderSummary();
    const groups = state.data.groups;
    const all = groups.flatMap(group => group.methods.map(method => ({group, method})));
    const traces = [];
    for (const id of state.methodIds) {
      const x = [], y = [], errorUp = [], errorDown = [], text = [];
      let methodTemplate = null;
      for (const group of groups) {
        const method = group.methods.find(m => m.method_id === id);
        if (!method) continue;
        methodTemplate = method;
        const stats = timingStats(method);
        x.push(`${caseName(group)}<br>${RECON[group.reconstruction] || group.reconstruction}`);
        y.push(stats.mean);
        errorUp.push(stats.mean === null ? null : stats.max - stats.mean);
        errorDown.push(stats.mean === null ? null : stats.mean - stats.min);
        text.push(`n=${stats.count}/${requested(method, group)}; ${timingEligible(method, group) ? 'original checks passed' : 'incomplete or failed original checks'}`);
      }
      if (methodTemplate && y.some(finite)) traces.push({type: 'bar', name: methodName(methodTemplate), x, y, marker: {color: color(methodTemplate)}, error_y: {type: 'data', array: errorUp, arrayminus: errorDown, color: themeValues().muted, thickness: 1, width: 3}, text, hovertemplate: '%{x}<br>%{fullData.name}: %{y:.5g} s<br>%{text}<extra></extra>'});
    }
    plot('all-timings', traces, layout('Complete solve · mean and min–max range', '', 'Warm wall time [s]', {barmode: 'group', margin: {l: 60, r: 20, t: 55, b: 100}, legend: {orientation: 'h', x: 0, y: -.3, font: {size: 9}}}));
    $('overview-table').innerHTML = all.map(({group, method}) => {
      const stats = timingStats(method), baseline = group.methods.find(m => m.method_id === 'sequence');
      const base = baseline ? timingStats(baseline) : null;
      const ratio = base?.mean && stats.mean ? stats.mean / base.mean : null;
      const sameRoot = method.method_id === 'sequence' ? null : agreement(method);
      const descriptive = method.method_id !== 'sequence' && !(timingEligible(method, group) && baseline && timingEligible(baseline, group) && sameRoot === true);
      const rep = representative(method);
      return `<tr><td>${escape(caseName(group))}</td><td>${escape(RECON[group.reconstruction] || group.reconstruction)}</td><td>${methodDot(method)}</td><td>${stats.count}/${escape(requested(method, group))}</td><td>${format(stats.mean, 4)}</td><td>${format(stats.min, 4)}–${format(stats.max, 4)}</td><td>${method.method_id === 'sequence' ? 'Reference' : ratio === null ? '—' : `${format(ratio)}×${descriptive ? '<span class="subline">descriptive</span>' : ''}`}</td><td>${badge(original(rep))}</td><td>${badge(branch(rep))}</td><td>${badge(analytical(rep))}${rep.assessment?.accuracy_assessed === false ? '<span class="subline">below recommended resolution</span>' : ''}</td><td>${method.method_id === 'sequence' ? 'Baseline' : badge(sameRoot)}</td></tr>`;
    }).join('');
    renderTuning();
    renderStartup();
  }

  function currentGroup() {
    return state.data.groups.find(g => g.case_id === $('physical-case').value && g.reconstruction === $('reconstruction').value);
  }

  function activeMethods(group) { return group.methods.filter(m => state.enabled.has(m.method_id)); }

  function profile(group, methods, key, name, units) {
    const traces = [];
    const reference = fields({fields: group.reference || group.reference_fields || {}});
    if (arr(reference[key]).length) traces.push({x: reference.coordinate, y: reference[key], type: 'scatter', mode: 'lines', name: 'Analytical reference', line: {color: themeValues().reference, width: 2, dash: 'dot'}, hovertemplate: `x=%{x:.5g} m<br>${name}=%{y:.6g} ${units}<extra>Analytical reference</extra>`});
    for (const method of methods) {
      const rep = representative(method), f = fields(rep);
      if (!arr(f[key]).length) continue;
      traces.push({x: f.coordinate, y: f[key], type: 'scatter', mode: 'lines', name: methodName(method) + (rep.success === false ? ' · failed state' : ''), line: {color: color(method), width: 1.7, dash: rep.success === false ? 'dash' : 'solid'}, hovertemplate: `x=%{x:.5g} m<br>${name}=%{y:.6g} ${units}<extra>%{fullData.name}</extra>`});
    }
    plot('profile-' + key, traces, layout(name, 'Axial coordinate [m]', `${name}${units ? ' [' + units + ']' : ''}`));
  }

  function workRows(record) {
    const work = record.work || {};
    let rows = work.levels || work.by_level || work.level_work || record.level_work;
    if (Array.isArray(rows)) return rows;
    if (rows && typeof rows === 'object') return Object.entries(rows).map(([cells, values]) => ({cells: Number(cells), ...(typeof values === 'object' ? values : {sweeps: values})}));
    if (Object.keys(work).some(key => /^\d+$/.test(key))) return Object.entries(work).filter(([key]) => /^\d+$/.test(key)).map(([cells, values]) => ({cells: Number(cells), ...values}));
    const stages = arr(record.stages || record.representative_stages);
    return stages.map(stage => ({cells: stage.cells, accepted_steps: arr(stage.history).filter(h => h.phase === 'accepted').length, ...stage.work}));
  }

  const WORK_LABELS = {jacobian_builds: 'Exact implicit Jacobian builds', jacobian_reuses: 'Reused Jacobians', jacobian_reuse_hits: 'Reused Jacobians', jacobian_refreshes: 'Jacobian refreshes', frozen_attempts: 'Frozen Jacobian attempts', aggregate_spatial_builds: 'Coarse spatial block projections', aggregate_fine_spatial_jacobian_builds: 'Fine spatial Jacobian AD builds', aggregate_implicit_builds: 'Aggregated implicit matrix builds', aggregate_jacobian_uses: 'Supplied-matrix step attempts', implicit_newton_iterations: 'Implicit Newton corrections', recovery_iterations: 'State recovery iterations', newton_or_recovery_iterations: 'Combined Newton / recovery iterations', residual_evaluations: 'Residual evaluations', residual_calls: 'Residual calls', accepted_steps: 'Accepted pseudo-steps', rejected_steps: 'Rejected pseudo-steps', sweeps: 'Recorded smoother sweeps', smoother_steps: 'Smoother steps', implicit_steps: 'Implicit steps', newton_iterations: 'Newton corrections', linear_solves: 'Linear solves', recovery_calls: 'State recovery calls', boundary_solves: 'Boundary solves'};
  const counterName = key => WORK_LABELS[key] || key.replaceAll('_', ' ');

  function workMetrics(rows) {
    const preferred = ['jacobian_builds', 'jacobian_refreshes', 'aggregate_fine_spatial_jacobian_builds', 'aggregate_spatial_builds', 'aggregate_implicit_builds', 'aggregate_jacobian_uses', 'frozen_attempts', 'implicit_newton_iterations', 'recovery_iterations', 'jacobian_reuses', 'jacobian_reuse_hits', 'residual_evaluations', 'residual_calls', 'accepted_steps', 'rejected_steps', 'implicit_steps', 'newton_iterations', 'linear_solves', 'recovery_calls'];
    const legacyOnly = key => /^(aggregate_|frozen_attempts$|jacobian_refreshes$)/.test(key);
    const keys = [...new Set(rows.flatMap(row => Object.keys(row).filter(key => finite(row[key]) && !key.endsWith('_s') && !['cells', 'last_cfl', 'initial_cfl'].includes(key))))]
      .filter(key => !legacyOnly(key) || rows.some(row => finite(row[key]) && row[key] !== 0))
      .filter(key => key !== 'newton_or_recovery_iterations' || !rows.some(row => finite(row.implicit_newton_iterations) && finite(row.recovery_iterations)));
    return [...preferred.filter(key => keys.includes(key)), ...keys.filter(key => !preferred.includes(key))];
  }

  function renderCaseCost(group, methods) {
    const timing = [];
    for (const method of methods) {
      const values = times(method);
      if (!values.length) continue;
      timing.push({type: 'box', name: methodName(method), y: values, boxpoints: 'all', jitter: .35, pointpos: 0, marker: {color: color(method), size: 5}, line: {color: color(method), width: 1.4}, boxmean: true, hovertemplate: '%{y:.6g} s<extra>%{fullData.name}</extra>'});
    }
    plot('case-timing', timing, layout('All measured complete-solve samples', '', 'Warm wall time [s]', {showlegend: false, margin: {l: 63, r: 20, t: 52, b: 75}}));
    const allRows = methods.flatMap(m => workRows(representative(m)));
    const keys = workMetrics(allRows), previous = $('work-metric').value;
    $('work-metric').innerHTML = keys.map(key => `<option value="${escape(key)}">${escape(counterName(key))}</option>`).join('');
    const metric = keys.includes(previous) ? previous : keys[0];
    if (metric) $('work-metric').value = metric;
    const work = metric ? methods.map(method => {
      const rows = workRows(representative(method));
      return {type: 'bar', name: methodName(method), x: rows.map(row => `${row.cells} cells`), y: rows.map(row => finite(row[metric]) ? row[metric] : null), customdata: rows.map(row => JSON.stringify(row)), marker: {color: color(method)}, hovertemplate: '%{x}<br>%{y}<br>%{customdata}<extra>%{fullData.name}</extra>'};
    }).filter(trace => trace.y.some(finite)) : [];
    plot('work-chart', work, layout('Recorded work by level', '', counterName(metric || 'recorded_operations'), {barmode: 'group'}));
    const durations = [...new Set(allRows.flatMap(row => Object.keys(row).filter(key => key.endsWith('_s') && finite(row[key]))))]
      .filter(key => !key.startsWith('aggregate_') || allRows.some(row => finite(row[key]) && row[key] !== 0));
    $('work-counter-head').innerHTML = '<tr><th>Strategy</th><th>Grid</th>' + keys.map(key => `<th>${escape(counterName(key))}</th>`).join('') + durations.map(key => `<th>${escape(key === 'elapsed_s' ? 'Level wall [s]' : counterName(key.slice(0, -2)) + ' [s]')}</th>`).join('') + '</tr>';
    $('work-counter-table').innerHTML = methods.flatMap(method => workRows(representative(method)).map(row => `<tr><td>${methodDot(method)}</td><td>${escape(row.cells)}</td>${keys.map(key => `<td>${format(row[key], 0)}</td>`).join('')}${durations.map(key => `<td>${format(row[key], 5)}</td>`).join('')}</tr>`)).join('');
    const firstSolve = [], compile = [], names = [], colors = [], info = [];
    for (const method of methods) {
      if (!method.warmup) continue;
      names.push(methodName(method)); colors.push(color(method)); firstSolve.push(elapsed(method.warmup));
      const observation = first(method.warmup, ['compilation.stats', 'compilation', 'compile_report']) || {};
      compile.push(first(observation, ['compile_or_load_s', 'backend_compile_load_s', 'duration_s']));
      const requests = first(observation, ['backend_requests', 'requests']);
      const hits = first(observation, ['persistent_cache_hits', 'cache_hits']);
      info.push(`${requests ?? 'unrecorded'} backend requests; ${hits ?? 'unrecorded'} persistent hits`);
    }
    const compilation = [];
    if (firstSolve.some(finite)) compilation.push({type: 'bar', x: names, y: firstSolve, name: 'Excluded complete solve', marker: {color: colors, opacity: .7}, customdata: info, hovertemplate: '%{y:.6g} s<br>%{customdata}<extra>Excluded solve</extra>'});
    if (compile.some(finite)) compilation.push({type: 'bar', x: names, y: compile, name: 'Observed backend compile/load', marker: {color: colors, pattern: {shape: '/'}}, customdata: info, hovertemplate: '%{y:.6g} s<br>%{customdata}<extra>Compile/load events</extra>'});
    plot('compilation-chart', compilation, layout('First-solve observations · not additive', '', 'Wall / event duration [s]', {barmode: 'group'}));
  }

  function renderConvergence(methods) {
    const mode = $('residual-mode').value, axis = $('residual-axis').value, scope = $('history-scope').value;
    const labels = {absolute: 'Worst original equation RMS', relative: 'Saved native relative residual', boundary: 'Boundary defect', global: 'Global balance defect'};
    const traces = [], noTime = [], shapes = [], annotations = [], stageRows = [];
    const initialization = methods.flatMap(method => initializationSummary(representative(method)).map(row => ({method, row})));
    $('initialization-table').innerHTML = initialization.map(({method, row}) => `<tr><td>${methodDot(method)}</td><td>${escape(row.cells)}</td><td>${escape(row.stage)}${row.observations > 1 ? ` (${row.observations} records)` : ''}</td><td>${format(row.level_smoother_steps, 0)}</td><td>${format(row.smoother_steps, 0)}</td><td>${format(row.elapsed_s, 5)}</td><td>${row.accepted === true ? 'Accepted' : row.accepted === false ? 'No update' : 'Unrecorded'}</td></tr>`).join('') || '<tr><td colspan="7">No saved nested initialization history is selected.</td></tr>';
    for (const method of methods) {
      const record = representative(method), stages = method.method_id === 'sequence' ? sequenceStages(record) : [];
      if (stages.length) {
        for (const stage of stages) {
          const endpoint = last(stage.points)?.row || {};
          stageRows.push(`<tr><td>${methodDot(method)}</td><td>${escape(stage.cells)}</td><td>${stage.accepted_steps}</td><td>${stage.rejected_steps}</td><td>${stage.start}–${stage.end}</td><td>${format(stage.elapsed_s, 5)}</td><td>${format(stage.completed_elapsed_s, 5)}</td><td>${format(residualValue(endpoint, mode))}</td></tr>`);
        }
        const selected = scope === 'fine' ? stages.slice(-1) : stages;
        if (axis === 'time') noTime.push(methodName(method));
        selected.forEach((stage, index) => {
          const title = `${methodName(method)} · N=${stage.cells} (${stage.accepted_steps} steps)`;
          if (axis === 'time') {
            const progress = arr(record.work?.stage_progress).find(row => row.cells === stage.cells);
            const value = progress ? residualValue(progress, mode) : null;
            if (finite(progress?.elapsed_s) && finite(value) && value > 0) traces.push({type: 'scatter', mode: 'markers', name: title + ' · stage end', x: [progress.elapsed_s], y: [value], marker: {color: color(method), size: 8, symbol: ['circle','square','diamond','triangle-up'][index % 4]}, hovertemplate: 'Actual stage completion: %{x:.6g} s<br>%{y:.5e}<extra>%{fullData.name}</extra>'});
            return;
          }
          const points = stage.points.filter(point => finite(residualValue(point.row, mode)) && residualValue(point.row, mode) > 0);
          if (points.length) traces.push({type: 'scatter', mode: 'lines+markers', name: title, x: points.map(point => scope === 'fine' ? point.local_step : point.cumulative_step), y: points.map(point => residualValue(point.row, mode)), customdata: points.map(point => [point.local_step, point.row.cfl, stage.cells]), line: {color: color(method), width: 1.8, dash: ['solid','dash','dot','dashdot'][index % 4]}, marker: {color: color(method), size: 3, symbol: ['circle','square','diamond','triangle-up'][index % 4]}, hovertemplate: 'Cumulative/view step=%{x}<br>Grid N=%{customdata[2]}, accepted step=%{customdata[0]}<br>Defect=%{y:.5e}<br>CFL=%{customdata[1]:.5g}<extra>%{fullData.name}</extra>'});
          if (scope === 'all') {
            if (index > 0) shapes.push({type: 'line', xref: 'x', yref: 'paper', x0: stage.start, x1: stage.start, y0: 0, y1: 1, line: {color: themeValues().line, width: 1, dash: 'dot'}});
            annotations.push({xref: 'x', yref: 'paper', x: (stage.start + stage.end) / 2, y: 1.02, text: `N=${stage.cells}: ${stage.accepted_steps}`, showarrow: false, font: {size: 9, color: color(method)}});
          }
        });
        continue;
      }
      const rows = history(record).filter(row => row.phase !== 'rejected' && row.accepted !== false && finite(residualValue(row, mode)));
      const x = rows.map((row, index) => axis === 'time' ? first(row, ['elapsed_s', 'wall_s', 'time_s', 'wall_elapsed_s']) : first(row, ['cycle', 'step', 'iteration', 'accepted_steps']) ?? index);
      const valid = rows.map((row, index) => finite(x[index]) && residualValue(row, mode) > 0);
      if (axis === 'time' && !x.some(finite)) { noTime.push(methodName(method)); continue; }
      if (valid.some(Boolean)) traces.push({type: 'scatter', mode: 'lines+markers', name: methodName(method), x: x.filter((_, index) => valid[index]), y: rows.map(row => residualValue(row, mode)).filter((_, index) => valid[index]), customdata: rows.map(row => row.phase || 'recorded state').filter((_, index) => valid[index]), line: {color: color(method), width: 1.8}, marker: {color: color(method), size: rows.map(row => row.phase === 'fmg_initialization' ? 7 : rows.length > 100 ? 2 : 3).filter((_, index) => valid[index]), symbol: rows.map(row => row.phase === 'fmg_initialization' ? 'diamond' : 'circle').filter((_, index) => valid[index])}, hovertemplate: '%{x:.6g}<br>%{y:.5e}<br>Phase: %{customdata}<extra>%{fullData.name}</extra>'});
    }
    plot('residual-chart', traces, layout(labels[mode], axis === 'time' ? 'Recorded elapsed wall time [s]' : scope === 'all' ? 'Cumulative accepted sequence steps / FAS cycle visits' : 'Finest-grid accepted steps / FAS cycle visits', labels[mode], {yaxis: {...layout('', '', '').yaxis, type: 'log'}, shapes, annotations, margin: {l: 62, r: 20, t: 66, b: 115}, legend: {orientation: 'h', x: 0, y: -.28, font: {size: 8}}}));
    $('sequence-stage-table').innerHTML = stageRows.join('') || '<tr><td colspan="8">No sequencing stage histories are selected.</td></tr>';
    $('history-caption').textContent = `One actual representative trajectory per method. ${scope === 'all' ? 'Grid sequencing includes every saved grid; separate traces and dotted boundaries mark changes of discretization. The x coordinate accumulates accepted steps across grids.' : 'The curve view is restricted to the finest grid; the stage ledger still reports every sequencing grid.'} FAS curves monitor the original finest-grid problem; auxiliary forced coarse residuals are not physical convergence tests. Cycle counts include visits that finish after pre-smoothing when the original fine target has already passed. A pseudo-step and a cycle visit are different units of work.${initialization.length ? ' FMG initialization occurs before fine cycle 1: an x=0 residual drop (diamond marker) includes charged initialization work. Both complete-solve timing and the recorded elapsed axis start before that work; the initialization ledger and per-level counters retain it.' : ''} Exact zeros remain unchanged in downloads and are omitted from log curves.${mode === 'relative' ? ' Native relative norms reset at each sequencing grid; compare absolute residuals when assessing their common scale.' : ''}${noTime.length ? ` No per-iterate sequence wall times are inferred: only saved stage-completion markers are shown where available. Missing per-iterate times: ${noTime.join(', ')}.` : ''}`;
  }

  function renderExplorer() {
    const group = currentGroup();
    if (!group) {
      $('case-notice').hidden = false; $('case-notice').textContent = 'This operating point / reconstruction combination was not recorded.';
      return;
    }
    const methods = activeMethods(group);
    const notices = group.methods.filter(m => !samples(m).length || representative(m).success === false).map(m => `${methodName(m)}: ${representative(m).reason || (!samples(m).length ? 'no measured samples' : 'unsuccessful')}`);
    $('case-notice').hidden = notices.length === 0;
    $('case-notice').textContent = notices.join(' · ');
    const cells = first(group, ['cells', 'finest_cells']) ?? last(arr(group.grids)) ?? first(state.data.configuration, ['finest_cells', 'cells']) ?? '400';
    $('case-meta').innerHTML = `<span>${escape(group.case_id)}</span><span>${escape(RECON[group.reconstruction] || group.reconstruction)}</span><span>${escape(cells)} finest cells</span><span>${escape(group.methods.length)} strategies retained</span>`;
    profile(group, methods, 'p', 'Static pressure', 'Pa'); profile(group, methods, 'M', 'Mach number', '');
    profile(group, methods, 'u', 'Velocity', 'm/s'); profile(group, methods, 'rho', 'Density', 'kg/m³');
    renderConvergence(methods);
    renderCaseCost(group, methods);
    $('case-table').innerHTML = group.methods.map(method => {
      const rep = representative(method), metrics = rep.metrics || {};
      const levels = workRows(rep);
      const rejected = levels.length && levels.every(row => finite(row.rejected_steps))
        ? levels.reduce((sum, row) => sum + row.rejected_steps, 0)
        : first(rep, ['work.rejected_steps', 'rejected_steps']);
      return `<tr><td>${methodDot(method)}</td><td>${badge(rep.success, 'Converged', 'Failed')}</td><td title="${escape(rep.reason)}">${escape(rep.reason || 'unrecorded')}</td><td>${format(maximum(metrics.equation_norm))}</td><td>${format(metrics.boundary_max)}</td><td>${format(maximum(metrics.global_defect))}</td><td>${escape(stepSummary(method))}</td><td>${format(rejected, 0)}</td></tr>`;
    }).join('');
    $('case-diagnostics').textContent = JSON.stringify(group.methods.map(method => ({method_id: method.method_id, requested_repeats: requested(method, group), measured_samples: samples(method).length, field_agreement: method.field_agreement ?? method.comparison ?? null, warmup: {success: method.warmup?.success, reason: method.warmup?.reason, elapsed_s: elapsed(method.warmup || {}), compilation: method.warmup?.compilation, assessment: method.warmup?.assessment}, representative: {success: representative(method).success, reason: representative(method).reason, metrics: representative(method).metrics, work: representative(method).work, initialization_history: representative(method).initialization_history || representative(method).work?.initialization_history || [], assessment: representative(method).assessment}})), null, 2);
  }

  function renderProtocol() {
    const metadata = state.data.metadata || {};
    const initialized = state.data.groups.flatMap(group => group.methods.filter(method => method.method_id !== 'sequence'));
    const settings = [...new Set(initialized.map(method => JSON.stringify(representative(method).fas_settings || {})))];
    const selected = settings.length === 1 ? JSON.parse(settings[0]) : {};
    const labels = {fmg_cycles: 'intermediate initialization cycles', fmg_prolongation: 'conservative initialization prolongation', pre_sweeps: 'pre-smoothing steps', post_sweeps: 'post-smoothing steps', coarse_steps: 'coarsest step cap', coarse_reconstruction: 'coarse reconstruction', cfl: 'initial CFL', cfl_growth: 'CFL growth', coarse_reduction: 'coarse defect-reduction goal', smoother: 'implicit smoother'};
    const displayed = Object.keys(labels).filter(key => key in selected).map(key => `${labels[key]}: ${selected[key] === null && key === 'coarse_reconstruction' ? 'same as finest' : selected[key]}`);
    if(finite(selected.coarse_cfl_max)) displayed.push(`forced coarse CFL cap: ${selected.coarse_cfl_max}`);
    if(selected.fmg_min_cfl > 0) displayed.push(`CFL floor after prolongation: ${selected.fmg_min_cfl}`);
    if(selected.finest_early_stop) displayed.push('finish a cycle after pre-smoothing when every original fine-grid check passes');
    if(selected.fmg_inexact_initialization) displayed.push('intentionally inexact initialization smoothing enabled');
    if(finite(selected.fmg_coarse_tolerance)) displayed.push(`auxiliary coarsest initialization target: ${selected.fmg_coarse_tolerance}`);
    if(selected.smoother && selected.smoother !== 'exact') for(const [key, value] of Object.entries(selected)) if(/^(smoother_|inner_|inexact)/.test(key)) displayed.push(`${key.replaceAll('_', ' ')}: ${value}`);
    $('fmg-settings').textContent = settings.length === 1
      ? `One FAS setting set is used across the reported groups. ${displayed.join('; ')}. Full settings and per-level work remain in Protocol & evidence and the case diagnostics.`
      : `${settings.length} distinct FAS setting sets occur in the saved groups. Their case mapping and complete values are retained in the configuration and case diagnostics; the report does not treat different settings as one universal configuration.`;
    const repeatCounts = [...new Set(state.data.groups.flatMap(group => group.methods.map(method => requested(method, group))))];
    $('timing-protocol').textContent = `Each successful configuration receives one excluded complete solve and ${repeatCounts.join(' / ')} requested measured complete ${repeatCounts.length === 1 && repeatCounts[0] === 1 ? 'solve' : 'solves'}. Warm totals charge all grid levels, transfers, recovery, boundary completion, monitoring and rejected work. Numerical state and controller histories reset for every repetition. The exact per-configuration completed counts appear in the ledger.`;
    $('configuration').textContent = JSON.stringify(state.data.configuration || state.data.protocol || {}, null, 2);
    $('metadata').textContent = JSON.stringify(metadata, null, 2);
    $('audits').textContent = JSON.stringify(state.data.audits || state.data.assessment || state.data.comparisons || {}, null, 2);
    $('startup-protocol').textContent = JSON.stringify(state.data.startup_probes || {records: [], note: 'No independent startup.json file was supplied.'}, null, 2);
    $('report-provenance').textContent = JSON.stringify(state.data.report_provenance || {}, null, 2);
    const revision = first(metadata, ['source.revision', 'source.git_commit', 'git_revision', 'revision', 'commit']) || 'See embedded provenance';
    const sourceUnchanged = first(metadata, ['source_unchanged', 'source.unchanged', 'src_unchanged']) ?? first(state.data, ['audits.source_unchanged', 'audits.package_source_unchanged']);
    const backend = first(metadata, ['backend', 'environment.backend', 'environment.jax_backend']) || 'See embedded environment';
    $('provenance-cards').innerHTML = `<article class="settings-card"><h3>Measured source</h3><p>${escape(revision)}</p><p>${sourceUnchanged === true ? badge(true, 'src/ unchanged') : sourceUnchanged === false ? badge(false, '', 'Source changed') : 'Source identity and before/after hashes are retained below.'}</p></article><article class="settings-card"><h3>Execution environment</h3><p>${escape(typeof backend === 'object' ? JSON.stringify(backend) : backend)}</p><p>Precision, hardware and library versions are campaign metadata.</p></article><article class="settings-card"><h3>Presentation identity</h3><p>Saved-data rebuild; no numerical solve.</p><p>Embedded source summary SHA-256: ${escape(Object.values(state.data.report_provenance?.input_sha256 || {})[0] || '—')}</p></article>`;
    $('footer-id').textContent = first(metadata, ['campaign_id', 'id']) || state.data.campaign_id || 'Complete campaign records embedded';
  }

  function download(name, value, mime = 'application/json') {
    const blob = new Blob([typeof value === 'string' ? value : JSON.stringify(value, null, 2)], {type: mime});
    const url = URL.createObjectURL(blob), anchor = document.createElement('a');
    anchor.href = url; anchor.download = name; document.body.appendChild(anchor); anchor.click(); anchor.remove(); setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  function timingCSV() {
    const rows = [['case_id','reconstruction','method_id','sample','elapsed_s','success','reason','original_checks','analytical_accuracy','field_agreement']];
    for (const group of state.data.groups) for (const method of group.methods) samples(method).forEach((record, index) => rows.push([group.case_id, group.reconstruction, method.method_id, index, elapsed(record), record.success, record.reason, original(record), analytical(record), agreement(method)]));
    return rows.map(row => row.map(value => '"' + String(value ?? '').replaceAll('"', '""') + '"').join(',')).join('\r\n');
  }

  function navigate(page) {
    state.activePage = page;
    document.querySelectorAll('.page').forEach(element => element.classList.toggle('active', element.id === page));
    document.querySelectorAll('.navigation button').forEach(button => { const active = button.dataset.page === page; button.classList.toggle('active', active); active ? button.setAttribute('aria-current', 'page') : button.removeAttribute('aria-current'); });
    if (page === 'overview') renderOverview();
    if (page === 'explorer') renderExplorer();
    if (page === 'theory' && window.MathJax?.typesetPromise) MathJax.typesetPromise([$('theory')]).catch(() => {});
    try { window.history.replaceState(null, '', '#' + page); } catch (_) {}
  }

  function setTheme(value) {
    state.theme = value;
    const dark = value === 'dark' || (value === 'auto' && window.matchMedia?.('(prefers-color-scheme: dark)').matches);
    document.documentElement.dataset.theme = dark ? 'dark' : 'light';
    try { localStorage.setItem('nh-multigrid-report-theme', value); } catch (_) {}
    if (state.data) { if (state.activePage === 'overview') renderOverview(); if (state.activePage === 'explorer') renderExplorer(); }
  }

  function bind() {
    const cases = [...new Map(state.data.groups.map(group => [group.case_id, group])).values()];
    $('physical-case').innerHTML = cases.map(group => `<option value="${escape(group.case_id)}">${escape(caseName(group))}</option>`).join('');
    const methodMap = new Map(state.data.groups.flatMap(group => group.methods).map(method => [method.method_id, method]));
    state.methodIds = [...methodMap.keys()]; state.enabled = new Set(state.methodIds);
    $('methods').innerHTML = state.methodIds.map(id => { const method = methodMap.get(id); return `<label class="method-chip"><input type="checkbox" value="${escape(id)}" checked><span class="dot" style="background:${color(method)}"></span>${escape(methodName(method))}</label>`; }).join('');
    $('methods').addEventListener('change', event => { event.target.checked ? state.enabled.add(event.target.value) : state.enabled.delete(event.target.value); renderExplorer(); });
    for (const id of ['physical-case','reconstruction','residual-mode','residual-axis','history-scope','work-metric']) $(id).addEventListener('change', renderExplorer);
    document.querySelectorAll('.navigation button').forEach(button => button.addEventListener('click', () => navigate(button.dataset.page)));
    $('theme').addEventListener('change', event => setTheme(event.target.value));
    $('download-data').addEventListener('click', () => download('multigrid_complete_campaign.json', state.data));
    $('download-csv').addEventListener('click', () => download('multigrid_timing_samples.csv', timingCSV(), 'text/csv'));
    $('download-fields').addEventListener('click', () => { const group = currentGroup(); if (group) download('multigrid_selected_profiles.json', {case_id: group.case_id, reconstruction: group.reconstruction, reference: group.reference, methods: activeMethods(group).map(method => ({method_id: method.method_id, success: representative(method).success, fields: fields(representative(method))}))}); });
    $('download-history').addEventListener('click', () => { const group = currentGroup(); if (group) download('multigrid_selected_histories.json', {case_id: group.case_id, reconstruction: group.reconstruction, methods: activeMethods(group).map(method => ({method_id: method.method_id, fine_history: history(representative(method)), stage_history: representative(method).stage_history || [], stage_summary: sequenceStages(representative(method)).map(({rows, points, ...stage}) => stage), stages: representative(method).stages || [], initialization_history: representative(method).initialization_history || representative(method).work?.initialization_history || [], work: representative(method).work}))}); });
    window.matchMedia?.('(prefers-color-scheme: dark)').addEventListener?.('change', () => { if (state.theme === 'auto') setTheme('auto'); });
    renderSummary();
    renderProtocol();
  }

  async function start() {
    let appearance = 'auto'; try { appearance = localStorage.getItem('nh-multigrid-report-theme') || 'auto'; } catch (_) {}
    if (!['auto','light','dark'].includes(appearance)) appearance = 'auto';
    $('theme').value = appearance; setTheme(appearance);
    const bytes = Uint8Array.from(atob($('report-data').textContent.trim()), c => c.charCodeAt(0));
    if (!window.DecompressionStream) throw new Error('This offline report needs a modern browser with gzip DecompressionStream support.');
    const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'));
    state.data = JSON.parse(await new Response(stream).text());
    bind();
    const page = location.hash.slice(1);
    navigate(['overview','explorer','theory','protocol'].includes(page) ? page : 'overview');
    window.MultigridReport = {state, navigate, renderSummary, renderExplorer, renderOverview, renderStartup, startupEvents, currentGroup, original, analytical, branch, agreement, timingStats, fields, history, sequenceStages, acceptedStep, stepSummary, initializationSummary, workRows, workMetrics, setTheme, timingCSV};
    document.documentElement.dataset.reportReady = 'true';
    window.dispatchEvent(new Event('multigrid-report-ready'));
  }

  start().catch(error => { $('load-error').hidden = false; $('load-error').textContent = 'The embedded report could not be opened: ' + error.message; console.error(error); });
})();
