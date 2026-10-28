"""Figures and a complete point-by-point report from saved backpressure runs."""

import numpy as np
import matplotlib.pyplot as plt
import jaxprop as jxp

from development.roberto.backpressure_sensitivity.scripts.reference import PerfectGasReference


def render(report, output):
    plt.rcParams.update({'font.size': 10, 'axes.grid': True, 'grid.alpha': .2})
    runs = [r for r in report['runs'] if r['mode'] == 'descending']
    cold = [r for r in report['runs'] if r['mode'] == 'cold']
    restarted = [r for r in report['runs'] if r['mode'] == 'supersonic_restart']
    ref = PerfectGasReference(jxp.FluidPerfectGas(name='Air'))
    x = np.asarray(report['x_m'])
    reference_runs = [ref.profile(x, r['p_back_ratio']) for r in runs]
    ratios = np.array([r['p_back_ratio'] for r in runs])
    cmap = plt.colormaps['plasma']
    colors = cmap(np.linspace(.2, .8, len(runs)))

    def finish(fig, name):
        fig.savefig(output / (name + '.png'), dpi=190, bbox_inches='tight')
        fig.savefig(output / (name + '.pdf'), bbox_inches='tight')
        plt.close(fig)

    def landmarks(ax):
        for r, label in [(ref.critical, 'choking'), (ref.shock_exit, 'shock at exit'), (ref.adapted, 'adapted')]:
            ax.axvline(r, color='.5', ls=':', lw=.9)
            ax.text(r, 1.01, label, rotation=65, transform=ax.get_xaxis_transform(),
                    va='bottom', fontsize=8, color='.3')
        ax.set_xlim(1.005, 0)
        ax.set_xlabel(r'Imposed backpressure $p_b/p_0$ (decreasing →)')

    def series(ax, key, exact=None, scale=1):
        values = np.array([np.nan if r[key] is None else r[key] for r in runs], dtype=float) / scale
        ax.plot(ratios, values, color=cmap(.35), lw=1.5, marker='o', ms=3, label='Numerical continuation')
        if exact is not None:
            ax.plot(ratios, np.asarray(exact) / scale, color='black', ls='--', lw=1.2, label='Exact nozzle interior')
        if cold:
            ax.scatter([r['p_back_ratio'] for r in cold], [np.nan if r[key] is None else r[key] / scale for r in cold],
                       edgecolor=cmap(.75), facecolor='none', s=48, label='Independent simple start', zorder=4)
        if restarted:
            ax.scatter([r['p_back_ratio'] for r in restarted],
                       [np.nan if r[key] is None else r[key] / scale for r in restarted],
                       color=cmap(.55), marker='^', s=40, label='Supersonic restart (branch probe)', zorder=4)
        failed = [r for r in runs + cold if not r['converged']]
        if failed:
            ax.scatter([r['p_back_ratio'] for r in failed], [np.nan if r[key] is None else r[key] / scale for r in failed],
                       color='red', marker='x', s=45, label='Not converged', zorder=5)
        landmarks(ax)

    fig, axes = plt.subplots(3, 2, figsize=(13, 12), layout='constrained')
    specs = [('mdot_kg_s', [r['mdot_exact_kg_s'] for r in runs], r'Mass flow $\dot m$ [kg/s]'),
             ('entropy_generation_W_K', [r['entropy_generation_exact_W_K'] for r in runs],
              r'Interior entropy-flux rise [W/K]'),
             ('exit_T_K', [r['exit_T_exact_K'] for r in runs], 'Last interior temperature [K]'),
             ('exit_p_ratio', [r['exit_p_exact_ratio'] for r in runs], r'Last interior pressure $p/p_0$'),
             ('throat_M', [r['M'][np.argmin(report['area_m2'])] for r in reference_runs], 'Throat Mach number'),
             ('exit_M', [r['M'][-2] for r in reference_runs], 'Last interior Mach number')]
    for ax, (key, exact, label) in zip(axes.flat, specs):
        series(ax, key, exact)
        ax.set_ylabel(label)
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].plot(ratios, [r['shock_entropy_generation_including_exit_W_K'] for r in runs],
                    color=cmap(.65), ls=':', lw=1, label='Exact shock loss incl. exit plane')
    axes[0, 1].legend(fontsize=8)
    fig.suptitle('Nozzle operating map: same source implicit kernel, perfect-gas properties', fontsize=14)
    finish(fig, 'operating_map')

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout='constrained')
    series(axes[0, 0], 'ghost_p_ratio')
    axes[0, 0].plot(ratios, ratios, 'k--', lw=1, label='Imposed backpressure')
    axes[0, 0].plot(ratios, [r['exit_p_ratio'] for r in runs], color=cmap(.7), label='Last physical point')
    axes[0, 0].set_ylabel(r'Outlet pressures $p/p_0$')
    axes[0, 0].legend(fontsize=8)
    series(axes[0, 1], 'outlet_pressure_weight')
    axes[0, 1].set_ylabel('Subsonic pressure-imposition weight w')
    series(axes[1, 0], 'shock_estimate_m', [np.nan if r['shock_exact_m'] is None else r['shock_exact_m'] for r in runs])
    axes[1, 0].set_ylabel('Internal shock position [m]')
    series(axes[1, 1], 'pressure_L1_p0')
    axes[1, 1].plot(ratios, [r['mach_L1'] for r in runs], color=cmap(.7), label='Mach L1')
    axes[1, 1].set_ylabel('Mean absolute reference error')
    axes[1, 1].legend(fontsize=8)
    finish(fig, 'boundary_behavior')

    # Six panels keep curves readable across very different flow regimes.
    groups = [lambda r: r >= ref.critical + .002,
              lambda r: abs(r - ref.critical) <= .0021,
              lambda r: ref.shock_exit + .0021 < r < ref.critical - .0021,
              lambda r: abs(r - ref.shock_exit) <= .0021,
              lambda r: ref.adapted + .0021 < r < ref.shock_exit - .0021,
              lambda r: r <= ref.adapted + .0021]
    titles = ['Subsonic / stagnant', 'Choking neighborhood', 'Internal shocks',
              'Shock-at-exit neighborhood', 'Overexpanded: exterior adjustment', 'Adapted / underexpanded']
    for quantity, filename, ylabel, factor in [('p', 'pressure_profiles', r'$p/p_0$', report['p0_Pa']),
                                               ('M', 'mach_profiles', 'Mach number', 1.0)]:
        fig, axes = plt.subplots(2, 3, figsize=(16, 10), layout='constrained')
        for ax, choose, title in zip(axes.flat, groups, titles):
            selected = [i for i, r in enumerate(runs) if choose(r['p_back_ratio'])]
            if len(selected) > 6:
                selected = [selected[i] for i in np.linspace(0, len(selected) - 1, 6, dtype=int)]
            for i in selected:
                row, exact, color = runs[i], reference_runs[i], colors[i]
                ax.plot(x[1:-1], np.array(row['flow_state'][quantity])[1:-1] / factor,
                        color=color, lw=1.5, label=f'{row["p_back_ratio"]:.6f}' + (' FAIL' if not row['converged'] else ''))
                ax.plot(x[1:-1], exact[quantity][1:-1] / factor, color=color, ls='--', lw=.9)
            ax.set(title=title, xlabel='Axial position [m]', ylabel=ylabel)
            if selected:
                ax.legend(title=r'$p_b/p_0$', fontsize=7, title_fontsize=8)
        fig.suptitle('Solid: numerical; dashed: analytic interior; boundary/ghost points excluded', fontsize=13)
        finish(fig, filename)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10), layout='constrained')
    for ax, choose, title in zip(axes.flat, groups, titles):
        selected = [i for i, r in enumerate(runs) if choose(r['p_back_ratio'])]
        if len(selected) > 7:
            selected = [selected[i] for i in np.linspace(0, len(selected) - 1, 7, dtype=int)]
        for i in selected:
            h = runs[i]['history']
            ax.semilogy([r['update'] for r in h], [max(r['spatial_rms'], 1e-18) for r in h],
                        color=colors[i], label=f'{runs[i]["p_back_ratio"]:.6f}')
        ax.axhline(1e-8, color='.5', ls=':')
        ax.set(title=title, xlabel='Accepted implicit updates', ylabel='Fixed-scale spatial RMS')
        if selected:
            ax.legend(fontsize=7)
    finish(fig, 'convergence_histories')

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout='constrained')
    series(axes[0], 'accepted_updates')
    axes[0].set_ylabel('Accepted updates (warm-start continuation)')
    series(axes[1], 'face_mass_spread_relative')
    axes[1].set_yscale('log')
    axes[1].set_ylabel('Relative range of actual numerical face mass flux')
    finish(fig, 'solver_work_and_conservation')

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout='constrained')
    series(axes[0, 0], 'mdot_kg_s', [r['mdot_exact_kg_s'] for r in runs])
    axes[0, 0].set(xlim=(1, .985), ylim=(-.001, .036), ylabel='Mass flow [kg/s]', title='Near-rest relative accuracy')
    series(axes[0, 1], 'throat_M', [r['M'][np.argmin(report['area_m2'])] for r in reference_runs])
    axes[0, 1].set(xlim=(.942, .930), ylim=(.74, 1.03), ylabel='Throat Mach', title='Delayed numerical choking')
    series(axes[1, 0], 'exit_p_ratio', [r['exit_p_exact_ratio'] for r in runs])
    axes[1, 0].plot(ratios, ratios, color='.5', ls='--', label='Imposed pressure')
    axes[1, 0].set(xlim=(.55, .40), ylim=(.06, .71), ylabel=r'Last physical pressure $p/p_0$', title='Outlet shock retention')
    series(axes[1, 1], 'exit_M', [r['M'][-2] for r in reference_runs])
    axes[1, 1].set(xlim=(.55, .40), ylim=(.4, 2.3), ylabel='Last physical Mach', title='Branch-dependent release')
    axes[1, 1].legend(fontsize=7)
    # Labels outside these narrow windows would be misleading and cluttered.
    for ax in axes.flat:
        low, high = sorted(ax.get_xlim())
        for text in list(ax.texts):
            if not low <= text.get_position()[0] <= high:
                text.remove()
    finish(fig, 'transition_zooms')

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout='constrained')
    targets = [ref.shock_exit, .45, .425, .42]
    zoom_colors = cmap(np.linspace(.2, .8, len(targets)))
    fields = [('p', report['p0_Pa'], r'$p/p_0$'), ('M', 1, 'Mach number'),
              ('T', 1, 'Temperature [K]'), ('ds', 1, r'$s-s_0$ [J/(kg K)]')]
    for target, color in zip(targets, zoom_colors):
        i = int(np.argmin(np.abs(ratios - target)))
        row, exact = runs[i], reference_runs[i]
        for ax, (key, scale, label) in zip(axes.flat, fields):
            ax.plot(x, np.asarray(row['flow_state'][key]) / scale, '-o', color=color,
                    markersize=2.8, label=f'{row["p_back_ratio"]:.6f}')
            ax.plot(x[1:-1], np.asarray(exact[key])[1:-1] / scale, '--', color=color, lw=.9)
            ax.scatter([x[-1]], [row['flow_state'][key][-1] / scale], marker='s', color=color, s=30)
    check = next((r for r in cold if abs(r['p_back_ratio'] - .42) < 1e-10), None)
    if check:
        for ax, (key, scale, label) in zip(axes.flat, fields):
            ax.plot(x, np.asarray(check['flow_state'][key]) / scale, ':', color=zoom_colors[-1],
                    lw=2, label='0.420000 cold start')
    for ax, (_, _, label) in zip(axes.flat, fields):
        ax.axvspan((x[-1] + x[-2]) / 2, x[-1], color='.9', alpha=.6)
        ax.set(xlim=(.190, .2003), xlabel='Axial position [m]', ylabel=label)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('Outlet close-up: squares are algebraic ghosts; dashed curves are analytic interior', fontsize=12)
    finish(fig, 'outlet_closeup')

    lines = ['# Backpressure operating-map results', '',
             f'Mesh: {report["mesh_points"]} points. Source unchanged: {report.get("source_unchanged", "run incomplete")}.', '',
             '| Mode | $p_b/p_0$ | Reference regime | Converged | Updates | RMS | $M_t$ | $M_e$ | $w$ | Mass flow [kg/s] | Entropy flux rise [W/K] |',
             '|---|---:|---|:---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in report['runs']:
        lines.append(f'| {r["mode"]} | {r["p_back_ratio"]:.8f} | {r["regime_exact"]} | {r["converged"]} | '
                     f'{r["accepted_updates"]} | {r["history"][-1]["spatial_rms"]:.2e} | {r["throat_M"]:.4f} | '
                     f'{r["exit_M"]:.4f} | {r["outlet_pressure_weight"]:.3g} | {r["mdot_kg_s"]:.7f} | {r["entropy_generation_W_K"]:.6g} |')
    lines.extend(['', 'The last interior point is used for exit diagnostics. Ghosts are not physical samples.', '',
                  'Entropy flux rise is the stationary advective estimate between the first and last interior points;',
                  'it excludes the external jet and is meaningful as a production rate only for a converged solution.', '',
                  '## Figures', ''])
    for name in ['operating_map', 'boundary_behavior', 'pressure_profiles', 'mach_profiles', 'convergence_histories',
                 'solver_work_and_conservation', 'transition_zooms', 'outlet_closeup']:
        lines.extend([f'![{name.replace("_", " ")}]({name}.png)', ''])
    (output / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
