"""Read retained Newton/PTC/custom-adapted evidence without solver imports."""

import hashlib
import json
import math
from pathlib import Path

STUDY=Path(__file__).resolve().parents[1]
ORDER=('subsonic','sonic','shock35','shock70','supersonic')
NAMES={'implicit':'Implicit integration','ptc':'Physical-storage PTC','newton':'Controlled Newton',
       'hybr':'Custom HYBR','lm':'Custom LM','slsqp':'Custom SLSQP'}


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def numerical_files(output,adapted):
    paths=[output/'summary.json',output/'results.csv',output/'performance_analysis/analysis.json',
           adapted/'summary.json',adapted/'results.csv',adapted/'audit.json']
    for case in ORDER:paths.extend(p for p in (output/case).rglob('*') if p.suffix in ('.json','.csv'))
    paths.extend(p for p in (adapted/'workers').rglob('*') if p.suffix in ('.json','.csv','.jsonl'))
    return {p:sha(p) for p in paths if p.is_file()}


def load(output,adapted):
    read=lambda path:json.loads(path.read_text(encoding='utf-8'))
    return read(output/'summary.json'),read(output/'performance_analysis/analysis.json'),read(adapted/'summary.json')


def cold_runs(case):
    runs=[dict(r) for r in case['runs'] if r['phase']=='simple']
    runs.append(dict(case['implicit_references']['simple'],method='implicit',phase='simple'))
    return sorted(runs,key=lambda r:('implicit','ptc','newton').index(r['method']))


def slim(run,method=None):
    """Exact final fields and scalar histories, without redundant state snapshots."""
    keys=('method','phase','initialization','wall_s','worker_elapsed_s','iterations','native_success',
          'root_success','same_root','same_shocked_root','mass_conservation_pass','analytical_pass',
          'authoritative_returned_point','native_message','message','wall_s_scope','nfev','njev')
    result={k:run[k] for k in keys if k in run}
    if method:result['method']=method
    result['diagnostics']=run['diagnostics']
    result['history']=[{k:v for k,v in h.items() if k in ('accepted_index','observation_index',
        'elapsed_s','residual_rms','cfl','effective_cfl','dt','shift','step_fraction','backtracks')}
        for h in run['history']]
    result['backend']={k:v for k,v in run.get('backend',{}).items() if k in ('library','method','effective_options','algorithm','status','ntrials','nrejected')}
    return result


def make_payload(cold,perf,adapted):
    cases=[]
    for cid in ORDER:
        case=next(c for c in cold['cases'] if c['id']==cid)
        cases.append(dict(id=cid,name=case['name'],x_m=case['x_m'],exact_x_m=case['exact_x_m'],
            exact_pressure_Pa=case['exact_pressure_Pa'],p_back_ratio=case['back_pressure']/case['reservoir_pressure'],
            runs=[slim(r) for r in cold_runs(case)]))
    start_cases=[]
    for case in adapted['cases']:
        start_cases.append(dict(id=case['id'],label=case['label'],x_m=case['x_m'],reference=case['reference'],
            seeds={k:v['diagnostics'] for k,v in case['seeds'].items()},
            implicit_reference=slim(case['implicit_reference'],'implicit'),
            implicit_from_adapted=dict(iterations=case['implicit_from_adapted']['iterations'],
                                       residual_rms=case['implicit_from_adapted']['history'][-1]['residual_rms']),
            adapted_residual_change=case['adapted_residual_change'],runs=[slim(r) for r in case['runs']]))
    replay={cid:{k:{key:v for key,v in value.items() if key not in ('x','all_trials')}
                 for k,value in variants.items()} for cid,variants in perf['replay'].items()}
    ablations={k:slim(v,'ptc') for k,v in perf['controller_ablations'].items()}
    return dict(cases=cases,names=NAMES,performance=dict(replay=replay,ablations=ablations,
        profile=perf['profile'],first_correction=perf['first_correction']),adapted=start_cases,
        seed_cost_s=adapted['adapted']['implicit']['wall_s'],
        seed_updates=adapted['adapted']['implicit']['iterations'],
        settings=cold['settings'],versions=cold['versions'])


def audit(cold,perf,adapted,cleanup):
    records=[]
    for case in cold['cases']:
        reference=case['implicit_references']['simple']
        runs=cold_runs(case)
        assert len({r['initial_sha256'] for r in runs})==1
        for run in runs:
            rms=run['diagnostics']['residual_rms'];root=rms<=cold['residual_rms_tol']
            flags=dict(recorded_root=root==run['root_success'],history_endpoint=abs(rms-run['history'][-1]['residual_rms'])<=1e-12,
                       field_dimensions=all(len(v)==len(case['x_m']) for v in run['diagnostics']['fields'].values()))
            if run['method']!='implicit':
                distance=math.sqrt(sum((a-b)**2 for a,b in zip(run['x'],reference['x']))/len(reference['x']))
                flags['distance']=abs(distance-run['scaled_solution_difference_from_implicit'])<=1e-12
                flags['same_root']=(root and distance<=cold['same_solution_rms_tol'])==run['same_root']
            records.append(dict(case=case['id'],method=run['method'],pass_=all(flags.values()),root=root,checks=flags))
    adapted_records=[]
    for case in adapted['cases']:
        for run in case['runs']:
            assert run['method'] in ('newton','hybr','lm','slsqp') and run['backend']['library']=='pysolver'
            rms=run['diagnostics']['residual_rms']
            root=run['authoritative_returned_point'] and rms<=adapted['metadata']['residual_rms_tol']
            reference=case['implicit_reference']['x']
            distance=math.sqrt(sum((a-b)**2 for a,b in zip(run['audited_point'],reference))/len(reference))
            flags=dict(root=root==run['root_success'],distance=abs(distance-run['scaled_distance_to_shocked_root'])<=1e-12,
                       requested_root=(root and distance<=1e-4)==run['same_shocked_root'],
                       seed=run['initial_point_sha256']==case['seeds'][run['initialization']]['initial_point_sha256'])
            adapted_records.append(dict(case=case['id'],method=run['method'],initialization=run['initialization'],
                pass_=all(flags.values()),equation_root=root,requested_shocked_root=run['same_shocked_root'],checks=flags))
        assert canonical(case['runs'])==cleanup['retained_custom_run_hashes'][case['id']]
        assert canonical({k:v for k,v in case.items() if k!='runs'})==cleanup['shared_reference_and_seed_hashes'][case['id']]
    replays=[]
    for cid,variants in perf['replay'].items():
        for key,run in variants.items():
            flags=dict(work=sum(t['inner_solves'] for t in run['all_trials'])==run['total_linear_solves'],
                       updates=len(run['accepted_history'])==run['updates'])
            if key=='default':flags['saved_state']=run['max_state_difference_from_saved']==0
            replays.append(dict(case=cid,variant=key,pass_=all(flags.values()),checks=flags))
    return dict(pass_=all(r['pass_'] for r in records+adapted_records+replays),
        kind='Saved-data consistency and cleanup-preservation checks; no numerical solver evaluations',
        cold_runs=len(records),cold_spatial_roots=sum(r['root'] for r in records),
        retained_custom_adapted_runs=len(adapted_records),
        adapted_equation_roots=sum(r['equation_root'] for r in adapted_records),
        adapted_requested_shocked_roots=sum(r['requested_shocked_root'] for r in adapted_records),
        records=records,adapted_records=adapted_records,replay_records=replays)
