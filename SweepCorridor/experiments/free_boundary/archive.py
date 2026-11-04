#!/usr/bin/env python3
"""Archive numeric evidence without duplicating large map snapshots."""
import gzip,hashlib,json,os,tarfile
from pathlib import Path
os.chdir(Path(__file__).resolve().parents[3])
scratch=Path('.scratch/free_boundary');out=Path('docs/validation/free_boundary');out.mkdir(parents=True,exist_ok=True)
data={}
for name in ('final_development','final_regression','balanced_development','balanced_regression','development','holdout','cap_qp','native_v1','native_v2','native_v3'):
    root=scratch/name
    if (root/'summary.json').exists():
        data[name]={k:json.loads((root/(k+'.json')).read_text()) for k in ('summary','manifest')}
data['regularization_scan']=json.loads((scratch/'regularization.json').read_text())
data['progress_regularization_scan']=json.loads((scratch/'progress_regularization.json').read_text())
data['firi']=json.loads((scratch/'firi/metrics.json').read_text())
data['profile']={mode:[json.loads(line) for line in (scratch/'profile'/mode/'rows.jsonl').read_text().splitlines()] for mode in ('baseline','free_boundary')}
data['validation']={name:(scratch/name).read_text() for name in ('tests.log','library.log','consumer.log','ros.log','search_test.log','volume_test.log','ros_smoke.log')}
data['validation']['geometry_metrics_test.log']=(scratch/'geometry_metrics_test.log').read_text()
data['validation']['curvature_resolution.log']=(scratch/'curvature_resolution.log').read_text()
data['commands']={name:(scratch/name).read_text() for name in ('validate.sh','integration.sh','regularization.py','progress_regularization.py','check_curvature.py')}
data['ros_smoke']={scene:json.loads(Path(f'.scratch/sweep_rviz_c3/{scene}_smoke.json').read_text()) for scene in ('perlin','maze')}
data['summary']=json.loads((out/'aggregate.json').read_text())
data['regularization_sources']={name:{str(p.relative_to(scratch/name)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (scratch/name/'include').rglob('*.hpp')} for name in ('energy3','energy10','progress003','progress01')}
with gzip.open(out/'results.json.gz','wt',encoding='utf-8') as file:json.dump(data,file,ensure_ascii=False)
inputs=json.loads(Path('docs/validation/inflation_round1/inputs.json').read_text())
(out/'inputs.json').write_text(json.dumps(inputs,indent=2)+'\n')
with tarfile.open(out/'geometry.tar.gz','w:gz') as archive:
    for case in [0,1,2,3,4,5,6,57,60,61,62,67]:
        split='final_development' if case in [0,1,2,3,4,5,6,67] else 'final_regression'
        for mode in ('baseline','free_boundary'):
            for name in ('corridor.csv','core.csv','support_planes.csv','rows.jsonl'):
                archive.add(scratch/split/f'case_{case:02d}'/mode/name,arcname=f'case_{case:02d}/{mode}/{name}')
        record=next(c for c in inputs['cases'] if c['case']==case)
        archive.add(record['seed'],arcname=f'case_{case:02d}/seed.txt')
    archive.add(scratch/'firi/firi_planes.csv',arcname='case_67/firi/firi_planes.csv')
    archive.add(scratch/'firi/metrics.json',arcname='case_67/firi/metrics.json')
artifacts={
    'sources':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for root in ['SweepCorridor/include','SweepCorridor/experiments/free_boundary'] for p in Path(root).rglob('*') if p.is_file() and '__pycache__' not in str(p)},
    'binaries':{name:hashlib.sha256((scratch/name).read_bytes()).hexdigest() for name in ('benchmark','baseline_benchmark','measure','tests')},
    'files':{str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file() and p.name!='artifacts.json'}}
(out/'artifacts.json').write_text(json.dumps(artifacts,indent=2)+'\n')
print('Archived',out)
