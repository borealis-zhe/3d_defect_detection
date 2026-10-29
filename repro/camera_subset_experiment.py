"""Server-only frozen pose_mv_v1 evaluation with estimated exterior cameras."""
import argparse
import hashlib
import importlib.util
import io
import json
import time
from pathlib import Path
from zipfile import ZipFile

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT.parent/'real3d-dinomaly-20261026'
CAMERAS=ROOT/'data/estimated_camera_v1'
RENDERS=ROOT/'data/exterior_camera_test_v1'
RESULT=ROOT/'results/exterior_camera_test_v1'
ARCHIVE=ROOT.parent/'real3d-mvr-matched-20261028/Real3D-AD-PCD.zip'
VIEWS=(0,1,2,3,6,9,13,14,16,17,18,22,23,25,26)
CATEGORIES=('airplane','candybar','car','chicken','diamond','duck','fish','gemstone','seahorse','shell','starfish','toffees')


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.partial')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    tmp.replace(path)


def virtual_points(raw,c):
    aligned=((raw@np.asarray(c['template_basis'])-c['template_center'])/c['template_scale'])@np.asarray(c['object_rotation_row'])+c['object_translation_row']
    return (aligned-c['target_template'])@np.asarray(c['frame_columns_right_up_outward'])/c['radius_template']


def render(categories,limit=0):
    spec=importlib.util.spec_from_file_location('original_processor',ROOT/'scripts/preprocess_pcd.py')
    p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
    renderer=p.Renderer();skipped=[];done=0
    try:
        with ZipFile(ARCHIVE) as archive:
            for category in categories:
                paths=sorted((SOURCE/'data/pcd_self_v1'/category/'test').glob('*/xyz/*/sample.json'))
                for meta_path in paths:
                    meta=json.loads(meta_path.read_text());name=meta_path.parent.name
                    cp=CAMERAS/category/'test'/name/'camera.json';c=json.loads(cp.read_text())
                    if c['raw_member']!=meta['raw_member']:raise ValueError('Camera/raw mismatch')
                    if c['status']!='ready':
                        skipped.append(dict(category=category,sample=name,reasons=c['reasons']));continue
                    folder=RENDERS/category/'test'/meta['defect']/'xyz'/name
                    manifest=folder/'sample.json';camera_sha=digest(cp)
                    if manifest.exists():
                        previous=json.loads(manifest.read_text())
                        if previous['camera_sha256']!=camera_sha:raise ValueError('Changed cached camera')
                        if not all((folder/f'view_{v:03d}.{ext}').exists() for v in VIEWS for ext in ['png','npy']):raise ValueError('Incomplete cached render')
                        continue
                    raw=p.read_pcd(archive.read(meta['raw_member']))
                    # Pose is fixed before annotations are read.
                    if meta['defect']=='good':labels=np.zeros(len(raw),dtype=np.int8)
                    else:
                        member=meta['raw_member'].replace('/test/','/gt/')[:-4]+'.txt'
                        labels,_=p.match_labels(raw,np.loadtxt(io.BytesIO(archive.read(member))))
                    points=virtual_points(raw.astype(float),c)
                    cloud=renderer.cloud(points);K,E=renderer.render(cloud)
                    ids,_=p.front_representatives(points,K,E);refs=points[ids]
                    folder.mkdir(parents=True,exist_ok=True);visibility=[];view_cameras=[]
                    A=np.asarray(c['template_basis'])@np.asarray(c['object_rotation_row'])@np.asarray(c['frame_columns_right_up_outward'])/(c['template_scale']*c['radius_template'])
                    b=((-np.asarray(c['template_center'])/c['template_scale'])@np.asarray(c['object_rotation_row'])+c['object_translation_row']-np.asarray(c['target_template']))@np.asarray(c['frame_columns_right_up_outward'])/c['radius_template']
                    scale=1/(c['template_scale']*c['radius_template'])
                    for v in VIEWS:
                        image,pos,vis,info=renderer.render(cloud,refs,p.ANGLES[v])
                        image.save(folder/f'view_{v:03d}.png');np.save(folder/f'view_{v:03d}.npy',pos);visibility.append(vis)
                        vr=np.asarray(info['rotation']);ve=np.asarray(info['extrinsic']);eraw=np.eye(4)
                        eraw[:3,:3]=ve[:3,:3]@vr@A.T/scale
                        eraw[:3,3]=(ve[:3,:3]@vr@b+ve[:3,3])/scale
                        info.update(view_id=v,world_to_camera_raw=eraw.tolist());view_cameras.append(info)
                    np.savez_compressed(folder/'points.npz',raw_point_ids=ids,xyz=refs.astype(np.float32),labels=labels[ids],visibility=np.stack(visibility))
                    save(folder/'cameras.json',dict(cameras=view_cameras))
                    save(manifest,dict(category=category,split='test',defect=meta['defect'],sample=name,raw_member=meta['raw_member'],camera_sha256=camera_sha,views=list(VIEWS)))
                    done+=1
                    if done%10==0:print(f'rendered {done} samples; latest {category}/{name}',flush=True)
                    if limit and done>=limit:return
                print(f'{category}: render stage complete',flush=True)
    finally:renderer.close()
    save(RESULT/'skipped.json',skipped)


def metrics(predictions):
    from sklearn.metrics import roc_auc_score,average_precision_score
    ys=[];ss=[];oy=[];os=[]
    for sid,item in predictions.items():
        ys.append(item['labels']);ss.append(item['scores']);oy.append(int('/good/' not in sid));os.append(float(item['scores'].max()))
    y=np.concatenate(ys);s=np.concatenate(ss);known=y>=0
    return dict(objects=len(predictions),points=int(known.sum()),positive_points=int((y==1).sum()),
                O_AUROC=float(roc_auc_score(oy,os)) if len(set(oy))==2 else None,
                P_AUROC=float(roc_auc_score(y[known],s[known])) if len(np.unique(y[known]))==2 else None,
                Point_AP=float(average_precision_score(y[known],s[known])) if (y==1).any() else None)


def evaluate(categories,limit=0):
    import torch
    from . import pose_experiment as experiment
    r=experiment.r
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; do not silently evaluate on CPU')
    reports=[]
    for category in categories:
        paths=sorted((RENDERS/category/'test').glob('*/xyz/*/sample.json'))
        expected=sum(json.loads(p.read_text())['status']=='ready' for p in (CAMERAS/category/'test').glob('*/camera.json'))
        if not limit and len(paths)!=expected:raise ValueError(f'Incomplete renders {category}: {len(paths)}/{expected}')
        if limit:paths=paths[:limit]
        if not paths:continue
        checkpoint=ROOT/'results/pose_mv_v1/train'/category/'final.pt';checkpoint_sha=digest(checkpoint)
        r.seed(1);model=r.build_model();state=torch.load(checkpoint,map_location='cpu',weights_only=True)
        model.bottleneck.load_state_dict(state['bottleneck'],strict=True);model.decoder.load_state_dict(state['decoder'],strict=True);model.eval()
        output=RESULT/'eval'/category;output.mkdir(parents=True,exist_ok=True);predictions={}
        for index,path in enumerate(paths):
            meta=json.loads(path.read_text());folder=path.parent;sid=f"{category}/test/{meta['defect']}/{folder.name}"
            destination=output/f'{folder.name}.npz'
            if destination.exists():
                with np.load(destination) as data:
                    if str(data['checkpoint_sha256'])!=checkpoint_sha or str(data['camera_sha256'])!=meta['camera_sha256']:raise ValueError('Prediction provenance changed')
                    predictions[sid]={k:data[k].copy() for k in ['scores','labels','raw_point_ids']}
                continue
            with np.load(folder/'points.npz') as data:labels=data['labels'].astype(np.int8);ids=data['raw_point_ids'];xyz=data['xyz']
            maps=[]
            with torch.inference_mode():
                for start in range(0,len(VIEWS),3):
                    batch=torch.stack([r.image_tensor(folder/f'view_{v:03d}.png') for v in VIEWS[start:start+3]]).cuda()
                    targets,recon=model(batch);maps.append(r.score_maps(targets,recon))
                positions=[np.load(folder/f'view_{v:03d}.npy') for v in VIEWS]
                scores=r.point_scores_from_maps(torch.cat(maps),positions).cpu().numpy()
            if not np.isfinite(scores).all() or len(scores)!=len(labels):raise ValueError('Invalid scores')
            np.savez_compressed(destination,sample_id=sid,scores=scores,labels=labels,raw_point_ids=ids,xyz=xyz,
                                checkpoint_sha256=checkpoint_sha,camera_sha256=meta['camera_sha256'])
            predictions[sid]=dict(scores=scores,labels=labels,raw_point_ids=ids)
            if (index+1)%10==0:print(f'{category}: inferred {index+1}/{len(paths)}',flush=True)
        old={}
        for path in (ROOT/'results/pose_mv_v1/eval'/category).glob('sample_*.npz'):
            with np.load(path) as data:
                sid=str(data['sample_id'])
                if sid in predictions:old[sid]={k:data[k].copy() for k in ['scores','labels','raw_point_ids']}
        if set(old)!=set(predictions):raise ValueError('Missing paired old predictions')
        report=dict(category=category,checkpoint='pose_mv_v1/final.pt',checkpoint_sha256=checkpoint_sha,
                    new_camera=metrics(predictions),old_camera_same_objects=metrics(old),
                    excluded_objects=(53 if category=='chicken' else 50)-len(predictions),
                    caveat='same checkpoint and object subset; point domains differ; uncertain poses excluded')
        save(output/'metrics.json',report);reports.append(report)
        print(json.dumps(report),flush=True)
        del model,state;torch.cuda.empty_cache()
    save(RESULT/'comparison.json',dict(reports=reports,trained=False,scope='ready cameras only'))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['render','evaluate','all'])
    parser.add_argument('--categories',nargs='+',choices=CATEGORIES,default=list(CATEGORIES));parser.add_argument('--limit',type=int,default=0)
    args=parser.parse_args();RESULT.mkdir(parents=True,exist_ok=True)
    if args.stage in ['render','all']:render(args.categories,args.limit)
    if args.stage in ['evaluate','all']:evaluate(args.categories,args.limit)


if __name__=='__main__':main()
