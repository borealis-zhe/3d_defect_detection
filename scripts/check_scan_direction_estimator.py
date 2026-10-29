"""Known-direction synthetic visibility checks of the coverage estimator."""
import csv
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree
from scripts.scan_direction_audit import OUT, estimate

rows=[]
for cat in ['fish','car','airplane']:
    data=np.load(OUT/cat/'template.npz')
    template,normals=data['xyz'],data['normals']
    spacing=float(np.median(cKDTree(template).query(template,k=2)[0][:,1]))
    for axis in range(3):
        for sign in [-1,1]:
            perpendicular=[i for i in range(3) if i!=axis]
            uv=template[:,perpendicular]
            cells=np.minimum(31,((uv-uv.min(0))/np.ptp(uv,axis=0)*32).astype(int))
            flat=cells[:,0]+32*cells[:,1]
            depth=sign*template[:,axis]
            front=np.full(1024,-np.inf);np.maximum.at(front,flat,depth)
            visible=depth>=front[flat]-.01
            part=template[visible]
            # Actual viewpoint has a known direction in the template frame;
            # apply an arbitrary rigid transform to challenge registration.
            rotation=Rotation.from_euler('xyz',[33,-51,24],degrees=True).as_matrix()
            query=part@rotation+np.array([.06,-.08,.04])
            result=estimate(query,template,normals)
            true=np.eye(3)[axis]*sign
            error=float(np.degrees(np.arccos(np.clip(np.dot(result['direction'],true),-1,1))))
            expected=normals[visible].mean(0);expected/=np.linalg.norm(expected)
            coverage_error=float(np.degrees(np.arccos(np.clip(np.dot(result['direction'],expected),-1,1))))
            uncertain=result['ambiguous'] or result['score']>2*spacing or result['strength']<.3
            row=dict(category=cat,true_axis='XYZ'[axis]+('+' if sign>0 else '-'),
                     points=len(part),angle_error=error,coverage_angle_error=coverage_error,
                     ambiguous=uncertain,score=result['score'],normal_strength=result['strength'])
            rows.append(row)
with (OUT/'synthetic_direction_checks.csv').open('w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
for row in rows:print(row,flush=True)
