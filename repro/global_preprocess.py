"""Scheme C: normal training data only, 26 global camera directions."""
import argparse,copy,hashlib,importlib.util,json
from pathlib import Path
from zipfile import ZipFile
import numpy as np
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/global26_v2'
CATEGORIES=('airplane','candybar','car','chicken','diamond','duck','fish','gemstone','seahorse','shell','starfish','toffees')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--limit',type=int,default=0);args=parser.parse_args()
    spec=importlib.util.spec_from_file_location('processor',ROOT/'scripts/preprocess_pcd.py');p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
    config=json.loads((ROOT/'global26_positions.json').read_text());ds=np.array(config['directions']);ups=np.array(config['image_up'])
    assert ds.shape==(26,3) and all(any(np.allclose(-v,w) for w in ds) for v in ds)
    renderer=p.Renderer();completed=0;reports=[]
    try:
        with ZipFile(ROOT.parent/'real3d-mvr-matched-20261028/Real3D-AD-PCD.zip') as archive:
            for cat in CATEGORIES:
                paths=sorted((ROOT/'data/pcd_self_v1'/cat/'train/good/xyz').glob('*/sample.json'));assert len(paths)==4
                for path in paths:
                    meta=json.loads(path.read_text());assert meta['split']=='train' and meta['defect']=='good'
                    posepath=ROOT/'data/pose_transforms'/cat/'train'/path.parent.name/'pose.json';pose=json.loads(posepath.read_text())
                    assert pose['raw_member']==meta['raw_member'] and '/train/' in meta['raw_member']
                    out=DATA/cat/'train/good/xyz'/path.parent.name;manifest=out/'sample.json'
                    provenance=dict(source_sha256=sha(path),pose_sha256=sha(posepath),camera_config_sha256=sha(ROOT/'global26_positions.json'),preprocessor_sha256=sha(__file__))
                    if manifest.exists():
                        prior=json.loads(manifest.read_text());assert all(prior[k]==v for k,v in provenance.items())
                        assert all((out/f'view_{i:03d}.png').exists() for i in range(26));reports.append(prior);continue
                    raw=p.read_pcd(archive.read(meta['raw_member']))
                    xyz=(raw.astype(float)@np.array(meta['basis'])-meta['oriented_center'])/meta['normalization_radius']
                    xyz=xyz@np.array(pose['rotation_row'])+pose['translation_row']
                    cloud=renderer.cloud(xyz);out.mkdir(parents=True,exist_ok=True);images=[];cameras=[]
                    for i,(d,u) in enumerate(zip(ds,ups)):
                        right=np.cross(u,d);frame=np.column_stack([right,u,d]);assert np.allclose(frame.T@frame,np.eye(3)) and np.linalg.det(frame)>.99999
                        rotated=copy.deepcopy(cloud);rotated.rotate(frame.T,center=[0,0,0]);rotated.orient_normals_towards_camera_location([0,0,10])
                        q=np.asarray(rotated.points)
                        required=float(np.max(q[:,2]+np.max(np.abs(q[:,:2]),axis=1)*(112/np.tan(np.pi/6))/93.0))+.02
                        distance=max(2.5,required)
                        rotated.scale(2.5/distance,center=[0,0,0])
                        im,_,_,cam=renderer.render(rotated,np.zeros((1,3)))
                        arr=np.asarray(im);foreground=(arr.min(2)<235);yy,xx=np.where(foreground)
                        assert len(xx)>100 and xx.min()>=14 and xx.max()<210 and yy.min()>=14 and yy.max()<210,('crop/blank',cat,path.parent.name,i)
                        im.save(out/f'view_{i:03d}.png');images.append(im)
                        cameras.append(dict(view=i,direction=d.tolist(),position=(distance*d).tolist(),distance=distance,framing_adjusted=distance>2.5,image_up=u.tolist(),world_to_camera_aligned=np.block([[np.array(cam['extrinsic'])[:3,:3]@frame.T,np.array(cam['extrinsic'])[:3,3:4]*(distance/2.5)],[np.zeros((1,3)),np.ones((1,1))]]).tolist()))
                    result=dict(category=cat,split='train',defect='good',sample=path.parent.name,raw_member=meta['raw_member'],views=list(range(26)),**provenance)
                    (out/'cameras.json').write_text(json.dumps(cameras,indent=2));manifest.write_text(json.dumps(result,indent=2));reports.append(result)
                    if completed==0 and not (DATA/'first_sample_26views.png').exists():
                        grid=Image.new('RGB',(13*180,2*205),'white');draw=ImageDraw.Draw(grid)
                        for i,im in enumerate(images):
                            x=(i%13)*180;y=(i//13)*205;grid.paste(im.resize((180,180)),(x,y+25));draw.text((x+7,y+5),f'{i:02d} '+str(tuple(np.sign(ds[i]).astype(int))),fill='black')
                        grid.save(DATA/'first_sample_26views.png')
                    completed+=1;print('RENDERED',cat,path.parent.name,26,flush=True)
                    if args.limit and completed>=args.limit:return
    finally:renderer.close()
    assert len(reports)==48
    (DATA/'render_manifest.json').write_text(json.dumps(dict(normal_objects=48,views_per_object=26,images=1248,records=reports),indent=2))
    print('VALIDATED 48 normal training clouds / 1248 images',flush=True)

if __name__=='__main__':main()
