"""Estimate an exterior virtual camera before rendering each partial test cloud.

No actual scanner calibration is recovered. Direction comes from a label-blind
partial-to-template fit; distance, intrinsics and roll are explicit conventions.
Uncertain fits receive a proposal but are blocked from automatic rendering.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from scipy.spatial import cKDTree

from .bundle_ablate import CATEGORIES, ROOT, VIEW_IDS
from .pose_registration import DATA
from .pose_render import find_sample, original_preprocessor, raw_labels
from .raw_coverage import RAW_ZIP, read_xyz

DEFAULT_FITS = ROOT / 'outputs/scan_direction_audit'
DEFAULT_OUTPUT = ROOT / 'outputs/estimated_camera_v1'


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def camera_frame(direction: np.ndarray) -> np.ndarray:
    """Columns: image right, image up, outward viewing position direction."""
    direction = np.asarray(direction, dtype=np.float64)
    if not np.isfinite(direction).all() or np.linalg.norm(direction) < 1e-9:
        raise ValueError('Invalid exterior direction')
    direction = direction / np.linalg.norm(direction)
    up = np.array([0., 1., 0.])
    if abs(np.dot(up, direction)) > .95:
        up = np.array([1., 0., 0.])
    right = np.cross(up, direction)
    right /= np.linalg.norm(right)
    up = np.cross(direction, right)
    return np.column_stack([right, up, direction])


def build_camera(raw: np.ndarray, template_meta: dict, fit: dict,
                 spacing: float) -> dict:
    basis = np.asarray(template_meta['basis'], dtype=np.float64)
    center = np.asarray(template_meta['oriented_center'], dtype=np.float64)
    scale = float(template_meta['normalization_radius'])
    rotation = np.asarray(fit['rotation_row'], dtype=np.float64)
    translation = np.asarray(fit['translation_row'], dtype=np.float64)
    for name, matrix in [('basis', basis), ('rotation', rotation)]:
        if not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(matrix), 1., atol=1e-6):
            raise ValueError(f'{name} must be a proper rigid rotation')
    aligned = ((raw @ basis - center) / scale) @ rotation + translation
    target = (aligned.min(0) + aligned.max(0)) / 2
    radius = float(np.linalg.norm(aligned - target, axis=1).max())
    if not np.isfinite(radius) or radius <= 1e-9:
        raise ValueError('Degenerate test cloud')
    frame = camera_frame(np.asarray(fit['direction']))
    distance = 2.5 * radius
    position = target + frame[:, 2] * distance
    # OpenCV camera coordinates: right, down, forward.
    world_rotation = np.diag([1., -1., -1.]) @ frame.T
    template_E = np.eye(4)
    template_E[:3, :3] = world_rotation
    template_E[:3, 3] = -world_rotation @ position
    raw_E = np.eye(4)
    raw_E[:3, :3] = world_rotation @ rotation.T @ basis.T
    raw_E[:3, 3] = world_rotation @ (scale * translation - center @ rotation) + scale * template_E[:3, 3]
    raw_position = ((position - translation) @ rotation.T * scale + center) @ basis.T
    focal = 336 / np.tan(np.pi / 6)
    K = np.array([[focal, 0., 335.5], [0., focal, 335.5], [0., 0., 1.]])
    reasons = []
    if fit['alternative_angle_deg'] > 45:
        reasons.append('near_equal_matches_disagree_on_side')
    if fit['normal_strength'] < .3:
        reasons.append('weak_surface_direction')
    if fit['score'] > 2 * spacing:
        reasons.append('partial_template_fit_too_large')
    # Keep any upstream ambiguity flag, including future additional checks.
    if fit['ambiguous'] and not reasons:
        reasons.append('upstream_direction_ambiguous')
    return dict(version='exterior_virtual_camera_v1',
                status='needs_review' if reasons else 'ready', reasons=reasons,
                is_measured_scanner_pose=False,
                pose_conventions='OpenCV world-to-camera: right/down/forward; row-vector input XYZ',
                estimated_from='normal-template surface correspondence; no anomaly labels',
                heuristic_parameters=dict(max_relative_fit_spacing=2., min_normal_strength=.3,
                                          max_alternative_angle_deg=45., distance_radius_ratio=2.5),
                direction_template=frame[:, 2].tolist(),
                target_template=target.tolist(), camera_position_template=position.tolist(),
                camera_position_raw=raw_position.tolist(), radius_template=radius,
                distance_template=distance, frame_columns_right_up_outward=frame.tolist(),
                world_to_camera_template=template_E.tolist(),
                world_to_camera_raw=raw_E.tolist(), camera_to_world_raw=np.linalg.inv(raw_E).tolist(),
                intrinsic_672=K.tolist(), image_size=[672,672],
                intrinsics_and_distance='chosen virtual rendering parameters, not recovered calibration',
                roll='template +Y projected to image; +X fallback near parallel',
                template_basis=basis.tolist(), template_center=center.tolist(), template_scale=scale,
                object_rotation_row=rotation.tolist(), object_translation_row=translation.tolist(),
                match_score=fit['score'], score_over_template_spacing=fit['score']/spacing,
                normal_strength=fit['normal_strength'], alternative_angle_deg=fit['alternative_angle_deg'])


def virtual_points(raw: np.ndarray, camera: dict) -> np.ndarray:
    aligned = ((raw @ np.asarray(camera['template_basis']) - camera['template_center']) /
               camera['template_scale']) @ np.asarray(camera['object_rotation_row']) + camera['object_translation_row']
    return ((aligned - camera['target_template']) @
            np.asarray(camera['frame_columns_right_up_outward']) / camera['radius_template'])


def raw_to_virtual_affine(camera: dict) -> tuple[np.ndarray, np.ndarray]:
    basis=np.asarray(camera['template_basis']);r=np.asarray(camera['object_rotation_row'])
    frame=np.asarray(camera['frame_columns_right_up_outward'])
    scale=camera['template_scale'];radius=camera['radius_template']
    matrix=basis @ r @ frame / (scale*radius)
    offset=((-np.asarray(camera['template_center'])/scale) @ r + camera['object_translation_row'] -
            np.asarray(camera['target_template'])) @ frame / radius
    return matrix,offset


def validate_camera(raw: np.ndarray, camera: dict) -> dict:
    q=raw[::max(1,len(raw)//2000)]
    virtual=virtual_points(q,camera)
    virtual_cam=virtual * np.array([1.,-1.,-1.])+np.array([0.,0.,2.5])
    E=np.asarray(camera['world_to_camera_raw']);K=np.asarray(camera['intrinsic_672'])
    raw_cam=q @ E[:3,:3].T+E[:3,3]
    def pixels(x):
        p=x @ K.T
        return p[:,:2]/p[:,2:3]
    error=float(np.max(np.abs(pixels(raw_cam)-pixels(virtual_cam))))
    if error>1e-6 or raw_cam[:,2].min()<=0:
        raise ValueError(f'Camera projection inconsistency: {error}')
    recovered_center=-E[:3,:3].T @ E[:3,3]
    if not np.allclose(recovered_center,camera['camera_position_raw'],atol=1e-6):
        raise ValueError('Extrinsic center mismatch')
    return dict(projection_error_pixels=error, positive_depth=True,
                camera_outside_bounding_sphere=camera['distance_template']>camera['radius_template'])


def render_estimated(category: str, sample: str, archive: ZipFile, processor,
                     camera_root: Path, output: Path, views=VIEW_IDS) -> dict:
    camera_path=camera_root/category/'test'/sample/'camera.json'
    camera=json.loads(camera_path.read_text())
    if camera['status']!='ready':
        raise ValueError(f'Camera requires review: {category}/{sample}: {camera["reasons"]}')
    meta_path=find_sample(category,'test',sample)
    meta=json.loads(meta_path.read_text())
    if camera['raw_member']!=meta['raw_member'] or camera['source_meta_sha256']!=sha(meta_path):
        raise ValueError('Camera/source provenance mismatch')
    raw=read_xyz(archive.read(meta['raw_member'])).astype(np.float64)
    points=virtual_points(raw,camera)
    # Annotation access happens only AFTER the camera has been fixed and accepted.
    labels=raw_labels(category,'test',sample,len(raw))
    destination=output/category/'test'/sample
    if (destination/'render.json').exists():
        old=json.loads((destination/'render.json').read_text())
        if old.get('camera_sha256')!=sha(camera_path) or old.get('views')!=list(views):
            raise ValueError('Existing render uses a different camera/view set; choose a new output')
    destination.mkdir(parents=True,exist_ok=True)
    renderer=processor.Renderer()
    visibility=[];cameras=[]
    affine,offset=raw_to_virtual_affine(camera)
    uniform_scale=1/(camera['template_scale']*camera['radius_template'])
    try:
        cloud=renderer.cloud(points)
        K,E=renderer.render(cloud)
        ids,_=processor.front_representatives(points,K,E)
        refs=points[ids]
        for view in views:
            image,pos,visible,info=renderer.render(cloud,refs,processor.ANGLES[view])
            image.save(destination/f'view_{view:03d}.png')
            np.save(destination/f'view_{view:03d}.npy',pos)
            visibility.append(visible)
            view_r=np.asarray(info['rotation']);view_e=np.asarray(info['extrinsic'])
            # Compose the legacy point rotation into an equivalent per-view
            # raw-world camera so every pixel has an explicit camera contract.
            eraw=np.eye(4)
            eraw[:3,:3]=(view_e[:3,:3] @ view_r @ affine.T)/uniform_scale
            eraw[:3,3]=(view_e[:3,:3] @ view_r @ offset+view_e[:3,3])/uniform_scale
            info.update(view_id=int(view),world_to_camera_raw=eraw.tolist(),
                        camera_to_world_raw=np.linalg.inv(eraw).tolist())
            cameras.append(info)
    finally:
        renderer.close()
    np.savez_compressed(destination/'points.npz',raw_point_ids=ids,xyz=refs.astype(np.float32),
                        labels=labels[ids],visibility=np.stack(visibility),view_ids=np.array(views))
    record=dict(category=category,sample=sample,views=list(views),camera=str(camera_path),
                camera_sha256=sha(camera_path),raw_member=meta['raw_member'],
                reference_points=len(ids),renderer='exterior_estimated_camera_v1')
    (destination/'cameras.json').write_text(json.dumps(dict(cameras=cameras),indent=2),encoding='utf-8')
    (destination/'render.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    print(f'Rendered {category}/{sample}: {len(views)} views from estimated exterior camera',flush=True)
    return record


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--categories',nargs='+',choices=CATEGORIES,default=list(CATEGORIES))
    parser.add_argument('--fit-root',type=Path,default=DEFAULT_FITS)
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--archive',type=Path,default=RAW_ZIP)
    parser.add_argument('--samples',nargs='*',default=[],help='Optional category/sample selection')
    parser.add_argument('--render-samples',nargs='*',default=[],help='category/sample pairs; uncertain cameras are blocked')
    parser.add_argument('--views',nargs='+',type=int,default=list(VIEW_IDS))
    args=parser.parse_args()
    if not args.views or len(set(args.views))!=len(args.views) or any(v<0 or v>=27 for v in args.views):
        raise ValueError('Views must be unique IDs in 0..26')
    summary=[]
    with ZipFile(args.archive) as archive:
        for category in args.categories:
            template_path=sorted((DATA/category/'train/good/xyz').glob('*/sample.json'))[0]
            template_meta=json.loads(template_path.read_text())
            # Cached label-blind fits are reused; a clean run can estimate them
            # directly from raw geometry before producing any camera or image.
            from scripts.scan_direction_audit import downsample, estimate, template_normals
            fit_folder=args.fit_root/category
            fit_folder.mkdir(parents=True,exist_ok=True)
            template_cache=fit_folder/'template.npz'
            if template_cache.exists():
                with np.load(template_cache) as data:
                    template=data['xyz'];normals=data['normals']
            else:
                template_raw=read_xyz(archive.read(template_meta['raw_member'])).astype(np.float64)
                template=downsample((template_raw@np.asarray(template_meta['basis'])-
                                     template_meta['oriented_center'])/template_meta['normalization_radius'],
                                    6000,template_meta['raw_member'])
                normals=template_normals(template)
                np.savez_compressed(template_cache,xyz=template,normals=normals)
            spacing=float(np.median(cKDTree(template).query(template,k=2)[0][:,1]))
            for path in sorted((DATA/category/'test').glob('*/xyz/*/sample.json')):
                if args.samples and f'{category}/{path.parent.name}' not in args.samples:
                    continue
                meta=json.loads(path.read_text());fit_path=args.fit_root/category/(path.parent.name+'.json')
                raw=read_xyz(archive.read(meta['raw_member'])).astype(np.float64)
                if not fit_path.exists():
                    query=downsample((raw@np.asarray(template_meta['basis'])-
                                      template_meta['oriented_center'])/template_meta['normalization_radius'],
                                     1800,meta['raw_member'])
                    result=estimate(query,template,normals)
                    fit=dict(raw_member=meta['raw_member'],template=template_meta['raw_member'],
                             rotation_row=result['rotation'].tolist(),translation_row=result['translation'].tolist(),
                             direction=result['direction'].tolist(),score=result['score'],
                             normal_strength=result['strength'],ambiguous=bool(result['ambiguous']),
                             alternative_angle_deg=result['alternative_angle_deg'],
                             method='one_way_trim90_ICP15_PCA8_plus_identity_v1')
                    fit_path.write_text(json.dumps(fit,indent=2),encoding='utf-8')
                fit=json.loads(fit_path.read_text())
                if fit['raw_member']!=meta['raw_member'] or fit['template']!=template_meta['raw_member']:
                    raise ValueError(f'Fit provenance mismatch: {path}')
                record=build_camera(raw,template_meta,fit,spacing)
                record.update(category=category,sample=path.parent.name,raw_member=meta['raw_member'],
                              source_meta_sha256=sha(path),fit_sha256=sha(fit_path),
                              template_meta_sha256=sha(template_path),template_raw_member=template_meta['raw_member'])
                record['validation']=validate_camera(raw,record)
                target=args.output/category/'test'/path.parent.name/'camera.json'
                target.parent.mkdir(parents=True,exist_ok=True)
                target.write_text(json.dumps(record,indent=2),encoding='utf-8')
                summary.append(dict(category=category,sample=path.parent.name,status=record['status'],
                                    reasons=';'.join(record['reasons']),
                                    projection_error_pixels=record['validation']['projection_error_pixels']))
            print(f'{category}: cameras estimated',flush=True)
        if not summary:
            raise ValueError('No test samples matched the requested selection')
        args.output.mkdir(parents=True,exist_ok=True)
        with (args.output/'camera_manifest.csv').open('w',newline='',encoding='utf-8') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
        if args.render_samples:
            processor=original_preprocessor()
            for item in args.render_samples:
                category,sample=item.split('/',1)
                render_estimated(category,sample,archive,processor,args.output,args.output/'renders',args.views)
    print(f'{len(summary)} cameras; {sum(r["status"]=="ready" for r in summary)} ready; '
          f'{sum(r["status"]!="ready" for r in summary)} need review',flush=True)


if __name__=='__main__':main()
