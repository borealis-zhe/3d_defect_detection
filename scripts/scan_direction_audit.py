"""Estimate observed-surface directions, not recorded scanner extrinsics.

Partial-to-full, one-way trimmed ICP; no defect labels or saved v1 poses.
Ambiguous alternative surface matches are retained as uncertainty.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from scipy.spatial import cKDTree

from repro.bundle_ablate import CATEGORIES, ROOT
from repro.pose_registration import DATA, AXIS_MAPS, pca_frame, sample_indices
from repro.raw_coverage import RAW_ZIP, read_xyz

OUT = ROOT / "outputs/scan_direction_audit"


def downsample(points, budget, key):
    return points[sample_indices(len(points), budget, key)]


def one_way_score(points, tree):
    distance = tree.query(points, workers=1)[0]
    keep = max(1, int(.9 * len(distance)))
    return float(np.partition(distance, keep - 1)[:keep].mean())


def refine(query, template, tree, rotation, translation, steps=15):
    rotation, translation = rotation.copy(), translation.copy()
    for _ in range(steps):
        moved = query @ rotation + translation
        distance, nearest = tree.query(moved, workers=1)
        good = distance <= np.quantile(distance, .9)
        a, b = moved[good], template[nearest[good]]
        am, bm = a.mean(0), b.mean(0)
        u, _, vt = np.linalg.svd((a - am).T @ (b - bm))
        delta = u @ np.diag([1., 1., np.linalg.det(u @ vt)]) @ vt
        rotation = rotation @ delta
        translation = (translation - am) @ delta + bm
    return rotation, translation


def template_normals(points):
    # Coarse outward orientation relative to prototype centroid. This is a
    # surface-direction proxy and can be unreliable in deep concavities.
    tree = cKDTree(points)
    nearest = tree.query(points, k=24, workers=1)[1]
    local = points[nearest]
    local -= local.mean(1, keepdims=True)
    _, vectors = np.linalg.eigh(np.einsum('nki,nkj->nij', local, local))
    normals = vectors[:, :, 0]
    radial = points - points.mean(0)
    normals *= np.where(np.sum(normals * radial, axis=1) >= 0, 1., -1.)[:, None]
    return normals


def direction(query, template, tree, normals, rotation, translation):
    moved = query @ rotation + translation
    distances, ids = tree.query(moved, workers=1)
    ids = np.unique(ids[distances <= np.quantile(distances, .9)])
    mean = normals[ids].mean(0)
    strength = float(np.linalg.norm(mean))
    unit = mean / max(strength, 1e-12)
    return unit, strength, ids


def estimate(query, template, normals):
    tree = cKDTree(template)
    qframe, _ = pca_frame(query)
    tframe, _ = pca_frame(template)
    starts = [(one_way_score(query, tree), np.eye(3), np.zeros(3), 'raw_identity')]
    for axis in AXIS_MAPS:
        r = qframe @ axis @ tframe.T
        if np.linalg.det(r) <= 0:
            continue
        t = template.mean(0) - query.mean(0) @ r
        starts.append((one_way_score(query @ r + t, tree), r, t, 'pca'))
    starts.sort(key=lambda x: x[0])
    # Raw coordinates sometimes already match the common object frame; retain
    # identity even if its initial translation makes its coarse cost large.
    selected = starts[:8]
    if not any(s[3] == 'raw_identity' for s in selected):
        selected.append(next(s for s in starts if s[3] == 'raw_identity'))
    results = []
    for _, r, t, source in selected:
        r, t = refine(query, template, tree, r, t)
        score = one_way_score(query @ r + t, tree)
        d, strength, ids = direction(query, template, tree, normals, r, t)
        results.append(dict(score=score, rotation=r, translation=t,
                            direction=d, strength=strength, matched_ids=ids,
                            start=source))
    results.sort(key=lambda x: x['score'])
    best = results[0]
    contenders = [x for x in results if x['score'] <= best['score'] * 1.10]
    angles = [float(np.degrees(np.arccos(np.clip(np.dot(best['direction'], x['direction']), -1, 1))))
              for x in contenders]
    best['alternative_angle_deg'] = max(angles)
    best['ambiguous'] = max(angles) > 45 or best['strength'] < .15
    best['candidate_directions'] = [dict(score=x['score'], direction=x['direction'].tolist(),
                                        strength=x['strength']) for x in results]
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--categories', nargs='+', default=list(CATEGORIES))
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    with ZipFile(RAW_ZIP) as archive:
        for category in args.categories:
            train_path = sorted((DATA / category / 'train/good/xyz').glob('*/sample.json'))[0]
            meta = json.loads(train_path.read_text())
            basis = np.asarray(meta['basis'])
            center = np.asarray(meta['oriented_center'])
            radius = meta['normalization_radius']
            raw = read_xyz(archive.read(meta['raw_member'])).astype(np.float64)
            template = downsample((raw @ basis - center) / radius, 6000, meta['raw_member'])
            normals = template_normals(template)
            folder = OUT / category
            folder.mkdir(exist_ok=True)
            np.savez_compressed(folder / 'template.npz', xyz=template, normals=normals)
            paths = sorted((DATA / category / 'test').glob('*/xyz/*/sample.json'))
            if args.limit:
                paths = paths[:args.limit]
            for i, path in enumerate(paths):
                destination = folder / (path.parent.name + '.json')
                if destination.exists():
                    continue
                sample = json.loads(path.read_text())
                raw = read_xyz(archive.read(sample['raw_member'])).astype(np.float64)
                query = downsample((raw @ basis - center) / radius, 1800, sample['raw_member'])
                result = estimate(query, template, normals)
                d = result['direction']
                axis = int(np.argmax(np.abs(d)))
                record = dict(category=category, sample=path.parent.name,
                              raw_member=sample['raw_member'], template=meta['raw_member'],
                              score=result['score'], direction=d.tolist(),
                              normal_strength=result['strength'], ambiguous=bool(result['ambiguous']),
                              alternative_angle_deg=result['alternative_angle_deg'],
                              sector='XYZ'[axis] + ('+' if d[axis] >= 0 else '-'),
                              rotation_row=result['rotation'].tolist(),
                              translation_row=result['translation'].tolist(),
                              candidates=result['candidate_directions'],
                              method='one_way_trim90_ICP15_PCA8_plus_identity_v1')
                destination.write_text(json.dumps(record, indent=2), encoding='utf-8')
                np.savez_compressed(folder / (path.parent.name + '.npz'),
                                    aligned_query=query @ result['rotation'] + result['translation'],
                                    matched_ids=result['matched_ids'])
                if (i + 1) % 10 == 0 or i + 1 == len(paths):
                    print(f'{category}: {i + 1}/{len(paths)}', flush=True)
    print('Direction fits saved.', flush=True)


if __name__ == '__main__':
    main()
