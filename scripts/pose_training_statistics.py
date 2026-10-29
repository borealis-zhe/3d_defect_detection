"""Object-block uncertainty and fish front/back failure audit; no inference."""
from pathlib import Path
import json
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_pose_training as a


def ap_for(items, indices):
    y = np.concatenate([items[i]['labels'] for i in indices])
    s = np.concatenate([items[i]['scores'] for i in indices])
    known = y >= 0
    return float(average_precision_score(y[known], s[known]))


def bootstrap_ap(old, new, names, repetitions=500):
    # Paired object blocks, stratified by object normal/anomaly label.
    aitems, bitems = [old[n] for n in names], [new[n] for n in names]
    strata = [np.array([i for i, n in enumerate(names) if ('/good/' in n) == normal])
              for normal in (True, False)]
    rng = np.random.default_rng(20261029)
    differences = []
    for _ in range(repetitions):
        ids = np.concatenate([rng.choice(group, len(group), replace=True) for group in strata])
        differences.append(ap_for(bitems, ids) - ap_for(aitems, ids))
    return dict(repetitions=repetitions,
                delta=ap_for(bitems, range(len(names))) - ap_for(aitems, range(len(names))),
                percentile_95_ci=np.quantile(differences, [.025, .975]).tolist())


def main():
    old = a.load_predictions('fish', a.OLD)
    new = a.load_predictions('fish', a.NEW)
    flip_names = []
    for p in (a.POSE/'fish/test').glob('*/pose.json'):
        pose = json.loads(p.read_text())
        side_angle = float(np.degrees(np.arccos(np.clip(np.array(pose['rotation_row'])[2, 2], -1, 1))))
        if side_angle > 90:
            flip_names.append(p.parent.name)
    names = sorted(old)
    unflipped = [n for n in names if n.split('/')[-1] not in flip_names]
    report = {'seed_count': 1, 'fish_flipped_samples': sorted(flip_names),
              'fish_flipped_count': len(flip_names), 'unflipped_objects': len(unflipped),
              'bootstrap_scope': 'paired normal/anomaly-stratified test-object blocks; fixed trained models; exploratory, own point domains',
              'fish_all_bootstrap': bootstrap_ap(old, new, names),
              'fish_unflipped_bootstrap': bootstrap_ap(old, new, unflipped)}
    for version, data in [('old', old), ('new', new)]:
        ordered = []
        for name, item in data.items():
            known = item['labels'] >= 0
            ordered.extend((float(score), int(label), name)
                           for score, label in zip(item['scores'][known], item['labels'][known]))
        ordered.sort(reverse=True)
        k = sum(y for s, y, n in ordered)
        top = ordered[:k]
        subset_items = [data[n] for n in unflipped]
        report[version] = dict(top_k=k, top_k_true=sum(y for s, y, n in top),
                               top_k_false=sum(y == 0 for s, y, n in top),
                               top_k_false_flipped=sum(y == 0 and n.split('/')[-1] in flip_names for s, y, n in top),
                               top_k_false_normal_objects=sum(y == 0 and '/good/' in n for s, y, n in top),
                               full_ap=ap_for([data[n] for n in names], range(len(names))),
                               unflipped_ap=ap_for(subset_items, range(len(subset_items))))
    (a.OUT/'fish_diagnosis.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    x = np.arange(2)
    axes[0].bar(x-.17, [100*report['old']['full_ap'],100*report['old']['unflipped_ap']],
                width=.34,color='#999999',label='Original')
    axes[0].bar(x+.17, [100*report['new']['full_ap'],100*report['new']['unflipped_ap']],
                width=.34,color='#1f78b4',label='Pose-trained')
    axes[0].set_xticks(x,['All 50 fish','41 without back-side flip\n(post-hoc diagnostic subset)'])
    axes[0].set_ylabel('Point AP (%)');axes[0].legend();axes[0].set_title('Regression is concentrated in 9 flipped fish')
    axes[1].bar(['9 flipped fish','Other 41 fish'],[report['new']['top_k_false_flipped'],
                 report['new']['top_k_false']-report['new']['top_k_false_flipped']],color=['#e66101','#5e3c99'])
    axes[1].set_ylabel('False-positive points in class-wide top 1,348 scores')
    axes[1].set_title('1,128 of 1,134 high-score false positives')
    fig.savefig(a.OUT/'figures/fish_flip_diagnosis.png',dpi=200);plt.close(fig)
    print(json.dumps(report, indent=2),flush=True)


if __name__=='__main__':
    main()
