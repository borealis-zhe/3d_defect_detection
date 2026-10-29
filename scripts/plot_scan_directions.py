"""Summarize inferred coverage directions and preserve uncertainty."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

from repro.bundle_ablate import CATEGORIES, ROOT

OUT = ROOT / 'outputs/scan_direction_audit'
SECTORS = ['X+', 'X-', 'Y+', 'Y-', 'Z+', 'Z-']


def main():
    allrows = []
    summary = []
    counts = []
    for category in CATEGORIES:
        rows = [json.loads(p.read_text()) for p in sorted((OUT / category).glob('*.json'))]
        if not rows:
            continue
        template=np.load(OUT/category/'template.npz')['xyz']
        spacing=float(np.median(cKDTree(template).query(template,k=2)[0][:,1]))
        for r in rows:
            r['candidate_ambiguous']=r['ambiguous']
            r['score_over_spacing']=r['score']/spacing
            r['poor_fit']=r['score_over_spacing']>2.0
            r['ambiguous']=r['ambiguous'] or r['poor_fit'] or r['normal_strength']<.3
        allrows.extend(rows)
        confident = [r for r in rows if not r['ambiguous']]
        n = [sum(r['sector'] == s for r in confident) for s in SECTORS]
        uncertain = len(rows) - len(confident)
        counts.append(n + [uncertain])
        directions = np.array([r['direction'] for r in confident])
        if len(directions):
            mean_length = np.linalg.norm(directions.mean(0))
            # Axis concentration remains useful when front/back signs differ.
            tensor = directions.T @ directions / len(directions)
            axis_concentration = np.linalg.eigvalsh(tensor)[-1]
        else:
            mean_length = axis_concentration = float('nan')
        record = dict(category=category,objects=len(rows),direction_uncertain=uncertain,
                      **{s:n[i] for i,s in enumerate(SECTORS)},
                      mean_vector_length=mean_length,
                      principal_axis_concentration=axis_concentration)
        summary.append(record)
    with (OUT / 'category_distribution.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
    flat = [{k:r[k] for k in ['category','sample','sector','score','normal_strength',
                              'alternative_angle_deg','ambiguous','candidate_ambiguous',
                              'score_over_spacing','poor_fit']} |
            dict(dx=r['direction'][0],dy=r['direction'][1],dz=r['direction'][2]) for r in allrows]
    with (OUT / 'sample_directions.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)

    values=np.array(counts)
    fig,ax=plt.subplots(figsize=(10,7),constrained_layout=True)
    im=ax.imshow(values,cmap='Blues',vmin=0,vmax=53,aspect='auto')
    ax.set_xticks(range(7),SECTORS+['Uncertain'])
    ax.set_yticks(range(len(summary)),[r['category'] for r in summary])
    ax.set_title('Estimated observed-surface direction per test cloud\n603 original test files; axes belong to each class template')
    for i in range(len(summary)):
        for j in range(7):
            ax.text(j,i,str(values[i,j]),ha='center',va='center',color='white' if values[i,j]>25 else 'black')
    fig.colorbar(im,ax=ax,label='Number of objects')
    fig.savefig(OUT/'direction_counts.png',dpi=180);plt.close(fig)

    fig=plt.figure(figsize=(15,11),layout='constrained')
    theta=np.linspace(0,2*np.pi,80)
    for i,category in enumerate(CATEGORIES):
        ax=fig.add_subplot(3,4,i+1,projection='3d')
        rows=[r for r in allrows if r['category']==category]
        for plane in range(3):
            xyz=np.zeros((3,len(theta)));xyz[(plane+1)%3]=np.cos(theta);xyz[(plane+2)%3]=np.sin(theta)
            ax.plot(*xyz,color='#cccccc',lw=.6)
        for ambiguous,color,marker in [(False,'#087f8c','o'),(True,'#dc8b2c','x')]:
            d=np.array([r['direction'] for r in rows if r['ambiguous']==ambiguous])
            if len(d):ax.scatter(*d.T,c=color,marker=marker,s=22,alpha=.7)
        ax.set(xlim=(-1.1,1.1),ylim=(-1.1,1.1),zlim=(-1.1,1.1),xlabel='X',ylabel='Y',zlabel='Z')
        ax.set_xticks([-1,0,1]);ax.set_yticks([-1,0,1]);ax.set_zticks([-1,0,1]);ax.set_box_aspect((1,1,1))
        ax.view_init(elev=20,azim=35)
        ax.set_title(category,fontsize=12)
    fig.suptitle('One dot = inferred visible-side direction in the template frame\nTeal: stable among tested matches; orange x: ambiguous / weak surface direction',fontsize=14)
    fig.savefig(OUT/'direction_spheres.png',dpi=160);plt.close(fig)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
