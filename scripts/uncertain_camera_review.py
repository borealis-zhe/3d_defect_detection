"""Geometry-only review; selection is independent of pending full evaluation."""
import json
import sys
from pathlib import Path
from zipfile import ZipFile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from repro.raw_coverage import RAW_ZIP,read_xyz
OUT=ROOT/'outputs/uncertain_camera_review'
REASONS={'partial_template_fit_too_large':'模板匹配误差偏大','near_equal_matches_disagree_on_side':'近似同分候选方向分歧','weak_surface_direction':'表面方向较弱'}
plt.rcParams['font.sans-serif']=['Microsoft YaHei','DejaVu Sans']
plt.rcParams['axes.unicode_minus']=False

def project_depth(points,C,R):
    q=(points-C)@R.T
    uv=q[:,:2]/q[:,2,None]*(112/np.tan(np.pi/6))+111.5
    ij=np.rint(uv).astype(int)
    ok=(q[:,2]>0)&(ij>=0).all(1)&(ij<224).all(1)
    depth=np.full(224*224,np.inf)
    np.minimum.at(depth,ij[ok,1]*224+ij[ok,0],q[ok,2])
    depth=depth.reshape(224,224);depth[~np.isfinite(depth)]=np.nan
    return depth

def equal_axes(ax,points,extra=None):
    p=points if extra is None else np.vstack([points,extra])
    lo=p.min(0);hi=p.max(0);mid=(lo+hi)/2;r=(hi-lo).max()*.55
    ax.set(xlim=(mid[0]-r,mid[0]+r),ylim=(mid[1]-r,mid[1]+r),zlim=(mid[2]-r,mid[2]+r),xlabel='X',ylabel='Y',zlabel='Z')
    ax.set_box_aspect((1,1,1))

def main():
    global OUT
    review=None
    if len(sys.argv)>1:
        result_dir=Path(sys.argv[1]);review=json.loads((result_dir/'review_summary.json').read_text())
        OUT=ROOT/'outputs/camera_all_human_review'
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'plotly.min.js').write_text(get_plotlyjs(),encoding='utf-8')
    selected=[]
    for cat in ([] if review else ['airplane','chicken','diamond','duck','gemstone','shell','toffees']):
        cs=[json.loads(p.read_text()) for p in (ROOT/'outputs/estimated_camera_v1'/cat/'test').glob('*/camera.json')]
        cs=[c for c in cs if c['status']!='ready']
        selected.append(max(cs,key=lambda c:(c['alternative_angle_deg']>45,c['alternative_angle_deg'],c['score_over_template_spacing'])))
    if review:
        for row in review['selected']:
            c=json.loads((ROOT/'outputs/estimated_camera_v1'/row['category']/'test'/row['sample']/'camera.json').read_text())
            c['evaluation']=row;selected.append(c)
    (OUT/'selection.json').write_text(json.dumps(selected,indent=2),encoding='utf-8')
    cards=[]
    with ZipFile(RAW_ZIP) as archive:
        for c in selected:
            cat,name=c['category'],c['sample'];key=cat+'_'+name
            raw=read_xyz(archive.read(c['raw_member'])).astype(float)
            mid=(raw.min(0)+raw.max(0))/2;scale=np.linalg.norm(raw-mid,axis=1).max()
            points=(raw-mid)/scale
            ids=np.random.default_rng(42).choice(len(raw),min(16000,len(raw)),replace=False);p=points[ids]
            C=(np.array(c['camera_position_raw'])-mid)/scale
            R=np.array(c['world_to_camera_raw'])[:3,:3]
            forward=R.T@np.array([0.,0.,1.]);up=R.T@np.array([0.,-1.,0.]);right=R.T@np.array([1.,0.,0.])
            distance=c['distance_template']*c['template_scale']/scale
            target=C+forward*distance
            opposite=target+forward*distance
            opposite_R=np.stack([-right,-up,-forward])
            corners=[C+forward*.6+right*x*.346+up*y*.346 for x,y in [(-1,-1),(1,-1),(1,1),(-1,1)]]
            edges=[(C,k) for k in corners]+[(corners[i],corners[(i+1)%4]) for i in range(4)]
            fig=plt.figure(figsize=(16,10),layout='constrained');gs=fig.add_gridspec(2,3)
            for col,(elev,azim,title) in enumerate([(25,-60,'点云斜视：保留真实壳体形态'),(8,25,'同一点云侧视：检查开口、厚度')]):
                ax=fig.add_subplot(gs[0,col],projection='3d');ax.scatter(*p.T,s=.35,c=p[:,2],cmap='cividis',alpha=.65,rasterized=True)
                equal_axes(ax,p);ax.view_init(elev,azim);ax.set_title(title)
            ax=fig.add_subplot(gs[0,2],projection='3d');ax.scatter(*p.T,s=.2,color='#708798',alpha=.35)
            for a,b in edges:ax.plot(*np.stack([a,b]).T,c='#d64b25',lw=1)
            ax.scatter(*C,c='#d64b25',marker='D',s=45);ax.quiver(*C,*forward,length=1,color='#d64b25',linewidth=2)
            ax.scatter(*opposite,c='#7c58a2',marker='x',s=35)
            equal_axes(ax,p,np.stack([C,opposite]));ax.view_init(20,-60);ax.set_title('红：待测试相机；紫：仅供对侧检查')
            for col,cam,rot,title in [(0,C,R,'候选相机看到的原始点云深度'),(1,opposite,opposite_R,'对侧诊断视图（未用于本轮推理）')]:
                ax=fig.add_subplot(gs[1,col]);d=project_depth(points,cam,rot);ax.imshow(d,cmap='cividis');ax.axis('off');ax.set_title(title+'\n几何投影，不是模型输入的灰度渲染')
            template=read_xyz(archive.read(c['template_raw_member'])).astype(float)
            ti=np.random.default_rng(1).choice(len(template),min(9000,len(template)),replace=False)
            template=(template[ti]@np.array(c['template_basis'])-c['template_center'])/c['template_scale']
            aligned=((raw[ids]@np.array(c['template_basis'])-c['template_center'])/c['template_scale'])@np.array(c['object_rotation_row'])+c['object_translation_row']
            ax=fig.add_subplot(gs[1,2],projection='3d');ax.scatter(*template.T,s=.15,c='#aab2b9',alpha=.2,label='完整正常模板');ax.scatter(*aligned.T,s=.25,c='#007fa3',alpha=.6,label='配准后的测试点云');equal_axes(ax,np.vstack([template,aligned]));ax.view_init(20,-60);ax.set_title('模板叠加：检查匹配错位');ax.legend(fontsize=8)
            reasons='；'.join(REASONS[r] for r in c['reasons']) or '通过原几何门槛（不保证方向正确）'
            subtitle='几何待复核，尚无本轮测试分数'
            if review:
                row=c['evaluation'];old_text='无异常点/AP不定义' if row['old_AP'] is None else f"{100*row['old_AP']:.1f}%"
                subtitle=f"{row['selection']}；AP {old_text} → {100*row['AP']:.1f}%"
            fig.suptitle(f'{cat} / {name}  |  {subtitle}\n{reasons}；候选方向分歧 {c["alternative_angle_deg"]:.1f}°；匹配误差/模板点间距 {c["score_over_template_spacing"]:.2f}',fontsize=13)
            fig.savefig(OUT/(key+'.png'),dpi=145);plt.close(fig)
            traces=[go.Scatter3d(x=p[:,0],y=p[:,1],z=p[:,2],mode='markers',marker=dict(size=1.5,color=p[:,2],colorscale='Cividis',opacity=.8),name='原始点云（平移缩放）')]
            lines=np.array([v for a,b in edges for v in [a,b,[np.nan]*3]])
            traces.append(go.Scatter3d(x=lines[:,0],y=lines[:,1],z=lines[:,2],mode='lines',line=dict(color='#d64b25',width=4),name='候选相机视锥'))
            for pos,label,color in [(C,'候选相机','#d64b25'),(opposite,'对侧诊断相机（未测试）','#7c58a2')]:
                traces.append(go.Scatter3d(x=[pos[0],target[0]],y=[pos[1],target[1]],z=[pos[2],target[2]],mode='lines+markers',line=dict(color=color,width=4),marker=dict(size=[7,2]),name=label))
            score_card=''
            if review:
                with np.load(result_dir/key/'prediction.npz') as z:
                    scores=z['scores'];labels=z['labels'];rid=z['raw_point_ids']
                refs=points[rid];pos=labels==1
                from repro.pose_render import raw_labels
                full_labels=raw_labels(cat,'test',name,len(raw));positive=np.flatnonzero(full_labels==1)
                gi=positive[np.linspace(0,len(positive)-1,min(5000,len(positive))).astype(int)] if len(positive) else positive
                traces.append(go.Scatter3d(x=points[gi,0],y=points[gi,1],z=points[gi,2],mode='markers',marker=dict(size=3,color='#e34a33'),name='真实异常点（原始点云）'))
                traces.append(go.Scatter3d(x=refs[:,0],y=refs[:,1],z=refs[:,2],mode='markers',marker=dict(size=3,color=scores,colorscale='Turbo',showscale=True,colorbar=dict(title='异常分数')),name='模型参考点分数',visible='legendonly'))
                sf,axs=plt.subplots(2,3,figsize=(15,9),layout='constrained',subplot_kw={})
                for i,v in enumerate([0,13,26]):
                    axs[0,i].imshow(Image.open(result_dir/key/f'view_{v:03d}.png'));axs[0,i].axis('off');axs[0,i].set_title(f'实际送入模型的灰度图 v{v:02d}')
                for i,title in enumerate(['原始点云真值：红色为全部异常区域','模型分数：仅当前参考点','参考点真值：红色为入选异常点']):
                    axs[1,i].remove();a=sf.add_subplot(2,3,i+4,projection='3d')
                    a.scatter(*p.T,s=.15,c='#b9c2c9',alpha=.12)
                    if i==0:a.scatter(*points[gi].T,s=2,c='#e34a33')
                    elif i==1:
                        art=a.scatter(*refs.T,s=2,c=scores,cmap='turbo');sf.colorbar(art,ax=a,shrink=.55,label='原始异常分数')
                    else:a.scatter(*refs.T,s=.35,c='#5795b1',alpha=.25);a.scatter(*refs[pos].T,s=2,c='#e34a33')
                    equal_axes(a,p);a.view_init(25,-60);a.set_title(title)
                common='无共同异常点' if row['common_AP'] is None else f"共同点 AP {100*row['old_common_AP']:.1f}% → {100*row['common_AP']:.1f}%"
                sf.suptitle(f"{key} | {subtitle}\n新旧参考点域不同；{common}；旧/新异常参考点 {row['old_positive_points']}/{row['positive_points']}；原始异常点 {len(positive)}",fontsize=13)
                sf.savefig(OUT/(key+'_scores.png'),dpi=140);plt.close(sf)
                score_card=f'<img src="{key}_scores.png" style="width:100%">'
            f=go.Figure(traces);f.update_layout(title=f'{cat}/{name}：拖动旋转、滚轮缩放；点击图例隐藏相机',scene=dict(aspectmode='data'),height=760,margin=dict(l=0,r=0,t=70,b=0))
            f.write_html(OUT/(key+'.html'),include_plotlyjs='plotly.min.js')
            cards.append(f'<section><h2>{cat} / {name}</h2><p>{subtitle}。{reasons}</p><a href="{key}.html">打开可旋转三维点云与相机（含真值/评分图层）</a><img src="{key}.png" style="width:100%">{score_card}<p>人工判断：□外表面正确 □从内部看壳体 □匹配错位 □仅平面旋转 □无法判断</p></section>')
            print(key,len(raw),flush=True)
    intro='已完成603个样本测试。案例按测试后分数筛选，含不确定类别的低AP案例（至少10个异常参考点）、海马退步、鱼类进步和不确定但表现较好的对照；属于探索性人工分析。' if review else '全量测试结果尚未导入。以下案例仅按几何歧义筛选。'
    (OUT/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>相机人工复核</title><style>body{font:17px sans-serif;max-width:1500px;margin:30px auto;background:#f7f9fb}section{background:white;padding:20px;margin:25px 0}</style><h1>点云、相机与异常定位：人工复核</h1><p>'+intro+'相机是估计的虚拟相机，不是真实扫描仪标定。点云仅整体平移缩放，未补洞、封闭或改变形状。三维图中点云为显示而抽样；计算没有使用该显示抽样。</p><p>请重点记录：红相机是否看向外表面？是否从开口看进内部？模板叠加是否错位？相反方向是否更合理？180°平面旋转应与正反面翻转分开判断。紫色对侧相机仅是诊断参照，并不表示它一定正确。深度图颜色表示离相机远近，不能当作异常分数；两张深度图各自缩放颜色范围。灰度缺失或点稀疏不能单独证明方向错误。配准候选方向分歧是在各候选模板坐标中计算，不能直接等同真实扫描方向差。</p>'+''.join(cards),encoding='utf-8')

if __name__=='__main__':main()
