"""Visualize proposed camera positions only; never changes experiment configs."""
import sys,json,itertools
from pathlib import Path
from zipfile import ZipFile
import numpy as np
import open3d as o3d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.graph_objects as go

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from repro.raw_coverage import RAW_ZIP,read_xyz
OUT=ROOT/'outputs/view_selection_proposals'
plt.rcParams['font.sans-serif']=['Microsoft YaHei','DejaVu Sans'];plt.rcParams['axes.unicode_minus']=False

def ring(deg,n,phase=0):
    t=np.deg2rad(deg);a=np.arange(n)*2*np.pi/n+phase
    return np.column_stack([np.sin(t)*np.cos(a),np.sin(t)*np.sin(a),np.full(n,np.cos(t))])

def up(d):
    y=np.array([0.,1.,0.])
    if abs(d@y)>.95:y=np.array([1.,0.,0.])
    u=y-d*(d@y);return u/np.linalg.norm(u)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    c=json.loads((ROOT/'outputs/estimated_camera_v1/airplane/test/635_sink/camera.json').read_text())
    with ZipFile(RAW_ZIP) as z:raw=read_xyz(z.read(c['template_raw_member'])).astype(float)
    cloud=(raw@np.array(c['template_basis'])-c['template_center'])/c['template_scale']
    cloud=cloud[np.random.default_rng(42).choice(len(cloud),10000,replace=False)]
    angles=list(itertools.product([0.,-15.,15.],repeat=3))
    rs=[o3d.geometry.get_rotation_matrix_from_xyz(np.deg2rad(a)) for a in angles]
    a=np.array([r.T@np.array([0.,0.,1.]) for r in rs]);au=np.array([r.T@np.array([0.,1.,0.]) for r in rs])
    north=np.vstack([[0,0,1],ring(15,4),ring(30,8,np.pi/8)])
    b=np.vstack([north,-north])
    dirs=np.array([v for v in itertools.product([-1,0,1],repeat=3) if any(v)],float);dirs/=np.linalg.norm(dirs,axis=1)[:,None]
    d=np.vstack([[0,0,1],ring(25,6),ring(45,8,np.pi/8)])
    methods=[('A 原方案：单侧小角度',a,au,'27 张 / 25 个位置；最大倾角约 21°'),
             ('B 候选：正反两侧',b,np.array([up(x) for x in b]),'26 个位置；每侧 1 + 4@15° + 8@30°'),
             ('C 候选：全方位',dirs,np.array([up(x) for x in dirs]),'26 个位置；6 面 + 12 边 + 8 顶点方向'),
             ('D 候选：测试外侧双圈',d,np.array([up(x) for x in d]),'15 个位置；1 + 6@25° + 8@45°')]
    assert len(np.unique(np.round(a,8),axis=0))==25
    assert all(np.isclose(np.linalg.norm(m[1],axis=1),1).all() for m in methods)
    assert all(any(np.allclose(-v,w) for w in dirs) for v in dirs)
    fig=plt.figure(figsize=(20,10),layout='constrained');gs=fig.add_gridspec(2,4,height_ratios=[1.25,1])
    for i,(name,ds,ups,note) in enumerate(methods):
        pos=ds*2.5;colors=['#1976b7' if q[2]>.001 else '#db7627' if q[2]<-.001 else '#258967' for q in ds]
        ax=fig.add_subplot(gs[0,i],projection='3d');ax.scatter(*cloud.T,s=.2,c='#718695',alpha=.3,rasterized=True)
        for zlevel in [-2.5,0,2.5]:
            if zlevel==0:
                t=np.linspace(0,2*np.pi,120);ax.plot(2.5*np.cos(t),2.5*np.sin(t),t*0,c='#dce1e5',lw=.8)
        t=np.linspace(0,2*np.pi,120)
        ax.plot(2.5*np.cos(t),t*0,2.5*np.sin(t),c='#e2e6e9',lw=.7)
        ax.scatter(*pos.T,s=32,c=colors,edgecolors='white',linewidths=.5,depthshade=False)
        ax.quiver(*pos.T,*(-ds*.35).T,color=colors,arrow_length_ratio=.32,linewidth=.8)
        for p,u,color in zip(pos,ups,colors):ax.plot(*np.stack([p,p+u*.15]).T,c=color,lw=1)
        ax.text(0,0,2.85,'主方向 +Z',fontsize=9,c='#1976b7');ax.text(0,0,-2.95,'对侧 −Z',fontsize=9,c='#db7627')
        ax.set(xlim=(-2.9,2.9),ylim=(-2.9,2.9),zlim=(-2.9,2.9),xlabel='X',ylabel='Y',zlabel='Z')
        ax.set_box_aspect((1,1,1));ax.view_init(18,-58);ax.set_xticks([-2,0,2]);ax.set_yticks([-2,0,2]);ax.set_zticks([-2,0,2]);ax.set_title(name+'\n'+note,fontsize=12,pad=20)
        ax=fig.add_subplot(gs[1,i]);ax.plot(2.5*np.cos(t),2.5*np.sin(t),c='#dce1e5',lw=1)
        ax.scatter(cloud[:,0],cloud[:,2],s=.3,c='#718695',alpha=.3)
        ax.scatter(pos[:,0],pos[:,2],s=45,c=colors,edgecolors='white',lw=.5)
        ax.axhline(0,c='#c5cdd4',lw=.7);ax.axvline(0,c='#c5cdd4',lw=.7)
        ax.set(xlim=(-2.9,2.9),ylim=(-2.9,2.9),xlabel='X',ylabel='Z',title='侧面投影 X–Z（不同 Y 的位置可能重叠）');ax.set_aspect('equal')
    fig.suptitle('相机位置方案对比｜同一飞机正常模板 128_template，同一观察中心与距离\n蓝色：主方向半球；橙色：对侧半球；绿色：侧面；箭头朝向物体。灰色线仅标示相机所在球面。',fontsize=17)
    fig.savefig(OUT/'camera_positions_comparison.png',dpi=160,bbox_inches='tight');plt.close(fig)
    f=go.Figure();f.add_trace(go.Scatter3d(x=cloud[:,0],y=cloud[:,1],z=cloud[:,2],mode='markers',marker=dict(size=1.5,color='#7e909d',opacity=.55),name='完整正常飞机模板'))
    for name,ds,ups,note in methods:
        pos=2.5*ds;colors=['#1976b7' if q[2]>.001 else '#db7627' if q[2]<-.001 else '#258967' for q in ds]
        custom=np.column_stack([np.arange(len(ds)),np.degrees(np.arccos(np.clip(ds[:,2],-1,1)))])
        f.add_trace(go.Scatter3d(x=pos[:,0],y=pos[:,1],z=pos[:,2],mode='markers+text',text=[str(i) for i in range(len(ds))],textposition='top center',marker=dict(size=6,color=colors),customdata=custom,hovertemplate='编号 %{customdata[0]}<br>相对 +Z 倾角 %{customdata[1]:.1f}°<br>位置 (%{x:.2f}, %{y:.2f}, %{z:.2f})<extra></extra>',name='相机位置',visible=False))
        line=np.array([v for p in pos for v in [p,p*.83,[np.nan]*3]])
        f.add_trace(go.Scatter3d(x=line[:,0],y=line[:,1],z=line[:,2],mode='lines',line=dict(color='#767f89',width=3),name='朝向中心',visible=False))
        line=np.array([v for p,u in zip(pos,ups) for v in [p,p+u*.23,[np.nan]*3]])
        f.add_trace(go.Scatter3d(x=line[:,0],y=line[:,1],z=line[:,2],mode='lines',line=dict(color='#8d559b',width=4),name='画面向上方向',visible=False))
    for i in range(1,4):f.data[i].visible=True
    buttons=[]
    for i,(name,ds,ups,note) in enumerate(methods):
        vis=[True]+[False]*12
        for j in range(1+i*3,4+i*3):vis[j]=True
        buttons.append(dict(label=name,method='update',args=[{'visible':vis},{'title':name+'｜'+note}]))
    f.update_layout(title=methods[0][0]+'｜'+methods[0][3],height=800,scene=dict(aspectmode='cube',xaxis=dict(range=[-3,3]),yaxis=dict(range=[-3,3]),zaxis=dict(range=[-3,3]),camera=dict(eye=dict(x=1.5,y=-1.6,z=.9))),updatemenus=[dict(buttons=buttons,x=0,y=1.12)],margin=dict(t=130,l=0,r=0,b=0),legend=dict(orientation='h',y=0))
    plot=f.to_html(full_html=False,include_plotlyjs=True)
    intro='<h1>相机位置方案：只供讨论，未运行新实验</h1><p>下拉框切换方案，拖动旋转、滚轮缩放。相机都位于半径2.5的球面并朝向物体中心。紫色短线表示画面朝上方向；相机位置相同但画面旋转不同的视图可能重叠。</p><p>A 按现有 Open3D 的旋转矩阵精确换算为物体不动、相机移动后的等价位姿。B 是新明确的26位置提案；C 是全方位26位置提案；D 是用于单面测试点云的15位置提案。B/C/D没有实际训练验证。</p><p>为了比较分布，四个方案都展示同一完整正常飞机模板。D 中 +Z 仅代表估计的外侧中心方向；真实测试时整个相机组随该样本估计方向旋转，并不固定在原始点云的 +Z。这里没有用完整模板假装测试点云，也没有声称球面位置覆盖了全部可见表面。</p>'
    (OUT/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>相机位置方案</title><style>body{font:17px sans-serif;max-width:1500px;margin:30px auto;background:#f7f9fb}p{line-height:1.7}</style>'+intro+plot+'<img src="camera_positions_comparison.png" style="width:100%">',encoding='utf-8')
    payload={name:dict(directions=ds.tolist(),camera_positions=(ds*2.5).tolist(),image_up=ups.tolist(),description=note) for name,ds,ups,note in methods}
    (OUT/'positions.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
    print(OUT)

if __name__=='__main__':main()
