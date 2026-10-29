import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
source=ROOT/'outputs/camera_all_results/review_summary.json'
r=json.loads(source.read_text())
lines=['# 全部603个样本：直接采用候选相机的测试','',
'模型为 pose_mv_v1 各类 final.pt，本轮没有训练或修正167个不确定位姿；仅取消排除并使用其已有候选相机。旧方案使用同一组物体和相同模型权重。', '',
'## 分组结果','', '|子集|物体数|类别数|指标|旧相机 %|新相机 %|变化（百分点）|','|---|---:|---:|---|---:|---:|---:|']
for g,items in r['groups'].items():
    for k in ['O_AUROC','P_AUROC','Point_AP']:
        pairs=[(x['old'][k],x['new'][k]) for x in items if x['old'][k] is not None and x['new'][k] is not None]
        a=sum(x[0] for x in pairs)/len(pairs)*100;b=sum(x[1] for x in pairs)/len(pairs)*100
        lines.append(f"|{g}|{sum(x['new']['objects'] for x in items)}|{len(pairs)}|{k}|{a:.2f}|{b:.2f}|{b-a:+.2f}|")
lines+=['','分组均为有定义且可配对的类别宏平均。needs_review只涉及7类，不能直接拿它的宏平均和全部12类宏平均相减。', '',
'## 全量逐类结果','', '|类别|数量|物体 AUROC 旧→新|点 AUROC 旧→新|Point AP 旧→新|','|---|---:|---:|---:|---:|']
for x in r['groups']['all']:
    vals=[f"{100*x['old'][k]:.2f} → {100*x['new'][k]:.2f}" for k in ['O_AUROC','P_AUROC','Point_AP']]
    lines.append(f"|{x['category']}|{x['new']['objects']}|"+'|'.join(vals)+'|')
lines+=['','## 人工复核案例','', '|类别/样本|选取原因|AP 旧→新|共同原始点AP 旧→新|异常参考点 旧→新|','|---|---|---:|---:|---:|']
def val(v):return '未定义' if v is None else f'{v*100:.2f}'
for x in r['selected']:
    lines.append(f"|{x['category']}/{x['sample']}|{x['selection']}|{val(x['old_AP'])} → {val(x['AP'])}|{val(x['old_common_AP'])} → {val(x['common_AP'])}|{x['old_positive_points']} → {x['positive_points']}|")
lines+=['','## 解释边界','',
'- 本轮不意味着167个不可靠方向已经修复。它检验的是直接使用这些候选的后果。',
'- 新旧参考点域不同；共同点AP有助于排除一部分选点变化，但交集也不能代表全部原始点云。',
'- 无异常参考点时单物体AP不定义，不能当作AP为零或定位成功。',
'- 人工复核例子依据测试分数事后筛选，适合诊断，不是独立验证集。人工修改后的指标应继续标为探索性结果。',
'- 估计相机不是数据集提供的真实扫描仪标定；对侧视图只用于诊断。',
'','[打开人工复核页面](../camera_all_human_review/index.html)']
(source.parent/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('\n'.join(lines[:20]))
