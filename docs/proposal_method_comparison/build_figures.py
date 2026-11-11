"""Render proposal diagrams only; never import or execute project models.

The same node/edge definitions produce Mermaid, SVG and 4K PNG assets.
Run: python docs/proposal_method_comparison/build_figures.py
"""
from pathlib import Path
import html
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, PathPatch
from matplotlib.path import Path as MplPath

OUT = Path(__file__).resolve().parent
FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
font_manager.fontManager.addfont(FONT)
plt.rcParams.update({'font.family': font_manager.FontProperties(fname=FONT).get_name(),
                     'svg.fonttype': 'path', 'axes.unicode_minus': False})
INK, MUTED = '#142B42', '#53677D'
STYLES = {
    'base': ('#EDF2F7', '#71859A', 'solid'),
    'done': ('#EAF3FF', '#3574B8', 'solid'),
    'partial': ('#FFF4DF', '#BB7A21', 'dashdot'),
    'plan': ('#F2ECFA', '#8560A9', 'dashed'),
    'unknown': ('#F2F2F2', '#8A8A8A', 'dotted'),
    'eval': ('#E7F5F0', '#2E8169', 'solid'),
}


def node(uid, x, y, w, h, title, body='', style='done', ref=''):
    return dict(id=uid, rect=[x, y, w, h], title=title, body=body, style=style, ref=ref)


def edge(a, b, points, label='', dashed=False):
    return dict(a=a, b=b, points=points, label=label, dashed=dashed)


def chain(nodes):
    result = []
    for a, b in zip(nodes, nodes[1:]):
        x, y, w, h = a['rect']; xx, yy, ww, hh = b['rect']
        result.append(edge(a['id'], b['id'], [(x+w, y+h/2), (xx, yy+hh/2)]))
    return result


def text(ax, x, y, value, size=12, color=INK, weight='normal', ha='left', va='center'):
    return ax.text(x, y, value, fontsize=size, color=color, weight=weight,
                   ha=ha, va=va, linespacing=1.55, zorder=5)


def panel(ax, x, y, w, h, title, lines, style='base', size=11.5):
    fill, stroke, dash = STYLES[style]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.015,rounding_size=0.10',
                              facecolor=fill, edgecolor=stroke, linestyle=dash, linewidth=1.1))
    text(ax, x+.18, y+h-.32, title, 13.5, stroke, 'bold')
    text(ax, x+.18, y+h-.71, '\n'.join(lines), size, va='top')


def draw(spec, pdf):
    fig = plt.figure(figsize=(16, 9), dpi=240, facecolor='#FAFCFE')
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 16); ax.set_ylim(0, 9); ax.axis('off')
    text(ax, .6, 8.55, spec['title'], 26, weight='bold')
    text(ax, .62, 8.04, spec['subtitle'], 12.5, MUTED)
    ax.plot([.6, 15.4], [7.68, 7.68], color='#DCE5EF', lw=1)
    for label in spec.get('labels', []):
        text(ax, *label)
    for p in spec.get('panels', []): panel(ax, **p)
    for e in spec['edges']:
        pts = e['points']
        vertices = pts; codes = [MplPath.MOVETO]+[MplPath.LINETO]*(len(pts)-1)
        ax.add_patch(FancyArrowPatch(path=MplPath(vertices, codes), arrowstyle='-|>',
                                    mutation_scale=13, linewidth=1.5, color=MUTED,
                                    linestyle='--' if e['dashed'] else '-', zorder=2))
        if e['label']:
            mid = pts[len(pts)//2]
            text(ax, mid[0], mid[1]+.16, e['label'], 9.3, MUTED, ha='center')
    node_texts=[]
    for n in spec['nodes']:
        x, y, w, h = n['rect']; fill, stroke, dash = STYLES[n['style']]
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.015,rounding_size=0.10',
                                  facecolor=fill, edgecolor=stroke, linewidth=1.5, linestyle=dash, zorder=3))
        artists=[text(ax,x+w/2,y+h-.30,n['title'],12.5,stroke,'bold',ha='center')]
        if n['body']: artists.append(text(ax,x+w/2,y+h-.67,n['body'],10.5,ha='center',va='top'))
        if n['ref']: artists.append(text(ax,x+w/2,y+.13,n['ref'],8.5,MUTED,ha='center'))
        node_texts.append((n,artists))
    text(ax,.62,.36,spec['footer'],9.5,MUTED)
    text(ax,15.35,.36,spec['number'],11,MUTED,ha='right')
    fig.savefig(OUT/(spec['name']+'.svg'), facecolor=fig.get_facecolor())
    fig.savefig(OUT/(spec['name']+'.png'), dpi=240, facecolor=fig.get_facecolor())
    pdf.savefig(fig, facecolor=fig.get_facecolor())
    renderer = fig.canvas.get_renderer()
    overflow = []
    for t in ax.texts:
        bb=t.get_window_extent(renderer)
        if bb.x0 < 0 or bb.y0 < 0 or bb.x1 > fig.bbox.width or bb.y1 > fig.bbox.height:
            overflow.append(t.get_text())
    node_issues=[]
    for n,artists in node_texts:
        x,y,w,h=n['rect']; low=ax.transData.transform((x,y)); high=ax.transData.transform((x+w,y+h))
        boxes=[a.get_window_extent(renderer) for a in artists]
        if any(b.x0<low[0] or b.y0<low[1] or b.x1>high[0] or b.y1>high[1] for b in boxes):
            node_issues.append({'node':n['id'],'issue':'text_outside_node'})
        if any(a.overlaps(b) for i,a in enumerate(boxes) for b in boxes[i+1:]):
            node_issues.append({'node':n['id'],'issue':'text_overlap'})
    plt.close(fig)
    mmd = ['flowchart LR']
    for n in spec['nodes']:
        label = '<br/>'.join(filter(None,[n['title'],n['body'].replace('\n','<br/>'),n['ref']]))
        label=label.replace('"','&quot;')
        mmd.append(f'    {n["id"]}["{label}"]:::{n["style"]}')
    for e in spec['edges']:
        arrow = '-.->' if e['dashed'] else '-->'
        label = f'|"{e["label"]}"|' if e['label'] else ''
        mmd.append(f'    {e["a"]} {arrow}{label} {e["b"]}')
    for i,p in enumerate(spec.get('panels',[])):
        label='<br/>'.join([p['title'],*p['lines']]).replace('"','&quot;')
        mmd.append(f'    note_{i}["{label}"]:::{p.get("style","base")}')
    for key,(fill,stroke,dash) in STYLES.items():
        mmd.append(f'    classDef {key} fill:{fill},stroke:{stroke},color:{INK}'+
                   (',stroke-dasharray: 5 3;' if dash!='solid' else ';'))
    (OUT/(spec['name']+'.mmd')).write_text('\n'.join(mmd)+'\n')
    return {'name':spec['name'],'nodes':len(spec['nodes']),'edges':len(spec['edges']),
            'canvas_overflow':overflow,'node_issues':node_issues,'png_size':[3840,2160]}


def salve():
    n=[node('s_rgb',.65,5.87,2.25,1.35,'整层全景输入','无序、稀疏全景\n姿态校正；已知相机高度','base','论文 §4.1'),
       node('s_layout',3.45,6.15,2.65,1.12,'布局与 W/D/O','房间轮廓、置信度、语义开口','base','§4.2'),
       node('s_depth',3.45,4.53,2.65,1.12,'HoHoNet 深度','用于地板／天花板纹理渲染','base','§4.3'),
       node('s_pose',6.65,6.15,3.05,1.12,'语义几何对齐假设','同类型配对；方向、位置与宽度','done','§4.2；不是已确认对应'),
       node('s_bev',6.65,4.53,3.05,1.12,'候选位姿下的 RGB BEV','两图 × 地板／天花板','done','§4.3 / Fig. 3'),
       node('s_verify',10.3,4.53,2.65,1.12,'CNN 对齐验证','接受或拒绝相对位姿假设','done','§4.3'),
       node('s_graph',10.3,2.62,2.65,1.26,'位姿图与全局定位','全景节点；可信相对约束\n初始化与鲁棒 PGO','done','§4.4'),
       node('s_fuse',6.65,2.62,3.05,1.26,'同房间轮廓融合','空间分组；高置信边界选择','done','§4.5'),
       node('s_out',3.45,2.62,2.65,1.26,'整体平面图','融合、拼接后的房间区域','eval','§4.5')]
    e=[edge('s_rgb','s_layout',[(2.9,6.55),(3.18,6.55),(3.18,6.71),(3.45,6.71)]),
       edge('s_rgb','s_depth',[(1.78,5.87),(1.78,5.09),(3.45,5.09)]),
       edge('s_layout','s_pose',[(6.1,6.71),(6.65,6.71)]),
       edge('s_pose','s_bev',[(8.18,6.15),(8.18,5.65)]),
       edge('s_depth','s_bev',[(6.1,5.09),(6.65,5.09)]),
       edge('s_rgb','s_bev',[(.65,6.55),(.38,6.55),(.38,4.18),(7.5,4.18),(7.5,4.53)],'RGB'),
       edge('s_bev','s_verify',[(9.7,5.09),(10.3,5.09)]),
       edge('s_verify','s_graph',[(11.63,4.53),(11.63,3.88)]),
       edge('s_graph','s_fuse',[(10.3,3.25),(9.7,3.25)]),
       edge('s_layout','s_fuse',[(4.78,7.27),(4.78,7.48),(13.48,7.48),(13.48,2.26),(8.18,2.26),(8.18,2.62)],'局部布局'),
       edge('s_fuse','s_out',[(6.65,3.25),(6.1,3.25)])]
    return dict(name='SALVe_技术路线',number='01',title='SALVe：先提出几何假设，再验证视觉对齐',
        subtitle='ECCV 2022  ·  原论文流程重绘  ·  几何用于生成假设，RGB 外观用于判断假设',nodes=n,edges=e,
        panels=[dict(x=.65,y=.91,w=14.7,h=1.05,title='汇报要点',lines=['SALVe 已使用视觉信息，也有轮廓置信度融合。比较重点是视觉证据的作用阶段与优化对象。'],size=12)],
        footer='来源：Lambert 等，SALVe，Fig. 2–3、§4.1–4.5；详细假设与官方源码索引见《技术对比分析》。')


def current():
    xs=[.65,3.65,6.65,9.65,12.65];w=2.7
    top=[node('c_input',xs[0],6.05,w,1.24,'单图感知｜已实现','Bi-Layout + Opening Head\n布局、开口、1D / 2D 特征','done','C01–C03 · 冻结前端'),
         node('c_repr',xs[1],6.05,w,1.24,'开口增强｜已实现','结构 + 上下文 + 几何\n融合、图内交互 → 描述子','done','C04 · 每张图编码一次'),
         node('c_pair',xs[2],6.05,w,1.24,'双图检索｜已实现','具体候选对条件化查询\n双向检索二维视觉记忆','done','C05 · 枚举全层无序图像对'),
         node('c_assign',xs[3],6.05,w,1.24,'分配与拒绝｜已实现','Sinkhorn + 无匹配节点\n互选、阈值、数值检查','done','C06 · 不是位姿验证'),
         node('c_pose',xs[4],6.05,w,1.24,'位姿多假设｜已实现','两种端点映射\nSim(2) / 尺度先验对照','done','C07 · 可替换端点几何')]
    lower=[node('c_verify',.65,3.97,2.7,1.35,'位姿评分｜已训练','几何 BEV + 视觉特征\n或几何规则策略','done','C08 · 学习工作点未通过'),
           node('c_strict',4.05,4.65,2.85,1.12,'严格模式','仅正式校准通过的边','done','C09 · 当前不承诺可靠接边'),
           node('c_explore',4.05,3.05,2.85,1.35,'实验模式','排序展示 / 工作点过滤\ntentative；未校准','partial','C10 · 过滤无工作点则不接边'),
           node('c_graph',7.62,4.0,2.22,1.35,'全局定位｜已实现','森林 / 鲁棒 PGO\n保留独立分量','done','C11 · 节点是全景'),
           node('c_out',10.25,4.0,2.22,1.35,'组装导出｜已实现','相机、区域、开口\n墙图、连通分量','done','C12 · 可选房间融合'),
           node('c_eval',12.88,4.0,2.47,1.35,'独立评估｜已实现','七项整层指标\n主分量单锚点规范','eval','C13 · GT 只在评价读取'),
           node('c_gt',12.88,2.18,2.47,.95,'GT｜仅离线评价','不回流到选边或组装','base')]
    e=chain(top)+[
        edge('c_pose','c_verify',[(14,6.05),(14,5.91),(2,5.91),(2,5.32)]),
        edge('c_verify','c_strict',[(3.35,4.89),(3.7,4.89),(3.7,5.25),(4.05,5.25)]),
        edge('c_verify','c_explore',[(3.35,4.4),(3.7,4.4),(3.7,3.8),(4.05,3.8)]),
        edge('c_strict','c_graph',[(6.9,5.25),(7.23,5.25),(7.23,4.96),(7.62,4.96)]),
        edge('c_explore','c_graph',[(6.9,3.8),(7.23,3.8),(7.23,4.37),(7.62,4.37)]),
        edge('c_graph','c_out',[(9.84,4.67),(10.25,4.67)]),
        edge('c_out','c_eval',[(12.47,4.67),(12.88,4.67)]),
        edge('c_gt','c_eval',[(14.12,3.13),(14.12,4.0)],dashed=True)]
    return dict(name='现有项目_真实流程',number='02',title='当前系统：推理链路已接通，可靠接边仍待解决',
        subtitle='代码状态 ≠ 效果验证  ·  基于当前分支及 2026-10-26 已有实验记录；本轮未运行模型',nodes=top+lower,edges=e,
        panels=[dict(x=.65,y=1.15,w=5.75,h=1.63,title='当前实验边界',lines=['R3 第 5000 步固定；V0 / V1 已训练。','两组工作点均失败；排序结果仅供诊断。','最新整层对照关闭房间融合。'],style='partial',size=11),
                dict(x=6.8,y=1.15,w=5.6,h=1.63,title='尚未完成的研究目标',lines=['端点拟合只改变位姿输入，不修正房间多边形。','PGO 优化相机变换，不优化局部墙段与端点。','学习房间关联有代码，本轮未训练。'],style='plan',size=11)],
        footer='蓝色实线：代码已实现；橙色点划线：实验策略；C01–C13 对应《代码证据清单》。')


def thesis():
    xs=[.65,3.65,6.65,9.65,12.65];w=2.7
    n=[node('t_front',xs[0],5.95,w,1.39,'01  单图感知','Bi-Layout 双布局\n候选开口与视觉特征','done','已有基础模型与前端适配'),
       node('t_match',xs[1],5.95,w,1.39,'02  研究一：开口对应','联合表征 + 双向检索\n匹配置信度与无匹配判断','done','已实现 · 可靠性待验证'),
       node('t_pose',xs[2],5.95,w,1.39,'03  初始位姿与关联','端点多假设 + 验证\n保留歧义与不连通分量','partial','已有流程 · 接边门槛未通过'),
       node('t_refine',xs[3],5.95,w,1.39,'04  研究二：局部修正','可信跨图观测约束\n墙段与开口端点更新','plan','拟研究 · 不等同于 PGO'),
       node('t_floor',xs[4],5.95,w,1.39,'05  整体平面重建','重估位姿 → 全局定位\n房间融合与结构输出','partial','后端已有 · 修正后闭环待接入')]
    e=chain(n)
    return dict(name='我的论文_完整技术路线',number='03',title='论文主线：先建立可靠关联，再约束局部几何',
        subtitle='输入：同一楼层全部 RGB 全景  →  输出：相机位姿、房间区域、开口连接与结构墙图',nodes=n,edges=e,
        labels=[(.65,5.40,'研究一回答：哪两个开口可以对应？',15,'#3574B8','bold'),
                (8.52,5.40,'研究二回答：可靠对应能修正哪里？',15,'#8560A9','bold')],
        panels=[dict(x=.65,y=2.47,w=7.25,h=2.51,title='开口实例驱动的视觉—几何匹配｜已实现',
                lines=['结构代理采样 + 邻域软覆盖上下文 + 几何属性',
                       '门控融合与图内交互 → 单图开口描述子',
                       '候选对条件化双向检索 → 联合分配与拒绝',
                       '验证目标：在控制误接受时保留更多跨房间正确对应'],style='done',size=12),
                dict(x=8.52,y=2.47,w=6.83,h=2.51,title='匹配约束下的局部结构更新｜拟研究',
                lines=['固定可靠对应与当前位姿，筛选有效跨图观测',
                       '优化墙段方向 / 位置及开口沿墙端点',
                       '本图证据 + 跨图一致性 + 结构保持约束',
                       '无有效观测保留原布局；异常更新回退'],style='plan',size=12),
                dict(x=.65,y=.95,w=14.7,h=1.15,title='验证闭环｜只设计，不在本轮执行',
                lines=['对应精确率 / 召回率与误接受 → 相对位姿质量 → 局部几何改善 → 七项整层指标；GT 仅用于离线监督及评价。'],style='eval',size=11.5)],
        footer='实线蓝：已实现；点划橙：部分闭环／实验状态；虚线紫：拟研究。方法差异是待验证假设，尚无优于 SALVe 的结论。')


def compare():
    xs=[.65,3.65,6.65,9.65,12.65];w=2.7
    upper=[node('u_front',xs[0],5.16,w,1.48,'布局与 W/D/O','单图布局、语义开口\nHoHoNet 深度','base','SALVe §4.1–4.3'),
           node('u_assoc',xs[1],5.16,w,1.48,'几何候选关联','同类语义与宽度筛选\n生成相对对齐假设','base','尚未确认对应'),
           node('u_verify',xs[2],5.16,w,1.48,'RGB BEV 验证','对齐地板 / 天花板纹理\nCNN 判断位姿假设','done','外观用于对齐验证'),
           node('u_geom',xs[3],5.16,w,1.48,'位姿定位后融合','位姿图优化\n同房间高置信轮廓选择','base','已有轮廓融合'),
           node('u_out',xs[4],5.16,w,1.48,'平面拼接输出','房间区域与相机位置\n原论文协议评测','eval','不能直接对比异协议数字')]
    lower=[node('v_front',xs[0],2.69,w,1.55,'双布局与候选开口','Bi-Layout 为基础\n共享前端与特征缓存','done','基础技术 · 非本课题原创'),
           node('v_assoc',xs[1],2.69,w,1.55,'研究一：开口实例匹配','结构 / 上下文 / 几何\n候选对检索 + 无匹配分配','done','先判断具体候选对应'),
           node('v_verify',xs[2],2.69,w,1.55,'位姿多假设与验证','几何 BEV + 视觉特征\n工作点筛选 / 诊断排序','partial','工作点未通过'),
           node('v_geom',xs[3],2.69,w,1.55,'研究二：局部形状修正','跨房间有效观测约束\n墙段位置 / 方向、端点','plan','拟研究；与相机 PGO 分开'),
           node('v_out',xs[4],2.69,w,1.55,'重定位与整体输出','PGO、区域融合、结构图\n七项指标与失败诊断','partial','已有后端；完整目标待验证')]
    return dict(name='SALVe_vs_我的方法',number='04',title='关键差异：视觉证据作用于哪里，几何变量优化什么',
        subtitle='逐阶段概念对齐；同列不代表两方法具有完全相同的输入、尺度假设或执行细节',nodes=upper+lower,edges=chain(upper)+chain(lower),
        labels=[(.65,7.22,'01 感知',12,MUTED), (3.65,7.22,'02 建立候选关联',12,MUTED),
                (6.65,7.22,'03 判断可接受关联',12,MUTED),(9.65,7.22,'04 几何处理',12,MUTED),
                (12.65,7.22,'05 整体输出',12,MUTED),(.65,6.91,'SALVe｜已有论文',14,INK,'bold'),
                (.65,4.65,'本课题｜当前实现 + 拟研究目标',14,INK,'bold')],
        panels=[dict(x=.65,y=.95,w=14.7,h=1.15,title='可检验的差异，而非已经证明的优势',
                lines=['H1：候选对视觉检索改善固定误接受率下的对应召回。  H2：可靠跨图约束改善局部几何，并减少整体重建误差。'],style='eval',size=12)],
        footer='共同基础：布局估计、相对几何、位姿图与拼接。不能把“使用视觉”“跨注意力”或“有优化”单独作为创新。')


def main():
    specs=[salve(),current(),thesis(),compare()]
    with PdfPages(OUT/'开题技术路线_四页.pdf') as pdf:
        results=[draw(s,pdf) for s in specs]
    (OUT/'figure_spec.json').write_text(json.dumps(specs,ensure_ascii=False,indent=2)+'\n')
    (OUT/'figure_checks.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
    cards='\n'.join(f'<section id="figure{i}"><h2>{html.escape(s["title"])}</h2>'
                   f'<img src="{s["name"]}.svg" alt="{html.escape(s["title"])}">'
                   f'<p><a href="{s["name"]}.png">4K PNG</a> · <a href="{s["name"]}.svg">SVG</a> · '
                   f'<a href="{s["name"]}.mmd">Mermaid</a></p></section>'
                   for i,s in enumerate(specs,1))
    links=' · '.join(f'<a href="{name}.md">{name}</a>' for name in [
        '技术对比分析','代码证据清单','实验验证设计','开题预演讲稿','导师可能追问','优化后的Codex提示词'])
    (OUT/'index.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1"><title>开题预演技术路线</title>'
        '<style>body{font:16px/1.8 system-ui,"Noto Sans CJK SC",sans-serif;margin:36px auto;max-width:1500px;'
        'padding:0 24px;background:#f4f7fb;color:#142b42}h1{font-size:30px}h2{font-size:22px}'
        'section{background:white;padding:22px;margin:28px 0;border-radius:12px}img{width:100%;height:auto}'
        'a{color:#24629e}p{margin:8px 0}@media print{section{break-after:page;margin:0;padding:0}'
        'body{margin:0;padding:0}h1,header,section>h2,section>p{display:none}}</style>'
        '<header><h1>开题预演：技术路线与研究贡献</h1><p>2026-10-26｜证据来自现有代码与论文；本轮未运行模型实验。</p>'
        '<p><strong>讲解顺序：</strong>SALVe → 当前实现 → 论文主图 → 双泳道对比。'
        '研究二为拟研究，可靠接边效果尚未建立。</p><p><a href="开题技术路线_四页.pdf">四页 PDF</a> · '
        '<a href="开题预演材料.zip">完整材料包</a></p><p>'+links+'</p></header>'+cards+'</html>')
    print(json.dumps(results,ensure_ascii=False,indent=2))


if __name__=='__main__': main()
