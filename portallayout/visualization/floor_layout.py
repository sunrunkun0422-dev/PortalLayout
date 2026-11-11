"""Display each disconnected component in its own axes; never change poses."""
import math
from pathlib import Path
import numpy as np
from portallayout.geometry.floor_geometry import polygon, rings


def render_floor(prediction, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.path import Path as PlotPath
    from matplotlib.patches import PathPatch
    from shapely.geometry.polygon import orient
    components = prediction['components']
    count = max(1, len(components)); columns = min(4, count); rows = int(math.ceil(count/columns))
    fig, axes = plt.subplots(rows, columns, figsize=(5*columns, 4.5*rows), squeeze=False)
    for ax in axes.flat: ax.set_visible(False)
    for ci, component in enumerate(components):
        ax = axes.flat[ci]; ax.set_visible(True); cid = component['component_id']
        rooms = [r for r in prediction['rooms'] if r['component_id'] == cid]
        for i, room in enumerate(rooms):
            p = polygon(room['geometry'])
            if p is None: continue
            # Opposite winding keeps unobserved interior holes transparent.
            for part in ([p] if p.geom_type == 'Polygon' else p.geoms):
                part = orient(part, sign=1.)
                paths = []
                for ring in [part.exterior]+list(part.interiors):
                    vertices = np.asarray(ring.coords)
                    codes = np.full(len(vertices), PlotPath.LINETO)
                    codes[0], codes[-1] = PlotPath.MOVETO, PlotPath.CLOSEPOLY
                    paths.append(PlotPath(vertices, codes))
                ax.add_patch(PathPatch(PlotPath.make_compound_path(*paths), alpha=.22,
                                       facecolor=plt.cm.tab20(i%20), edgecolor='none'))
        for wall in prediction['structural_graphs'][cid]['walls']:
            p = np.array(wall['endpoints']); ax.plot(p[:, 0], p[:, 1], color='#263238', linewidth=1.)
        if prediction.get('inference_mode') == 'exploratory':
            tree = set(component['tree_edge_ids'])
            for edge in prediction['pose_graph']['accepted_edges']:
                if edge['edge_id'] not in tree: continue
                a, b = [np.asarray(component['poses'][uid])[:2,2] for uid in (edge['a'],edge['b'])]
                ax.plot([a[0],b[0]],[a[1],b[1]],'--',color='#6a7180',linewidth=.7,alpha=.6)
        for o in prediction['observations']:
            if o['component_id'] != cid: continue
            p = np.array(o['T_local_to_component']); t = p[:2, 2]; direction = p[:2, 1]
            ax.plot(t[0], t[1], 'o', color='#d84315', markersize=3)
            direction = direction/max(np.linalg.norm(direction), 1e-9)*.12
            ax.arrow(*t, *direction, color='#d84315', width=.005)
            ax.annotate(o['observation_id'][-5:], t, fontsize=6)
        for opening in prediction['openings']:
            if opening['component_id'] == cid and opening.get('endpoints_component'):
                p = np.array(opening['endpoints_component']); ax.plot(p[:, 0], p[:, 1], color='#00897b', linewidth=2.)
        title = 'MAIN' if ci == 0 else 'Relative location unknown'
        ax.set_title('%s | %d views'%(title, len(component['observation_ids'])), fontsize=10)
        ax.set_aspect('equal'); ax.set_xlabel('component x'); ax.set_ylabel('component z'); ax.grid(alpha=.15)
    title = ('PRELIMINARY / UNCALIBRATED\nDashed: tentative pose constraints' if
             prediction.get('inference_mode') == 'exploratory' else prediction['status'])
    fig.suptitle(title+'\n'+prediction['coordinate_spec']['units'], fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, .97))
    output = Path(output)
    for suffix in ('svg', 'png'): fig.savefig(output/('floor_bev.'+suffix), dpi=140, bbox_inches='tight')
    plt.close(fig)


def render_evaluated_floor(prediction, ground_truth, metrics, output):
    """Offline display of the evaluator's single anchor alignment; no refitting."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from portallayout.geometry.floor_geometry import transform_shape, points_transform
    alignment = metrics['alignment']
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    scale = ground_truth.get('metric_scale') or 1.
    conversion = np.diag([scale, scale, 1.])
    def draw(ax, shape, color, alpha=.12):
        for part in ([shape] if shape.geom_type == 'Polygon' else shape.geoms):
            for ring in [part.exterior]+list(part.interiors):
                xy = np.asarray(ring.coords)
                ax.plot(xy[:,0], xy[:,1], color=color, linewidth=1.)
            # Boundaries carry holes faithfully; no blanket fill over holes.
    for value in ground_truth['rooms'].values():
        shape = transform_shape(polygon(value), conversion)
        for ax in (axes[0], axes[2]): draw(ax, shape, '#687585')
    if alignment is not None:
        alignment = np.asarray(alignment)
        for room in prediction['rooms']:
            if room['component_id'] != prediction['main_component_id']: continue
            shape = polygon(room.get('geometry'))
            if shape is not None:
                for ax in (axes[1], axes[2]): draw(ax, transform_shape(shape, alignment), '#007f86')
        for obs in prediction['observations']:
            if obs['component_id'] != prediction['main_component_id']: continue
            pose = alignment @ np.asarray(obs['T_local_to_component']); p = pose[:2,2]
            axes[1].plot(*p, 'o', color='#d75236', markersize=3)
            axes[2].plot(*p, 'o', color='#d75236', markersize=3)
    for ax,title in zip(axes,['Ground truth (offline only)', 'Predicted main component', 'Single-anchor aligned overlay']):
        ax.set_title(title); ax.set_aspect('equal'); ax.grid(alpha=.15)
        ax.set_xlabel('meters' if ground_truth.get('metric_scale') else 'native GT units')
    m = metrics['metrics']; iou = m['overall_iou']; coverage = m['localization_coverage']
    fig.suptitle('{} | {} | IoU={} | correct localization={}\nUncalibrated preliminary experiment; disconnected views remain in evaluation denominator'.format(
        ground_truth['house_id'],prediction['inference_mode'], 'N/A' if iou is None else '{:.3f}'.format(iou),
        'N/A' if coverage is None else '{:.1%}'.format(coverage)),fontsize=11)
    fig.tight_layout(rect=(0,0,1,.89));output=Path(output);output.mkdir(parents=True,exist_ok=True)
    for suffix in ('svg','png'):fig.savefig(output/('gt_comparison.'+suffix),dpi=180)
    plt.close(fig)


def render_preliminary_materials(summaries, output, pipeline_labels=None):
    """Export vector/raster presentation assets from actual metric summaries."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    with plt.rc_context({'font.family':'Noto Sans CJK JP', 'axes.unicode_minus':False, 'svg.fonttype':'path'}):
        fig,ax=plt.subplots(figsize=(15,5));ax.set(xlim=(0,15),ylim=(0,5));ax.axis('off')
        labels=['整层 RGB 全景\n冻结单图前端','局部布局与开口\n真实 1D / 2D 特征','R3 开口匹配\n保留无匹配项','Sim(2) 双解\n预测几何＋独立路径','全局定位\n森林 / robust PGO','区域融合与评价\n几何聚类＋七指标']
        if pipeline_labels is not None: labels = pipeline_labels
        positions=[(.2,2.8),(5.2,2.8),(10.2,2.8),(10.2,.6),(5.2,.6),(.2,.6)]
        for i,((x,y),label) in enumerate(zip(positions,labels)):
            ax.add_patch(FancyBboxPatch((x,y),4.5,1.2,boxstyle='round,pad=.08',facecolor='#e3f2f0' if i<3 else '#fff1d8',edgecolor='#287b7b'))
            ax.text(x+2.25,y+.6,label,ha='center',va='center',fontsize=13)
            if i:
                px,py=positions[i-1]
                if py==y:begin,end=((px+4.6,py+.6),(x-.1,y+.6)) if x>px else ((px-.1,py+.6),(x+4.6,y+.6))
                else:begin,end=(px+2.25,py-.1),(x+2.25,y+1.3)
                ax.add_patch(FancyArrowPatch(begin,end,arrowstyle='-|>',mutation_scale=16,color='#455a64'))
        ax.text(7.5,4.6,'研究点一：整层布局初步实验流程',ha='center',fontsize=18)
        ax.text(7.5,.08,'全预测推理；GT 只用于独立评价。试探性连接未经严格校准，训练采用分阶段方式。',ha='center',fontsize=10)
        for ext in ('svg','png'):fig.savefig(output/('pipeline.'+ext),dpi=180,bbox_inches='tight')
        plt.close(fig)
        names=['整体 IoU ↑','定位覆盖率 ↑','平移 cm ↓','旋转 ° ↓','Wall F1 ↑','Junction F1 ↑','墙距 cm ↓']
        keys=['overall_iou','localization_coverage','translation_error_cm','rotation_error_degrees','wall_f1','junction_f1','wall_distance_cm']
        rows=[]
        for role,variants in summaries.items():
            for name,summary in variants.items():
                rows.append([role+' / '+name]+['N/A' if summary['metrics'][k] is None else '{:.3f}'.format(summary['metrics'][k]) for k in keys])
        fig,ax=plt.subplots(figsize=(17,1.4+.55*len(rows)));ax.axis('off')
        table=ax.table(cellText=rows,colLabels=['开发数据 / 方法']+names,loc='center',cellLoc='center',colWidths=[.25]+[.105]*7)
        table.auto_set_font_size(False);table.set_fontsize(10);table.scale(1,1.8)
        for (r,c),cell in table.get_celld().items():
            if r==0:cell.set_facecolor('#dceeed')
        ax.set_title('初步整层结果（未经严格校准；非独立测试集）',pad=18,fontsize=15)
        fig.text(.5,.01,'S 尺度轨道；每层仅主分量锚点一次规范。有效数量、失败原因与逐层结果见 JSON / CSV。',ha='center',fontsize=9)
        for ext in ('svg','png'):fig.savefig(output/('results_table.'+ext),dpi=180,bbox_inches='tight')
        plt.close(fig)
