"""Build the opening defense from the supplied midterm, without model execution."""
import sys, copy, json, hashlib, math, xml.etree.ElementTree as ET
from pathlib import Path
sys.path.insert(0, '/tmp/bi_layout_ppt_deps')
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.dml import MSO_LINE_DASH_STYLE
from pptx.oxml.xmlchemy import OxmlElement
from PIL import Image

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
TEMPLATE=Path('/home/feixia/Downloads/cross_scene_stitching_midterm_defense_zh.pptx')
prs=Presentation(TEMPLATE)
headers=[copy.deepcopy(s._element) for s in list(prs.slides[1].shapes)[:3]]
coverline=copy.deepcopy(prs.slides[0].shapes[3]._element)
layout=prs.slides[1].slide_layout
for sid in list(prs.slides._sldIdLst):
    prs.part.drop_rel(sid.rId);prs.slides._sldIdLst.remove(sid)
R='A61E24'; BLUE='1A73E8'; GREEN='188038'; GRAY='5F6368'; DARK='202124'
PINK='FCE8E6'; LIGHT='F1F3F4'; PALE='E8F0FE'; GOLD='F9AB00'
FONT='Microsoft YaHei'; specs=[]; sources={}

def record(path):
    p=Path(path);sources[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest();return p

record(TEMPLATE)
for f in ['项目总览与学习手册.md','ZInD-ObservablePairs-v1构建报告.md','experiments/zind_observable_v1/README.md','experiments/zind_observable_v1/results/训练结果摘要.md','experiments/zind_observable_v1/results/best_metrics.json','experiments/ppt_real_outputs_audit/README.md','experiments/zind_e2e_v3/g0_seed123/T5_bev_verifier/preflight/REPORT.md','models/cross_scene_matcher.py','utils/candidate_bev_rasterizer.py','dataset/zind_observable_pairs.py','dataset/zind_pair_mining.py','dataset/zind_bipair_builder.py']:
    record(ROOT/f)
metrics=json.loads((ROOT/'experiments/zind_observable_v1/results/best_metrics.json').read_text())
m=metrics['validation']['predicted_openings']
assert int(m['positive_pair_count'])==1176
assert round(m['end_to_end_candidate_top1']*1176)==710
assert round((1-m['positive_candidate_recall'])*1176)==231
cases=json.loads(record(ROOT/'src/output/matcher_progress_ppt_20260910/case_selection_revised.json').read_text())['cases']

def style(p,size=18,color=DARK,bold=False):
    p.font.name=FONT;p.font.size=Pt(size);p.font.bold=bold;p.font.color.rgb=RGBColor.from_string(color)
    p.space_after=Pt(5)
    for run in p.runs:
        rp=run._r.get_or_add_rPr()
        for tag in ['a:latin','a:ea','a:cs']:
            e=rp.find(tag,{'a':'http://schemas.openxmlformats.org/drawingml/2006/main'})
            if e is None:e=OxmlElement(tag);rp.append(e)
            e.set('typeface',FONT)

def txt(s,x,y,w,h,t,size=18,color=DARK,bold=False,align=None):
    sh=s.shapes.add_textbox(Inches(x),Inches(y),Inches(w),Inches(h));tf=sh.text_frame
    tf.clear();tf.word_wrap=True
    tf.margin_left=tf.margin_right=tf.margin_top=tf.margin_bottom=Pt(0)
    for i,line in enumerate(str(t).split('\n')):
        p=tf.paragraphs[0] if i==0 else tf.add_paragraph();p.text=line;style(p,size,color,bold)
        if align is not None:p.alignment=align
    return sh

def box(s,x,y,w,h,fill='FFFFFF',line=None,dash=False):
    sh=s.shapes.add_shape(MSO_SHAPE.RECTANGLE,Inches(x),Inches(y),Inches(w),Inches(h))
    sh.fill.solid();sh.fill.fore_color.rgb=RGBColor.from_string(fill)
    if line:
        sh.line.color.rgb=RGBColor.from_string(line);sh.line.width=Pt(1.2)
        if dash:sh.line.dash_style=MSO_LINE_DASH_STYLE.DASH
    else:sh.line.fill.background()
    return sh

def line(s,x1,y1,x2,y2,color=GRAY,dash=False,arrow=False,width=1.4):
    sh=s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT,Inches(x1),Inches(y1),Inches(x2),Inches(y2));sh.line.color.rgb=RGBColor.from_string(color);sh.line.width=Pt(width)
    if dash:sh.line.dash_style=MSO_LINE_DASH_STYLE.DASH
    if arrow:
        end=OxmlElement('a:tailEnd');end.set('type','triangle');sh._element.spPr.get_or_add_ln().append(end)
    return sh

def node(s,x,y,w,h,title,sub='',color=BLUE,dash=False):
    box(s,x,y,w,h,PALE if color==BLUE else PINK if color==R else LIGHT,color,dash)
    txt(s,x+.12,y+.15,w-.24,.4,title,19,color,True,PP_ALIGN.CENTER)
    if sub:txt(s,x+.14,y+.68,w-.28,h-.72,sub,16,DARK,False,PP_ALIGN.CENTER)

def band(s,t,y=6.35,size=18):
    box(s,.8,y,11.7,.55,'FFFFFF',R);txt(s,.98,y+.12,11.35,.36,t,size,R,True)

def notes(s,t):
    ns=s.notes_slide
    if ns.notes_text_frame is None:
        ph=OxmlElement('p:sp')
        nv=OxmlElement('p:nvSpPr');cn=OxmlElement('p:cNvPr');cn.set('id','99');cn.set('name','Notes body');nv.append(cn);nv.append(OxmlElement('p:cNvSpPr'))
        np=OxmlElement('p:nvPr');pp=OxmlElement('p:ph');pp.set('type','body');pp.set('idx','1');np.append(pp);nv.append(np);ph.append(nv);ph.append(OxmlElement('p:spPr'))
        tb=OxmlElement('p:txBody');tb.append(OxmlElement('a:bodyPr'));tb.append(OxmlElement('a:lstStyle'));tb.append(OxmlElement('a:p'));ph.append(tb);ns.shapes._spTree.append(ph)
    ns.notes_text_frame.text=t
    specs[-1]['notes']=t

def new(title,sub,section='03 方法设计',source='研究方案；来源及条件详见演讲者备注'):
    s=prs.slides.add_slide(layout)
    for sh in list(s.shapes):s.shapes._spTree.remove(sh._element)
    s.background.fill.solid();s.background.fill.fore_color.rgb=RGBColor(255,255,255)
    for e in headers:s.shapes._spTree.insert_element_before(copy.deepcopy(e),'p:extLst')
    for sh,t in zip(list(s.shapes),[section,title,sub]):
        for p in sh.text_frame.paragraphs:
            for r in list(p._p):
                if r.tag.endswith('}r') or r.tag.endswith('}br'):p._p.remove(r)
        p=sh.text_frame.paragraphs[0];p.text=t
        idx=list(s.shapes).index(sh) if False else 0
    # Preserve actual original title coordinates and set explicit source typography.
    for i,sh in enumerate(list(s.shapes)[:3]):
        for p in sh.text_frame.paragraphs:style(p,[8.5,26,12.5][i],[GRAY,R,GRAY][i],i<2)
    txt(s,.62,7.16,11.65,.22,source,9,GRAY)
    txt(s,12.3,7.12,.38,.25,f'{len(prs.slides):02d}',10,GRAY,align=PP_ALIGN.RIGHT)
    specs.append({'page':len(prs.slides),'title':title,'source':source})
    return s

def bullets(s,items,x=.95,y=1.85,w=11.5,gap=1.30,size=21):
    for i,(head,body) in enumerate(items):
        yy=y+i*gap;txt(s,x,yy,.2,.35,'•',size,R,True);txt(s,x+.32,yy,w-.32,.42,head,size,DARK,True)
        if body:txt(s,x+.32,yy+.53,w-.45,.65,body,size-3,GRAY)

def table(s,heads,rows,widths,x=.8,y=1.7,rh=.7,size=17):
    sh=s.shapes.add_table(len(rows)+1,len(heads),Inches(x),Inches(y),Inches(sum(widths)),Inches(rh*(len(rows)+1)))
    tb=sh.table
    for c,w in zip(tb.columns,widths):c.width=Inches(w)
    for i,row in enumerate([heads]+rows):
        tb.rows[i].height=Inches(rh)
        for j,v in enumerate(row):
            c=tb.cell(i,j);c.text=str(v);c.margin_left=c.margin_right=Inches(.12);c.margin_top=c.margin_bottom=Inches(.08);c.vertical_anchor=MSO_ANCHOR.MIDDLE
            c.fill.solid();c.fill.fore_color.rgb=RGBColor.from_string(LIGHT if i==0 else 'FFFFFF' if i%2 else 'F8F9FA')
            for p in c.text_frame.paragraphs:style(p,size,R if i==0 else DARK,i==0)
    return sh

def pic(s,path,x,y,w,h,crop=None):
    path=record(path)
    with Image.open(path) as im:iw,ih=im.size
    if crop:
        l,t,r,b=crop;ratio=iw*(1-l-r)/(ih*(1-t-b))
    else:ratio=iw/ih
    ww=min(w,h*ratio);hh=ww/ratio
    sh=s.shapes.add_picture(str(path),Inches(x+(w-ww)/2),Inches(y+(h-hh)/2),Inches(ww),Inches(hh))
    if crop:sh.crop_left,sh.crop_top,sh.crop_right,sh.crop_bottom=crop
    return sh

def simple_rooms(s,x,y,scale=1,shift=False):
    # Explicit method schematic, not a model prediction.
    a=[(0,0),(1.9,0),(1.9,1.2),(1.15,1.2),(.85,1.2),(0,1.2),(0,0)]
    b=[(.85,1.2),(.85,2.1),(2.6,2.1),(2.6,1.2),(1.15,1.2)]
    for pts,col in [(a,BLUE),(b,GREEN)]:
        for p,q in zip(pts[:-1],pts[1:]):line(s,x+p[0]*scale,y+p[1]*scale,x+q[0]*scale,y+q[1]*scale,col,width=2)
    line(s,x+.85*scale,y+1.2*scale,x+1.15*scale,y+1.2*scale,R,width=4)

# 01 cover: original title anchor, original red decorative rule.
s=prs.slides.add_slide(layout)
for sh in list(s.shapes):s.shapes._spTree.remove(sh._element)
s.background.fill.solid();s.background.fill.fore_color.rgb=RGBColor(255,255,255)
txt(s,.62,.55,11.8,1.25,'面向稀疏室内全景图的\n开口引导跨房间布局重建方法研究',30,R,True)
s.shapes._spTree.insert_element_before(coverline,'p:extLst')
txt(s,.64,2.48,10,.6,'硕士学位论文开题汇报',23,GRAY)
txt(s,.64,3.55,6,.5,'汇报人：孙润坤',22,DARK)
txt(s,.64,4.22,8,.5,'研究单元：双房间匹配、拒绝与布局对齐',18,GRAY)
txt(s,.66,6.55,11.9,.3,'稀疏全景  ·  开口表示  ·  跨图视觉匹配  ·  候选对齐验证',12,GRAY)
specs.append({'page':1,'title':'封面','source':str(TEMPLATE)})
notes(s,'题目、姓名以用户提供为准。沿用中期汇报原文件的16:9画幅、标题锚点、暗红色、Microsoft YaHei及红色装饰横线，在原文件副本中替换内容。原稿无导师、日期和其他可确认身份，因此不添加。')

# 02
s=new('研究任务：判断连接，并恢复双房间布局','输入不预设相邻；可靠匹配与合理拒绝是同一任务的两种结果','01 研究定位')
for y,t in [(1.85,'全景图 A'),(3.35,'全景图 B')]:node(s,.85,y,2.65,1.12,t,'同楼层；相对位姿未知')
line(s,3.5,2.4,4.25,3.05,arrow=True);line(s,3.5,3.9,4.25,3.05,arrow=True)
node(s,4.25,2.38,3.2,1.4,'开口引导匹配','对应集合 + 匹配置信度',R)
line(s,7.45,3.08,8.25,2.38,arrow=True);line(s,7.45,3.08,8.25,4.28,arrow=True)
node(s,8.25,1.72,4.1,1.45,'接受：生成对齐布局','共享接口与相对变换',GREEN)
node(s,8.25,3.6,4.1,1.45,'拒绝：保留独立布局','非相邻或证据不足，避免强行拼接',GRAY)
band(s,'当前以双房间为主要研究单元，多房间整合作为后续扩展。')
notes(s,'输入是同楼层、相对位姿未知的RGB全景图对。研究输出共享开口对应、置信度和统一坐标布局；拒绝与连接判断同步评价。证据不足时的拒绝是模型弃权，不把几何相连但不可辨识的正例改成负标签。现有Matcher主要输出最佳候选对，数据保留多GT集合；集合输出能力作为后续完善。来源：项目总览1、4、5节；models/cross_scene_matcher.py。')

# 03
s=new('研究背景与意义','以稀疏观测恢复结构化室内空间','01 研究定位','来源：ZInD 官方项目；SALVe（ECCV 2022）；应用为研究目标')
bullets(s,[('探索室内平面图的快速构建','基于全景相机的稀疏观测，减少对连续、密集采集的依赖。'),('为机器人空间感知和导航规划提供结构先验','恢复房间轮廓与连接关系，表达可通行空间及跨房间路径。'),('服务房产展示、室内地图与空间理解','将离散全景组织为具有空间关系的布局表达。')],gap=1.4)
band(s,'“快速构建”是研究目标；当前不报告实时速度，也未部署到机器人。',size=17)
notes(s,'来源：https://github.com/zillow/zind ，官方使用消费级360相机稀疏采集住宅；https://johnwlambert.github.io/salve/ 。机器人导航为潜在应用和结构先验目标，不代表已完成机器人系统或实地部署。不编造时延、FPS或采集效率提升。')

# 04
s=new('关键挑战：从看见开口到可靠对齐','三类误差分别影响候选覆盖、对应判别和布局一致性','01 研究定位','本页均为可编辑方法示意，非实验输出')
for i,(title,body) in enumerate([('开口遗漏','窄开口、遮挡、全景接缝\n真实接口未进入候选集'),('视觉对应歧义','重复门框、相似走廊\n局部外观不足以唯一判断'),('几何对齐误差','端点、方向与尺度偏差\n匹配正确仍可能布局错位')]):
    x=.85+i*4;txt(s,x,1.75,3.6,.45,title,22,R,True);box(s,x,2.45,3.6,2.5,LIGHT)
    if i==0:
        line(s,x+.25,3.2,x+1.6,3.2,BLUE,width=3);line(s,x+1.85,3.2,x+3.3,3.2,BLUE,width=3)
        box(s,x+1.4,3.45,.8,.55,'FFFFFF',R,True);txt(s,x+.4,4.25,2.9,.4,'窄区间：弱响应',16,GRAY)
    elif i==1:
        for dx in [.45,1.45,2.45]:box(s,x+dx,2.9,.65,1.25,'FFFFFF',BLUE)
        txt(s,x+.4,4.4,2.9,.4,'相似候选：哪一扇？',16,GRAY)
    else:
        box(s,x+.4,2.9,1.65,1.25,'FFFFFF',BLUE);box(s,x+1.1,3.3,1.8,1.15,'FFFFFF',GREEN,True)
    txt(s,x,5.2,3.7,.95,body,18,DARK)
notes(s,'示意图仅用于解释问题，无模型或数据含义。最新ObservablePairs内部监控同时存在231个候选遗漏与235个召回后的决策失败，见第19页。几何误差应独立评价，不能用匹配正确率替代拼接成功率。')

# 05
s=new('单图布局研究：为开口提供结构线索','单图布局解决房间边界表达，跨房间对应仍需要视觉证据','02 相关研究','来源：HorizonNet（CVPR 2019）；Bi-Layout（CVPR 2024）；仓库 README')
table(s,['代表方法','主要作用','对本研究的启示'],[
 ['HorizonNet','用一维边界表示学习全景房间布局','提供沿全景水平方向组织的布局线索'],
 ['Bi-Layout','预测闭合布局与外延可见布局\n表达开口附近的标注歧义','双布局差异可提示潜在开口区域'],
 ['本课题拟研究','融合布局线索与视觉特征\n构建面向跨图匹配的开口表示','同时关注候选覆盖与边界定位']],[2,5,4.7],rh=1.02,size=18)
band(s,'Bi-Layout 的双布局及差异提示属于已有方法；本课题研究其跨图匹配用途。',size=17)
notes(s,'HorizonNet: Sun et al., HorizonNet: Learning Room Layout with 1D Representation and Pano Stretch Data Augmentation, CVPR 2019. 原论文位于/home/feixia/Downloads/horizonNet.pdf。Bi-Layout: Tsai et al., No More Ambiguity in 360° Room Layout via Bi-Layout Estimation, CVPR 2024, https://arxiv.org/abs/2404.09993；仓库README已核对其双布局与歧义区域检测属性。外延布局不保证恢复完整相邻房间。代码：models/bi_layout.py、models/modules/horizon_net_feature_extractor.py。')

# 06
s=new('SALVe：候选对齐后的视觉验证','语义接口提出假设；对齐后的真实纹理检验假设','02 相关研究','图源：Lambert et al., SALVe, ECCV 2022, Figure 2；非本项目输出')
pic(s,OUT/'assets/salve_page4.png',.8,1.7,8.15,3.55,(.215,.145,.21,.68))
bullets(s,[('语义候选','门、窗与开口生成对齐假设。'),('BEV 纹理','共同坐标下的地面、天花板鸟瞰纹理。'),('学习式验证','卷积网络判断假设是否可信。')],x=9.1,y=1.85,w=3.3,gap=1.2,size=18)
band(s,'借鉴“假设 → 视觉证据 → 验证”；拟贡献落实到开口表示与上下文交互。',size=17)
notes(s,'SALVe原论文已阅读第3–4节与Figure 2/3：https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136910632.pdf；项目页：https://johnwlambert.github.io/salve/ 。同类语义接口配对产生旋转、平移假设；利用预测深度将两图地面和天花板纹理投影至共同BEV。四张RGB图共12通道输入ResNet验证器，进行匹配/不匹配分类。论文假定相机高度已知，不能无条件迁移其米制设置。本课题不将该视觉验证思想声称为原创；候选条件下的局部和全局上下文融合、有效投影处理是拟研究设计。')

# 07
s=new('研究目标：由可靠候选建立可靠连接','两个研究内容共同服务双房间布局重建','03 方法设计')
node(s,.95,1.95,5.1,2.25,'研究内容一：可靠候选','布局与视觉融合的开口表示\n提高覆盖，改善边界与位置',R,True)
node(s,7.15,1.95,5.1,2.25,'研究内容二：可靠连接','开口上下文视觉匹配\n候选对齐后的视觉验证',BLUE,True)
line(s,6.1,3.1,7.1,3.1,BLUE,arrow=True,width=2)
txt(s,1.2,4.75,4.6,.85,'回答：共享开口是否被可靠提出？',21,R,True)
txt(s,7.35,4.75,4.7,.85,'回答：哪一条连接值得接受？',21,BLUE,True)
band(s,'视觉承担主要对应判断与假设验证；几何提供投影、对齐及空间合理性辅助。',size=17)
notes(s,'两项为拟研究贡献，已有融合开口头和Matcher作为基线，不将集成本身视为已证明创新。最终须用第20页控制变量实验支持。候选质量包含覆盖和定位；连接质量包含正例决策、负例拒绝及置信度。')

# 08
s=new('总体框架：视觉主线，几何辅助','实线框为已有模块；虚线框为拟研究模块；箭头表示数据依赖','03 方法设计')
xs=[.8,3.25,5.7,8.15,10.6];ws=[2.05]*5
for x,t,b,d in zip(xs,['单图编码','开口候选','跨图匹配','视觉验证','布局输出'],['视觉特征\n双布局预测','融合开口头\n预测区间','视觉交互\n评分与拒绝','候选条件\n纹理 / 特征','统一坐标\n共享接口'],[False,False,False,True,False]):
    node(s,x,2.05,2.05,1.7,t,b,BLUE,d)
for i in range(4):line(s,xs[i]+2.05,2.9,xs[i+1],2.9,BLUE,dash=i>=2,arrow=True,width=2)
node(s,1.7,4.7,2.7,1.05,'表示与定位增强','拟研究',R,True)
node(s,5.65,4.7,2.7,1.05,'局部—全局交互','拟研究',R,True)
node(s,9.55,4.7,2.85,1.05,'候选相对对齐','已有解析几何',GRAY)
line(s,3.1,4.7,4.2,3.75,R,True,True);line(s,6.9,4.7,6.7,3.75,R,True,True)
line(s,6.8,3.75,6.8,4.2,GRAY);line(s,6.8,4.2,11,4.2,GRAY);line(s,11,4.2,11,4.7,GRAY,arrow=True)
line(s,9.55,5.2,9.15,5.2,GRAY);line(s,9.15,5.2,9.15,3.75,GRAY,True,True)
band(s,'已有流程可从匹配直接进入几何输出；拟新增视觉验证用于候选重排与拒绝。',size=17)
notes(s,'真实代码先对256-token特征做双向cross-attention，再按候选mask与开口guidance池化cross features；不是先候选池化再做候选级attention。这里总体框架用功能模块表示，细节见第10页。已有模块：Bi_Layout、OpeningSignalHead、OpeningGuidedCrossAttentionMatcher、candidate_local_portal_geometry、joint_layout、cross_scene_pipeline。候选几何对齐可对拟保留的Top-N分别计算，产生条件验证输入。新视觉验证尚未实现；现有几何栅格不能代表纹理验证。')

# 09
s=new('研究内容一：布局与视觉融合的开口表示','已有融合开口头为基线；拟增强面向匹配的候选覆盖与定位','03 方法设计')
node(s,.85,1.9,3.25,1.55,'双布局线索','闭合 / 外延深度\n差异与比例',GRAY)
node(s,.85,3.9,3.25,1.55,'全景视觉特征','开口邻域外观\n边界与上下文',BLUE)
node(s,5.05,2.7,3.2,1.9,'融合开口表示','已有：特征 + 双深度\n拟：边界与置信度增强',R,True)
line(s,4.1,2.7,5.05,3.45,arrow=True);line(s,4.1,4.65,5.05,3.65,BLUE,arrow=True)
line(s,8.25,3.65,9.15,3.65,arrow=True)
bullets(s,[('候选区间','覆盖共享接口，控制冗余。'),('全景接缝','环形区间与跨界合并。'),('窄开口','细粒度边界与局部视觉。')],x=9.2,y=1.85,w=3.1,gap=1.38,size=18)
band(s,'评价：候选召回率 + 开口区间重叠 + 端点定位；固定后续评估方式。',size=17)
notes(s,'代码核对：OpeningSignalHead输入LayerNorm(F)及delta、relative_delta、log_ratio三个几何通道；1×1卷积、环形padding与卷积生成opening_logits和depth_scale。已有环形区间支持接缝，不能称该支持本身是新贡献。拟研究：候选端点细化、局部视觉采样、多尺度上下文与置信度校准。候选Recall-TopK历史诊断为实验性方案，不作为已部署生产结果。候选实验固定匹配器及评价契约，避免把后续模型变化误归因给候选改进。')

# 10
s=new('研究内容二：开口上下文驱动的视觉匹配','Cross-Attention（交叉注意力）连接两图上下文，输出评分与拒绝','03 方法设计')
for x,t,sub in [(1.0,'全景特征与开口权重','水平特征单元 + 开口引导权重'),(4.85,'双向交叉注意力','A 查询 B；B 查询 A'),(8.7,'候选池化与评分','候选区间内聚合交互特征')]:node(s,x,2,3.55,1.6,t,sub,BLUE)
line(s,4.55,2.8,4.85,2.8,BLUE,arrow=True);line(s,8.4,2.8,8.7,2.8,BLUE,arrow=True)
box(s,.95,4.2,7.1,1.52,'FFFFFF',R,True)
txt(s,1.15,4.4,6.7,.4,'拟改进：开口邻域与全景上下文显式交互',21,R,True)
txt(s,1.15,5.0,6.65,.48,'局部细节辨别相似门框，全局语义约束对应关系。',18,GRAY)
node(s,8.7,4.2,3.55,1.52,'匹配 / 拒绝','双向无匹配类别（dustbin）',GREEN)
line(s,10.5,3.6,10.5,4.2,arrow=True)
band(s,'已有：特征单元交互 → 候选池化 → 双向拒绝；拟增强：局部与全局视觉交互。',size=17)
notes(s,'来源：models/cross_scene_matcher.py，第437行起Matcher、_cross_attention与forward。先token attention，后OpeningTokenPooler；opening probability偏置与candidate mask作用于attention。候选分数为双向assignment及开口权重组合，dustbin是可学习无匹配项。pair_match_probability与pair_no_match_probability不是互补且已校准的邻接概率；不能用显示分数的0.5阈值代替正式双向决策。拟研究显式局部窗口、开口外邻域和全景全局token交互，不声称此改进已训练。')

# 11
s=new('拟研究：候选对齐后的视觉验证','比较对齐后的真实纹理或图像特征，学习每个假设的可信度','03 方法设计','拟研究方案；借鉴 SALVe 的视觉验证思想，非已有实验结果')
for x,t,sub in [(1,'候选条件','候选开口 + 预测几何\n生成相对对齐假设'),(4.9,'对齐视觉证据','真实纹理 / 视觉特征\n有效投影与观测掩码'),(8.8,'学习式可信度','局部与跨图上下文\n重排候选并决定拒绝')]:node(s,x,2,3.5,2.05,t,sub,BLUE,True)
line(s,4.5,3,4.9,3,BLUE,True,True);line(s,8.4,3,8.8,3,BLUE,True,True)
bullets(s,[('掩码约束','剔除无效深度、投影越界及未观测区域；记录有效证据覆盖。'),('几何辅助','提供共同坐标、相对变换和空间合理性，不替代视觉判断。')],x=1,y=4.55,w=11.4,gap=.85,size=18)
notes(s,'本页全部为拟研究。第一步由预测候选端点、预测布局及必要尺度建立每个对齐假设；第二步投影地面、天花板真实图像纹理或视觉特征，保存各自有效掩码、共同可观测掩码与覆盖率，避免把无效填充值当纹理，极少重叠时允许证据不足/拒绝；第三步融合开口邻域和跨图上下文学习可信度。训练监督可离线使用GT生成对齐正确性标签，但推理不得使用GT位姿/布局/候选。参考SALVe第4.3节；本课题具体实现与收益均尚未验证。现有7通道栅格只有占据、边界、重叠、接口和validity，不含RGB或视觉特征。')

# 12
s=new('几何辅助与布局输出','相对变换负责空间对齐；视觉证据负责主要对应判别','03 方法设计','流程示意；解析几何已有实现，最终新方案效果待实验验证')
bullets(s,[('从接口恢复相对变换','估计旋转、平移与必要的相对尺度，将 B 变换到 A。'),('检查空间合理性','接口位置、方向及布局关系提供辅助证据。'),('输出统一坐标布局','保留房间轮廓、共享接口、置信度与拒绝原因。')],x=.95,y=1.8,w=6.8,gap=1.34,size=20)
simple_rooms(s,8.5,2.3,1.22);txt(s,8.5,5.3,3.5,.4,'统一坐标的双房间示意',16,GRAY)
band(s,'尺度依据不足时使用归一化评价；不将全景开口位移角写成相机旋转误差。',size=17)
notes(s,'已有几何：utils/candidate_local_portal_geometry.py、utils/joint_layout.py。变换方向B→A，p_A=sR p_B+t。相机归一化高度不等于真实1米；米制平移需要A局部到楼层尺度与楼层米制比例。缺尺度时不伪造米制误差，报告归一化平移、相对尺度、角度与布局误差及有效率。Matcher portal token shift含视点效应，不是相机yaw。本轮ObservablePairs没有完整相对位姿或布局成功率，图为方法示意。')

# 13
s=new('数据基础：ZInD 与 ObservablePairs','保留官方几何与来源，统一组织两类连接关系','04 数据与协议','来源：ZInD 官方项目；ObservablePairs train_auto_v2 构建报告')
txt(s,.95,1.75,11.45,.55,'ZInD：67,448 张全景、1,575 个住宅；提供布局、接口与相机位姿。',20,DARK)
table(s,['关系范围','自动索引 / 对','任务含义'],[
 ['跨完整房间连接','192,571','不同完整房间之间的门 / 开口关系'],
 ['同完整房间内子空间连接','59,079','同一完整房间内不同子空间的关系'],
 ['合计','251,650','含正、负与未知；不是已核验连接数']],[4.1,2.3,5.3],y=2.65,rh=.76,size=18)
band(s,'自动索引已完成；人工核验 0/300，人工确认核心集为空。',size=20)
notes(s,'ZInD统计采用官方README当前公开数字：https://github.com/zillow/zind 。ObservablePairs只对官方train的1260个房屋自动构建，251650是选中panorama pairs而非独立连接数；原始枚举850243种视点组合、106404个房间单元对。两类关系统一记录但分开统计；partial room为标注子空间，不直接当成完整物理房间。自动正41169、负98299、unknown112182；两侧观测代理通过35659，不等于人工确认RGB可辨识。')

# 14
s=new('为什么重新组织数据集','让训练与评价贴近“从图像判断连接”的实际任务','04 数据与协议')
bullets(s,[('几何相连，不等于双侧图像都能辨识','开口可能被遮挡、视角过窄，或缺少唯一视觉证据。'),('固定主视点，不一定适合共享开口','主视点面向整体房间标注，主 / 辅视点组合需面向接口重选。'),('把证据问题与模型问题分开','分别记录数据证据不足、候选遗漏、候选命中后的决策错误。')],gap=1.4)
band(s,'几何正例看不清，仍不能直接改成负例；观测与可辨识状态需独立记录。',size=17)
notes(s,'构建报告第2–6节：primary旨在整体房间标注，不证明对特定共享接口最优。视点重选由几何观测代理执行，家具和门板遮挡没有完整建模；不宣称已完成RGB可辨识筛选。新数据将adjacency、visibility、identifiability三维标签分开；真实像素可见性当前uncertain，可辨识性unreviewed。目标不是证明旧标签普遍错误，而是补充任务需要的证据与诊断。')

# 15
s=new('新旧数据组织方式的区别','依据构建代码比较覆盖范围、目标契约与核验流程','04 数据与协议','来源：三套数据构建代码及 ObservablePairs 构建报告；“新增”不等于已人工完成')
table(s,['比较项','原 BiPair','原 ZInD_matching','ObservablePairs'],[
 ['连接范围','同完整房间的\n不同子空间','不同完整房间','统一记录两类关系\n分别统计'],
 ['视点选择','严格主视点筛选','子空间内主视点\n优先选择','主 / 辅组合枚举重选\n每单元对最多3组'],
 ['关系标签','正负样本与过滤','正负与阈值间隔\n不安全负例不纳入','正 / 负 / 未知\n显式保留冲突'],
 ['对应目标','每对共享开口','每对目标候选索引','多对应集合\n训练适配保留集合'],
 ['核验机制','几何配对与检查','楼层坐标匹配\n及负例筛选','新增图像盲审→真值核验\n工具已备，真人待完成']],[1.55,2.8,3.15,4.2],y=1.65,rh=.82,size=16)
notes(s,'已读dataset/zind_bipair_builder.py、dataset/zind_pair_mining.py、dataset/zind_observable_pairs.py。BiPair只用layout_raw.openings，完整/visible/raw及米制信息要求严格；matching用doors+openings，优先primary，存在正负阈值间隔，不能称“所有未匹配都标负”。matching的target_candidate_a/b是单目标索引；BiPair监督为单个共享portal。ObservablePairs几何一对多冲突保留unknown，确定存在多条GT时保留集合。新表中的人工机制是已实现的待审核工具和协议，尚非已完成数据核验。')

# 16
s=new('数据构建的科学依据与待完成环节','从可复现自动标签，走向独立核验和冻结评价协议','04 数据与协议')
node(s,.9,1.8,3.5,1.95,'自动构建：已完成','官方几何 + 坐标核对\n多条件一致；冲突记为未知',GREEN)
node(s,4.9,1.8,3.5,1.95,'训练试点：待核验','300 对分层样本\n双人图像盲审 → 真值核验',R,True)
node(s,8.9,1.8,3.5,1.95,'正式协议：待冻结','仅训练房屋校准规则\n冻结后构建验证 / 测试集',R,True)
line(s,4.4,2.75,4.9,2.75,arrow=True);line(s,8.4,2.75,8.9,2.75,arrow=True)
bullets(s,[('使用统一评估集验证改进','固定评价对象，再比较旧 / 新权重与数据组织方式。'),('处理相关性与不确定性','按关系类型分层；考虑房屋聚类、重复视点及抽样权重。')],x=1,y=4.45,w=11.4,gap=.95,size=20)
notes(s,'300对试点覆盖261个训练房屋，10层各30对；正120、负120、unknown60；按房屋后样本的分层概率抽样，保存包含概率和权重。第一阶段两名审核者只看RGB，自由选择对应；归档后第二阶段发GT并检查完整集合，分歧保留与裁决。目前0/300完成，不能估计总体可辨识率。388个train全景与官方坐标实现核对，误差近浮点精度；多条件包括端点、方向、宽度、高度、边界与空间关系。阈值仍draft_train_only，无冻结记录，不构建正式val/test。未来统一评估考虑房屋bootstrap、按连接聚合、多视点非独立性，不能将最多3组视点作为独立证据。')

# 17
s=new('已有研究基础：联合训练与双房间输出','已具备可运行的候选、匹配、拒绝和解析几何链路','05 研究基础','图：历史 T3 / 官方 train / house 0665 / floor_03；仅定性；GT 不参与预测与几何')
svg=record(ROOT/'src/output/zind_room_group_debug/t3_best_0665_floor03_candidate_local_m2/pairs/0665_floor_03_pano_34_pano_37_3_pre_manhattan_joint.svg')
sr=ET.fromstring(svg.read_text());sc=4.25/900;ox=1.05;oy=1.5
for poly in sr.findall('{http://www.w3.org/2000/svg}polygon'):
    pts=[tuple(map(float,p.split(','))) for p in poly.attrib['points'].split()]
    # Original SVG coordinates, uniform scale; native editable freeform.
    fb=s.shapes.build_freeform(pts[0][0],pts[0][1],scale=Inches(sc))
    fb.add_line_segments(pts[1:],close=True)
    sh=fb.convert_to_shape(Inches(ox),Inches(oy));sh.fill.background();sh.line.color.rgb=RGBColor.from_string(poly.attrib['stroke'][1:]);sh.line.width=Pt(2)
ln=sr.find('{http://www.w3.org/2000/svg}line')
line(s,ox+float(ln.attrib['x1'])*sc,oy+float(ln.attrib['y1'])*sc,ox+float(ln.attrib['x2'])*sc,oy+float(ln.attrib['y2'])*sc,R,width=3)
txt(s,2.35,3.0,1.8,.4,'房间 01',20,BLUE,True)
txt(s,4.8,4.2,1.8,.4,'共享开口',18,R,True)
line(s,4.75,4.65,3.08,4.66,R,width=1)
txt(s,4.8,5.0,1.8,.4,'房间 04',18,GREEN,True)
line(s,4.7,5.25,3.12,5.13,GREEN,width=1)
txt(s,.95,5.72,5.75,.48,'pano 34 ← 37；预测候选 A1–B0；开口几何对齐',15,GRAY)
bullets(s,[('联合训练已跑通','布局、开口头与匹配器已完成联合训练。'),('预测驱动输出已有','预测开口匹配、拒绝、双房间对齐及可视化。'),('视觉验证器尚未完成','历史 T5 为几何预检，不是纹理视觉验证。')],x=7.3,y=1.85,w=5.1,gap=1.42,size=20)
band(s,'图为历史模型定性案例，不代表最新 ObservablePairs 的拼接成功率。',size=17)
notes(s,'原图与口径：experiments/ppt_real_outputs_audit/README.md。为解决原图文字重叠，本页从对应pre_manhattan_joint.svg逐点等比例转为可编辑轮廓，保留原始几何和共享接口，重新放置中文标签。T3 checkpoint checkpoints/End_To_End/zind_e2e_v2/g0_seed123/T3/best.pt，completed epoch8，SHA256 ca083dbbc5a05d738dad36404ad708c623c272aef7a64398b4de3a793278cc5d。0665 floor03 complete_room01←04，pano34←37，预测候选(1,0)，Rank1，candidate score 0.8816207647；不是校准概率。candidate-local M2使用模型选中的预测开口几何生成B→A，GT不参与。此图是train-only qualitative/debug，文字为原始输出，本页旁注作清晰补充。没有采用GT oracle图。历史T5：已有CPU logistic/MLP及7-channel geometry raster，CNN未训练，也没有真实纹理视觉验证器。')

# 18
s=new('最新匹配结果：ObservablePairs 内部监控','最佳第15轮（epoch=14）；预测开口与预测引导权重，无真值（GT）候选注入','05 研究基础','来源：experiments/zind_observable_v1/results/best_metrics.json → validation.predicted_openings')
table(s,['指标','结果','统计口径'],[
 ['正样本总体匹配正确率',f"{m['end_to_end_candidate_top1']*100:.2f}%",'710 / 1,176'],
 ['正样本候选召回率',f"{m['positive_candidate_recall']*100:.2f}%",'945 / 1,176'],
 ['候选命中后的条件 Top-1',f"{m['conditional_candidate_top1']*100:.2f}%",'710 / 945'],
 ['负样本拒绝率',f"{m['negative_rejection_rate']*100:.2f}%",'1,094 / 1,176']],[5.2,2.1,4.4],y=1.65,rh=.67,size=19)
txt(s,.95,5.27,11.4,.7,'训练 27,354 对；内部监控 2,352 对，正负各 1,176。\n监控来自官方 train，与本轮训练按房屋互斥；预训练可能见过这些房屋。',17,GRAY)
band(s,'人工核验未完成，正式 val/test 未构建或使用；仅为匹配结果，不含新视觉验证器。',size=16)
notes(s,'模型：checkpoints/End_To_End/zind_observable_v1/best1_balanced_seed123_20260910/best.pt。最佳epoch14（从0计数），实际21轮，早停；本页只用validation.predicted_openings而非teacher_forced_manifest_candidates。guidance_mode=predicted，candidate_source=predicted，GT注入计数0。最新远程研究分支与本地均为8484b620d1850a27776445178c7f0f2e753213ac。内部监控100个房屋来自官方train，房屋与本轮训练互斥但预训练污染不能排除。不是正式测试，也不是位姿或拼接成功率。历史ZInD_matching曾使用官方val，与本轮ObservablePairs正式val/test未构建是不同协议，不矛盾。不直接比较新旧分数并归因于数据或模型改进。')

# 19
s=new('误差分析：候选遗漏与决策失败同时存在','1,176 个正样本的互斥分解，指向两个研究内容','05 研究基础','统计：ObservablePairs 内部监控；右侧为历史 T3 / 官方 val 的事后选择定性案例')
total=1176;x=.95;y=2.1;W=7.1
for n,c in [(710,GREEN),(231,GOLD),(235,R)]:
    ww=W*n/total;box(s,x,y,ww,.72,c);txt(s,x+.05,y+.18,ww-.1,.4,str(n),22,'FFFFFF',True,PP_ALIGN.CENTER);x+=ww
for yy,n,col,t in [(3.25,710,GREEN,'匹配成功'),(4.2,231,GOLD,'正确候选未召回 → 候选覆盖'),(5.15,235,R,'召回后决策失败 → 匹配与验证')]:
    box(s,1,yy,.16,.32,col);txt(s,1.35,yy-.04,6.55,.53,f'{t}：{n/total*100:.2f}%',20,DARK)
case=cases[2]
pic(s,case['image_paths'][0],8.65,2.12,3.45,1.73)
pic(s,case['image_paths'][1],8.65,3.95,3.45,1.73)
txt(s,8.65,1.65,3.8,.36,'真实决策失败案例：0769',16,R,True)
txt(s,8.65,5.85,3.85,.6,'预测 A0–B1；GT A1–B1\nGT 已召回，但未被选中',15,GRAY)
notes(s,'本页分解严格互斥：710+231+235=1176。945个候选命中，710/945=75.13%；80.36%×75.13%≈60.37%。235包含接受错误候选或拒绝正确候选，不能都写成候选排序错误。因标签尚未人工核验，自动判定失败中仍可能包含证据不足，不据此断言视觉信息论上限。右侧独立历史案例：0769_floor_02_pano_46_pano_71_6，ZInD_matching官方val，冻结T3，现有case_selection_revised.json；GT A1-B1排名3，最终错误A0-B1，score=0.5535287857；GT只用于事后诊断，未参与模型决定。原图不含GT叠画，为真实输入；案例是事后选择，不属于左边235例，不能据此作因果证明。')

# 20
s=new('实验设计：用控制变量验证各模块贡献','候选实验固定后续评价；匹配实验固定候选集合','06 实验与计划')
table(s,['验证问题','对照设计','主要指标'],[
 ['融合能否改善\n候选质量？','仅布局 / 仅视觉 / 融合 / 融合增强\n固定后续匹配器与评价方式','候选召回、区间重叠\n端点定位、候选数量'],
 ['视觉验证能否\n改善连接决策？','同一候选与同一对齐假设\n基础匹配 / +纹理或特征验证','条件Top-1、总体匹配\n负样本拒绝、置信度'],
 ['几何辅助带来\n多少额外收益？','同一视觉主干与投影条件\n无几何评分 / +几何辅助评分','匹配与拒绝；旋转、平移\n尺度、布局误差及有效率']],[3.1,5.2,3.4],y=1.75,rh=1.12,size=17)
band(s,'按房屋隔离、按关系类型分层；冻结规则后统一测试，报告置信区间与失败案例。',size=16)
notes(s,'候选实验固定后续Matcher/协议，若增强候选需要重训则与固定权重实验分别报告。匹配实验固定预测候选、Top-N及几何对齐假设，使视觉验证增益不被候选覆盖变化混淆。几何辅助消融移除额外评分信息而保留必要投影几何，否则无法形成视觉验证输入；另做纯几何基线以检验视觉主导结论。局部、全局、交互、validity mask可分别消融。报告总体正例正确率、候选命中条件Top1、负例拒绝、set correspondence precision/recall、coverage/abstention、位姿/布局误差及有效分母；米制仅在有尺度依据时。训练房屋校准并冻结阈值，val负责预设开发，test最终一次评价；多随机种子与按房屋bootstrap，控制同连接的多视点相关性。SALVe迁移比较须在相同候选与评价集下重建协议，不跨论文分数宣称领先。')

# 21
s=new('拟创新点','以下为拟研究贡献，需统一对照与消融实验支持','06 实验与计划')
box(s,.95,1.85,11.35,1.65,'FFFFFF',R,True)
txt(s,1.18,2.07,10.9,.5,'1. 面向跨房间匹配的布局与视觉融合开口表示',23,R,True)
txt(s,1.18,2.8,10.85,.48,'融合双布局线索、开口邻域视觉与边界信息，改善候选覆盖及定位质量。',19,DARK)
box(s,.95,4,11.35,1.75,'FFFFFF',BLUE,True)
txt(s,1.18,4.22,10.9,.5,'2. 开口上下文驱动的视觉匹配与对齐验证',23,BLUE,True)
txt(s,1.18,4.95,10.85,.62,'联合局部与全局上下文、候选条件下的视觉证据，以几何辅助重排与拒绝。',19,DARK)
band(s,'不将 Bi-Layout 双布局或 SALVe 已有的 BEV 纹理验证写成本课题原创。',size=17)
notes(s,'拟创新1必须超越已有OpeningSignalHead的简单特征+深度拼接，落实到面向匹配的候选表示与定位设计，并以独立候选指标及匹配指标支持。拟创新2以真实图像纹理/视觉特征为主要判断依据，几何只负责投影、对齐和空间辅助；不能把已有几何逻辑回归或占据栅格换名为视觉创新。两项均为开题阶段拟贡献，是否成立须通过第20页实验，避免原创性或收益的过度声明。')

# 22
s=new('研究计划与预期成果','按可验收阶段推进，不预设未经确认的完成日期','06 实验与计划')
table(s,['阶段','主要工作','预期产物'],[
 ['一：数据与协议','完成300对核验，校准并冻结规则','核验记录、正式划分与评价协议'],
 ['二：方法实现','开口表示增强；上下文与视觉验证','可复现双房间方法原型'],
 ['三：统一实验','控制变量对照、消融及误差分析','同口径结果表、案例与复现材料'],
 ['四：论文与原型','整理方法论证、实验结论及演示','硕士论文与可编辑图表、演示原型']],[2.25,5.4,4.05],y=1.85,rh=.88,size=18)
band(s,'多房间整合视双房间方法与实验进度开展，作为后续扩展。',size=19)
notes(s,'计划不继承中期原稿未经确认的9–11月时间判断。先补独立人工审核与数据协议，再实现视觉验证，统一实验后形成论文与原型。预期成果是研究交付类型，不承诺未获证据的论文接收、专利、速度或指标。多房间需进一步解决边一致性与全局累积误差，不作为已完成研究结果。')

# 23
s=new('恳请老师批评指正','从可辨识的共享开口出发，建立可靠的跨房间视觉连接与布局对齐。','07 结束')
txt(s,1.05,2.2,11.1,1.3,'可靠候选  →  可靠连接  →  布局对齐',30,R,True,PP_ALIGN.CENTER)
txt(s,1.35,4.05,10.5,.8,'视觉判断与验证为主，几何对齐与合理性为辅',23,DARK,False,PP_ALIGN.CENTER)
txt(s,1.35,5.3,10.5,.55,'汇报人：孙润坤',20,GRAY,False,PP_ALIGN.CENTER)
notes(s,'结束页为第23页正文。研究主线：稀疏全景开口候选、跨图视觉对应、候选对齐与视觉验证、双房间布局输出。数据人工核验和新增视觉验证模块尚待完成。后续第24页为参考文献备份页。')

# 24 references
s=new('参考文献与项目证据','备份页｜文献归属、实验口径与详细路径见各页演讲者备注','备份 参考文献','本地与远程研究分支核验：8484b620d1850a27776445178c7f0f2e753213ac')
refs=[
('HorizonNet','Sun et al. HorizonNet: Learning Room Layout with 1D Representation\nand Pano Stretch Data Augmentation. CVPR, 2019.','https://github.com/sunset1995/HorizonNet'),
('Bi-Layout','Tsai et al. No More Ambiguity in 360° Room Layout\nvia Bi-Layout Estimation. CVPR, 2024.','https://arxiv.org/abs/2404.09993'),
('SALVe','Lambert et al. SALVe: Semantic Alignment Verification for Floorplan\nReconstruction from Sparse Panoramas. ECCV, 2022.','https://johnwlambert.github.io/salve/'),
('ZInD','Cruz et al. Zillow Indoor Dataset: Annotated Floor Plans\nwith 360° Panoramas and 3D Room Layouts. CVPR, 2021.','https://github.com/zillow/zind'),
('本项目','sunrunkun0422-dev/bi-layout\n研究分支：agent/zind-bipair-opening-matcher','https://github.com/sunrunkun0422-dev/bi-layout/tree/agent/zind-bipair-opening-matcher')]
for i,(label,citation,url) in enumerate(refs):
    y=1.68+i*.96;txt(s,.85,y,1.75,.4,label,18,R,True);sh=txt(s,2.7,y,9.8,.82,citation,17,DARK)
    sh.text_frame.paragraphs[0].runs[0].hyperlink.address=url
notes(s,'项目证据优先级：最新best_metrics.json与训练摘要 > 数据构建及审计报告 > 代码 > 历史汇报。当前提交8484b620，本地/远程一致。历史T5预检的HEAD 50a6ae5，不替代最新ObservablePairs匹配结果。SALVe Figure2来自原论文PDF第4页。完整引用链接已写入本页对象超链接及各页备注。生成时未训练、未加载模型、未覆盖实验。')

prs.core_properties.title='面向稀疏室内全景图的开口引导跨房间布局重建方法研究'
prs.core_properties.subject='硕士开题汇报；23页正文及1页参考文献'
prs.core_properties.author='孙润坤'
prs.core_properties.keywords='稀疏全景,开口匹配,视觉验证,布局重建'
assert len(prs.slides)==24
dest=OUT/'孙润坤_硕士开题汇报.pptx';prs.save(dest)
(OUT/'slide_sources_and_notes.json').write_text(json.dumps(specs,ensure_ascii=False,indent=2))
(OUT/'source_hashes.json').write_text(json.dumps(sources,ensure_ascii=False,indent=2))
(OUT/'演讲者备注.md').write_text('\n\n'.join(f"## 第{x['page']}页 {x['title']}\n\n{x['notes']}" for x in specs))
print(dest)
