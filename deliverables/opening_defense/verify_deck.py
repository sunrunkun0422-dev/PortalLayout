import sys,json,re,subprocess,hashlib
from pathlib import Path
sys.path.insert(0,'/tmp/bi_layout_ppt_deps')
from pptx import Presentation
from PIL import Image,ImageDraw
OUT=Path(__file__).resolve().parent
p=Presentation(OUT/'孙润坤_硕士开题汇报.pptx')
raw=subprocess.check_output(['pdftotext','-raw',str(OUT/'孙润坤_硕士开题汇报.pdf'),'-']).decode().split('\f')
if not raw[-1].strip():raw.pop()
norm=lambda x:re.sub(r'\s+','',x)
missing=[];bounds=[];font_issues=[]
for i,s in enumerate(p.slides):
    assert s.notes_slide.notes_text_frame.text
    for sh in s.shapes:
        if sh.left<0 or sh.top<0 or sh.left+sh.width>p.slide_width+1000 or sh.top+sh.height>p.slide_height+1000:bounds.append((i+1,sh.name))
        tfs=[]
        if sh.has_text_frame:tfs.append(sh.text_frame)
        if sh.has_table:
            tfs.extend(c.text_frame for row in sh.table.rows for c in row.cells)
        for tf in tfs:
            for pp in tf.paragraphs:
                if pp.text and norm(pp.text) not in norm(raw[i]):missing.append({'page':i+1,'text':pp.text})
                for r in pp.runs:
                    if r.font.name!='Microsoft YaHei':font_issues.append((i+1,r.text,r.font.name))
hashes=json.loads((OUT/'source_hashes.json').read_text())
unchanged=all(hashlib.sha256(Path(f).read_bytes()).hexdigest()==v for f,v in hashes.items())
result={'slides':len(p.slides),'pdf_pages':len(raw),'body_pages':23,'reference_pages':1,'missing_rendered_text':missing,'outside_slide':bounds,'font_setting_issues':font_issues,'sources_unchanged':unchanged,'font_preview_limit':'PPTX使用Microsoft YaHei；本机无该字体，PDF由Noto Sans CJK替代渲染。','checked_commit':'8484b620d1850a27776445178c7f0f2e753213ac'}
(OUT/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
for chunk in range(4):
    im=Image.new('RGB',(1600,3*475),'#ddd');d=ImageDraw.Draw(im)
    for j in range(6):
        i=chunk*6+j;path=OUT/'preview'/f'page-{i+1:02d}.png'
        if path.exists():
            slide=Image.open(path).resize((800,450));im.paste(slide,((j%2)*800,(j//2)*475));d.text(((j%2)*800+10,(j//2)*475+452),f'Page {i+1}',fill='black')
    im.save(OUT/'preview'/f'contact-{chunk+1}.png')
print(json.dumps(result,ensure_ascii=False,indent=2))
