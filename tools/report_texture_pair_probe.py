"""Summarize all scored hypotheses of the preselected texture pair probes."""
import argparse
import html
import json
import re
from datetime import datetime, timezone, timedelta
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    args = p.parse_args(); root = args.run.resolve()
    cases = json.loads((root/'all_pairs.json').read_text())
    results = []
    for building, ids in [('0180', [(59, 60)]), ('0988', [(9, 11), (9, 10)]),
                          ('0726', [(7, 27)]), ('0561', [(33, 34)]),
                          ('0060', [(26, 29)]), ('0430', [(5, 6)])]:
        probe = root/('probe_'+building)
        audit = json.loads((probe/'verifier/audit.json').read_text())
        assert not audit['incomplete_groups']
        for pair in ids:
            def pano(path):
                return int(re.search(r'_pano_(\d+)\.jpg$',path).group(1))
            rows = [r for r in audit['rows'] if tuple(sorted(pano(f) for f in r['files'][:2])) == pair]
            case = next(c for c in cases if c['house_id']==building and tuple(sorted(pano(f) for f in c['images']))==pair)
            true_rows = [r for r in rows if r['label_for_evaluation_only']==1]
            false_rows = [r for r in rows if r['label_for_evaluation_only']==0]
            assert rows and true_rows, 'Cannot assess rejection without a positive candidate'
            assert len(rows)*4 == len({f for r in rows for f in r['files']})
            results.append(dict(building=building, panos=pair, pair_id=case['pair_id'],
                hypothesis_count=len(rows), correct_hypothesis_count=len(true_rows),
                salve_max_correct_score=max(r['positive_probability'] for r in true_rows),
                salve_correct_accepted=sum(r['positive_probability']>=.93 for r in true_rows),
                salve_wrong_accepted=sum(r['positive_probability']>=.93 for r in false_rows),
                r3_weak_positive_scores=[m['confidence'] for m in case['matches'] if m['target']==1],
                r3_unknown_matches=sum(m['target']==-1 for m in case['matches']),
                r3_known_wrong_matches=case['known_wrong_matches'],
                visual_texture_assessment={
                    '0180':'different floor materials; exclude as typical similar-floor example',
                    '0988':'pale ceilings and wood floors; no semantic masks',
                    '0726':'pale ceilings and warm-colored flooring; visible texture/lighting differences',
                    '0561':'pale textured ceilings and warm wood-toned floors; dark/light finish differences',
                    '0060':'pale ceilings; mixed wood-toned flooring and carpet, not a clean same-material example',
                    '0430':'pale ceilings and light square-tile floors; wood floor visible in adjacent area'
                }[building],
                human_correspondence_verified=False,
                salve_rows=rows))
    report=dict(baseline='SALVe official verifier and MHNet candidates + DAV2 metric depth adaptation',
        threshold=.93, threshold_tuned=False, selection='first three texture-ranked R3 weak-positive pairs, then next highest per new house (0726 and 0561); after 0726 success continued to 0060 and 0430. Each pair selected before its own SALVe scores; outcome-driven exploratory search, not an unbiased cohort',
        independent_test=False, pose_success_of_r3_verified=False,
        texture_failure_causality_verified=False, cases=results)
    (root/'paired_probe.json').write_text(json.dumps(report,indent=2)+'\n')
    lines=['| 房屋/全景对 | SALVe适配版正确候选最高分 | 接受正确/错误候选 | R3弱正对应分数 |',
           '|---|---:|---:|---|']
    for r in results:
        lines.append('| '+r['building']+' / '+'↔'.join(map(str,r['panos']))+
            f" | {r['salve_max_correct_score']:.4f} | {r['salve_correct_accepted']}/{r['salve_wrong_accepted']} | "+
            ', '.join(f'{s:.4f}' for s in r['r3_weak_positive_scores'])+' |')
    now=datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')
    notes='''
已完成 3,394 个开发集跨房间对的外观初筛，导出 28 个可视候选，先测试前三个 R3 弱正命中案例，再增加不同房屋 0726、0561 各一对。0726 正确对齐被接受，保留为反例；为继续寻找用户要求的案例，按剩余房屋相似度继续测试 0060、0430。0430 也被正确接受。新增展示候选为 0561 与 0060：后者可见木色地面与地毯混合，不能作为严格同材质纹理案例。每对均在自身 SALVe 评分前固定，但继续搜索由先前结果推动，属于事后探索，不能用命中比例估计总体优势。全部结果保留。0988 两对共享 pano 9，属于相关案例，不能算两次独立实验。

注意：SALVe 分数是对齐验证器概率，R3 是开口对应分数，两者不能直接比大小。表格比较的是各自原有规则下的行为；R3 未重新训练或调阈值。0180 的地板材质不同，不能作为纹理相似的典型。0988 两对各有一个额外 target=-1 的未知匹配，不能称整对完全正确。

结论边界：这不是原版 SALVe 失败、完整 Bi_Layout 正确重建的证明。原版缺少 HoHoNet 权重；当前替代深度可能影响 BEV 和分数。R3 正确性目前来自自动几何弱标签，未人工确认，也未验证由该对应得到的位姿。案例按 R3 命中选择，不能估计总体优势。没有通过消融证明纹理相似导致失败。

后续要形成论文证据：核实同一物理开口；使用双方相同的对应/位姿成功定义；在按纹理选定而不看模型输出的独立房屋子集上评估全部成功和失败；增加保留几何、改变或去掉纹理的对照实验。原版权重恢复前只能标注 SALVe-DAV2 adapted baseline。

文件：`index.html` 初筛页，`comparison.html` 全部已测案例比较，`all_pairs.json` 全量排序，`paired_probe.json` 全部新推理分数，`probe_*/` 原始候选、深度、BEV及验证器结果。`all_hypotheses` 是整屋导出；`hypotheses` 只按全景ID过滤，保留该对所有预测候选，排除 gt_alignment_exact。没有使用 GT 深度或 GT 对齐代替预测输入。

重建初筛与报告：

```bash
/home/feixia/anaconda3/envs/bi_layout/bin/python tools/mine_texture_ambiguity.py --output experiments/texture_ambiguity_20261029
python3 tools/report_texture_pair_probe.py --run experiments/texture_ambiguity_20261029
```
'''
    path=root/'README.md'
    history=path.read_text().split('\n\n',1)[0]+'\n' if path.exists() else ''
    path.write_text(history+'修改记录：'+now+'（Asia/Shanghai）—增加四对不同房屋实测，保留0726与0430反例，标注0060混合材质及搜索选择记录。\n\n# 纹理相似房间对探索实验\n\n'+'\n'.join(lines)+'\n'+notes)
    blocks=[]
    for r in results:
        outcome=('SALVe适配版接受了正确候选（保留反例）' if r['salve_correct_accepted'] else 'SALVe适配版拒绝了全部正确候选')
        blocks.append('<h2 id="case-'+r['pair_id']+'">'+('新增：' if r['building'] not in ('0180','0988') else '')+r['building']+' / '+' ↔ '.join(map(str,r['panos']))+'</h2><p>'+outcome+'；R3命中弱正对应。分数不能直接跨模型比较。</p><img width="100%" src="images/'+r['pair_id']+'.jpg"><pre>'+html.escape(json.dumps({k:v for k,v in r.items() if k!='salve_rows'},ensure_ascii=False,indent=2))+'</pre>')
    (root/'comparison.html').write_text('<!doctype html><meta charset="utf-8"><title>同对探索实验</title><style>body{max-width:1100px;margin:30px auto;font:16px sans-serif}pre{white-space:pre-wrap}</style><h1>同对探索实验：SALVe-DAV2 / R3</h1><p>新增：<a href="#case-42bfe7af587e93bf84318d15">0561 / 33 ↔ 34</a> · <a href="#case-2669006eeb1ea88dec5fe636">0430 / 5 ↔ 6（反例）</a> · <a href="#case-8603d3ce7e6003c6ae917db2">0060 / 26 ↔ 29</a> · <a href="#case-01c4f6129e63e2b5398f5c70">0726 / 7 ↔ 27（反例）</a></p><p>阈值固定 0.93；R3命中自动弱标签，不等于人工确认或正确位姿。开发案例，按R3命中预选。0988两对不独立。0180地板不相似；0060含混合材质。新增案例是在寻找指定结果的探索中筛选，不能据此估计总体优势。纹理因果关系未验证。</p><a href="index.html">全部初筛案例（含失败）</a>'+''.join(blocks))
    summary=json.loads((root/'summary.json').read_text())
    summary['paired_comparison_count']=len(results)
    summary['paired_comparison_report']='paired_probe.json'
    summary['conclusion']=f'{len(results)} adapted-baseline probes completed; original SALVe superiority and texture causality unverified'
    (root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    # Keep the screening page explicit about its frozen ranking, with a current result link.
    index=root/'index.html'; content=index.read_text()
    content=content.replace('查看已补跑的三对比较', '查看全部已补跑比较').replace('查看已补跑的五对比较', '查看全部已补跑比较')
    marker='<!-- paired-probe-link -->'
    if marker not in content:
        content=content.replace('<h1>',marker+'<p><a href="comparison.html">查看已补跑的三对比较</a>：下面初筛元数据保留评分前状态。</p><h1>',1)
    index.write_text(content)
    print('\n'.join(lines))


if __name__=='__main__':
    main()
