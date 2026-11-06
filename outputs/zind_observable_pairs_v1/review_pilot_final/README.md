# 双阶段人工核验材料

先向两位独立审核者分别提供 `stage1_package.zip`，仅含匿名 RGB 与自由框选界面。第一阶段导出后交由协调者归档，之后才提供 `stage2_package.zip`。第二阶段必须导入本人第一阶段 JSON，记录其文件 SHA-256，不修改第一次选择。

不要提前共享 stage2 包或协调者文件；本地静态材料依靠分包发放隔离，不是访问控制服务器。解压后可打开 index.html；若浏览器限制本地 WebCrypto/图片，分别在当前阶段目录使用本地 HTTP 服务。

审核者需填写独立 ID 与依据，可选择无对应或无法确定，也可在任意 RGB 区域框选多组对应。跨接缝区域勾选对应选项，取拖框外侧区间。没有自动填入或伪造的人审记录。普通负样本与困难代理负样本均需复核；几何 positive/negative 仍是 auto_proposed。

实际人审文件回收后使用 `tools/import_zind_observable_reviews.py` 生成独立 reviewed 快照及严格视图。两位审核者分歧应记录第三方裁决，不能默认为一致。
