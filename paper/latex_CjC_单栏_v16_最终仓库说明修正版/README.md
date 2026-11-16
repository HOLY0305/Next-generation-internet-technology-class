# MoE 动态放置课程报告 v16｜最终仓库说明修正版

本项目以同级目录上一级的 `MoE动态放置_个人课程报告_最终提交稿_v16.md` 为同步正文，基于 v15 AI 使用与仓库说明修正版，保留五章正文、中文摘要、图 1、图 2、表 1、九条参考文献和 AI 使用及其代码仓库说明；当前版不排版英文标题、英文摘要和英文关键词。

本版在 v15 基础上删除第 4.2 节指定句子，并更新 AI 使用说明正文；图注留白、说明文字字号、正文文献上标和两张 SVG 均沿用 v14。

## Overleaf 使用

1. 选择 **New Project → Upload Project**，上传本目录中的 ZIP。
2. 在 **Menu → Compiler** 选择 **XeLaTeX**。
3. 主文件使用根目录的 `main.tex`，点击 Recompile。
4. 先编辑 `config/metadata.tex`，再按需要替换 `main.tex` 的中文摘要和关键词，以及 `sections/` 下的章节内容。

ZIP 使用 Overleaf 项目根目录结构：`main.tex`、`CjC.cls`、依赖样式文件和 `references.bib` 位于根目录，其他源码位于对应子目录。`CjC.cls` 与主文件同级，因此不需要额外设置 TeX 搜索路径。

## Windows 本地编译

在 PowerShell 中进入解压后的项目目录并运行：

    .\build\build.ps1

需要 XeLaTeX、CTeX、AMS 等宏包。本项目参考文献采用 `backmatter/references.tex` 中的原生 `thebibliography`，因此构建脚本不会运行 BibTeX；`references.bib` 仅作为模板资源保留。入口文件为 `main.tex`，输出为 `main.pdf`。

## 内容修改位置

- 作者、学号、学院、专业、课程、教师、日期和页眉：`config/metadata.tex`。当前英文姓名仍为 `[To be filled]`，其余本轮提供的中文个人信息已写入。
- 页面、字体、摘要间距、公式和图表样式：`config/style.tex`、`config/layout.tex`。
- 中文摘要、关键词和章节顺序：`main.tex`。
- 五个示例章节：`sections/`。可按课程要求增删文件，并同步调整 `main.tex` 中的 `\input{...}`。
- 代码与数据、生成式人工智能说明：`backmatter/declarations.tex`。请据实填写；不适用时可删除相应小节。
- 参考文献：`backmatter/references.tex`，保留九条已核验文献和数字编号；`references.bib` 未启用。
- 图片：`figures/figure1.svg`、`figures/figure2.svg` 是 v12 已验证的 960×720（4:3）裁边 SVG 副本；`figures/original_uncropped/` 保留裁边前的原始未裁边备份；`figure1.pdf`、`figure2.pdf` 是由裁边 SVG 转换得到的矢量 PDF，供 XeLaTeX 插入。

论文使用等比例矢量 PDF，在 A4 纵向单栏中按版心宽度浮动排版。v15 不改变 SVG 内部图元；图注留白、文献上标和页面排版沿用 v14。

## 提交前检查

如需补充英文姓名，请编辑 `config/metadata.tex` 后运行 `.\build\build.ps1`。当前版不排版英文标题、摘要和关键词。第 4.3 节的合成 Trace 结果是相对成本代理，不是真实 GPU Step Time、网络测量、训练吞吐或收敛实验。

## 默认排版

A4 纵向、正文单栏；图 1 和图 2 均为纵向页面中的等比例矢量图；正文约 11 pt、约 1.25 倍行距；摘要与关键词使用窄于正文的版心；公式连续编号；图表自动编号；本 v15 实际使用并嵌入 TeX Gyre Termes，中文字体由 CTeX 按系统选择。页面尺寸、页眉高度和间距可在 `config/layout.tex` 调整。

## 模板来源与许可

模板来源、保留的上游文件及许可状态见 `TEMPLATE_NOTICE.md`。此工程适合本地课程论文复用；若要将包含 `CjC.cls` 及其依赖的模板再次公开发布，请先核对各文件的授权条件。

