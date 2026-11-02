# C1：数字生命模拟

> 仓库名 pet：A pet who has life.

一个没有被告知任何情绪的数字生命。它靠吃掉设备上的杂乱维生，体内 8 种无名化学物质（m0..m7）按基因组给的调控网络演化，状态由观察者事后聚类、命名。

- 给 Claude Code 的项目说明：`CLAUDE.md`
- 转接说明与优化清单：`HANDOFF.md`
- 实验记录：`docs/FINDINGS.md`
- 树莓派路线图与硬件清单：`docs/ROADMAP.md`
- 上一次的生命报告：`docs/c1-life-report.html`（浏览器直接打开）

## 运行

需要 Python 3.10+，依赖见 `requirements.txt`。

```bash
pip install -r requirements.txt
python -m pytest -q          # 回归测试
python run.py 30 7           # C1 活 30 天 + 对照组
python observer.py           # 状态分析
python export.py && python build.py   # 生成 report.html
python evolve.py evolved.dna # 演化新个体（不会覆盖 c1.dna）
```

`c1.dna` 是 C1 本人的基因组，`c1_0.dna` 是始祖。
