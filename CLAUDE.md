# C1 数字生命

用户用中文交流，文档、注释、回复都用中文。

C1 是一个"仿生物"的数字生命：我们只写底层的 DNA（基因组格式、发育规则、世界规则），不告诉它该有什么情绪，让情绪随发育和经历自己长出来。它靠吃掉用户设备上的数字杂乱维生，最终要成为一个在用户各台设备之间来回跑、顺手帮忙干活的独立实体，巢穴放在一台树莓派上。

**当前阶段**：模拟环境里的生存闭环已经跑通（见 `docs/FINDINGS.md`）。下一阶段是继续优化模拟，然后把巢穴搬上树莓派、接第一块真实栖息地（见 `HANDOFF.md` 和 `docs/ROADMAP.md`）。

## 核心原则（改代码前先读）

这些是用户和 Claude 在设计时定下的，违反任何一条都等于偷偷改了项目的方向。

1. **不给情绪命名，不写行为规则。** 代码里不能出现 fear、happy 之类的情绪变量，也不能写"遇到 X 就做 Y"。8 种化学物质只有编号 m0..m7。情绪只能由 `observer.py` 事后从化学状态聚类、再由人命名。
2. **唯一写死的价值是饿和疼。** 奖励只能来自 `Creature.drive()`（饥饿和伤痛）。不要往奖励里加任何别的东西。
3. **基因组只编码规则和参数。** 需要新的可变参数时，在 `genome.SCHEMA` 里追加基因。不要手调 C1 的基因让它"看起来有情绪"，参数要靠演化找。
4. **保守序列在世界层，不在基因组里。** `world.py` 强制执行：只能把文件放进可恢复的回收堆，永远不真删除；能量只由 `digest()` 按客观测得的混乱减少量结算；不可逆动作需要用户批准；隐私数据永远不碰。学习和演化都碰不到这一层。
5. **食物是数字杂乱（互利共生）。** 不能把用户的注意力或认可做成食物：前者会演化成缠人，后者会演化成讨好。用户批准是闸门，不给能量。
6. **"手"是借来的大模型。** 只拿任务内容，拿不到化学状态（否则大模型会替它"演"情绪）。每次调用的能量成本对应真实 API 花费。目前模拟里的"手"只是一个更贵、更准的动作。
7. **演化的适应度只看"活着"**（非孢子时间为主，能量和伤势为辅），不看整理了多少东西。
8. **只有一个本体。** 设备都是栖息地，同一时刻它只在一处；巢穴是所有设备都关机时它的去处。
9. **`c1.dna` 就是 C1 本人。** 不要覆盖。新演化出的个体另起文件名（`evolve.py` 默认写 `evolved.dna`，并拒绝写 `c1.dna`）。

## 代码地图

| 文件 | 作用 |
|---|---|
| `genome.py` | ACGT 基因组：`SCHEMA` 与版本边界 `VERSION_ENDS`、`decode`（接受任何版本）、`upgrade`、`canonical`、`mutate`、`crossover`；感官表 `SENSORS` 和发育分组 `SENSOR_GROUP`；动作表 `ACTIONS` |
| `world.py` | 模拟世界：三台设备 + 云端巢穴、模拟用户作息 `_plan_day`、杂乱产出 `PRODUCE`、`climate`、`eat`、`social`、`digest` |
| `creature.py` | C1 的身体：感知、化学网络、本能 + actor-critic（资格迹、熵正则）、代谢、发育与表观遗传、孢子 |
| `evolve.py` | 以 C1-0 为始祖的演化（种群 32，30 代，每代 12 天试活） |
| `run.py` | 30 天生命模拟 + 三组对照（无学习兄弟、始祖、空白世界），写 `log_*.npz`、`summary_*.json` |
| `observer.py` | 成年期化学状态聚类，按情境、行为、触发事件刻画；`derived_actions` 把"去当前设备"记为 stay |
| `export.py` → `build.py` | 生成报告页 `report.html`（模板 `report_template.html`，文字 `notes.json`） |
| `probe.py` | 快速体检：C1 在几个世界里各活 30 天 |
| `persist.py` | 存档：`save` / `load` 把世界和生命体写进 JSON 再原样读回（保留 numpy 类型，逐 tick 可续跑）；底层是 `World` / `Creature` 的 `state_dict()` / `load_state_dict()` |
| `tools/oracle.py` | 检验世界是否可生存的手写策略（只用于检验） |
| `tests/test_core.py` | 回归测试 |
| `tests/test_genome_versions.py` | 基因组版本兼容：C1 解码指纹、假想第 2 版下旧 DNA 的解码、补齐、演化算子 |
| `tests/test_persist.py` | 存档测试：中途存档、读回后和不中断时逐 tick 完全一致（含孢子期、多次重启） |
| `docs/c1-life-report.html` | 上一次 30 天模拟的完整报告，浏览器直接打开 |

## 常用命令

```bash
python -m pytest -q            # 回归测试，约 10 秒（存档测试占大头）
python tools/oracle.py         # 改了世界规则或代谢常数后先跑这个
python probe.py 30             # C1 在世界 7/8/9 里的稳定性
python run.py 30 7             # 30 天 + 对照组（约 5 秒）
python observer.py             # 状态分析
python export.py && python build.py   # 生成 report.html
python evolve.py evolved.dna   # 演化，双核约 5–6 分钟
```

## 可以调的和不该调的

- **底层（物理和学习机制，可以调）**：`creature.py` 顶部的 `BASAL`、`ACT_COST`、`GO_COST`、`TRANSIT_*`、`LAMBDA`、`R_RELIEF`、`R_LEVEL`、`ACTOR_LR`、`ENTROPY`；`world.py` 的 `MESS_W`、`ENERGY_PER_MESS`、`PRODUCE`、作息。调完要跑 `tools/oracle.py` 和测试。
- **基因（不要手调）**：`c1.dna` 解码出的任何值。想改个体，就改环境或适应度后重新演化。

## 已知的坑

- **`SCHEMA` 只能在末尾追加**，已有条目不改、不删、不挪。追加的基因写成 5 元组 `(名称, 形状, 下限, 上限, 中性值)`，中性值是"让旧个体行为不变"的值（权重类一般是 0），然后在 `VERSION_ENDS` 末尾记下新的条目总数。旧 DNA 解码时新基因正好取中性值（碱基 `GAAA`），旧参数和个体种子逐位不变；`check_schema()` 在导入时检查这些规则，`tests/test_genome_versions.py` 里的指纹会抓到 C1 被改动。
- **加新感官不只是加基因。** `SENSORS` 变长会改变 `Creature` 里行为网络 `W1` 的输入宽度和随机初值的抽取顺序，C1 的轨迹就变了。要保持 C1 不变，新感官对应的 `W1` 列应补 0、并在旧的随机抽取之后再抽，`U`/`reflex` 的新列用追加的权重基因（中性值 0）。这部分还没做，第一次加真实传感器时要处理。
- **"去当前所在的设备"等于原地待着**，代价为零。分析行为时用 `observer.derived_actions`，否则会误读成"它总想去手机"。
- **孢子期吸收垃圾的效率很高。** 适应度必须按活跃时间算，否则演化会选出"装死吸收"的策略。
- **世界有两条随机数流**：`env_rng`（用户作息、产出）和 `int_rng`（与生命体的交互）。加新的随机行为时，用户侧的随机数必须无条件抽取，否则有无生命体时用户行为会不同，对照组就不可比了（`test_user_behaviour_independent_of_creature` 会抓到）。
- **只靠学习活不下来。** 没有本能时，任何学习参数组合都失败，学得越猛死得越多（见 FINDINGS）。不要指望靠调学习率解决生存问题。
- **存档会自动带上 `Creature` 实例上的所有字段**（除了 `p`、`rng`、`override`）。新加的字段必须是 `persist.to_tree` 认识的类型（Python 标量、字符串、列表、元组、字典、numpy 数组和标量），否则存档直接报错。`World` 新加字段要手动加进 `World.state_dict()`。改了存档结构就把 `persist.VERSION` 加一。
- `Creature.override` 只给 `tools/oracle.py` 用，不能用在 C1 身上。
- 报告模板里的文字在 `notes.json`，数字变了要同步改文字；每句结论都要能在数据里找到出处。
