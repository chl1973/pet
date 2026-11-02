"""C1 的基因组。

一串 ACGT 碱基，每 4 个碱基编码一个基因值（0-255），再按 SCHEMA 解码成参数。
基因组只编码规则和参数：化学物质之间怎么互相调控、感官怎么连到化学物质、
各感官什么时候发育、学习快慢、敏感期多长……
它不编码任何情绪名称，也不编码"遇到 X 就做 Y"的行为规则。

权限边界（不能删除、不可逆动作要批准、不碰隐私）不在这里，
而在 world.py 的世界规则里，生命体的学习和演化都碰不到。
"""
import hashlib
import numpy as np

BASES = "ACGT"
BASE_VAL = {b: i for i, b in enumerate(BASES)}

# 感官。分组决定发育时间：0 组出生即有，其他组的开启时间由基因决定。
SENSORS = [
    "hunger", "pain", "toxin",                                   # 0 内感受
    "loc_laptop", "loc_desktop", "loc_phone", "loc_nest",        # 0 体位感（知道自己在哪块栖息地）
    "local_smell", "warmth", "crowding", "in_transit", "novelty",  # 1 环境感官
    "user_present", "cursor_near", "touch",                      # 2 对用户的感官
    "mem_laptop", "mem_desktop", "mem_phone",                    # 3 对远处食物的记忆
    "tod_sin", "tod_cos",                                        # 4 昼夜节律
]
SENSOR_GROUP = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4]
GROUP_MAX_ONSET_DAYS = {1: 1.0, 2: 3.0, 3: 3.0, 4: 5.0}

ACTIONS = ["rest", "eat_quick", "eat_careful", "hide", "approach",
           "go_laptop", "go_desktop", "go_phone"]

NS, NM, NA = len(SENSORS), 8, len(ACTIONS)   # 感官数、化学物质数(m0..m7)、动作数

# ---------- 基因表 ----------
# 只许在末尾追加，已有的条目永远不改、不删、不挪位置。
#
# 第 1 版的基因是 4 元组 (名称, 形状, 下限, 上限)，按下限到上限线性解码。c1.dna 就是这一版。
# 第 2 版起追加的基因必须是 5 元组 (名称, 形状, 下限, 上限, 中性值)：
#   中性值 = 让旧个体的行为保持不变的值（权重类基因一般就是 0）。
#   这类基因以碱基 GAAA（数值 128）为锚点解码，GAAA 正好解出中性值，
#   向两边线性铺到下限和上限。旧 DNA 缺的基因一律补 GAAA。
# 追加一批基因后，在 VERSION_ENDS 末尾记下新的条目总数，版本号就加了一。
SCHEMA = [
    ("W", (NM, NM), -1.5, 1.5),          # 化学物质之间的调控网络
    ("U", (NM, NS), -1.2, 1.2),          # 感官 -> 化学物质
    ("bias", (NM,), -0.5, 0.5),
    ("rate", (NM,), 0.01, 0.4),          # 每种化学物质的更新速度（快的像反射，慢的像心境）
    ("arousal_cost", (NM,), 0.0, 0.02),  # 化学物质活跃时额外消耗的能量
    ("temp_gate", (NM,), -0.8, 0.8),     # 化学状态如何调节行为的随机性
    ("lr_gate", (NM,), -1.0, 1.0),       # 化学状态如何调节学习速度
    ("hue_gate", (NM,), -1.0, 1.0),      # 化学状态 -> 体色（给以后的可视化用）
    ("speed_gate", (NM,), -1.0, 1.0),    # 化学状态 -> 动作节奏
    ("onset", (4,), 0.0, 1.0),           # 感官组 1-4 的开启时间（乘以各组上限天数）
    ("cp_end_days", (), 1.0, 4.0),       # 敏感期长度
    ("epi_k", (), 0.0, 3.0),             # 早期经历对后期敏感度的改写强度
    ("base_lr", (), 0.003, 0.03),
    ("gamma", (), 0.95, 0.995),
    ("base_tau", (), 0.6, 1.6),
    ("mouth_acc", (), 0.4, 0.95),        # 嘴的分辨力：认出"这不是垃圾"的概率
    ("growth", (), 0.2, 1.0),
    ("pain_gain", (), 0.5, 2.0),
    ("init_scale", (), 0.05, 0.4),
    # 本能：先天的行为倾向。初值随机，由演化筛选，不由我们手写。
    ("reflex", (NA, NS), -2.0, 2.0),     # 感官 -> 动作倾向
    ("reflex_bias", (NA,), -1.0, 1.0),
    ("chem_act", (NA, NM), -2.0, 2.0),   # 化学状态 -> 动作倾向（化学物质借此真正影响行为）
]


# 每个版本的 SCHEMA 到第几条为止（条目数，不是基因数）
VERSION_ENDS = [22]
NEUTRAL_CODON = "GAAA"
NEUTRAL_V = 128


def _size(shape):
    return int(np.prod(shape)) if shape else 1


def schema_version():
    return len(VERSION_ENDS)


def check_schema():
    """基因表本身的约束：版本边界递增、第 1 版之后的基因都带中性值且在范围内。"""
    assert VERSION_ENDS == sorted(set(VERSION_ENDS)) and VERSION_ENDS[-1] == len(SCHEMA)
    names = [e[0] for e in SCHEMA]
    assert len(names) == len(set(names)), "基因名重复"
    for i, e in enumerate(SCHEMA):
        if i < VERSION_ENDS[0]:
            assert len(e) == 4, f"{e[0]}：第 1 版的基因不能改格式"
        else:
            assert len(e) == 5, f"{e[0]}：追加的基因必须给出中性值"
            _, _, lo, hi, neutral = e
            assert lo <= neutral <= hi and lo < hi, f"{e[0]}：中性值不在范围内"


def n_genes(version=None):
    """某个版本的基因数（默认当前版本）。"""
    v = schema_version() if version is None else version
    return sum(_size(e[1]) for e in SCHEMA[:VERSION_ENDS[v - 1]])


def version_of(dna):
    """按长度认出 DNA 属于哪个版本。只追加的规则保证每个版本长度不同。"""
    bad = set(dna) - set(BASES)
    if bad:
        raise ValueError(f"DNA 里有非 ACGT 的字符：{sorted(bad)!r}")
    for v in range(1, schema_version() + 1):
        if len(dna) == n_genes(v) * 4:
            return v
    raise ValueError(f"基因组长度 {len(dna)} 不属于任何版本，"
                     f"可用长度：{[n_genes(v) * 4 for v in range(1, schema_version() + 1)]}")


def upgrade(dna):
    """把旧版本 DNA 补成当前版本：缺的基因补中性碱基，解码结果和直接解码旧 DNA 完全一样。"""
    return dna + NEUTRAL_CODON * (n_genes() - n_genes(version_of(dna)))


def canonical(dna):
    """去掉末尾整版都是中性碱基的部分。upgrade 前后的 DNA 规范形式相同。"""
    v = version_of(dna)
    while v > 1 and dna[n_genes(v - 1) * 4:] == NEUTRAL_CODON * (n_genes(v) - n_genes(v - 1)):
        v -= 1
        dna = dna[:n_genes(v) * 4]
    return dna


def random_dna(seed):
    rng = np.random.default_rng(seed)
    return "".join(rng.choice(list(BASES), size=n_genes() * 4))


def mutate(dna, rate, seed):
    """旧版本 DNA 先补成当前版本再突变，新基因也能被演化改动。"""
    dna = upgrade(dna)
    rng = np.random.default_rng(seed)
    out = list(dna)
    for i in range(len(out)):
        if rng.random() < rate:
            out[i] = rng.choice([b for b in BASES if b != out[i]])
    return "".join(out)


def crossover(a, b, seed):
    """按基因（4 个碱基）为单位的均匀交叉。"""
    a, b = upgrade(a), upgrade(b)
    rng = np.random.default_rng(seed)
    pick = rng.random(n_genes()) < 0.5
    return "".join(a[g * 4:(g + 1) * 4] if pick[g] else b[g * 4:(g + 1) * 4]
                   for g in range(n_genes()))


def decode(dna):
    """解码成参数字典。接受任何版本的 DNA；缺的基因取中性值。

    p["seed"] 由规范形式算出，所以旧 DNA 补齐前后是同一个个体（行为网络初值、随机数都一样）。
    """
    v = version_of(dna)
    full = upgrade(dna)
    codes = np.empty(n_genes())
    for g in range(n_genes()):
        x = 0
        for b in full[g * 4:(g + 1) * 4]:
            x = x * 4 + BASE_VAL[b]
        codes[g] = x
    p, i = {}, 0
    for e in SCHEMA:
        name, shape, lo, hi = e[:4]
        n = _size(shape)
        c = codes[i:i + n]
        if len(e) == 4:
            chunk = lo + (hi - lo) * (c / 255.0)
        else:
            neutral = e[4]
            up = neutral + (hi - neutral) * ((c - NEUTRAL_V) / (255 - NEUTRAL_V))
            down = neutral - (neutral - lo) * ((NEUTRAL_V - c) / NEUTRAL_V)
            chunk = np.where(c >= NEUTRAL_V, up, down)
        i += n
        p[name] = chunk.reshape(shape) if shape else float(chunk[0])
    p["seed"] = int(hashlib.sha256(canonical(dna).encode()).hexdigest()[:8], 16)
    p["dna_version"] = v
    return p


check_schema()
