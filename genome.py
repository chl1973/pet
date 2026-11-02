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

# (名称, 形状, 下限, 上限)
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


def n_genes():
    return sum(int(np.prod(s)) if s else 1 for _, s, _, _ in SCHEMA)


def random_dna(seed):
    rng = np.random.default_rng(seed)
    return "".join(rng.choice(list(BASES), size=n_genes() * 4))


def mutate(dna, rate, seed):
    rng = np.random.default_rng(seed)
    out = list(dna)
    for i in range(len(out)):
        if rng.random() < rate:
            out[i] = rng.choice([b for b in BASES if b != out[i]])
    return "".join(out)


def crossover(a, b, seed):
    """按基因（4 个碱基）为单位的均匀交叉。"""
    rng = np.random.default_rng(seed)
    pick = rng.random(n_genes()) < 0.5
    return "".join(a[g * 4:(g + 1) * 4] if pick[g] else b[g * 4:(g + 1) * 4]
                   for g in range(n_genes()))


def decode(dna):
    assert len(dna) == n_genes() * 4, "基因组长度不对"
    vals = np.empty(n_genes())
    for g in range(n_genes()):
        v = 0
        for b in dna[g * 4:(g + 1) * 4]:
            v = v * 4 + BASE_VAL[b]
        vals[g] = v / 255.0
    p, i = {}, 0
    for name, shape, lo, hi in SCHEMA:
        n = int(np.prod(shape)) if shape else 1
        chunk = lo + (hi - lo) * vals[i:i + n]
        i += n
        p[name] = chunk.reshape(shape) if shape else float(chunk[0])
    p["seed"] = int(hashlib.sha256(dna.encode()).hexdigest()[:8], 16)
    return p
