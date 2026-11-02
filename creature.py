"""C1 的身体。

写死的只有两种最底层的好坏信号：饿（能量不足）和疼（身体受损）。
学习的目标是让这两样少一点。除此之外，没有任何情绪名称、没有行为规则。

体内有 8 种无名化学物质 m0..m7，按基因组给的调控网络自己演化。
它们影响：行为选择、行为的随机性、学习快慢、额外能耗、体色和动作节奏。
"""
import copy
import numpy as np
from genome import (decode, canonical, NS, NM, NA, SENSORS, SENSOR_GROUP,
                    GROUP_MAX_ONSET_DAYS, ACTIONS)
from world import TPD, DEVICES

SI = {n: i for i, n in enumerate(SENSORS)}
BASAL = 0.25                      # 基础代谢（体型 1.0 时每 tick）
ACT_COST = dict(rest=0.0, eat_quick=0.3, eat_careful=1.5, hide=0.2, approach=0.2)
GO_COST, TRANSIT_TICKS, TRANSIT_COST = 1.0, 3, 0.5
H = 24                            # 行为网络隐藏层大小
LAMBDA = 0.8                      # 资格迹衰减（学习机制本身，属于底层，不由基因决定）
R_RELIEF, R_LEVEL = 5.0, 0.05     # 奖励：饿/疼减少量的权重，和持续饿/疼的惩罚
ACTOR_LR, ENTROPY = 0.1, 0.02     # 行为网络学习步长（相对价值网络）、探索倾向强度


def sig(x):
    return 1.0 / (1.0 + np.exp(-x))


class Creature:
    def __init__(self, name, dna, learn=True, birth="desktop"):
        self.name, self.dna, self.learn = name, dna, learn
        self.p = p = decode(dna)
        self.rng = np.random.default_rng(p["seed"])
        self.m = np.zeros(NM)
        self.size = 0.6
        self.E = self.Emax()              # 出生时带满"卵黄"
        self.damage, self.toxin = 0.0, 0.0
        self.age = 0
        self.loc, self.transit = birth, None
        self.mem = {d: (0.0, -10 ** 9) for d in DEVICES}
        self.last_social = dict(cursor_near=0, touch=0.0)
        # 孢子（饿到零时的休眠形态）
        self.spore, self.spore_count, self.spore_nest_ticks = False, 0, 0
        # 发育与表观遗传
        self.U_mult = np.ones(NS)
        self.pain_mult = 1.0
        self.cp_stress, self.cp_n, self.epi = 0.0, 0, None
        # 行为网络（权重初值由基因组种子决定，之后靠学习改变）
        nx = NS + NM + 1
        s = p["init_scale"]
        self.W1 = self.rng.normal(0, s, (H, nx))
        self.W2 = self.rng.normal(0, s, (NA, H))
        self.w3 = np.zeros(H)
        self.prev = None
        self.acc_r, self.acc_k = 0.0, 0
        self.event = ""
        self._reset_traces()
        self.D_prev = self.drive()

    # ---------- 存档与恢复 ----------
    # 不进存档的：p 由 dna 解码得到；rng 单独存状态；override 只给 tools/oracle.py 用。
    _NOT_SAVED = ("p", "rng", "override")

    def state_dict(self):
        """这条生命的完整状态（深拷贝）：化学浓度、行为网络权重、资格迹、发育与表观遗传标记、
        孢子状态、随机数状态……除了上面几项，实例上的每个字段都会自动进存档，
        以后加新字段不会被漏掉。"""
        sd = {k: copy.deepcopy(v) for k, v in vars(self).items() if k not in self._NOT_SAVED}
        sd["rng"] = copy.deepcopy(self.rng.bit_generator.state)
        return sd

    def load_state_dict(self, sd):
        assert canonical(sd["dna"]) == canonical(self.dna), "存档的基因组和这个个体不是同一个"
        fresh = set(vars(self)) - set(self._NOT_SAVED)
        missing = fresh - set(sd)
        assert not missing, f"存档缺少字段：{sorted(missing)}"
        for k, v in sd.items():
            if k == "rng":
                self.rng.bit_generator.state = copy.deepcopy(v)
            elif k not in self._NOT_SAVED:
                setattr(self, k, copy.deepcopy(v))

    @classmethod
    def from_state_dict(cls, sd):
        c = cls(sd["name"], sd["dna"], learn=sd["learn"])
        c.load_state_dict(sd)
        return c

    def _reset_traces(self):
        self.za_W1, self.za_W2 = np.zeros_like(self.W1), np.zeros_like(self.W2)
        self.zc_W1, self.zc_w3 = np.zeros_like(self.W1), np.zeros_like(self.w3)

    # ---------- 身体 ----------
    def Emax(self):
        return 100.0 * self.size

    def age_days(self):
        return self.age / TPD

    def hunger(self):
        return float(np.clip(1 - self.E / self.Emax(), 0, 1))

    def drive(self):
        """唯一写死的价值：饿和疼。"""
        return 2 * self.hunger() ** 2 + self.p["pain_gain"] * self.pain_mult * self.damage

    def expression(self):
        d, e = self.age_days(), np.ones(NS)
        for i, g in enumerate(SENSOR_GROUP):
            if g > 0:
                onset = self.p["onset"][g - 1] * GROUP_MAX_ONSET_DAYS[g]
                e[i] = sig((d - onset) / 0.15)
        return e

    def phenotype(self):
        return (float(360 * sig(self.p["hue_gate"] @ self.m)),
                float(0.5 + sig(self.p["speed_gate"] @ self.m)))

    def in_critical_period(self):
        return self.age_days() < self.p["cp_end_days"]

    # ---------- 感知 ----------
    def sense(self, w):
        s = np.zeros(NS)
        s[SI["hunger"]] = self.hunger()
        s[SI["pain"]] = self.damage
        s[SI["toxin"]] = min(1.0, self.toxin)
        if self.transit:
            s[SI["in_transit"]] = 1
        else:
            s[SI["loc_" + self.loc]] = 1
            if self.loc != "nest":
                h = w.h[self.loc]
                sm = min(1.0, h.smell() / 30)
                s[SI["local_smell"]] = sm
                self.mem[self.loc] = (sm, w.t)
                s[SI["user_present"]] = 1.0 if h.user else 0.0
                s[SI["novelty"]] = min(1.0, w.new_files[self.loc])
            _, warmth, crowd = w.climate(self.loc)
            s[SI["warmth"]], s[SI["crowding"]] = warmth, crowd
        s[SI["cursor_near"]] = self.last_social["cursor_near"]
        s[SI["touch"]] = self.last_social["touch"]
        for d in DEVICES:
            v, t0 = self.mem[d]
            s[SI["mem_" + d]] = v * np.exp(-(w.t - t0) / (TPD / 2))
        ang = 2 * np.pi * (w.t % TPD) / TPD
        s[SI["tod_sin"]], s[SI["tod_cos"]] = np.sin(ang), np.cos(ang)
        return s * self.expression()

    def chemistry(self, s):
        p = self.p
        drive_in = p["W"] @ self.m + (p["U"] * self.U_mult) @ s + p["bias"]
        self.m = (1 - p["rate"]) * self.m + p["rate"] * np.tanh(drive_in)

    # ---------- 学习 ----------
    def lr(self):
        p = self.p
        dev = 3.0 if self.in_critical_period() else 1.0
        return p["base_lr"] * dev * (0.25 + 1.5 * sig(p["lr_gate"] @ self.m))

    def learn_from(self, V_now):
        """TD 误差 × 资格迹：结果出来时，把功过分给最近做过的那些决定。"""
        V_prev = self.prev
        g = self.p["gamma"]
        delta = float(np.clip(self.acc_r + g ** self.acc_k * V_now - V_prev, -1, 1))
        lr = self.lr()
        self.w3 += lr * delta * self.zc_w3
        self.W1 += lr * delta * self.zc_W1 + ACTOR_LR * lr * delta * self.za_W1
        self.W2 += ACTOR_LR * lr * delta * self.za_W2

    def _trace(self, x, h, pi, a, tau):
        # 探索倾向：始终轻推行为分布不要塌缩成单一动作（熵正则，直接更新、不进资格迹）
        logp = np.log(pi + 1e-12)
        Hpi = -(pi * logp).sum()
        g_ent = -pi * (logp + Hpi) / tau
        step = ACTOR_LR * self.lr() * ENTROPY
        self.W2 += step * np.outer(g_ent, h)
        self.W1 += step * np.outer((self.W2.T @ g_ent) * (1 - h ** 2), x)
        d = (self.p["gamma"] * LAMBDA) ** self.acc_k
        ga = -pi / tau
        ga[a] += 1 / tau
        self.za_W2 = d * self.za_W2 + np.outer(ga, h)
        self.za_W1 = d * self.za_W1 + np.outer((self.W2.T @ ga) * (1 - h ** 2), x)
        self.zc_w3 = d * self.zc_w3 + h
        self.zc_W1 = d * self.zc_W1 + np.outer(self.w3 * (1 - h ** 2), x)

    # ---------- 一个 tick ----------
    def step(self, w):
        self.age += 1
        self.event = ""
        if self.spore:
            return self._spore_step(w)
        # 设备关机：被弹回云端巢穴
        if not self.transit and self.loc != "nest" and not w.h[self.loc].online:
            self.loc, self.event = "nest", "evicted"

        s = self.sense(w)
        self.chemistry(s)
        g = self.p["gamma"]
        cost, gained, resting, ate = 0.0, 0.0, False, None

        if self.transit:                        # 迁徙途中，不做决策
            dest, left = self.transit
            cost += TRANSIT_COST
            if left <= 1:
                self.transit = None
                self.loc = dest if w.h[dest].online else "nest"
                self.event = "arrived"
            else:
                self.transit = (dest, left - 1)
            a = -1
            self.last_social = dict(cursor_near=0, touch=0.0)
        else:
            x = np.concatenate([s, self.m, [1.0]])
            h = np.tanh(self.W1 @ x)
            p = self.p
            instinct = p["reflex"] @ s + p["reflex_bias"] + p["chem_act"] @ self.m
            logits, V = self.W2 @ h + instinct, float(self.w3 @ h)
            tau = self.p["base_tau"] * np.exp(np.clip(self.p["temp_gate"] @ self.m, -1.5, 1.5))
            z = logits / tau
            z -= z.max()
            pi = np.exp(z)
            pi /= pi.sum()
            if self.prev is not None and self.learn:
                self.learn_from(V)
            a = int(self.rng.choice(NA, p=pi))
            if getattr(self, "override", None):        # 仅用于检验世界是否可生存
                a = self.override(self, w)
            if self.learn:
                self._trace(x, h, pi, a, tau)
            name = ACTIONS[a]

            if name.startswith("go_"):
                dest = name[3:]
                if dest == self.loc:
                    pass                                  # 原地打转
                elif w.h[dest].online:
                    cost += GO_COST
                    self.transit = (dest, TRANSIT_TICKS)
                    self.event = "depart"
                else:
                    cost += 0.5                           # 目的地关机，白跑
            else:
                cost += ACT_COST[name]
                resting = name == "rest"
                if name.startswith("eat") and self.loc != "nest":
                    res = w.eat(self.loc, name == "eat_careful", self.p["mouth_acc"])
                    gained, ate = res["energy"], res
                    if res["toxic"]:
                        self.toxin += 1.0
                        self.event = "toxic"
            ev = w.social(self.loc, name) if not self.transit else dict(cursor_near=0, touch=0.0, damage=0.0, rough=False, task=0)
            self.damage = min(1.0, self.damage + ev["damage"])
            if ev.get("rough"):
                self.event = "rough"
            elif ev.get("task"):
                self.event = "task"
            self.last_social = dict(cursor_near=ev["cursor_near"], touch=ev["touch"])
            self.prev = V
            self.acc_r, self.acc_k = 0.0, 0

        # 代谢
        where = "nest" if self.transit else self.loc
        mult, _, _ = w.climate(where)
        basal = BASAL * self.size ** 0.75 * mult * (0.6 if resting else 1.0)
        arousal = float(self.p["arousal_cost"] @ np.abs(self.m))
        self.E = min(self.Emax(), self.E - basal - arousal - cost + gained)
        self.damage = min(1.0, self.damage + 0.02 * self.toxin)
        self.toxin *= 0.97
        self.damage = max(0.0, self.damage - 0.002 * (3 if resting else 1))

        # 奖励只来自饿和疼：主要看它们减少了多少，再加一点对"一直饿着/疼着"的惩罚
        D_now = self.drive()
        r = R_RELIEF * (self.D_prev - D_now) - R_LEVEL * D_now
        self.D_prev = D_now
        self.acc_r += g ** self.acc_k * r
        self.acc_k += 1

        self._develop()
        self.spent = dict(basal=basal, arousal=arousal, action=cost, gained=gained)
        self.action, self.ate = a, ate
        if self.E <= 0:
            self._sporulate()

    # ---------- 发育 ----------
    def _develop(self):
        d = self.age_days()
        mat = 1 - np.exp(-d / 3)
        target = 0.6 + (0.3 + 0.7 * self.p["growth"]) * mat
        nour = self.E / self.Emax()
        if nour > 0.4 and self.size < target:
            self.size += (target - self.size) * 0.003 * nour
        if self.in_critical_period():
            self.cp_stress += self.drive()
            self.cp_n += 1
        elif self.epi is None:
            stress = self.cp_stress / max(1, self.cp_n)
            k = self.p["epi_k"]
            for n in ["pain", "touch", "cursor_near", "user_present"]:
                self.U_mult[SI[n]] *= 1 + k * stress
            self.pain_mult *= 1 + 0.5 * k * stress
            self.epi = dict(day=round(d, 2), stress=round(stress, 3),
                            social_gain=round(1 + k * stress, 3),
                            pain_mult=round(self.pain_mult, 3))

    # ---------- 孢子 ----------
    def _sporulate(self):
        # 最后一次学习：孢子期间一直处在最大饥饿（drive=2）
        if self.prev is not None and self.learn:
            self.learn_from(-R_LEVEL * 2.0 / (1 - self.p["gamma"]))
        self._reset_traces()
        self.E = 0.0
        self.spore = True
        self.spore_count += 1
        if self.transit:
            self.loc, self.transit = "nest", None
        self.prev, self.spore_nest_ticks = None, 0
        self.event = "spore"

    def _spore_step(self, w):
        self.action, self.ate = -1, None
        self.spent = dict(basal=0.0, arousal=0.0, action=0.0, gained=0.0)
        if self.loc != "nest" and not w.h[self.loc].online:
            self.loc = "nest"
        if self.loc == "nest":
            self.spore_nest_ticks += 1
            if self.spore_nest_ticks >= TPD // 4:          # 随"风"落到一台在线设备上
                online = [d for d in DEVICES if w.h[d].online]
                self.loc = online[self.rng.integers(len(online))]
                self.spore_nest_ticks = 0
        elif self.age % 6 == 0:                            # 缓慢渗透吸收垃圾
            c = w.h[self.loc].c
            if c["temp"] + c["dup"] > 0:
                k = "temp" if c["temp"] > 0 else "dup"
                before = w.h[self.loc].mess()
                c[k] -= 1
                from world import digest
                g = digest(before, w.h[self.loc].mess())
                self.E += g
                self.spent["gained"] = g
        self.damage = max(0.0, self.damage - 0.002)
        if self.E >= 0.25 * self.Emax():
            self.spore = False
            self.pain_mult *= 1.1                           # 复苏后留下一点后遗症
            self.event = "revive"
            self.D_prev = self.drive()
