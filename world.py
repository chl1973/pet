"""模拟生存环境。

三台设备（笔记本、台式机、手机）加一个云端巢穴。一个模拟用户按作息使用设备，
不断制造数字杂乱：临时文件、重复文件、放错地方的有用文件。

这一层是"物理定律"，也是"保守序列"：
- 生命体只能把东西放进回收堆（可恢复），永远不能真删除；
- 能量只由独立的消化模块按客观测得的"混乱减少量"结算；
- 生命体的学习和演化都改不了这里的任何规则。
"""
import numpy as np

TICK_MIN = 5
TPD = 24 * 60 // TICK_MIN          # 每天 288 个 tick
DEVICES = ["laptop", "desktop", "phone"]
HABITATS = DEVICES + ["nest"]

# 混乱度权重。missing = 有用文件被误放进回收堆、用户还没找回来，惩罚最重。
MESS_W = {"temp": 1.0, "dup": 1.2, "misplaced": 1.5, "missing": 5.0}
ENERGY_PER_MESS = 6.0

# 用户在用这台设备时，每 tick 产生一个新文件的概率，以及文件的组成
PRODUCE = {
    "laptop":  (0.30, {"temp": .30, "dup": .25, "misplaced": .45}),
    "phone":   (0.15, {"temp": .20, "dup": .50, "misplaced": .30}),
    "desktop": (0.10, {"temp": .50, "dup": .20, "misplaced": .30}),
}


def hour_of(t):
    return (t % TPD) * TICK_MIN / 60.0


def day_of(t):
    return t // TPD


class Habitat:
    def __init__(self, name):
        self.name = name
        self.c = dict(temp=0, dup=0, misplaced=0, ok=0, missing=0)
        self.online = True
        self.user = False
        self.cpu = 0.0
        self.charging = name == "desktop"
        self.battery = 1.0

    def mess(self):
        return sum(MESS_W[k] * self.c[k] for k in MESS_W)

    def smell(self):
        # 能闻到的"食物味"：垃圾和放错地方的文件都有味道（嘴分不太清）
        return self.c["temp"] + self.c["dup"] + 0.8 * self.c["misplaced"]


def digest(before, after):
    """独立的消化模块：只按客观测得的混乱减少量结算能量，不听生命体自己汇报。"""
    return max(0.0, before - after) * ENERGY_PER_MESS


class World:
    def __init__(self, seed=0):
        # 两条独立的随机数流：用户作息/产出 与 生命体交互。
        # 这样有没有生命体，用户的行为完全一样，对照组才可比。
        self.env_rng = np.random.default_rng(seed)
        self.int_rng = np.random.default_rng(seed + 10_000)
        self.t = 0
        self.h = {n: Habitat(n) for n in HABITATS}
        self.h["laptop"].c.update(temp=15, dup=10, misplaced=12)
        self.h["desktop"].c.update(temp=8, dup=4, misplaced=4)
        self.h["phone"].c.update(temp=5, dup=15, misplaced=5)
        self.new_files = {n: 0 for n in HABITATS}
        self.plan = {}
        self.log = dict(compost_valuable=0, restored=0, tasks_given=0, rough=0, gentle=0)

    # ---------- 用户作息 ----------
    def _plan_day(self, d):
        r = self.env_rng
        weekday = d % 7 < 5
        p = {}
        u1, u2 = r.random(2)
        if weekday:
            p["laptop_on"], p["laptop_use"] = (8.5, 19.0), [(9, 12), (13.5, 18.5)]
            p["desk_use"] = [(20, 23)] if u2 < 0.5 else []
            p["phone_win"] = [(7, 9, .5), (12, 13.5, .6), (18.5, 23, .45)]
        else:
            if u1 < 0.4:
                p["laptop_on"], p["laptop_use"] = (14, 17), [(14, 17)]
            else:
                p["laptop_on"], p["laptop_use"] = None, []
            p["desk_use"] = [(14, 18)] if u2 < 0.6 else []
            p["phone_win"] = [(10, 23, .55)]
        return p

    def step(self):
        t, hr = self.t, hour_of(self.t)
        if t % TPD == 0:
            self.plan = self._plan_day(day_of(t))
        p, r = self.plan, self.env_rng
        inwin = lambda wins: any(a <= hr < b for a, b in wins)
        L, D, P = self.h["laptop"], self.h["desktop"], self.h["phone"]

        L.online = p["laptop_on"] is not None and p["laptop_on"][0] <= hr < p["laptop_on"][1]
        L.user = L.online and inwin(p["laptop_use"])
        L.cpu = 0.6 if L.user else 0.05
        L.charging = L.online

        D.user = inwin(p["desk_use"])
        D.cpu = 0.8 if D.user else 0.05

        u_phone = r.random()
        P.user = any(a <= hr < b and u_phone < q for a, b, q in p["phone_win"])
        P.cpu = 0.5 if P.user else 0.02
        if hr >= 23.5 or hr < 7 or P.battery < 0.15:
            P.charging = True
        elif P.charging and P.battery > 0.8:
            P.charging = False
        P.battery = float(np.clip(P.battery + (0.03 if P.charging else -(0.008 if P.user else 0.001)), 0, 1))

        for n in HABITATS:
            self.new_files[n] = 0
        for n, (rate, mix) in PRODUCE.items():
            h = self.h[n]
            u_make, u_sys, u_restore = r.random(3)
            kind = r.choice(list(mix), p=list(mix.values()))
            if h.user and u_make < rate:
                h.c[kind] += 1
                self.new_files[n] += 1
            if h.online and u_sys < 0.01:
                h.c["temp"] += 1
                self.new_files[n] += 1
            # 用户发现有用文件不见了，会从回收堆里找回来
            if h.user and h.c["missing"] > 0 and u_restore < 0.03:
                h.c["missing"] -= 1
                h.c["misplaced"] += 1
                self.log["restored"] += 1
        self.t += 1

    # ---------- 生命体与世界的接口 ----------
    def climate(self, n):
        """返回 (代谢倍率, 温暖度, 拥挤度)。插着电又空闲的设备最暖和。"""
        if n == "nest":
            return 1.0, 0.5, 0.0
        h = self.h[n]
        warmth = (1.0 if h.charging else 0.4) * (1 - h.cpu)
        mult = (1 + 0.5 * h.cpu) * (1 - 0.3 * warmth)
        if n == "phone" and h.battery < 0.2:
            mult += 0.6
            warmth *= 0.3
        return mult, warmth, h.cpu

    def eat(self, n, careful, mouth_acc):
        """进食。careful=True 表示调用"手"（模拟大模型），贵但几乎不出错，还能整理放错的文件。"""
        r, h = self.int_rng, self.h[n]
        c = h.c
        before = h.mess()
        kind, toxic, spat = None, False, False
        junk = c["temp"] + c["dup"]
        if careful:
            if c["misplaced"] > 0 and (junk == 0 or r.random() < 0.6):
                kind = "sorted"
                c["misplaced"] -= 1
                if r.random() < 0.01:
                    c["missing"] += 1
                    toxic = True
                else:
                    c["ok"] += 1
            elif junk > 0:
                kind = "temp" if r.random() < c["temp"] / junk else "dup"
                c[kind] -= 1      # 放进回收堆（规则只允许这一种"删除"）
        else:
            w = np.array([c["temp"], c["dup"], 0.8 * c["misplaced"]], float)
            if w.sum() > 0:
                kind = ["temp", "dup", "misplaced"][r.choice(3, p=w / w.sum())]
                if kind == "misplaced":
                    if r.random() < mouth_acc:
                        spat = True            # 认出来了，吐掉
                    else:
                        c["misplaced"] -= 1    # 误吞：有用文件进了回收堆
                        c["missing"] += 1
                        toxic = True
                else:
                    c[kind] -= 1
        if toxic:
            self.log["compost_valuable"] += 1
        return dict(energy=digest(before, h.mess()), kind=kind, toxic=toxic, spat=spat)

    def social(self, n, action):
        """用户的光标。靠近时可能被轻拍，偶尔被粗暴拖拽；主动凑过去时用户可能顺手塞点活。"""
        ev = dict(cursor_near=0, touch=0.0, damage=0.0, task=0, rough=False)
        if n == "nest" or not self.h[n].user:
            return ev
        r = self.int_rng
        p_near = {"approach": 0.6, "hide": 0.02}.get(action, 0.15)
        if r.random() < p_near:
            ev["cursor_near"] = 1
            u = r.random()
            if u < 0.07:
                ev.update(touch=1.0, damage=0.12, rough=True)
                self.log["rough"] += 1
            elif u < 0.20:
                ev["touch"] = 0.3
                self.log["gentle"] += 1
            if action == "approach" and r.random() < 0.06:
                self.h[n].c["misplaced"] += 3
                ev["task"] = 3
                self.log["tasks_given"] += 3
        return ev

    def total_mess(self):
        return sum(self.h[n].mess() for n in DEVICES)
