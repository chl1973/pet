"""把日志整理成报告页用的数据（report_data.json）。"""
import json
import numpy as np
from genome import SCHEMA, decode, ACTIONS, SENSORS
from observer import derived_actions, ACTIONS_EXT
from world import TPD

RUNS = ["C1", "C1_nolearn", "C1_0", "baseline"]
L = {k: dict(np.load(f"log_{k}.npz")) for k in RUNS}
S = {k: json.load(open(f"summary_{k}.json")) for k in RUNS}
states = json.load(open("states.json"))
lab = np.load("state_labels.npy")
evo = json.load(open("evolution.json"))
dna = open("c1.dna").read()
founder = open("c1_0.dna").read()
p = decode(dna)
T = len(L["C1"]["E"])
days = T // TPD


def hourly(x):
    return [round(float(v), 4) for v in np.asarray(x, float).reshape(-1, 12).mean(1)]


out = dict(days=days, tick_min=5, actions=ACTIONS_EXT)
out["energy"] = {k: hourly(L[k]["E"] / np.maximum(L[k]["Emax"], 1e-9)) for k in RUNS if k != "baseline"}
out["spore_h"] = {k: hourly(L[k]["spore"]) for k in RUNS if k != "baseline"}
out["mess"] = {k: hourly(L[k]["mess"].sum(1)) for k in RUNS}

# 活动谱：每个 tick 一位数字。0 笔记本 1 台式机 2 手机 3 巢穴 4 迁徙中 5 孢子
R = L["C1"]
code = np.where(R["spore"] == 1, 5, np.where(R["transit"] == 1, 4, R["loc"]))
out["acto"] = "".join(map(str, code.astype(int)))
out["acto_loc"] = "".join(map(str, R["loc"].astype(int)))
out["acto_action"] = "".join("-" if a < 0 else str(a) for a in derived_actions(R))
out["user_dev"] = ["".join(map(str, R["user"][:, i].astype(int))) for i in range(3)]
out["online_dev"] = ["".join(map(str, R["online"][:, i].astype(int))) for i in range(3)]
out["mess_dev"] = [hourly(R["mess"][:, i]) for i in range(3)]
out["hue_h"] = hourly(R["hue"])
out["speed_h"] = hourly(R["speed"])
out["acto_state"] = "".join("-" if s < 0 else str(s) for s in lab)
out["user_any"] = "".join(map(str, (R["user"].max(1) > 0).astype(int)))
ev_ticks = {}
from run import EVENT_ID
for name in ["rough", "toxic", "task", "evicted", "spore", "revive"]:
    ev_ticks[name] = np.where(R["event"] == EVENT_ID[name])[0].tolist()
out["events"] = ev_ticks

# 发育
first = 7 * TPD
out["expr"] = [hourly(R["expr"][:first, g]) for g in range(1, 5)]
out["size"] = hourly(R["size"])
out["epi"] = S["C1"]["epi"]
out["cp_end_days"] = round(p["cp_end_days"], 2)
out["onset_days"] = [round(float(p["onset"][i] * m), 2) for i, m in enumerate([1, 3, 3, 5])]

# 每日统计
def daily(x):
    return [int(v) for v in np.asarray(x).reshape(days, TPD).sum(1)]
out["daily_eaten"] = daily((R["ate"] > 0) & (R["ate"] != 3))
out["daily_toxic"] = daily(R["ate"] == 3)
out["daily_sorted"] = daily(R["ate"] == 4)

# 能量收支（C1，非孢子时）
act = R["spore"] == 0
out["budget"] = dict(gained=float(R["gained"].sum()), basal=float(R["basal"].sum()),
                     arousal=float(R["arousal"].sum()), action=float(R["action_cost"].sum()))

out["summary"] = S
out["states"] = states
out["evolution"] = evo
out["genome"] = dict(dna=dna, founder=founder, length=len(dna),
                     diff=sum(a != b for a, b in zip(dna, founder)),
                     segments=[])
gi = 0
for name, shape, *_ in SCHEMA:
    n = int(np.prod(shape)) if shape else 1
    out["genome"]["segments"].append(dict(name=name, start=gi * 4, len=n * 4))
    gi += n
out["genome"]["params"] = {k: round(float(p[k]), 4) for k in
                           ["base_lr", "gamma", "base_tau", "mouth_acc", "cp_end_days", "epi_k", "growth", "pain_gain"]}
out["genome"]["rate"] = [round(float(x), 3) for x in p["rate"]]
out["hue_mean"] = float(np.mean(R["hue"][R["spore"] == 0]))
json.dump(out, open("report_data.json", "w"), ensure_ascii=False, separators=(",", ":"))
print("report_data.json", round(len(json.dumps(out)) / 1024), "KB")
