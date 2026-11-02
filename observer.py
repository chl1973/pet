"""观察者：不给任何状态预设名字，只从数据里找 C1 反复回到的化学状态，
再看每个状态出现在什么情境、伴随什么行为、被什么事件触发。命名留给人事后做。"""
import json
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from genome import ACTIONS, NM
from world import TPD
from run import EVENT_ID

EV_NAMES = {v: k for k, v in EVENT_ID.items()}
LOCS = ["laptop", "desktop", "phone", "nest", "transit"]
ACTIONS_EXT = ACTIONS + ["stay"]       # "去当前所在的设备" 实际是原地待着，单独记一类


def derived_actions(R):
    a = R["action"].copy()
    for i, d in enumerate(["laptop", "desktop", "phone"]):
        go = ACTIONS.index("go_" + d)
        a[(R["action"] == go) & (R["transit"] == 0) & (R["loc"] == i)] = len(ACTIONS)
    return a


def run_lengths(labels, k):
    out = {i: [] for i in range(k)}
    cur, n = labels[0], 1
    for x in labels[1:]:
        if x == cur:
            n += 1
        else:
            out[cur].append(n)
            cur, n = x, 1
    out[cur].append(n)
    return {i: float(np.mean(v)) if v else 0.0 for i, v in out.items()}


def analyze(R, start_day):
    T = len(R["E"])
    idx = np.arange(start_day * TPD, T)
    idx = idx[R["spore"][idx] == 0]
    M = R["m"][idx]
    mu, sd = M.mean(0), M.std(0) + 1e-9
    Z = (M - mu) / sd
    rng = np.random.default_rng(0)
    sub = rng.choice(len(Z), min(4000, len(Z)), replace=False)
    best = None
    for k in range(3, 7):
        km = KMeans(k, n_init=10, random_state=0).fit(Z)
        sc = silhouette_score(Z[sub], km.labels_[sub])
        if best is None or sc > best[0]:
            best = (sc, k, km)
    sil, k, km = best
    lab = km.labels_
    # 按占比从大到小重新编号
    order = np.argsort(-np.bincount(lab, minlength=k))
    remap = np.empty(k, int)
    remap[order] = np.arange(k)
    lab = remap[lab]
    full = np.full(T, -1)
    full[idx] = lab

    loc = np.where(R["transit"][idx] == 1, 4, R["loc"][idx])
    hour = ((idx % TPD) * 5 / 60).astype(int)
    act = derived_actions(R)[idx]
    hunger = 1 - R["E"][idx] / R["Emax"][idx]
    dwell = run_lengths(lab, k)

    states = []
    for c in range(k):
        s = lab == c
        a = act[s][act[s] >= 0]
        ad = np.bincount(a, minlength=len(ACTIONS_EXT)) / max(1, len(a))
        states.append(dict(
            id=c, share=float(s.mean()),
            chem=[float(x) for x in M[s].mean(0)],
            chem_z=[float(x) for x in Z[s].mean(0)],
            hunger=float(hunger[s].mean()), damage=float(R["damage"][idx][s].mean()),
            toxin=float(R["toxin"][idx][s].mean()),
            user=float(R["user_here"][idx][s].mean()), cursor=float(R["cursor"][idx][s].mean()),
            warmth=float(R["warmth"][idx][s].mean()), crowd=float(R["crowd"][idx][s].mean()),
            smell=float(R["local_smell"][idx][s].mean()),
            speed=float(R["speed"][idx][s].mean()), hue=float(R["hue"][idx][s].mean()),
            hours=[float(x) for x in np.bincount(hour[s], minlength=24) / max(1, s.sum())],
            locs=[float(x) for x in np.bincount(loc[s], minlength=5) / max(1, s.sum())],
            actions=[float(x) for x in ad],
            dwell_min=dwell[c] * 5,
        ))
    # 事件触发：事件发生后 30 分钟内，各状态出现比例相对基线的倍数
    base = np.bincount(lab, minlength=k) / len(lab)
    triggers = {}
    for ev in ["rough", "toxic", "evicted", "arrived", "task"]:
        ts = np.where(R["event"] == EVENT_ID[ev])[0]
        ts = ts[ts >= start_day * TPD]
        win = [full[t + 1:t + 7] for t in ts if t + 7 < T]
        win = np.concatenate(win) if win else np.array([], int)
        win = win[win >= 0]
        if len(win) >= 6:
            dist = np.bincount(win, minlength=k) / len(win)
            triggers[ev] = dict(n=int(len(ts)), lift=[float(x) for x in dist / np.maximum(base, 1e-9)])
    # 用户出现时（光标开始靠近）
    cur = R["cursor"]
    on = np.where((cur[1:] == 1) & (cur[:-1] == 0))[0] + 1
    on = on[on >= start_day * TPD]
    win = np.concatenate([full[t:t + 6] for t in on if t + 6 < T]) if len(on) else np.array([], int)
    win = win[win >= 0]
    if len(win):
        triggers["cursor"] = dict(n=int(len(on)), lift=[float(x) for x in (np.bincount(win, minlength=k) / len(win)) / base])

    # 状态是否真的影响行为：状态与动作的互信息，对比打乱后的基线
    def mi(x, y):
        j = np.zeros((x.max() + 1, y.max() + 1))
        np.add.at(j, (x, y), 1)
        j /= j.sum()
        px, py = j.sum(1, keepdims=True), j.sum(0, keepdims=True)
        nz = j > 0
        return float((j[nz] * np.log(j[nz] / (px @ py)[nz])).sum())
    ok = act >= 0
    mi_real = mi(lab[ok], act[ok])
    mi_shuf = float(np.mean([mi(rng.permutation(lab[ok]), act[ok]) for _ in range(20)]))
    # 与"只看情境"相比：同一地点 + 同样饥饿档位里，状态是否还能区分行为
    hb = np.digitize(hunger, [0.3, 0.6])
    ctx = (loc * 3 + hb)
    within = []
    for cval in np.unique(ctx[ok]):
        sel = ok & (ctx == cval)
        if sel.sum() > 200 and len(np.unique(lab[sel])) > 1:
            within.append((sel.sum(), mi(lab[sel], act[sel]),
                           np.mean([mi(rng.permutation(lab[sel]), act[sel]) for _ in range(10)])))
    w_real = float(sum(n * a for n, a, _ in within) / max(1, sum(n for n, _, _ in within)))
    w_shuf = float(sum(n * b for n, _, b in within) / max(1, sum(n for n, _, _ in within)))
    return dict(k=k, silhouette=float(sil), start_day=start_day, states=states, triggers=triggers,
                mi_state_action=mi_real, mi_shuffled=mi_shuf,
                mi_within_context=w_real, mi_within_shuffled=w_shuf), full


if __name__ == "__main__":
    R = dict(np.load("log_C1.npz"))
    summ = json.load(open("summary_C1.json"))
    start = int(np.ceil(summ["epi"]["day"])) + 1 if summ.get("epi") else 5
    res, lab = analyze(R, start)
    json.dump(res, open("states.json", "w"), ensure_ascii=False, indent=1)
    np.save("state_labels.npy", lab)
    print("k =", res["k"], "silhouette", round(res["silhouette"], 3))
    print("状态-行为互信息", round(res["mi_state_action"], 3), "打乱基线", round(res["mi_shuffled"], 3))
    print("同情境内互信息", round(res["mi_within_context"], 3), "打乱基线", round(res["mi_within_shuffled"], 3))
    for s in res["states"]:
        top = np.argsort(s["actions"])[::-1][:3]
        print(f"状态{s['id']} 占{s['share']:.2f} 停留{s['dwell_min']:.0f}分钟 饿{s['hunger']:.2f} 伤{s['damage']:.2f} "
              f"用户{s['user']:.2f} 光标{s['cursor']:.2f} 暖{s['warmth']:.2f} 挤{s['crowd']:.2f} 气味{s['smell']:.1f} | "
              + " ".join(f"{ACTIONS_EXT[i]}{s["actions"][i]:.2f}" for i in top)
              + " | 地点 " + " ".join(f"{LOCS[i]}{s['locs'][i]:.2f}" for i in range(5) if s['locs'][i] > 0.05))
    for ev, d in res["triggers"].items():
        print(ev, d["n"], [round(x, 2) for x in d["lift"]])
