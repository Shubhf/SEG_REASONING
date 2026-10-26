import glob, json, os, sys
from collections import defaultdict

FALLBACK = {  # task type -> list of key paths tried in order (inside the split dict)
    "STS": [("cos_sim", "spearman")],
    "Classification": [("accuracy",)],
    "Clustering": [("v_measure",)],
    "PairClassification": [("cos_sim", "ap")],
    "Reranking": [("map",)],
    "Retrieval": [("ndcg_at_10",)],
}

def dig(d, path):
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d if isinstance(d, (int, float)) else None

def score_of(split, ttype):
    cands = [split] + [split[k] for k in ("en-en", "en", "default") if isinstance(split.get(k), dict)]
    for c in cands:
        if isinstance(c.get("main_score"), (int, float)):
            return c["main_score"]
    for c in cands:
        for path in FALLBACK.get(ttype, []):
            v = dig(c, path)
            if v is not None:
                return v
    return None

def load_run(root):
    res = defaultdict(dict)  # (ttype, task) -> {dim: score}
    for f in glob.glob(os.path.join(root, "dim*", "*", "*.json")):
        parts = f.split(os.sep)
        dim = int(parts[-3][3:]); ttype = parts[-2]; task = os.path.basename(f)[:-5]
        if task == "model_meta":
            continue
        data = json.load(open(f))
        split = data.get("test")
        if not isinstance(split, dict):
            continue
        s = score_of(split, ttype)
        if s is not None:
            res[(ttype, task)][dim] = 100 * s
    return res

def main(roots):
    runs = {os.path.basename(r.rstrip("/")): load_run(r) for r in roots}
    dims = sorted({d for r in runs.values() for t in r.values() for d in t}, reverse=True)
    for name, res in runs.items():
        print(f"\n=== {name} ===")
        print(f"{'type':<20}{'task':<34}" + "".join(f"{d:>8}" for d in dims))
        for (tt, task), sc in sorted(res.items()):
            print(f"{tt:<20}{task:<34}" + "".join(f"{sc[d]:>8.2f}" if d in sc else f"{'-':>8}" for d in dims))
        full = [d for d in dims if all(d in sc for sc in res.values())]
        print(f"{'AVERAGE (tasks with all dims)':<54}" + "".join(
            f"{sum(sc[d] for sc in res.values())/len(res):>8.2f}" if d in full else f"{'-':>8}" for d in dims))
        print(f"tasks found: {len(res)}")

main(sys.argv[1:])
