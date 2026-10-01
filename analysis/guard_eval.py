#!/usr/bin/env python3
"""guard_eval.py — offline evaluation of the value-surprisal guard from shadow-pairs.jsonl.

Reads the (signal trace, truth label) pairs written by bench/shadow-run.mjs and prints every
number reported in the paper's guard section: judge fidelity, edge accuracy, AUROC of the
candidate decision statistics, and the held-out operating points. Deterministic: the
train/eval split is a seeded shuffle STRATIFIED BY WORKLOAD (70/30), thresholds are fit on
train only and applied unchanged to eval.

    python3 analysis/guard_eval.py [bench/results-zen5-run2/shadow-pairs.jsonl]
"""
import json, sys, math, random, bisect, collections

PATH = sys.argv[1] if len(sys.argv) > 1 else 'bench/results-zen5-run2/shadow-pairs.jsonl'
rows = [json.loads(l) for l in open(PATH)]
header, pairs = rows[0], rows[1:]
assert header.get('kind') == 'provenance'
LABEL = 'correct_truth' if all('correct_truth' in p for p in pairs) else 'correct'

def wilson(k, n, z=1.959964):
    if n == 0: return (float('nan'), float('nan'))
    p = k / n; d = 1 + z*z/n; c = (p + z*z/(2*n))/d; h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return (max(0, c-h), min(1, c+h))
def fmt(k, n): lo, hi = wilson(k, n); return f"{100*k/n:5.1f}% ({k}/{n}) [{100*lo:.1f}, {100*hi:.1f}]" if n else "  n/a"
STAT = {
    'window-mean (first 6)': lambda s: sum(s[:6]) / min(6, len(s)),
    'mean-all':              lambda s: sum(s) / len(s),
    'max-all':               lambda s: max(s),
}
def auroc(sc, err):
    pos = [s for s, e in zip(sc, err) if e]; neg = sorted(s for s, e in zip(sc, err) if not e)
    if not pos or not neg: return float('nan')
    return sum(bisect.bisect_left(neg, p) + 0.5*(bisect.bisect_right(neg, p) - bisect.bisect_left(neg, p)) for p in pos) / (len(pos)*len(neg))
def threshold(scores, rate):
    """escalate the top `rate` fraction under strict '>' (midpoint between boundary points)"""
    s = sorted(scores); n = len(s); cut = int((1-rate)*n)
    return s[0]-1 if cut <= 0 else s[-1]+1 if cut >= n else (s[cut-1]+s[cut])/2

# ── split: stratified by workload, seeded ──
rng = random.Random(20261002); train, ev = [], []
for kind in sorted({p['kind'] for p in pairs}):
    ks = sorted((p for p in pairs if p['kind'] == kind), key=lambda p: p['req_id']); rng.shuffle(ks)
    n_ev = round(0.3*len(ks)); ev += ks[:n_ev]; train += ks[n_ev:]
kinds = sorted({p['kind'] for p in pairs})
print(f"source: {PATH}\nedge: {header['edge_model']['id']} (vLLM {header['edge_model']['vllm']}, {header['edge_model']['gpu']})   judge: {header['judge']['provider']}/{header['judge']['id']}")
print(f"labels: {'constructed truth' if LABEL == 'correct_truth' else 'frontier judge'}   n={len(pairs)}   train={len(train)}  eval={len(ev)}  (stratified 70/30 by workload, seed 20261002)\n")

if LABEL == 'correct_truth':
    print("== judge fidelity (judge vs constructed truth) ==")
    print("  all        ", fmt(sum(p['judge_truth'] for p in pairs), len(pairs)))
    for k in kinds: sub = [p for p in pairs if p['kind'] == k]; print(f"  {k:14s}", fmt(sum(p['judge_truth'] for p in sub), len(sub)))
    print("  judge-relative label == truth label:", fmt(sum(p['correct'] == p['correct_truth'] for p in pairs), len(pairs)))
print("\n== edge accuracy (all 500) ==")
print("  all        ", fmt(sum(p[LABEL] for p in pairs), len(pairs)))
for k in kinds: sub = [p for p in pairs if p['kind'] == k]; print(f"  {k:14s}", fmt(sum(p[LABEL] for p in sub), len(sub)))
if 'style' in pairs[0]:
    for st in sorted({p['style'] for p in pairs}): sub = [p for p in pairs if p['style'] == st]; print(f"  style {st:10s}", fmt(sum(p[LABEL] for p in sub), len(sub)))
if 'diff_edge' in pairs[0]:
    c = collections.Counter((p['kind'], k) for p in pairs for k in p['diff_edge'])
    print("  wrong fields:", ", ".join(f"{k[1]} {v}" for k, v in c.most_common()))

print("\n== AUROC for detecting an edge error (higher = better; 0.5 = chance) ==")
print(f"  {'statistic':24s} {'eval all':>9s} " + " ".join(f"{k[:10]:>11s}" for k in kinds) + f" {'| all 500':>10s}")
for name, f in STAT.items():
    row = [auroc([f(p['s']) for p in ev], [not p[LABEL] for p in ev])]
    row += [auroc([f(p['s']) for p in ev if p['kind'] == k], [not p[LABEL] for p in ev if p['kind'] == k]) for k in kinds]
    row += [auroc([f(p['s']) for p in pairs], [not p[LABEL] for p in pairs])]
    print(f"  {name:24s} {row[0]:9.3f} " + " ".join(f"{x:11.3f}" for x in row[1:-1]) + f" {row[-1]:10.3f}")

def operate(stat, rate, per_kind):
    f = STAT[stat]
    thr = {k: threshold([f(p['s']) for p in train if (p['kind'] == k or not per_kind)], rate) for k in kinds}
    esc = [f(p['s']) > thr[p['kind']] for p in ev]; err = [not p[LABEL] for p in ev]
    n = len(ev); E = sum(err); ne = sum(esc); kept = n - ne
    return thr, ne, n, sum(1 for e, x in zip(esc, err) if not e and not x), kept, sum(1 for e, x in zip(esc, err) if e and x), E
print("\n== held-out operating points (thresholds fit on train, applied to eval) ==")
base = sum(p[LABEL] for p in ev)
print(f"  no guard: local precision = edge accuracy = {fmt(base, len(ev))}")
for stat in ('mean-all', 'window-mean (first 6)'):
    for per_kind in (True, False):
        print(f"\n  statistic = {stat};  threshold = {'per workload' if per_kind else 'single, pooled'}")
        print(f"  {'target':>6s}  {'escalated':>28s}  {'local precision':>30s}  {'error recall':>28s}")
        for rate in (0.1, 0.2, 0.3, 0.4, 0.5):
            thr, ne, n, kc, kept, ec, E = operate(stat, rate, per_kind)
            print(f"  {int(rate*100):5d}%  {fmt(ne, n):>28s}  {fmt(kc, kept):>30s}  {fmt(ec, E):>28s}" + (f"   thr={ {k[:4]: round(v, 4) for k, v in thr.items()} }" if rate == 0.2 else ""))

# ── isotonic calibration of the signal (P(correct) non-increasing in surprisal) ──
def pava_decreasing(xs, ys):
    pts = collections.OrderedDict()
    for x, y in sorted(zip(xs, ys)):                       # pool ties in x first
        s, w = pts.get(x, (0.0, 0)); pts[x] = (s + y, w + 1)
    blocks = [[s, w, x] for x, (s, w) in pts.items()]      # [sum_y, weight, x_left]
    i = 0
    while i < len(blocks) - 1:                             # enforce mean non-increasing
        if blocks[i][0]/blocks[i][1] < blocks[i+1][0]/blocks[i+1][1]:
            blocks[i][0] += blocks[i+1][0]; blocks[i][1] += blocks[i+1][1]; del blocks[i+1]; i = max(i-1, 0)
        else: i += 1
    return [(b[2], b[0]/b[1]) for b in blocks]
def predict(model, x):
    xs = [m[0] for m in model]; j = bisect.bisect_right(xs, x) - 1
    return model[max(j, 0)][1]
def ece(probs, labels, M=10):
    bins = [[] for _ in range(M)]
    for p, y in zip(probs, labels): bins[min(M-1, int(p*M))].append((p, y))
    return sum(len(b)/len(probs) * abs(sum(y for _, y in b)/len(b) - sum(p for p, _ in b)/len(b)) for b in bins if b)
f = STAT['mean-all']
model = pava_decreasing([f(p['s']) for p in train], [1 if p[LABEL] else 0 for p in train])
pe = [predict(model, f(p['s'])) for p in ev]; ye = [1 if p[LABEL] else 0 for p in ev]
base_rate = sum(1 if p[LABEL] else 0 for p in train) / len(train)
print(f"\n== calibration of the mean-all signal (isotonic fit on train, evaluated on eval) ==")
print(f"  held-out ECE (10 bins): {ece(pe, ye):.3f}   vs constant train base rate: {ece([base_rate]*len(ye), ye):.3f}   isotonic steps: {len(model)}")
