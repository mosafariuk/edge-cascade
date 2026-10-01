#!/usr/bin/env python3
"""guard_confirm.py — PRE-REGISTERED confirmation analysis (committed before the data existed).

Hypothesis under test, from the exploratory grid (results-zen5-run2/CONCLUSIONS.md §8):
  H1  Few-shot prompt conventions are the first-order lever on edge accuracy.
  H2  The value-surprisal guard is a second-order filter on the residual errors.

Design (all fixed here, none chosen after seeing the confirmation data):
  corpus      payloads/synthetic-confirm — 2,000 records, seed 20261004, generated after the
              few-shot examples were frozen; constructed truth, mechanically verified texts
  arms        T = 0 with 0-shot, 1-shot, 2-shot (the frozen examples in bench/optimize-guard.mjs)
  statistic   mean surprisal over all value tokens (the deployed guard). Nothing else.
  split       stratified by workload, seeded shuffle (20261004), 70% train / 30% held-out
  threshold   fit on TRAIN, per arm, at a 20% escalation target; applied unchanged to held-out.
              Surprisal is exactly 0 for many records, so the quantile may sit inside a tie;
              the achieved escalation rate is reported as measured.
  metrics     held-out: edge accuracy, error count, AUROC, escalation rate, local precision,
              error recall, silent-error rate (wrong AND kept local, share of all records),
              and the same without a guard. Wilson 95% intervals.
  tests       paired bootstrap over held-out records (10,000 resamples, seed 20261004):
                H1: accuracy(k-shot) − accuracy(0-shot),      k = 1, 2
                H2: silent-error with guard − without guard,   within each arm
                    silent-error with guard, k-shot − 0-shot
  decision    adopt k-shot as the deployed prompt iff its held-out accuracy gain over 0-shot
              has a 95% interval excluding zero; between 1 and 2 shots prefer the lower
              held-out silent-error rate with the guard (tie → fewer shots).

    python3 analysis/guard_confirm.py [bench/results-zen5-run2/guard-confirm.jsonl]
"""
import json, sys, math, random, bisect, collections

PATH = sys.argv[1] if len(sys.argv) > 1 else 'bench/results-zen5-run2/guard-confirm.jsonl'
rows = [json.loads(l) for l in open(PATH)]
header, recs = rows[0], rows[1:]
assert header.get('kind') == 'guard-grid'
ARMS = ['T0_shot0', 'T0_shot1', 'T0_shot2']; TARGET = 0.20; SPLIT_SEED = 20261004

def wilson(k, n, z=1.959964):
    if n == 0: return (float('nan'), float('nan'))
    p = k/n; d = 1 + z*z/n; c = (p + z*z/(2*n))/d; h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return (max(0, c-h), min(1, c+h))
def pc(k, n): lo, hi = wilson(k, n); return f"{100*k/n:5.1f}% ({k}/{n}) [{100*lo:.1f}, {100*hi:.1f}]" if n else "   n/a"
def auroc(sc, err):
    pos = [s for s, e in zip(sc, err) if e]; neg = sorted(s for s, e in zip(sc, err) if not e)
    if not pos or not neg: return float('nan')
    return sum(bisect.bisect_left(neg, p) + 0.5*(bisect.bisect_right(neg, p) - bisect.bisect_left(neg, p)) for p in pos) / (len(pos)*len(neg))
def threshold(scores, rate):
    s = sorted(scores); n = len(s); cut = int((1-rate)*n)
    return s[0]-1 if cut <= 0 else s[-1]+1 if cut >= n else (s[cut-1]+s[cut])/2
BIG = 1e6
sig = lambda r: sum(r['s'])/len(r['s']) if r['s'] and r['status'] == 'done' else BIG   # unusable trace → escalate

by = {a: {} for a in ARMS}
for r in recs:
    if r['cfg'] in by: by[r['cfg']][r['req_id']] = r
ids = collections.defaultdict(set)
for r in by[ARMS[0]].values(): ids[r['kind']].add(r['req_id'])
rng = random.Random(SPLIT_SEED); ev = []
for kind in sorted(ids):
    ks = sorted(ids[kind]); rng.shuffle(ks); ev += ks[:round(0.3*len(ks))]
ev = sorted(ev); evs = set(ev); n_all = len(by[ARMS[0]])
for a in ARMS: assert set(by[a]) == set(by[ARMS[0]]), f"arm {a} does not cover the same records"
print(f"source: {PATH}   dry_run={header.get('dry_run')}\nedge: {header['edge_model']['id']} (vLLM {header['edge_model']['vllm']})   corpus: {header['corpus']}   n={n_all}  train={n_all-len(ev)}  held-out={len(ev)}\n")

res = {}
for a in ARMS:
    tr = [r for k, r in by[a].items() if k not in evs]; te = [by[a][k] for k in ev]
    thr = threshold([sig(r) for r in tr], TARGET)
    esc = {k: sig(by[a][k]) > thr for k in ev}; err = {k: not by[a][k]['correct_truth'] for k in ev}
    res[a] = dict(thr=thr, esc=esc, err=err, te=te,
                  auc=auroc([sig(r) for r in te], [not r['correct_truth'] for r in te]),
                  auc_tr=auroc([sig(r) for r in tr], [not r['correct_truth'] for r in tr]),
                  acc_all=sum(r['correct_truth'] for r in by[a].values()), fail=sum(r['status'] != 'done' for r in by[a].values()))

print("== held-out metrics per arm (threshold fit on train at the 20% target) ==")
for a in ARMS:
    R = res[a]; n = len(ev); E = sum(R['err'].values()); ne = sum(R['esc'].values())
    caught = sum(1 for k in ev if R['esc'][k] and R['err'][k]); kept = n - ne; keptok = sum(1 for k in ev if not R['esc'][k] and not R['err'][k])
    print(f"\n{a}   (all {n_all}: accuracy {pc(R['acc_all'], n_all)};  non-parsed {R['fail']})")
    print(f"  edge accuracy        {pc(n-E, n)}        errors: {E}")
    for kind in sorted(ids):
        sub = [k for k in ev if by[a][k]['kind'] == kind]; print(f"    {kind:15s}    {pc(sum(1 for k in sub if not R['err'][k]), len(sub))}")
    print(f"  AUROC (held-out)     {R['auc']:.3f}    (train {R['auc_tr']:.3f})     s_max = {R['thr']:.4f} nats")
    print(f"  escalated            {pc(ne, n)}")
    print(f"  local precision      {pc(keptok, kept)}")
    print(f"  error recall         {pc(caught, E)}")
    print(f"  silent errors        with guard {pc(E-caught, n)}    |  without guard {pc(E, n)}")
    c = collections.Counter(f for r in R['te'] for f in r['diff_edge']); print("  wrong fields (held-out): " + (", ".join(f"{k} {v}" for k, v in c.most_common()) or "none"))

rb = random.Random(SPLIT_SEED); B = 10000
def boot(f):
    v = []
    for _ in range(B):
        s = [ev[rb.randrange(len(ev))] for _ in ev]; v.append(f(s))
    v.sort(); return sum(v)/len(v), v[int(.025*B)], v[int(.975*B)]
def line(name, m, lo, hi, unit='pts'): print(f"  {name:58s} {100*m:+6.1f} {unit}  95% [{100*lo:+.1f}, {100*hi:+.1f}]   {'*' if lo > 0 or hi < 0 else 'n.s.'}")
acc = lambda a, s: sum(1 for k in s if not res[a]['err'][k])/len(s)
silent = lambda a, s: sum(1 for k in s if res[a]['err'][k] and not res[a]['esc'][k])/len(s)
noguard = lambda a, s: sum(1 for k in s if res[a]['err'][k])/len(s)
print("\n== paired bootstrap on held-out records (10,000 resamples); * = 95% interval excludes zero ==")
print("H1 — few-shot as first-order lever (edge accuracy):")
for a in ARMS[1:]: line(f"accuracy {a} − T0_shot0", *boot(lambda s, a=a: acc(a, s) - acc(ARMS[0], s)))
line("accuracy T0_shot2 − T0_shot1", *boot(lambda s: acc(ARMS[2], s) - acc(ARMS[1], s)))
print("H2 — guard as second-order filter (silent-error rate):")
for a in ARMS: line(f"{a}: with guard − without guard", *boot(lambda s, a=a: silent(a, s) - noguard(a, s)))
for a in ARMS[1:]: line(f"with guard: {a} − T0_shot0", *boot(lambda s, a=a: silent(a, s) - silent(ARMS[0], s)))
line("with guard: T0_shot2 − T0_shot1", *boot(lambda s: silent(ARMS[2], s) - silent(ARMS[1], s)))

h1 = {a: boot(lambda s, a=a: acc(a, s) - acc(ARMS[0], s)) for a in ARMS[1:]}
ok = [a for a in ARMS[1:] if h1[a][1] > 0]
print("\n== decision (pre-registered) ==")
if not ok: print("  No few-shot arm improves held-out accuracy with an interval excluding zero → keep 0-shot; H1 not confirmed.")
else:
    full = {a: silent(a, ev) for a in ok}; best = min(ok, key=lambda a: (round(full[a], 6), ARMS.index(a)))
    print(f"  H1 confirmed for: {', '.join(ok)}.  Deploy: {best}  (held-out silent-error with guard {100*full[best]:.1f}%, s_max = {res[best]['thr']:.4f} nats)")
