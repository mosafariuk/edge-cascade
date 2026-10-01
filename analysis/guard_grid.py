#!/usr/bin/env python3
"""guard_grid.py — compare guard configurations from bench/optimize-guard.mjs.

For every (temperature, few-shot) configuration and every candidate decision statistic it
reports, on the SAME stratified 70/30 split used for paper Table VIII:
  - edge accuracy (a config that detects errors better by making more of them is not a win)
  - AUROC on the training and held-out splits
  - at thresholds fit on TRAIN for a 20% and a 30% escalation target: held-out escalation
    rate, local precision, error recall, and the SILENT-ERROR RATE = wrong-and-kept-local
    as a share of all held-out records — the quantity a deployment actually pays for.

Selection rule (fixed before looking at held-out data): the winner is the configuration ×
statistic with the lowest TRAIN silent-error rate at the 20% target (ties → higher train
AUROC). Its held-out numbers are then compared with the baseline (T0_shot0, mean surprisal)
by a paired bootstrap over held-out records. Adopt it only if the 95% interval of the
improvement excludes zero.

    python3 analysis/guard_grid.py [bench/results-zen5-run2/guard-grid.jsonl]
"""
import json, sys, math, random, bisect, collections

PATH = sys.argv[1] if len(sys.argv) > 1 else 'bench/results-zen5-run2/guard-grid.jsonl'
rows = [json.loads(l) for l in open(PATH)]
header, recs = rows[0], rows[1:]
assert header.get('kind') == 'guard-grid'
if header.get('dry_run'): print("*** DRY RUN FILE — numbers are synthetic and meaningless ***\n")

def wilson(k, n, z=1.959964):
    if n == 0: return (float('nan'), float('nan'))
    p = k/n; d = 1 + z*z/n; c = (p + z*z/(2*n))/d; h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return (max(0, c-h), min(1, c+h))
def auroc(sc, err):
    pos = [s for s, e in zip(sc, err) if e]; neg = sorted(s for s, e in zip(sc, err) if not e)
    if not pos or not neg: return float('nan')
    return sum(bisect.bisect_left(neg, p) + 0.5*(bisect.bisect_right(neg, p) - bisect.bisect_left(neg, p)) for p in pos) / (len(pos)*len(neg))
def threshold(scores, rate):
    s = sorted(scores); n = len(s); cut = int((1-rate)*n)
    return s[0]-1 if cut <= 0 else s[-1]+1 if cut >= n else (s[cut-1]+s[cut])/2
BIG = 1e6   # a record with no usable trace (parse/schema failure) is always escalated
STAT = {
    'mean surprisal':  lambda r: sum(r['s'])/len(r['s']) if r['s'] and r['status'] == 'done' else BIG,
    'max surprisal':   lambda r: max(r['s']) if r['s'] and r['status'] == 'done' else BIG,
    'mean entropy@k':  lambda r: sum(r['h'])/len(r['h']) if r['h'] and r['status'] == 'done' else BIG,
    'min margin (neg)': lambda r: -min(r['m']) if r['m'] and r['status'] == 'done' else BIG,
}

# ── the Table VIII split: stratified by workload, seeded shuffle, 30% held out ──
ids = {}
for r in recs: ids.setdefault(r['kind'], set()).add(r['req_id'])
rng = random.Random(20261002); ev_ids = set()
for kind in sorted(ids):
    ks = sorted(ids[kind]); rng.shuffle(ks); ev_ids |= set(ks[:round(0.3*len(ks))])
by_cfg = collections.OrderedDict()
for r in recs: by_cfg.setdefault(r['cfg'], []).append(r)
n_all = len(next(iter(by_cfg.values()))); n_ev = sum(1 for r in next(iter(by_cfg.values())) if r['req_id'] in ev_ids)
print(f"source: {PATH}\nedge: {header['edge_model']['id']}  corpus: {header['corpus']} ({n_all} records)  split: train={n_all-n_ev} eval={n_ev} (same as Table VIII)\n")

def evaluate(rs, stat, rate):
    f = STAT[stat]; tr = [r for r in rs if r['req_id'] not in ev_ids]; ev = [r for r in rs if r['req_id'] in ev_ids]
    thr = threshold([f(r) for r in tr], rate)
    def op(part):
        esc = [f(r) > thr for r in part]; err = [not r['correct_truth'] for r in part]
        n = len(part); E = sum(err); ne = sum(esc); caught = sum(1 for a, b in zip(esc, err) if a and b)
        keptok = sum(1 for a, b in zip(esc, err) if not a and not b)
        return dict(n=n, esc=ne, kept=n-ne, keptok=keptok, E=E, caught=caught, silent=E-caught)
    return dict(thr=thr, tr=op(tr), ev=op(ev),
                auc_tr=auroc([f(r) for r in tr], [not r['correct_truth'] for r in tr]),
                auc_ev=auroc([f(r) for r in ev], [not r['correct_truth'] for r in ev]))

print("== per configuration (statistic = mean surprisal, the deployed guard) ==")
print(f"{'config':14s} {'acc all':>8s} {'acc eval':>8s} {'fail':>4s} | {'AUROC tr':>8s} {'AUROC ev':>8s} | "
      f"{'@20%: esc':>9s} {'prec':>6s} {'recall':>6s} {'SILENT':>7s} | {'@30%: esc':>9s} {'prec':>6s} {'recall':>6s} {'SILENT':>7s}")
table = {}
for cfg, rs in by_cfg.items():
    acc = sum(r['correct_truth'] for r in rs)/len(rs); ev = [r for r in rs if r['req_id'] in ev_ids]
    acc_ev = sum(r['correct_truth'] for r in ev)/len(ev); fail = sum(r['status'] != 'done' for r in rs)
    line = f"{cfg:14s} {100*acc:7.1f}% {100*acc_ev:7.1f}% {fail:4d} | "
    for stat in STAT:
        for rate in (0.2, 0.3): table[(cfg, stat, rate)] = evaluate(rs, stat, rate)
    e2, e3 = table[(cfg, 'mean surprisal', 0.2)], table[(cfg, 'mean surprisal', 0.3)]
    line += f"{e2['auc_tr']:8.3f} {e2['auc_ev']:8.3f} | "
    for e in (e2, e3):
        o = e['ev']; line += f"{100*o['esc']/o['n']:8.1f}% {100*o['keptok']/max(1,o['kept']):5.1f}% {100*o['caught']/max(1,o['E']):5.1f}% {100*o['silent']/o['n']:6.1f}% | "
    print(line.rstrip(' |'))
print("  SILENT = wrong AND kept local, as % of all held-out records (lower is better; no guard = 100% − accuracy).")

print("\n== held-out AUROC by statistic ==")
print(f"{'config':14s} " + " ".join(f"{s:>17s}" for s in STAT))
for cfg in by_cfg: print(f"{cfg:14s} " + " ".join(f"{table[(cfg, s, 0.2)]['auc_ev']:17.3f}" for s in STAT))

# ── does temperature change REPORTED logprobs, or only which token is sampled? ──
base = {r['req_id']: r for r in by_cfg.get('T0_shot0', [])}
if base:
    print("\n== temperature check: records whose value tokens are IDENTICAL to T0_shot0 ==")
    for cfg, rs in by_cfg.items():
        if cfg == 'T0_shot0' or not cfg.endswith('shot0'): continue
        same = [(r, base[r['req_id']]) for r in rs if r['req_id'] in base and r['toks'] == base[r['req_id']]['toks'] and r['toks']]
        if not same: print(f"  {cfg}: no token-identical records"); continue
        d = [abs(a - b) for r, b0 in same for a, b in zip(r['s'], b0['s'])]
        print(f"  {cfg}: {len(same)}/{len(rs)} token-identical; mean |Δ surprisal| per value token = {sum(d)/len(d):.5f} nats"
              + ("   → identical: logprobs are PRE-temperature, T only changes sampling" if sum(d)/len(d) < 1e-4
                 else "   → negligible (< 0.01 nats): T has no material effect on reported logprobs" if sum(d)/len(d) < 0.01
                 else "   → logprobs are POST-temperature"))

# ── selection on TRAIN, report on EVAL, paired bootstrap vs baseline ──
def train_key(k): e = table[k]; return (e['tr']['silent']/e['tr']['n'], -e['auc_tr'])
cands = [k for k in table if k[2] == 0.2]
win = min(cands, key=train_key); BASE = ('T0_shot0', 'mean surprisal', 0.2)
print("\n== selection (lowest TRAIN silent-error rate at the 20% target) ==")
for name, k in (('baseline', BASE), ('selected', win)):
    if k not in table: print(f"  {name}: {k[0]} not in this file"); continue
    e = table[k]; o = e['ev']; lo, hi = wilson(o['keptok'], o['kept']); rlo, rhi = wilson(o['caught'], o['E'])
    print(f"  {name}: {k[0]} / {k[1]}   train silent {100*e['tr']['silent']/e['tr']['n']:.1f}%  | HELD-OUT: AUROC {e['auc_ev']:.3f}  escalated {100*o['esc']/o['n']:.1f}%  "
          f"precision {100*o['keptok']/max(1,o['kept']):.1f}% [{100*lo:.1f}, {100*hi:.1f}]  recall {100*o['caught']/max(1,o['E']):.1f}% [{100*rlo:.1f}, {100*rhi:.1f}]  silent {100*o['silent']/o['n']:.1f}%")
if BASE in table and win != BASE:
    fb, fw = STAT[BASE[1]], STAT[win[1]]
    eb = {r['req_id']: r for r in by_cfg[BASE[0]] if r['req_id'] in ev_ids}; ew = {r['req_id']: r for r in by_cfg[win[0]] if r['req_id'] in ev_ids}
    keys = sorted(eb); tb, tw = table[BASE]['thr'], table[win]['thr']; rb = random.Random(7); dA, dS = [], []
    for _ in range(4000):
        samp = [keys[rb.randrange(len(keys))] for _ in keys]
        a_b = auroc([fb(eb[k]) for k in samp], [not eb[k]['correct_truth'] for k in samp]); a_w = auroc([fw(ew[k]) for k in samp], [not ew[k]['correct_truth'] for k in samp])
        s_b = sum(1 for k in samp if not eb[k]['correct_truth'] and not fb(eb[k]) > tb)/len(samp); s_w = sum(1 for k in samp if not ew[k]['correct_truth'] and not fw(ew[k]) > tw)/len(samp)
        if not (math.isnan(a_b) or math.isnan(a_w)): dA.append(a_w - a_b)
        dS.append(s_w - s_b)
    q = lambda v, p: sorted(v)[int(p*len(v))]
    print(f"\n  paired bootstrap on held-out (4000 resamples), selected − baseline:")
    print(f"    ΔAUROC            = {sum(dA)/len(dA):+.3f}   95% [{q(dA, .025):+.3f}, {q(dA, .975):+.3f}]")
    print(f"    Δ silent-error    = {100*sum(dS)/len(dS):+.1f} pts  95% [{100*q(dS, .025):+.1f}, {100*q(dS, .975):+.1f}]   (negative = fewer silent errors)")
    ok = q(dS, .975) < 0
    print(f"  VERDICT: {'IMPROVEMENT — interval excludes zero; confirm on a fresh corpus before changing the paper' if ok else 'NOT DISTINGUISHABLE from the baseline on held-out data — the paper stays as is'}")
print(f"\nNote: {len(by_cfg)} configurations × {len(STAT)} statistics were compared; only the pre-registered selection above is a valid claim.")
