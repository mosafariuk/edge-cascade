#!/usr/bin/env python3
"""
From-scratch isotonic-regression calibration for a router confidence score.

CLAIM UNDER TEST: raw router confidence is miscalibrated (ECE ~0.12); a monotone
isotonic map fit on labeled (score, correct?) pairs drops ECE to ~0.03.

IMPORTANT MODELING NOTE (correcting the premise of the question):
Isotonic regression is NOT fit against the embedding vector. It is a 1-D monotone
map  g: s -> p  from the router's SCALAR uncalibrated score s in [0,1] to a
calibrated probability p. The vector produces s (via the classifier); calibration
acts on s alone. That is why it is O(log n) per query and needs little data.

Pure stdlib. Deterministic (seeded).
"""
import random, math, bisect

rnd = random.Random(7)

# ---- synthesize a realistic miscalibrated router ------------------------------
# Latent difficulty u ~ U(0,1). True P(local model correct | u) = u.
# The router REPORTS an overconfident score s = u**0.5 (concave -> inflates).
# label y ~ Bernoulli(u).  So reported s systematically exceeds true accuracy.
def make(n):
    S, Y = [], []
    for _ in range(n):
        u = rnd.random()
        s = u**0.5                      # overconfident reported score
        s = min(1.0, max(0.0, s + rnd.gauss(0, 0.03)))  # sensor noise
        y = 1 if rnd.random() < u else 0
        S.append(s); Y.append(y)
    return S, Y

def ece(scores, labels, M=15):
    """Expected Calibration Error with M equal-width bins."""
    bins = [[] for _ in range(M)]
    for s, y in zip(scores, labels):
        b = min(M-1, int(s*M))
        bins[b].append((s, y))
    n = len(scores); e = 0.0
    for b in bins:
        if not b: continue
        conf = sum(s for s, _ in b)/len(b)
        acc  = sum(y for _, y in b)/len(b)
        e += (len(b)/n)*abs(acc-conf)
    return e

def pava(x, y):
    """
    Pool Adjacent Violators. Fit non-decreasing step fn minimizing sum (y-g(x))^2.
    Returns (knots_x, knots_y) defining a right-continuous step function.
    O(n) after the O(n log n) sort. This is the entire 'training' cost.
    """
    pairs = sorted(zip(x, y))
    xs = [p[0] for p in pairs]
    # blocks: [sum_y, weight, left_x]
    blocks = []
    for xi, yi in pairs:
        blocks.append([yi, 1.0, xi])
        # merge while monotonicity violated
        while len(blocks) >= 2 and (blocks[-2][0]/blocks[-2][1]) > (blocks[-1][0]/blocks[-1][1]):
            s2, w2, l2 = blocks.pop()
            s1, w1, l1 = blocks.pop()
            blocks.append([s1+s2, w1+w2, l1])
    knots_x, knots_y = [], []
    for s, w, lx in blocks:
        knots_x.append(lx); knots_y.append(s/w)
    return knots_x, knots_y

def predict(knots_x, knots_y, s):
    """Calibrated prob for score s: O(log n) binary search. This runs per query."""
    i = bisect.bisect_right(knots_x, s) - 1
    if i < 0: i = 0
    return knots_y[i]

print("="*72)
print("B1. ECE BEFORE vs AFTER isotonic calibration (held-out test set)")
print("="*72)
Strain, Ytrain = make(5000)
Stest,  Ytest  = make(20000)
kx, ky = pava(Strain, Ytrain)
raw_ece = ece(Stest, Ytest)
cal_scores = [predict(kx, ky, s) for s in Stest]
cal_ece = ece(cal_scores, Ytest)
print(f"  raw   ECE (test) = {raw_ece:.4f}")
print(f"  calib ECE (test) = {cal_ece:.4f}   (isotonic fit on 5,000 labeled pairs)")
print(f"  reduction        = {100*(raw_ece-cal_ece)/raw_ece:.0f}%")

print()
print("="*72)
print("B2. HOW MUCH DATA to recalibrate? (test ECE vs training-set size)")
print("   isotonic sup-norm error shrinks at the ~n^(-1/3) cube-root rate")
print("="*72)
print(f"{'train n':>8} {'test ECE':>9}")
for n in [100, 300, 1000, 3000, 10000, 30000]:
    st, yt = make(n)
    kx, ky = pava(st, yt)
    cs = [predict(kx, ky, s) for s in Stest]
    print(f"{n:>8} {ece(cs, Ytest):>9.4f}")

print()
print("="*72)
print("B3. PER-QUERY CALIBRATION COST (wall-clock, pure python, no numpy)")
print("="*72)
import time
kx, ky = pava(*make(5000))
probe = [rnd.random() for _ in range(200000)]
t0 = time.perf_counter()
acc = 0.0
for s in probe:
    acc += predict(kx, ky, s)
dt = time.perf_counter() - t0
print(f"  {len(probe):,} calibration lookups in {dt*1000:.1f} ms"
      f"  ->  {dt/len(probe)*1e9:.0f} ns/query  ({len(probe)/dt/1e6:.2f} M/s)")
print("  (a compiled/bucketed lookup in Rust/C is ~10-50 ns; either way: free)")
