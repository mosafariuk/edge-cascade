// audit-sample.mjs — draw a human-audit subset from shadow-pairs.jsonl to bound
// FRONTIER-JUDGE FIDELITY (Threats §iii). Stratifies around the decision threshold
// so the audited records are the ones where the guard's call actually mattered, then
// emits a CSV a human labels; audit-score.mjs (below) compares human vs. judge.
//
//   node bench/audit-sample.mjs [N=50] [MAX_SURPRISAL=0]
// Reads bench/shadow-pairs.jsonl, writes bench/audit-todo.csv (blank `human_correct`).
import { readFileSync, writeFileSync, existsSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const N = Number(process.argv[2] || 50);
const THR = Number(process.argv[3] || process.env.MAX_SURPRISAL ?? 0);
const src = join(here, 'shadow-pairs.jsonl');
if (!existsSync(src)) { console.error('run bench/shadow-run.mjs first to produce shadow-pairs.jsonl'); process.exit(1); }

// line 1 is the provenance header written by shadow-run.mjs; only pair lines are data
const pairs = readFileSync(src, 'utf8').trim().split('\n').filter(Boolean).map((l) => JSON.parse(l))
  .filter((o) => o.kind !== 'provenance');
// stratify: half nearest below the threshold (kept-local, borderline), half nearest above
const below = pairs.filter((p) => p.signal != null && p.signal <= THR).sort((a, b) => (THR - a.signal) - (THR - b.signal));
const above = pairs.filter((p) => p.signal != null && p.signal > THR).sort((a, b) => (a.signal - THR) - (b.signal - THR));
const pick = [...below.slice(0, Math.ceil(N / 2)), ...above.slice(0, Math.floor(N / 2))];

const esc = (s) => `"${String(s).replace(/"/g, '""')}"`;
const rows = [['req_id', 'kind', 'signal', 'judge_correct', 'human_correct'].join(',')];
for (const p of pick) rows.push([esc(p.req_id), esc(p.kind), p.signal.toFixed(4), p.correct ? 1 : 0, ''].join(','));
writeFileSync(join(here, 'audit-todo.csv'), rows.join('\n') + '\n');
console.log(`wrote ${pick.length} records to bench/audit-todo.csv — fill the human_correct column (1/0), then run audit-score.mjs`);
