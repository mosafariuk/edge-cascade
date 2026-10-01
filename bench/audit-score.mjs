// audit-score.mjs — after a human fills audit-todo.csv's human_correct column,
// report frontier-JUDGE FIDELITY: how often the frontier adjudicator agreed with the
// human. This bounds the "ground truth is an LLM" threat with a real number.
//   node bench/audit-score.mjs
import { readFileSync, existsSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { wilson } from '../src/calibration.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const f = join(here, 'audit-todo.csv');
if (!existsSync(f)) { console.error('run bench/audit-sample.mjs first'); process.exit(1); }

const lines = readFileSync(f, 'utf8').trim().split('\n');
const hdr = lines[0].split(',');
const ji = hdr.indexOf('judge_correct'), hi = hdr.indexOf('human_correct');
let n = 0, agree = 0, blank = 0;
for (const line of lines.slice(1)) {
  const c = line.split(',');
  const h = (c[hi] ?? '').trim();
  if (h !== '0' && h !== '1') { blank++; continue; }
  n++; if (Number(c[ji]) === Number(h)) agree++;
}
if (!n) { console.error('no human labels found — fill the human_correct column (1/0)'); process.exit(1); }
const [lo, hiCI] = wilson(agree, n);
console.log(`frontier-judge fidelity: ${(100 * agree / n).toFixed(1)}%  (${agree}/${n})  95% CI [${(100*lo).toFixed(1)}, ${(100*hiCI).toFixed(1)}]`);
if (blank) console.log(`(${blank} rows still unlabeled)`);
console.log('→ report this as the judge-fidelity bound in Threats §iii.');
