// validate-samples.mjs — run the inbound admission gate over every dropped sample.
// Confirms each real payload routes correctly and matches its inbound contract
// BEFORE it would ever reach the model. Reports accept/reject with reasons.
import { readdirSync, readFileSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { admit } from '../src/schemas.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const dirs = ['property_audit', 'telemetry'];
let ok = 0, bad = 0;

for (const d of dirs) {
  let files;
  try { files = readdirSync(join(here, d)).filter((f) => f.endsWith('.json')); }
  catch { continue; }
  for (const f of files) {
    let payload;
    try { payload = JSON.parse(readFileSync(join(here, d, f), 'utf8')); }
    catch (e) { console.log(`✗ ${d}/${f}  invalid JSON: ${e.message}`); bad++; continue; }
    const gate = admit(payload);
    if (gate.ok) { console.log(`✓ ${d}/${f}  → ${gate.entry.key}`); ok++; }
    else { console.log(`✗ ${d}/${f}  rejected: ${gate.reason}${gate.errors ? ' ' + JSON.stringify(gate.errors) : ''}`); bad++; }
  }
}
console.log(`\n${ok} accepted, ${bad} rejected`);
process.exit(bad ? 1 : 0);
