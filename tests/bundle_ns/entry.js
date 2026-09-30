// P1-13 e2e fixture — entry: consume both re-export forms end-to-end.
// ns is the namespace of dep.js ONLY (star-as semantics): alpha/beta/default.
// gamma comes through the plain-star re-export, flattened onto mid's exports.
import { ns, gamma } from './mid.js';

if (ns.alpha !== 1 || ns.beta !== 2 || ns.default !== 42) {
  console.error('FAIL bundle_ns: ns.alpha=' + ns.alpha + ' ns.beta=' + ns.beta +
    ' ns.default=' + ns.default);
  process.exit(1);
}
if (gamma !== 3) {
  console.error('FAIL bundle_ns: gamma=' + gamma);
  process.exit(1);
}
console.log('bundle_ns_ok');
