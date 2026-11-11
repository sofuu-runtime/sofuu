// Real QTSQ store checks; each invocation is a separate Sofuu process.
'use strict';

async function main() {
  const phase = process.argv[2];
  const path = process.argv[3];
  const marker = process.argv[4];
  const vector = new Float32Array(64);
  vector[3] = 1;
  if (!sofuu.memory || typeof sofuu.memory.open !== 'function' ||
      (sofuu.build && sofuu.build.missing && sofuu.build.missing.indexOf('qtsq') >= 0)) {
    console.error('BLOCKED: this build does not expose the QTSQ store');
    process.exit(2);
  }
  if (phase === 'write') {
    const store = sofuu.memory.open(path, 64, { plaintext: true });
    store.remember(vector, marker);
    const result = store.flush();
    if (!result || result.ok !== true) throw new Error('store did not acknowledge its flush');
    store.close();
  } else if (phase === 'read') {
    const store = sofuu.memory.open(path, 64, { readonly: true, plaintext: true });
    const hits = store.recall(vector, 5);
    if (!hits.some(hit => hit.text === marker)) throw new Error('persisted marker was not found');
    let error = null;
    try { store.remember(vector, 'must not be saved'); } catch (e) { error = e; }
    if (!error || !error.code) throw new Error('readonly store accepted a write');
    store.close();
  } else if (phase === 'empty') {
    const store = sofuu.memory.open(path, 64, { plaintext: true });
    if (store.recall(vector, 5).some(hit => hit.text === marker)) throw new Error('separate store leaked the marker');
    store.close();
  } else if (phase === 'fail-flush') {
    const directory = process.argv[5];
    const store = sofuu.memory.open(path, 64, { plaintext: true });
    store.remember(vector, marker);
    await sofuu.fs.chmod(directory, 0o500);
    let error = null;
    try { store.flush(); } catch (e) { error = e; }
    finally { await sofuu.fs.chmod(directory, 0o700); }
    if (!error || !error.code) throw new Error('flush did not report the filesystem failure');
    store.close();
  } else {
    throw new Error('unknown test phase');
  }
}
main().catch(function (error) {
  console.error('FAIL real memory check:', String(error && error.message || error));
  process.exit(1);
});
