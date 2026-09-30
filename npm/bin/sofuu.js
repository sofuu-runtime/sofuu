#!/usr/bin/env node
//
// bin/sofuu.js — the shim npm links onto your PATH.
//
// All it does is exec the real binary that install.js downloaded (into
// bin/runtime/sofuu), passing argv through untouched and streaming stdio so
// interactive chat, streaming LLM output, and Ctrl-C all behave as if you
// had run the binary directly. The binary is kept in a subdirectory so it
// never collides with this shim.
//
'use strict';

const path = require('path');
const fs = require('fs');
const { spawnSync } = require('child_process');

const isWin = process.platform === 'win32';
const real = path.join(__dirname, 'runtime', isWin ? 'sofuu.exe' : 'sofuu');

if (!fs.existsSync(real)) {
  // Almost always means postinstall was skipped (--ignore-scripts, or an
  // offline install). Say so specifically rather than surfacing ENOENT.
  console.error(
    'sofuu: the runtime binary is missing from this package.\n' +
    '  It normally downloads during `npm install`. To fix, run:\n' +
    '    npm rebuild sofuu\n' +
    '  or build from source and point at it:\n' +
    '    SOFUU_BINARY_PATH=/path/to/sofuu npm rebuild sofuu\n'
  );
  process.exit(1);
}

const result = spawnSync(real, process.argv.slice(2), { stdio: 'inherit' });
if (result.error) {
  console.error('sofuu: failed to launch the runtime:', result.error.message);
  process.exit(1);
}
// Propagate the child's exit code faithfully — a chat session ended with
// Ctrl-C must not report success to a wrapping script.
process.exit(result.status === null ? 1 : result.status);
