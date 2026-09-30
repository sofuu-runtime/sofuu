#!/usr/bin/env node
//
// install.js — fetch the Sofuu binary for this machine and put it on disk.
//
// This is the same download the `curl | sh` installer does, wrapped so that
// `npm i -D sofuu` (or `npx sofuu`) just works for the JavaScript ecosystem
// the way every other serious runtime does. It is the same shape esbuild and
// Deno use: resolve platform → download a prebuilt → verify its SHA-256 →
// place it next to the shim. No compile step, no postinstall surprises.
//
// Rules this script holds to:
//   * It never installs an unverified binary. If the checksum is missing or
//     does not match, it fails loudly and leaves nothing behind.
//   * It respects SOFUU_BINARY_PATH (point it at a local build to skip the
//     download entirely) and SOFUU_SKIP_DOWNLOAD (CI matrix jobs).
//   * `--verify-only` checks an existing install without fetching, so
//     `npm test` after install is a real assertion, not a no-op.
//
'use strict';

const fs = require('fs');
const os = require('os');
const path = require('path');
const https = require('https');
const { execFileSync } = require('child_process');

const BASE_URL = process.env.SOFUU_DOWNLOAD_BASE || 'https://sofuu.xyz/downloads';
const verifyOnly = process.argv.includes('--verify-only');

function fail(msg) {
  console.error(`\n  ✗ sofuu install failed: ${msg}\n`);
  process.exit(1);
}

function info(msg) { console.log(`  → ${msg}`); }

// ── Resolve platform/arch the same way the release tarballs are named ──
function resolveTarget() {
  let platform = os.platform();
  let arch = os.arch();

  if (platform === 'win32') platform = 'windows';
  if (platform === 'darwin') platform = 'darwin';

  // We ship arm64 + x86_64 only.
  if (arch === 'x64') arch = 'x86_64';
  if (arch === 'arm64') arch = 'arm64';

  if (!['darwin', 'linux', 'windows'].includes(platform)) {
    fail(`unsupported platform: ${os.platform()}`);
  }
  if (!['x86_64', 'arm64'].includes(arch)) {
    fail(`unsupported architecture: ${os.arch()}. Sofuu ships x86_64 and arm64.`);
  }
  return { platform, arch };
}

// A single dropped connection should not fail an install; retry a few times
// with a short backoff, then give up with the actionable message below.
async function fetchWithRetry(archive, attempts = 3) {
  let lastErr;
  for (let i = 0; i < attempts; i++) {
    try {
      return await Promise.all([
        get(`${BASE_URL}/${archive}`),
        get(`${BASE_URL}/${archive}.sha256`),
      ]);
    } catch (e) {
      lastErr = e;
      if (i < attempts - 1) {
        await new Promise((r) => setTimeout(r, 500 * (i + 1)));
      }
    }
  }
  throw lastErr;
}

function get(url, redirects = 0) {
  return new Promise((resolve, reject) => {
    if (redirects > 5) return reject(new Error('too many redirects'));
    https
      .get(url, { headers: { 'user-agent': 'sofuu-npm' } }, (res) => {
        if (res.statusCode >= 300 && res.statusCode < 400 && res.headers.location) {
          res.resume();
          return resolve(get(res.headers.location, redirects + 1));
        }
        if (res.statusCode !== 200) {
          res.resume();
          return reject(new Error(`HTTP ${res.statusCode} for ${url}`));
        }
        const chunks = [];
        res.on('data', (c) => chunks.push(c));
        res.on('end', () => resolve(Buffer.concat(chunks)));
      })
      .on('error', reject);
  });
}

function sha256(buf) {
  return require('crypto').createHash('sha256').update(buf).digest('hex');
}

// Extract a .tar.gz using the system tar (present on macOS, Linux, and
// Windows 10+). We avoid a JS tar dependency on purpose: this package ships
// zero runtime dependencies, so `npm i` stays instant and auditable.
//
// The published archive currently holds ONE entry named
// `sofuu-<platform>-<arch>` (the bare binary, no wrapping directory), but
// older/newer release scripts have used a wrapping directory — so handle
// both: find the extracted file, and if a directory was used, look inside it.
function extract(tarball, destDir) {
  const tmp = path.join(destDir, '.extract');
  fs.rmSync(tmp, { recursive: true, force: true });
  fs.mkdirSync(tmp, { recursive: true });
  execFileSync('tar', ['-xzf', tarball, '-C', tmp], { stdio: 'inherit' });

  // Depth-first search for the first regular file named sofuu* (or *.exe).
  function find(dir) {
    for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
      const full = path.join(dir, entry.name);
      if (entry.isDirectory()) {
        const hit = find(full);
        if (hit) return hit;
      } else if (entry.isFile() && /^sofuu(\.exe)?$/.test(entry.name)) {
        return full;
      } else if (entry.isFile() && /^sofuu-/.test(entry.name)) {
        return full;
      }
    }
    return null;
  }

  const binPath = find(tmp);
  if (!binPath) fail(`archive did not contain a sofuu binary (${assetHint()})`);
  return binPath;
}

function assetHint() {
  const { platform, arch } = resolveTarget();
  return `sofuu-${platform}-${arch}`;
}

async function main() {
  const pkgDir = path.dirname(__filename);
  const binDir = path.join(pkgDir, 'bin');
  const runtimeDir = path.join(binDir, 'runtime');
  const binaryName = process.platform === 'win32' ? 'sofuu.exe' : 'sofuu';
  const binaryPath = path.join(runtimeDir, binaryName);

  if (process.env.SOFUU_BINARY_PATH) {
    info(`using SOFUU_BINARY_PATH=${process.env.SOFUU_BINARY_PATH}`);
    fs.mkdirSync(runtimeDir, { recursive: true });
    fs.copyFileSync(process.env.SOFUU_BINARY_PATH, binaryPath);
    fs.chmodSync(binaryPath, 0o755);
    return;
  }

  if (verifyOnly) {
    if (!fs.existsSync(binaryPath)) {
      const junk = fs.existsSync(runtimeDir) ? fs.readdirSync(runtimeDir) : [];
      fail(
        `no runtime binary at bin/runtime/${binaryName}` +
        ` (found: ${junk.join(', ') || 'nothing'}).\n` +
        '  Run `npm rebuild sofuu` to (re)download it.'
      );
    }
    try {
      const v = execFileSync(binaryPath, ['version'], { encoding: 'utf8' });
      info(`version: ${v.trim().split('\n')[0]}`);
    } catch (e) {
      fail('installed binary did not run');
    }
    return;
  }

  if (process.env.SOFUU_SKIP_DOWNLOAD) {
    info('SOFUU_SKIP_DOWNLOAD set — not fetching a binary.');
    return;
  }

  const { platform, arch } = resolveTarget();
  const asset = `sofuu-${platform}-${arch}`;
  const archive = `${asset}.tar.gz`;
  info(`downloading ${asset} …`);

  let tarball, checksumFile;
  try {
    [tarball, checksumFile] = await fetchWithRetry(archive);
  } catch (e) {
    fail(
      `could not download ${BASE_URL}/${archive}\n` +
      `    ${e.message}\n` +
      `    Published builds: https://sofuu.xyz/downloads\n` +
      `    Building from source instead: cargo build --release`
    );
  }

  // The .sha256 file is "<hash>  <filename>" (coreutils format).
  const expected = checksumFile.toString('utf8').trim().split(/\s+/)[0];
  const actual = sha256(tarball);
  if (!/^[0-9a-f]{64}$/.test(expected)) {
    fail(`checksum file is malformed: ${checksumFile.toString('utf8').slice(0, 120)}`);
  }
  if (expected !== actual) {
    fail(`checksum mismatch for ${archive}\n` +
      `    expected ${expected}\n` +
      `    actual   ${actual}\n` +
      `    Refusing to install an unverified binary.`);
  }
  info('checksum verified');

  fs.mkdirSync(runtimeDir, { recursive: true });
  const tarPath = path.join(runtimeDir, archive);
  fs.writeFileSync(tarPath, tarball);
  const extracted = extract(tarPath, runtimeDir);
  fs.rmSync(binaryPath, { force: true });
  fs.renameSync(extracted, binaryPath);
  fs.chmodSync(binaryPath, 0o755);
  fs.rmSync(tarPath, { force: true });
  fs.rmSync(path.join(runtimeDir, '.extract'), { recursive: true, force: true });

  const v = execFileSync(binaryPath, ['version'], { encoding: 'utf8' }).trim();
  info(`installed sofuu ${v}`);
  console.log('\n  Try:  npx sofuu            (chat)   ·   npx sofuu run app.ts\n');
}

main().catch((e) => fail(e && e.stack ? e.stack : String(e)));
