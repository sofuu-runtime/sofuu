/**
 * production-agent-example/lib/tools/ast_scan.js
 *
 * High-performance, resilient static security and code quality scanner tool.
 * Detects hardcoded secrets, injection vectors, prototype pollution, path traversal,
 * and unsafe cryptography across JavaScript, TypeScript, Python, and shell scripts.
 */

import fs from 'fs';
import path from 'path';

export const SEVERITY_LEVELS = {
  CRITICAL: 4,
  HIGH: 3,
  MEDIUM: 2,
  LOW: 1,
};

export const SECURITY_RULES = [
  {
    id: 'SEC-001',
    category: 'Secret Leak',
    severity: 'CRITICAL',
    name: 'Hardcoded API Key / Token',
    regex: /(?:sk-[a-zA-Z0-9_-]{20,}|ghp_[a-zA-Z0-9]{36}|xox[baprs]-[0-9a-zA-Z-]{20,}|AIza[0-9A-Za-z-_]{35}|AKIA[0-9A-Z]{16})/i,
    description: 'Potential live production API key, GitHub token, or AWS access key committed directly in code.',
    remediation: 'Move sensitive credentials into environment variables or secrets manager (e.g. AWS Secrets Manager, Doppler).',
  },
  {
    id: 'SEC-002',
    category: 'Secret Leak',
    severity: 'HIGH',
    name: 'Hardcoded Password / Private Key Assignment',
    regex: /(?:password|passwd|secret|jwt_secret|private_key)\s*[:=]\s*["'][^"']{6,}["']/i,
    description: 'Hardcoded password, JWT secret, or private key string literal.',
    remediation: 'Load secret values at runtime from process.env or a secure key-vault.',
  },
  {
    id: 'SEC-003',
    category: 'Code Injection',
    severity: 'CRITICAL',
    name: 'Dynamic Code Evaluation (eval / Function)',
    regex: /\b(?:eval\s*\(|new\s+Function\s*\(|vm\.runIn[A-Za-z]+\s*\()/i,
    description: 'Dynamic code execution primitives can lead to Remote Code Execution (RCE) if evaluated over untrusted user input.',
    remediation: 'Refactor to safe static parsing (e.g. JSON.parse), expression trees, or sandboxed interpreters.',
  },
  {
    id: 'SEC-004',
    category: 'Command Injection',
    severity: 'HIGH',
    name: 'Unsanitized Shell Execution',
    regex: /(?:exec|execSync|spawnSync)\s*\(\s*`[^`]*\$\{[^}]+\}[^`]*`\s*\)/i,
    description: 'Shell command constructed via string template interpolation without argument escaping.',
    remediation: 'Pass arguments as an explicit array using spawn() or execFile(), or enforce strict alphanumeric allowlisting.',
  },
  {
    id: 'SEC-005',
    category: 'Path Traversal',
    severity: 'HIGH',
    name: 'Unvalidated Path Concatenation',
    regex: /(?:readFile|writeFile|createReadStream|createWriteStream|unlink)\s*\([^)]*\+\s*req\.(?:query|body|params)/i,
    description: 'Filesystem operation directly concatenates untrusted request parameters, risking directory traversal (CWE-22).',
    remediation: 'Canonicalize paths with path.resolve() and verify they remain within the intended base directory with startsWith().',
  },
  {
    id: 'SEC-006',
    category: 'Prototype Pollution',
    severity: 'MEDIUM',
    name: 'Prototype Property Manipulation',
    regex: /\[(?:'|")__(?:proto|lookupGetter|lookupSetter)__(?:'|")\]|\.constructor\.prototype/i,
    description: 'Direct access to Object prototype chains can enable prototype pollution and privilege escalation (CWE-1321).',
    remediation: 'Use Object.create(null) for dictionary maps, or freeze object prototypes with Object.freeze().',
  },
  {
    id: 'SEC-007',
    category: 'Weak Cryptography',
    severity: 'MEDIUM',
    name: 'Insecure Hashing Algorithm (MD5 / SHA1)',
    regex: /(?:createHash\s*\(\s*["'](?:md5|sha1)["']|hashlib\.(?:md5|sha1)\s*\()/i,
    description: 'Collision-vulnerable hashing algorithm used. Inappropriate for digital signatures or credential storage (CWE-327).',
    remediation: 'Upgrade to SHA-256 / SHA-3 for integrity checks, or Argon2id / bcrypt for password storage.',
  },
  {
    id: 'SEC-008',
    category: 'SQL Injection',
    severity: 'CRITICAL',
    name: 'Raw SQL String Concatenation',
    regex: /["'](?:SELECT|INSERT|UPDATE|DELETE)\s+[^"']*["']\s*\+\s*[a-zA-Z0-9_]+/i,
    description: 'Direct SQL query assembly via string concatenation without parameterized bindings (CWE-89).',
    remediation: 'Use prepared statements or parameterized queries ($1, ?, or named parameters).',
  },
];

const IGNORED_DIRS = new Set([
  'node_modules',
  '.git',
  '.sofuu',
  'dist',
  'build',
  'target',
  'coverage',
  '.cargo',
]);

const SCANNABLE_EXTS = new Set([
  '.js',
  '.ts',
  '.jsx',
  '.tsx',
  '.mjs',
  '.cjs',
  '.py',
  '.sh',
  '.bash',
  '.json',
]);

function collectFiles(dirPath, maxFiles = 300) {
  const results = [];
  function walk(current) {
    if (results.length >= maxFiles) return;
    let entries = [];
    try {
      entries = fs.readdirSync(current);
    } catch (_) {
      return;
    }
    for (const name of entries) {
      if (results.length >= maxFiles) break;
      const fullPath = path.join(current, name);
      let stat;
      try {
        stat = fs.statSync(fullPath);
      } catch (_) {
        continue;
      }
      if (stat.isDirectory()) {
        if (!IGNORED_DIRS.has(name) && !name.startsWith('.')) {
          walk(fullPath);
        }
      } else if (stat.isFile()) {
        const ext = path.extname(name).toLowerCase();
        if (SCANNABLE_EXTS.has(ext)) {
          results.push(fullPath);
        }
      }
    }
  }
  walk(dirPath);
  return results;
}

function scanFile(filePath, minSeverityWeight = 1) {
  const findings = [];
  let content = '';
  try {
    content = fs.readFileSync(filePath, 'utf8');
  } catch (_) {
    return findings;
  }

  // Avoid scanning massive minified bundle files
  if (content.length > 500 * 1024) return findings;

  const lines = content.split('\n');
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    if (line.length > 1000) continue; // Skip minified long lines

    for (const rule of SECURITY_RULES) {
      const weight = SEVERITY_LEVELS[rule.severity] || 1;
      if (weight < minSeverityWeight) continue;

      if (rule.regex.test(line)) {
        findings.push({
          ruleId: rule.id,
          category: rule.category,
          severity: rule.severity,
          name: rule.name,
          file: filePath,
          line: i + 1,
          snippet: line.trim(),
          description: rule.description,
          remediation: rule.remediation,
        });
      }
    }
  }

  return findings;
}

/**
 * The AST/Static Analysis Tool definition for sofuu.agent
 */
export const astScanTool = {
  name: 'ast_scan',
  description:
    'Scans a target directory or file for security vulnerabilities, secret leaks, injection flaws, and unsafe patterns using deterministic static rules.',
  parameters: {
    type: 'object',
    properties: {
      targetPath: {
        type: 'string',
        description: 'Path to directory or file to scan (relative or absolute).',
      },
      severityFloor: {
        type: 'string',
        enum: ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL'],
        description: 'Minimum severity level to report (default: LOW).',
      },
      maxFiles: {
        type: 'number',
        description: 'Maximum number of files to inspect (default: 100).',
      },
    },
    required: ['targetPath'],
  },
  readOnly: true,
  parallel: true,
  execute: async function (args) {
    const target = path.resolve(process.cwd(), args.targetPath || '.');
    const floor = (args.severityFloor || 'LOW').toUpperCase();
    const minWeight = SEVERITY_LEVELS[floor] || 1;
    const maxFiles = Math.min(300, Math.max(1, args.maxFiles || 100));

    if (!fs.existsSync(target)) {
      return JSON.stringify({
        ok: false,
        error: `Target path does not exist: ${target}`,
      });
    }

    const stat = fs.statSync(target);
    const files = stat.isDirectory() ? collectFiles(target, maxFiles) : [target];

    const allFindings = [];
    for (const file of files) {
      const f = scanFile(file, minWeight);
      if (f.length > 0) allFindings.push(...f);
    }

    const summary = {
      critical: allFindings.filter((x) => x.severity === 'CRITICAL').length,
      high: allFindings.filter((x) => x.severity === 'HIGH').length,
      medium: allFindings.filter((x) => x.severity === 'MEDIUM').length,
      low: allFindings.filter((x) => x.severity === 'LOW').length,
    };

    return JSON.stringify({
      ok: true,
      targetPath: target,
      scannedFilesCount: files.length,
      totalFindings: allFindings.length,
      severityCounts: summary,
      findings: allFindings.slice(0, 50), // Cap payload to prevent context blowout
    });
  },
};

export default astScanTool;
