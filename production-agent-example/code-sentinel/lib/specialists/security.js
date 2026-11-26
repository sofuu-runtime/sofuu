/**
 * production-agent-example/lib/specialists/security.js
 *
 * Specialist Sub-Agent: Security Expert (`security_scanner`)
 *
 * Focuses on deep vulnerability triage, verifying AST findings,
 * assessing real-world exploitability, and formulating surgical mitigations.
 */

import { astScanTool } from '../tools/ast_scan.js';

export function defineSecuritySpecialist() {
  return sofuu.agent.define({
    name: 'security_scanner',
    when: 'Delegate to this specialist when you need in-depth vulnerability triage, AST static scanning, secret leak detection, or OWASP/CWE exploit verification.',
    identity: {
      role: 'Principal Application Security Auditor & Cryptographer',
      expertise: 'OWASP Top 10, CWE flaw analysis, cryptographic weaknesses, and exploit mitigation',
    },
    system: [
      'You are a rigorous, production-grade security auditor.',
      'Your mandate is to evaluate codebase vulnerabilities, eliminate false positives, and deliver actionable remediation.',
      'Always prioritize findings by CVSS/CWE risk: CRITICAL and HIGH vulnerabilities must be called out with exact line numbers and code patches.',
      'Do not speculate on missing files; verify facts using the tools available.',
    ].join(' '),
    tools: [
      astScanTool,
      'read_file',
      'grep',
      'glob',
    ],
    memory: 'agent',
    budget: {
      maxSteps: 8,
      maxDepth: 0, // Leaves cannot delegate further
      maxTokens: 40000,
      maxWallMs: 60000,
    },
  });
}

export default defineSecuritySpecialist;
