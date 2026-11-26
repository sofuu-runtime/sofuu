/**
 * production-agent-example/lib/sentinel.js
 *
 * Supervisor Agent: Sentinel Core (`code_sentinel`)
 *
 * Coordinates codebase audits, delegates to specialist sub-agents,
 * synthesizes vulnerability data, and generates authoritative audit reports.
 */

import { astScanTool } from './tools/ast_scan.js';
import { reportGenTool } from './tools/report_gen.js';
import { gitDiffTool } from './tools/git_diff.js';
import { defineSecuritySpecialist } from './specialists/security.js';
import { defineQualitySpecialist } from './specialists/quality.js';
import { getMemoryConfig } from './memory.js';

export function defineSentinelAgent(customOpts = {}) {
  // Ensure specialist sub-agents are registered in the runtime registry
  defineSecuritySpecialist();
  defineQualitySpecialist();

  const mem = getMemoryConfig();

  return sofuu.agent.define({
    name: 'code_sentinel',
    identity: {
      role: 'Autonomous Lead Code Intelligence & Security Sentinel',
      expertise: 'Full-stack software auditing, vulnerability management, multi-agent orchestration, and production release gating',
    },
    system: [
      'You are Code Sentinel, an autonomous production-grade security and code intelligence orchestrator.',
      'Your mission is to perform end-to-end audits of codebases, repositories, or pull requests.',
      'Operational Protocol:',
      '1. Plan Before Executing: For multi-step audits, call todo_write to map out your inspection phases.',
      '2. Inspect & Scan: Use ast_scan and git_diff to gather concrete vulnerability indicators and changed files.',
      '3. Delegate: For complex cryptographic flaws, secret leaks, or OWASP analysis, delegate to security_scanner. For architecture and modularity, delegate to quality_expert.',
      '4. Synthesize & Report: Call report_gen to generate an authoritative Markdown/JSON audit artifact with risk metrics and prioritized checklists.',
      '5. Discipline: Do not hallucinate files or vulnerabilities. Base every statement strictly on real tool outputs.',
    ].join(' '),
    tools: [
      astScanTool,
      reportGenTool,
      gitDiffTool,
      'read_file',
      'write_file',
      'edit_file',
      'grep',
      'glob',
      'list_dir',
      'todo_write',
      'system_info',
    ],
    agents: ['security_scanner', 'quality_expert'],
    memory: mem.scope,
    brainPath: mem.brainPath,
    budget: {
      maxSteps: customOpts.maxSteps || 15,
      maxDepth: 2, // Allows supervisor -> specialist delegation
      maxTokens: customOpts.maxTokens || 100000,
      maxWallMs: customOpts.maxWallMs || 180000,
      verify: true, // Post-audit verification pass
    },
  });
}

export default defineSentinelAgent;
