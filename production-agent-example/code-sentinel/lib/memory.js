/**
 * production-agent-example/lib/memory.js
 *
 * Long-term memory integration for the Sentinel Agent.
 * Stores and recalls security patterns, past audit baselines, and project-specific
 * exceptions across runs using Sofuu's CMA (Continuity Memory Architecture).
 */

import path from 'path';

export const BRAIN_PATH = path.resolve(process.cwd(), '.sofuu/brains/sentinel.brain');

/**
 * Initializes and retrieves the agent's memory store if QTSQ is available.
 */
export function getMemoryConfig() {
  const hasMemory =
    typeof sofuu !== 'undefined' &&
    sofuu.memory &&
    typeof sofuu.memory.open === 'function';

  return {
    available: hasMemory,
    scope: hasMemory ? 'agent' : 'off',
    brainPath: BRAIN_PATH,
  };
}

/**
 * Store a discovered vulnerability pattern into the agent's permanent brain.
 */
export async function rememberFinding(finding) {
  if (typeof sofuu === 'undefined' || !sofuu.memory || typeof sofuu.memory.remember !== 'function') {
    return false;
  }
  try {
    const text = `Vulnerability found in ${finding.file}:${finding.line} - [${finding.severity}] ${finding.name}: ${finding.description}`;
    await sofuu.memory.remember(text, {
      entity: 'agent:code_sentinel',
      category: 'security_finding',
      tags: [finding.category, finding.severity],
    });
    return true;
  } catch (_) {
    return false;
  }
}

/**
 * Recall past findings or guidelines for a specific file or topic.
 */
export async function recallPastFindings(query, limit = 5) {
  if (typeof sofuu === 'undefined' || !sofuu.memory || typeof sofuu.memory.recall !== 'function') {
    return [];
  }
  try {
    const results = await sofuu.memory.recall(query, {
      entity: 'agent:code_sentinel',
      limit: limit,
    });
    return Array.isArray(results) ? results : [];
  } catch (_) {
    return [];
  }
}
