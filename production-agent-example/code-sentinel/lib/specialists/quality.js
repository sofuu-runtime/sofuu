/**
 * production-agent-example/lib/specialists/quality.js
 *
 * Specialist Sub-Agent: Code Architect (`quality_expert`)
 *
 * Focuses on software architecture, modularity, type safety,
 * dead code elimination, and maintainability.
 */

export function defineQualitySpecialist() {
  return sofuu.agent.define({
    name: 'quality_expert',
    when: 'Delegate to this specialist when you need architecture evaluation, dependency analysis, dead code auditing, or modularity review.',
    identity: {
      role: 'Staff Systems Architect & Clean Code Specialist',
      expertise: 'Software modularity, decoupled system design, dependency hygiene, and fault tolerance',
    },
    system: [
      'You are a senior software architect specializing in robust, maintainable systems.',
      'Your responsibility is to analyze software boundaries, identify technical debt, verify error handling resilience, and ensure production standards.',
      'Provide concrete refactoring recommendations with minimal blast radius.',
    ].join(' '),
    tools: [
      'read_file',
      'grep',
      'glob',
      'list_dir',
    ],
    memory: 'agent',
    budget: {
      maxSteps: 8,
      maxDepth: 0,
      maxTokens: 40000,
      maxWallMs: 60000,
    },
  });
}

export default defineQualitySpecialist;
