/**
 * production-agent-example/lib/tools/report_gen.js
 *
 * Production audit report synthesizer tool.
 * Produces structured Markdown and JSON reports with executive summaries,
 * risk scores, and remediation checklists.
 */

import fs from 'fs';
import path from 'path';

export function calculateRiskScore(severityCounts) {
  const c = (severityCounts.critical || 0) * 25;
  const h = (severityCounts.high || 0) * 10;
  const m = (severityCounts.medium || 0) * 4;
  const l = (severityCounts.low || 0) * 1;
  const rawScore = 100 - (c + h + m + l);
  return Math.max(0, Math.min(100, rawScore));
}

function getRelPath(p) {
  if (!p) return '';
  const cwd = process.cwd();
  if (p.startsWith(cwd)) {
    const rel = p.slice(cwd.length);
    return rel.startsWith('/') ? rel.slice(1) : rel;
  }
  return path.basename(p);
}

export function renderMarkdownReport({
  title,
  targetDir,
  executiveSummary,
  severityCounts,
  findings,
  recommendations,
}) {
  const score = calculateRiskScore(severityCounts);
  const status = score >= 85 ? 'PASSED (LOW RISK)' : score >= 60 ? 'WARNING (MEDIUM RISK)' : 'FAILED (HIGH RISK)';
  const date = new Date().toISOString();

  let md = `# Security & Code Audit Report: ${title || 'Application Review'}\n\n`;
  md += `> **Audit Date:** \`${date}\`  \n`;
  md += `> **Target Directory:** \`${targetDir || '.'}\`  \n`;
  md += `> **Security Posture Score:** **${score}/100** — \`${status}\`  \n\n`;

  md += `## 1. Executive Summary\n\n${executiveSummary || 'No summary provided.'}\n\n`;

  md += `## 2. Vulnerability Metrics\n\n`;
  md += `| Severity | Count | Weight Impact |\n`;
  md += `| :--- | :--- | :--- |\n`;
  md += `| **CRITICAL** | ${severityCounts.critical || 0} | Immediate blocker |\n`;
  md += `| **HIGH**     | ${severityCounts.high || 0} | Release gate fail |\n`;
  md += `| **MEDIUM**   | ${severityCounts.medium || 0} | Prioritize sprint |\n`;
  md += `| **LOW**      | ${severityCounts.low || 0} | Code health task |\n\n`;

  md += `## 3. Discovered Vulnerabilities & Findings\n\n`;
  if (!findings || findings.length === 0) {
    md += `*No actionable vulnerabilities detected within the scanned boundaries.*\n\n`;
  } else {
    findings.forEach((f, idx) => {
      const relFile = getRelPath(f.file || '');
      md += `### ${idx + 1}. [${f.severity}] ${f.name || f.ruleId}\n\n`;
      md += `- **Location:** \`${relFile}:${f.line || '?'}\`\n`;
      md += `- **Category:** \`${f.category || 'General'}\`\n`;
      md += `- **Description:** ${f.description || ''}\n`;
      if (f.snippet) {
        md += `\n\`\`\`javascript\n${f.snippet}\n\`\`\`\n`;
      }
      md += `- **Remediation Action:** ${f.remediation || 'Review and refactor implementation.'}\n\n`;
    });
  }

  md += `## 4. Prioritized Remediation Roadmap\n\n`;
  if (recommendations && recommendations.length > 0) {
    recommendations.forEach((rec, i) => {
      md += `${i + 1}. [ ] **${rec.title}**: ${rec.action}\n`;
    });
  } else {
    md += `1. [ ] Remediate any critical secrets or injection points immediately.\n`;
    md += `2. [ ] Add automated pre-commit scanning hooks to prevent future leaks.\n`;
  }
  md += `\n---\n*Report compiled by Sofuu Production Sentinel Agent.*\n`;

  return md;
}

export const reportGenTool = {
  name: 'report_gen',
  description:
    'Synthesizes audit findings, executive summary, and recommendations into an audit report and writes it to disk.',
  parameters: {
    type: 'object',
    properties: {
      title: { type: 'string', description: 'Title of the audit report' },
      targetDir: { type: 'string', description: 'Directory that was audited' },
      executiveSummary: { type: 'string', description: 'High-level executive evaluation' },
      severityCounts: {
        type: 'object',
        properties: {
          critical: { type: 'number' },
          high: { type: 'number' },
          medium: { type: 'number' },
          low: { type: 'number' },
        },
      },
      findings: {
        type: 'array',
        description: 'Array of finding objects',
      },
      recommendations: {
        type: 'array',
        description: 'Array of { title, action } recommendation items',
      },
      outputPath: {
        type: 'string',
        description: 'File path to save the generated report (default: ./security-audit-report.md)',
      },
    },
    required: ['executiveSummary'],
  },
  writes: true,
  execute: async function (args) {
    const outPath = path.resolve(process.cwd(), args.outputPath || 'security-audit-report.md');
    const severityCounts = args.severityCounts || { critical: 0, high: 0, medium: 0, low: 0 };
    const findings = Array.isArray(args.findings) ? args.findings : [];

    const md = renderMarkdownReport({
      title: args.title || 'Security & Code Audit',
      targetDir: args.targetDir || process.cwd(),
      executiveSummary: args.executiveSummary,
      severityCounts,
      findings,
      recommendations: args.recommendations || [],
    });

    try {
      fs.writeFileSync(outPath, md, 'utf8');
      const score = calculateRiskScore(severityCounts);
      return JSON.stringify({
        ok: true,
        reportPath: outPath,
        bytesWritten: Buffer.byteLength(md),
        riskScore: score,
        status: score >= 85 ? 'PASSED' : 'FAILED',
      });
    } catch (err) {
      return JSON.stringify({
        ok: false,
        error: `Failed to write audit report: ${err.message}`,
      });
    }
  },
};

export default reportGenTool;
