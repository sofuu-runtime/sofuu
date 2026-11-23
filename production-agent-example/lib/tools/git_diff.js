/**
 * production-agent-example/lib/tools/git_diff.js
 *
 * Git inspection tool for checking modified, untracked, and staged files
 * using Sofuu's safe process spawning primitives.
 */

import path from 'path';

function runGit(args, cwd) {
  return new Promise((resolve) => {
    let stdout = '';
    let stderr = '';
    try {
      sofuu.spawn({
        command: 'git',
        args: args,
        cwd: cwd || process.cwd(),
        onStdout: (chunk) => {
          stdout += String(chunk);
        },
        onStderr: (chunk) => {
          stderr += String(chunk);
        },
        onExit: (code) => {
          resolve({ code, stdout: stdout.trim(), stderr: stderr.trim() });
        },
      });
    } catch (err) {
      resolve({ code: 1, stdout: '', stderr: err.message });
    }
  });
}

export const gitDiffTool = {
  name: 'git_diff',
  description:
    'Retrieves git status and diff summaries for recent changes, modified files, and staged changes.',
  parameters: {
    type: 'object',
    properties: {
      stagedOnly: {
        type: 'boolean',
        description: 'Only inspect staged changes (git diff --staged).',
      },
      fileLimit: {
        type: 'number',
        description: 'Max number of changed files to list (default: 20).',
      },
      repoPath: {
        type: 'string',
        description: 'Path to repository root (default: current directory).',
      },
    },
  },
  readOnly: true,
  execute: async function (args) {
    const cwd = path.resolve(process.cwd(), args.repoPath || '.');
    const staged = args.stagedOnly === true;
    const limit = args.fileLimit || 20;

    // 1. Get short status
    const statusRes = await runGit(['status', '--short'], cwd);
    if (statusRes.code !== 0) {
      return JSON.stringify({
        ok: false,
        error: statusRes.stderr || 'Not a git repository or git command failed',
      });
    }

    const statusLines = statusRes.stdout ? statusRes.stdout.split('\n') : [];
    const changedFiles = statusLines.slice(0, limit).map((l) => l.trim());

    // 2. Get diff stat summary
    const diffArgs = staged ? ['diff', '--staged', '--stat'] : ['diff', '--stat'];
    const statRes = await runGit(diffArgs, cwd);

    return JSON.stringify({
      ok: true,
      totalChangedFiles: statusLines.length,
      files: changedFiles,
      diffSummary: statRes.stdout || 'Clean working tree',
    });
  },
};

export default gitDiffTool;
