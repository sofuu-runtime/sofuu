import { defineTool, z } from 'sofuu';

export default defineTool({
  name: 'github_repo',
  description: 'Fetch real-time stats and metadata for a public GitHub repository',
  permissions: { net: ['api.github.com'] },
  schema: z.object({
    owner: z.string().describe('Repository owner (e.g. facebook)'),
    repo: z.string().describe('Repository name (e.g. react)')
  }),
  timeoutMs: 8000,
  async run(args, ctx) {
    var url = 'https://api.github.com/repos/' + encodeURIComponent(args.owner) + '/' + encodeURIComponent(args.repo);
    var res = await ctx.fetch(url, {
      headers: {
        'Accept': 'application/vnd.github.v3+json',
        'User-Agent': 'sofuu-realworld-test/0.3'
      }
    });
    if (!res || !res.ok) {
      throw new Error('GitHub API error (HTTP ' + (res ? res.status : 'unknown') + ')');
    }
    var d = await res.json();
    return {
      fullName: d.full_name,
      description: d.description,
      stars: d.stargazers_count,
      forks: d.forks_count,
      openIssues: d.open_issues_count,
      license: d.license ? d.license.spdx_id : 'none'
    };
  }
});
