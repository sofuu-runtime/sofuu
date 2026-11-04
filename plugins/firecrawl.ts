import { defineTool, z } from 'sofuu';

export default defineTool({
  name: 'firecrawl_scrape',
  description: 'Scrape and extract clean markdown content from any web page using Firecrawl API',
  permissions: { net: ['api.firecrawl.dev'] },
  schema: z.object({
    url: z.string().describe('The URL to scrape and convert to markdown'),
    formats: z.array(z.string()).default(['markdown']).describe('Formats to extract (e.g. markdown, html)')
  }),
  timeoutMs: 15000,
  requiresApproval: false,
  async run(args, ctx) {
    var apiKey = (typeof process !== 'undefined' && process.env && process.env.FIRECRAWL_API_KEY) || '';
    if (!apiKey) {
      throw new Error('FIRECRAWL_API_KEY environment variable is required to use Firecrawl');
    }
    var res = await ctx.fetch('https://api.firecrawl.dev/v1/scrape', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer ' + apiKey
      },
      body: JSON.stringify({ url: args.url, formats: args.formats || ['markdown'] })
    });
    if (!res || !res.ok) {
      var errText = res ? await res.text() : 'network failure';
      throw new Error('Firecrawl API error (HTTP ' + (res ? res.status : 0) + '): ' + errText);
    }
    var data = await res.json();
    return {
      url: args.url,
      markdown: (data.data && data.data.markdown) || data.markdown || '',
      title: (data.data && data.data.metadata && data.data.metadata.title) || ''
    };
  }
});
