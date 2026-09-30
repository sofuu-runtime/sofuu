// P1-13 e2e fixture — the two re-export forms under test.
// `export * as ns from './dep.js'` used to be copied verbatim into the __d
// factory (SyntaxError), and './dep.js' never joined the module graph
// ("missing module" stub). './extra.js' is the pre-existing plain-star form.
export * as ns from './dep.js';
export * from './extra.js';
