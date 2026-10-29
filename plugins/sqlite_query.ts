// plugins/sqlite_query.ts - read-only query tool for the local structured store.
//
// THERE IS NO SQLITE IN THIS RUNTIME. That is deliberate, not an omission:
// the only mention of sqlite anywhere in crates/sofuu-core is a memory note
// saying it was replaced by QTSQ. What Sofuu ships is the native QTSQ
// structured store (src/js/storage.js, reachable as sofuu.storage): columnar
// tables with filter/order/limit queries. This tool therefore exposes that
// store through a deliberately narrow, read-only SELECT subset, so an agent
// can inspect sessions, traces and KV rows with no new dependency, no
// subprocess and no write path.
//
// Supported grammar (case-insensitive, exactly one statement, no joins):
//   SELECT * FROM <table>
//   SELECT <col>[, <col>...] FROM <table>
//   SELECT count(*) FROM <table>
//     [WHERE <col> = <string | number | true | false>]
//     [ORDER BY <col> [ASC | DESC]]
//     [LIMIT <n>]
//
// Every write or DDL verb (INSERT, UPDATE, DELETE, DROP, ALTER, ATTACH,
// PRAGMA, CREATE, REPLACE, VACUUM, TRUNCATE, BEGIN, COMMIT, ROLLBACK) is
// refused with E_DENIED before anything is parsed or opened.
//
// Standalone: no import beyond the SDK primitives, no shelling out.

import { defineTool, z } from 'sofuu';
import type { SofuuDatabase, SofuuStorage, StorageQueryOptions } from 'sofuu';

// -- local types (this block is stripped before the runtime sees the file) -

interface PluginError extends Error {
  code: string;
}

interface QueryArgs {
  sql: string;
  path?: string;
  limit?: number;
}

interface QueryResult {
  engine: string;
  sql: string;
  table: string;
  columns: string[];
  rows: Row[];
  rowCount: number;
  truncated: boolean;
}

interface ParsedQuery {
  table: string;
  columns: string[];
  countOnly?: boolean;
  filterCol?: string | null;
  filterVal?: string | null;
  orderCol?: string | null;
  orderDesc?: boolean;
  limit?: number | null;
}

type Row = Record<string, unknown>;
type RowList = Row[];
type MaybePath = string | undefined;
type MaybeDriver = SofuuStorage | null;

// -- rails ----------------------------------------------------------------

var ENGINE = 'qtsq-structured-store';
var MAX_ROWS = 500;
var DEFAULT_STORE = '.sofuu/storage';

// Anything that could mutate, reindex, or open a second database.
var WRITE_VERBS = /\b(insert|update|delete|drop|alter|attach|detach|pragma|create|replace|vacuum|truncate|begin|commit|rollback|reindex|grant|revoke)\b/i;

function fail(code: string, message: string): PluginError {
  var err = new Error(message) as PluginError;
  err.code = code;
  return err;
}

function refuseWrites(sql: string): void {
  var match = String(sql || '').match(WRITE_VERBS);
  if (match) {
    throw fail('E_DENIED', 'read-only tool: ' + match[1].toUpperCase() + ' is refused');
  }
}

function resolveStorePath(raw: MaybePath, cwd: string): string {
  var given = (raw && String(raw).trim()) ? String(raw).trim() : DEFAULT_STORE;
  if (given.indexOf('..') >= 0) throw fail('E_DENIED', 'store path may not escape the workspace: ' + given);
  if (given.charAt(0) === '/') {
    if (cwd && given.indexOf(cwd) !== 0) {
      throw fail('E_DENIED', 'store path outside the working directory: ' + given);
    }
    return given;
  }
  return given;
}

// -- parser ---------------------------------------------------------------

function literalValue(raw: string): string {
  var text = String(raw || '');
  if (text.length >= 2) {
    var first = text.charAt(0);
    if (first === String.fromCharCode(39) || first === String.fromCharCode(34)) {
      return text.slice(1, text.length - 1);
    }
  }
  return text;
}

function parseColumns(raw: string): string[] {
  var out: string[] = [];
  if (raw === '*') return out;
  var parts = raw.split(',');
  for (var i = 0; i < parts.length; i++) {
    var name = parts[i].trim();
    if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(name)) {
      throw fail('E_SCHEMA', 'unsupported column list: ' + raw);
    }
    out.push(name);
  }
  return out;
}

var SELECT_RE = /^select\s+([\s\S]+?)\s+from\s+([A-Za-z_][A-Za-z0-9_]*)([\s\S]*)$/i;
var COUNT_RE = /^count\s*\(\s*\*\s*\)$/i;
var WHERE_RE = /\bwhere\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*('(?:[^']*)'|"(?:[^"]*)"|-?\d+(?:\.\d+)?|true|false)/i;
var ORDER_RE = /\border\s+by\s+([A-Za-z_][A-Za-z0-9_]*)(?:\s+(asc|desc))?/i;
var LIMIT_RE = /\blimit\s+(\d+)/i;

function parseQuery(sql: string): ParsedQuery {
  var text = String(sql || '').trim().replace(/;\s*$/, '');
  if (!text) throw fail('E_SCHEMA', 'sql is required');
  if (text.indexOf(';') >= 0) throw fail('E_SCHEMA', 'one statement per call');
  var select = text.match(SELECT_RE);
  if (!select) throw fail('E_SCHEMA', "expected 'SELECT <columns> FROM <table>'");
  // Built by assignment: an object literal full of literal values would be
  // rewritten by the runtime stripper (", key: false" collapses to ", key").
  var parsed: ParsedQuery = { table: select[2], columns: [] };
  parsed.countOnly = false;
  parsed.filterCol = null;
  parsed.filterVal = null;
  parsed.orderCol = null;
  parsed.orderDesc = false;
  parsed.limit = null;
  var columnsRaw = select[1].trim();
  if (COUNT_RE.test(columnsRaw)) parsed.countOnly = true;
  else parsed.columns = parseColumns(columnsRaw);

  var rest = select[3] || '';
  var where = rest.match(WHERE_RE);
  if (where) {
    parsed.filterCol = where[1];
    parsed.filterVal = literalValue(where[2]);
    rest = rest.replace(where[0], ' ');
  }
  var order = rest.match(ORDER_RE);
  if (order) {
    parsed.orderCol = order[1];
    parsed.orderDesc = String(order[2] || 'asc').toLowerCase() === 'desc';
    rest = rest.replace(order[0], ' ');
  }
  var limit = rest.match(LIMIT_RE);
  if (limit) {
    var parsedLimit = parseInt(limit[1], 10);
    if (isNaN(parsedLimit) || parsedLimit < 0) throw fail('E_SCHEMA', 'LIMIT must be a positive integer');
    parsed.limit = parsedLimit;
    rest = rest.replace(limit[0], ' ');
  }
  if (rest.replace(/\s+/g, '') !== '') {
    throw fail('E_SCHEMA', 'unsupported clause: ' + rest.trim());
  }
  return parsed;
}

// -- execution ------------------------------------------------------------

function storageDriver(): MaybeDriver {
  if (typeof sofuu === 'undefined' || !sofuu.storage) return null;
  return sofuu.storage;
}

function pickColumns(rows: RowList, columns: string[]): RowList {
  if (!columns.length) return rows;
  var out: RowList = [];
  for (var i = 0; i < rows.length; i++) {
    var picked: Row = {};
    for (var j = 0; j < columns.length; j++) {
      var key = columns[j];
      picked[key] = rows[i][key];
    }
    out.push(picked);
  }
  return out;
}

function resultOf(sql: string, parsed: ParsedQuery, rows: RowList, wasTruncated: boolean): QueryResult {
  var truncated = wasTruncated;
  var result: QueryResult = {
    engine: ENGINE,
    sql: sql,
    table: parsed.table,
    columns: parsed.columns,
    rows: rows,
    rowCount: rows.length,
    truncated: truncated
  };
  return result;
}

export default defineTool({
  name: 'sqlite_query',
  description: 'Read-only SELECT against the local Sofuu structured store (QTSQ, not SQLite)',
  timeoutMs: 10000,
  permissions: { fs: 'workspace' },
  schema: z.object({
    sql: z.string().describe('One read-only SELECT statement, e.g. SELECT * FROM messages LIMIT 5'),
    path: z.string().optional().describe('Store path (default .sofuu/storage)'),
    limit: z.number().int().positive().optional().describe('Row cap, max 500')
  }),
  async run(args, ctx): Promise<QueryResult> {
    var sql = String(args.sql || '');
    refuseWrites(sql);
    var parsed = parseQuery(sql);
    var driver = storageDriver();
    if (!driver) {
      throw fail('E_CAPABILITY', 'sofuu.storage is not available in this build');
    }
    var storePath = resolveStorePath(args.path, ctx.cwd);
    var cap = (typeof args.limit === 'number' && args.limit > 0) ? Math.min(args.limit, MAX_ROWS) : MAX_ROWS;
    // SQL LIMIT narrows the tool cap; it never widens it.
    if (typeof parsed.limit === 'number' && parsed.limit > 0) cap = Math.min(cap, parsed.limit);
    var db: SofuuDatabase = driver.open(storePath);
    try {
      if (parsed.countOnly) {
        var total = db.count(parsed.table);
        var truncated = false;
        /* A same-name local: ", rowCount: total" would be rewritten to the
           shorthand ", rowCount" by the runtime stripper. */
        var rowCount = total;
        var counted: QueryResult = {
          engine: ENGINE,
          sql: sql,
          table: parsed.table,
          columns: [],
          rows: [],
          rowCount: rowCount,
          truncated: truncated
        };
        return counted;
      }
      if (cap === 0) return resultOf(sql, parsed, [], false);
      var options: StorageQueryOptions = {};
      options.limit = cap;
      if (parsed.filterCol && parsed.filterVal !== null) {
        options.filterCol = parsed.filterCol;
        options.filterVal = parsed.filterVal;
      }
      if (parsed.orderDesc === true) options.orderDesc = true;
      var rows = await db.query(parsed.table, options);
      var list: RowList = (rows && rows.length) ? rows : [];
      if (parsed.orderCol && list.length > 1) {
        list = sortRows(list, parsed.orderCol, parsed.orderDesc === true);
      }
      var projected = pickColumns(list, parsed.columns);
      ctx.log('sqlite_query: ' + parsed.table + ' -> ' + projected.length + ' row(s)');
      return resultOf(sql, parsed, projected, projected.length >= cap);
    } finally {
      try { db.close(); } catch (e) { /* already closed */ }
    }
  }
});

// The native query already orders when orderDesc is set; an explicit ORDER BY
// on another column is applied here so the grammar stays honest.
function sortRows(rows: RowList, column: string, desc: boolean): RowList {
  var copy: RowList = rows.slice(0);
  copy.sort(function (left: Row, right: Row) {
    var a = left[column];
    var b = right[column];
    if (a === b) return 0;
    var cmp = (a === undefined || a === null) ? -1 : (b === undefined || b === null) ? 1 : (a < b ? -1 : 1);
    return desc ? -cmp : cmp;
  });
  return copy;
}
