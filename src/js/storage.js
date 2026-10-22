// storage.js — Native QTSQ Structured Storage Driver for Sofuu (CMA v2).
//
// Provides high-performance, strict schema columnar tables, weekly rotation,
// checkpoint indexing, key-value state, and application document collections.
//
// Author: Haruhito

(function() {
  const global = globalThis;
  if (!global.sofuu) global.sofuu = {};

  class TableHandle {
    constructor(db, name) {
      this.db = db;
      this.name = name;
    }

    insert(row) {
      return this.db.insert(this.name, row);
    }

    query(options = {}) {
      return this.db.query(this.name, options);
    }

    count() {
      return this.db.count(this.name);
    }
  }

  class MessagesCollection extends TableHandle {
    constructor(db) {
      super(db, 'messages');
    }

    async log(msg) {
      return this.insert(msg);
    }

    async list(options = {}) {
      const q = {};
      if (options.sessionId) {
        q.filterCol = 'session_id';
        q.filterVal = String(options.sessionId);
      }
      if (options.limit) q.limit = options.limit;
      if (options.offset) q.offset = options.offset;
      if (options.minTime) q.minTimestamp = options.minTime;
      if (options.maxTime) q.maxTimestamp = options.maxTime;
      if (options.order === 'desc') q.orderDesc = true;
      return this.query(q);
    }
  }

  class TracesCollection extends TableHandle {
    constructor(db) {
      super(db, 'traces');
    }

    async log(trace) {
      return this.insert(trace);
    }

    async list(options = {}) {
      const q = {};
      if (options.sessionId) {
        q.filterCol = 'session_id';
        q.filterVal = String(options.sessionId);
      }
      if (options.runId) {
        q.filterCol = 'session_id';
        q.filterVal = String(options.runId);
      }
      if (options.limit) q.limit = options.limit;
      if (options.offset) q.offset = options.offset;
      if (options.minTime) q.minTimestamp = options.minTime;
      if (options.maxTime) q.maxTimestamp = options.maxTime;
      if (options.order === 'desc') q.orderDesc = true;
      return this.query(q);
    }
  }

  class KeyValueStore {
    constructor(db) {
      this.db = db;
    }

    async set(key, value) {
      await this.db.insert('_kv_store', {
        key: String(key),
        val: value,
        timestamp: Date.now()
      });
      return true;
    }

    async get(key) {
      const results = await this.db.query('_kv_store', {
        filterCol: 'key',
        filterVal: String(key),
        limit: 1,
        orderDesc: true
      });
      if (results && results.length > 0) {
        return results[0].val;
      }
      return null;
    }

    async del(key) {
      return this.set(key, null);
    }
  }

  class Database {
    constructor(options = {}) {
      let path = typeof options === 'string' ? options : (options.path || options.file || '.sofuu/storage');
      let password = options.password || null;

      this.handle = global.__sofuu_storage_open(path, password);
      this.path = path;
      this.messages = new MessagesCollection(this);
      this.traces = new TracesCollection(this);
      this.kv = new KeyValueStore(this);
      this._tables = new Map();
      this._flushTimer = null;
    }

    insert(table, row) {
      if (!this.handle) throw new Error('Database is closed');
      const id = global.__sofuu_storage_insert(this.handle, table, JSON.stringify(row));
      this._scheduleDebouncedFlush();
      return id;
    }

    query(table, options = {}) {
      if (!this.handle) throw new Error('Database is closed');
      return global.__sofuu_storage_query(this.handle, table, JSON.stringify(options));
    }

    count(table) {
      if (!this.handle) return 0;
      return global.__sofuu_storage_count(this.handle, table);
    }

    table(name) {
      if (!this._tables.has(name)) {
        this._tables.set(name, new TableHandle(this, name));
      }
      return this._tables.get(name);
    }

    collection(name) {
      return this.table(name);
    }

    flush() {
      if (this._flushTimer) {
        clearTimeout(this._flushTimer);
        this._flushTimer = null;
      }
      if (!this.handle) return false;
      return global.__sofuu_storage_flush(this.handle);
    }

    close() {
      if (this._flushTimer) {
        clearTimeout(this._flushTimer);
        this._flushTimer = null;
      }
      if (this.handle) {
        global.__sofuu_storage_close(this.handle);
        this.handle = null;
      }
    }

    _scheduleDebouncedFlush() {
      if (!this._flushTimer) {
        this._flushTimer = setTimeout(() => {
          this._flushTimer = null;
          this.flush();
        }, 500);
      }
    }

    get activeSegment() {
      if (!this.handle) return null;
      return global.__sofuu_storage_active_segment(this.handle);
    }
  }

  const storage = {
    Database,
    open(path, options) {
      return new Database(typeof path === 'string' ? { path, ...options } : path);
    }
  };

  global.sofuu.storage = storage;
  if (!global.Sofuu) global.Sofuu = {};
  global.Sofuu.storage = storage;
})();
