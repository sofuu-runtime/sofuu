// rt/async_context.rs — Native AsyncLocalStorage & Agent Context Propagation (Phase 4).
//
// Implements TC39 / Node.js AsyncLocalStorage semantics seamlessly integrated
// into QuickJS promise reactions, await resumptions, microtasks, and libuv timers.

use std::ffi::c_int;
use sofuu_ffi::qjs::{self, JSContext, JSValue, JSValueConst};

unsafe extern "C" fn js_get_async_context(
    ctx: *mut JSContext,
    _this_val: JSValueConst,
    _argc: c_int,
    _argv: *const JSValueConst,
) -> JSValue {
    let rt = qjs::JS_GetRuntime(ctx);
    qjs::JS_GetAsyncContext(rt)
}

unsafe extern "C" fn js_set_async_context(
    ctx: *mut JSContext,
    _this_val: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    let rt = qjs::JS_GetRuntime(ctx);
    if argc >= 1 {
        qjs::JS_SetAsyncContext(rt, *argv);
    } else {
        qjs::JS_SetAsyncContext(rt, qjs::sofuu_js_undefined());
    }
    qjs::sofuu_js_undefined()
}

pub unsafe fn mod_async_context_register(ctx: *mut JSContext) {
    let global = qjs::sofuu_js_get_global_object(ctx);

    let fn_get = qjs::sofuu_js_new_cfunction(
        ctx,
        js_get_async_context,
        c"__sofuu_get_async_context".as_ptr(),
        0,
    );
    qjs::sofuu_js_set_property_str(ctx, global, c"__sofuu_get_async_context".as_ptr(), fn_get);

    let fn_set = qjs::sofuu_js_new_cfunction(
        ctx,
        js_set_async_context,
        c"__sofuu_set_async_context".as_ptr(),
        1,
    );
    qjs::sofuu_js_set_property_str(ctx, global, c"__sofuu_set_async_context".as_ptr(), fn_set);

    qjs::sofuu_js_free_value(ctx, global);

    // Bootstrap JavaScript AsyncLocalStorage class and sofuu.context namespace
    let bootstrap_src = c"(function() {
    var getCtx = globalThis.__sofuu_get_async_context;
    var setCtx = globalThis.__sofuu_set_async_context;

    var idSeq = 1;
    function AsyncLocalStorage() {
        this._id = Symbol('als_' + (idSeq++));
    }

    AsyncLocalStorage.prototype.run = function(store, callback) {
        var args = [];
        for (var i = 2; i < arguments.length; i++) {
            args.push(arguments[i]);
        }
        var prev = getCtx();
        var next = new Map(prev instanceof Map ? prev : []);
        next.set(this._id, store);
        setCtx(next);
        try {
            return callback.apply(undefined, args);
        } finally {
            setCtx(prev);
        }
    };

    AsyncLocalStorage.prototype.exit = function(callback) {
        var args = [];
        for (var i = 1; i < arguments.length; i++) {
            args.push(arguments[i]);
        }
        var prev = getCtx();
        if (!(prev instanceof Map) || !prev.has(this._id)) {
            return callback.apply(undefined, args);
        }
        var next = new Map(prev);
        next.delete(this._id);
        setCtx(next);
        try {
            return callback.apply(undefined, args);
        } finally {
            setCtx(prev);
        }
    };

    AsyncLocalStorage.prototype.getStore = function() {
        var ctx = getCtx();
        if (ctx instanceof Map) {
            return ctx.get(this._id);
        }
        return undefined;
    };

    AsyncLocalStorage.prototype.enterWith = function(store) {
        var prev = getCtx();
        var next = new Map(prev instanceof Map ? prev : []);
        next.set(this._id, store);
        setCtx(next);
    };

    AsyncLocalStorage.prototype.set = function(storeOrKey, value) {
        if (arguments.length === 1) {
            this.enterWith(storeOrKey);
        } else {
            var s = this.getStore();
            if (!s || typeof s !== 'object') {
                s = {};
                this.enterWith(s);
            }
            if (s instanceof Map) {
                s.set(storeOrKey, value);
            } else {
                s[storeOrKey] = value;
            }
        }
    };

    AsyncLocalStorage.prototype.disable = function() {
        var prev = getCtx();
        if (prev instanceof Map && prev.has(this._id)) {
            var next = new Map(prev);
            next.delete(this._id);
            setCtx(next);
        }
    };

    AsyncLocalStorage.snapshot = function() {
        var captured = getCtx();
        return function(fn) {
            var args = [];
            for (var i = 1; i < arguments.length; i++) {
                args.push(arguments[i]);
            }
            var prev = getCtx();
            setCtx(captured);
            try {
                return fn.apply(undefined, args);
            } finally {
                setCtx(prev);
            }
        };
    };

    AsyncLocalStorage.bind = function(fn) {
        var snap = AsyncLocalStorage.snapshot();
        return function() {
            var args = [];
            for (var i = 0; i < arguments.length; i++) {
                args.push(arguments[i]);
            }
            return snap.apply(undefined, [fn].concat(args));
        };
    };

    var defaultAls = new AsyncLocalStorage();

    var contextObj = {
        AsyncLocalStorage: AsyncLocalStorage,
        run: function(store, callback) {
            var args = [store, callback];
            for (var i = 2; i < arguments.length; i++) args.push(arguments[i]);
            return defaultAls.run.apply(defaultAls, args);
        },
        exit: function(callback) {
            var args = [callback];
            for (var i = 1; i < arguments.length; i++) args.push(arguments[i]);
            return defaultAls.exit.apply(defaultAls, args);
        },
        getStore: function() {
            return defaultAls.getStore();
        },
        enterWith: function(store) {
            return defaultAls.enterWith(store);
        },
        get: function(key) {
            var s = defaultAls.getStore();
            if (!s) return undefined;
            if (s instanceof Map) return s.get(key);
            if (typeof s === 'object') return s[key];
            return undefined;
        },
        set: function(keyOrStore, value) {
            if (arguments.length === 1) {
                defaultAls.enterWith(keyOrStore);
            } else {
                var s = defaultAls.getStore();
                if (!s || typeof s !== 'object') {
                    s = {};
                    defaultAls.enterWith(s);
                }
                if (s instanceof Map) {
                    s.set(keyOrStore, value);
                } else {
                    s[keyOrStore] = value;
                }
            }
        },
        snapshot: function() {
            return AsyncLocalStorage.snapshot();
        },
        bind: function(fn) {
            return AsyncLocalStorage.bind(fn);
        }
    };

    Object.defineProperty(contextObj, 'traceId', {
        get: function() { return contextObj.get('traceId'); },
        set: function(v) { contextObj.set('traceId', v); },
        configurable: true,
        enumerable: true
    });
    Object.defineProperty(contextObj, 'sessionId', {
        get: function() { return contextObj.get('sessionId'); },
        set: function(v) { contextObj.set('sessionId', v); },
        configurable: true,
        enumerable: true
    });
    Object.defineProperty(contextObj, 'signal', {
        get: function() { return contextObj.get('signal'); },
        set: function(v) { contextObj.set('signal', v); },
        configurable: true,
        enumerable: true
    });

    function AbortSignal() {
        this.aborted = false;
        this.reason = undefined;
        this._listeners = [];
    }
    AbortSignal.prototype.addEventListener = function(type, fn) {
        if (type === 'abort') {
            if (this.aborted) fn();
            else this._listeners.push(fn);
        }
    };
    AbortSignal.prototype.removeEventListener = function(type, fn) {
        if (type === 'abort') {
            this._listeners = this._listeners.filter(function(l) { return l !== fn; });
        }
    };
    AbortSignal.prototype.throwIfAborted = function() {
        if (this.aborted) {
            throw this.reason || new Error('This operation was aborted');
        }
    };
    AbortSignal.abort = function(reason) {
        var sig = new AbortSignal();
        sig.aborted = true;
        sig.reason = reason || new Error('This operation was aborted');
        return sig;
    };
    AbortSignal.timeout = function(ms) {
        var ctrl = new AbortController();
        setTimeout(function() { ctrl.abort(new Error('TimeoutError: The operation timed out')); }, ms);
        return ctrl.signal;
    };

    function AbortController() {
        this.signal = new AbortSignal();
    }
    AbortController.prototype.abort = function(reason) {
        if (this.signal.aborted) return;
        this.signal.aborted = true;
        this.signal.reason = reason || new Error('This operation was aborted');
        var list = this.signal._listeners.slice();
        for (var i = 0; i < list.length; i++) {
            try { list[i](); } catch (e) {}
        }
    };

    if (!globalThis.AbortController) globalThis.AbortController = AbortController;
    if (!globalThis.AbortSignal) globalThis.AbortSignal = AbortSignal;
    contextObj.AbortController = AbortController;
    contextObj.AbortSignal = AbortSignal;

    if (!globalThis.sofuu) globalThis.sofuu = {};
    globalThis.sofuu.context = contextObj;
    globalThis.AsyncLocalStorage = AsyncLocalStorage;
})();";

    let res = qjs::JS_Eval(
        ctx,
        bootstrap_src.as_ptr(),
        bootstrap_src.to_bytes().len(),
        c"<sofuu-async-context>".as_ptr(),
        qjs::JS_EVAL_TYPE_GLOBAL,
    );
    if qjs::is_exception(res) {
        let exc = qjs::sofuu_js_get_exception(ctx);
        let s = qjs::sofuu_js_to_cstring(ctx, exc);
        if !s.is_null() {
            eprintln!("[async_context] bootstrap error: {:?}", std::ffi::CStr::from_ptr(s));
            qjs::sofuu_js_free_cstring(ctx, s);
        }
        qjs::sofuu_js_free_value(ctx, exc);
    }
    qjs::sofuu_js_free_value(ctx, res);
}
