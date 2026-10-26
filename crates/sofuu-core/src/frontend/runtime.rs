// crates/sofuu-core/src/frontend/runtime.rs
//
// Embedded micro-runtime for the browser client (`/_sofuu/ui.js`).
// Provides complete primitives for modern reactive web applications:
//   - Minimal reactive VDOM: `h()`, `createElement()`, `Fragment`, `render()`, `createPortal()`
//   - Full standard declarative hooks:
//     `useState()`, `useReducer()`, `useEffect()`, `useLayoutEffect()`,
//     `useCallback()`, `useMemo()`, `useRef()`, `useId()`, `useContext()`, `createContext()`,
//     `forwardRef()`, `useImperativeHandle()`, `memo()`, `useSyncExternalStore()`,
//     `useTransition()`, `startTransition()`, `useDeferredValue()`, `Children`
//   - Client-side SPA Router: `Link`, `useRouter()`, `usePathname()`, `useSearchParams()`
//   - Data Fetching & Cache: `useQuery()`, `useMutation()`
//   - Global Reactive Store: `createStore()`, `useStore()`
//   - AI-Native reactive primitives:
//     `useChat()`, `useCompletion()`, `<ThinkingBlock />`, `<Markdown />`, `<CodeBlock />`
//   - Utilities: `clsx()`, `cn()`
//   - Error Boundary fallback & full SVG namespace support

pub const UI_JS: &str = r#"
// Sofuu Client Micro-Runtime (/_sofuu/ui.js)
// Zero external dependencies - pure ESM reactive VDOM, Router, Store & AI Streaming Hooks

let currentComponent = null;
let hookIndex = 0;
let rootContainer = null;
let rootVNode = null;
let idCounter = 0;

export function Fragment(props) {
  return props.children;
}

export function h(type, props, ...children) {
  props = props || {};
  const flatChildren = [];
  function flatten(arr) {
    for (let i = 0; i < arr.length; i++) {
      const item = arr[i];
      if (item === null || item === undefined || typeof item === 'boolean') continue;
      if (Array.isArray(item)) {
        flatten(item);
      } else if (typeof item === 'object') {
        flatChildren.push(item);
      } else {
        flatChildren.push(String(item));
      }
    }
  }
  flatten(children);
  return { type, props: { ...props, children: flatChildren } };
}

export const createElement = h;

export function useState(initialValue) {
  const comp = currentComponent;
  const idx = hookIndex++;
  if (!comp.hooks[idx]) {
    comp.hooks[idx] = {
      state: typeof initialValue === 'function' ? initialValue() : initialValue,
    };
  }
  const hook = comp.hooks[idx];
  const setState = (newValue) => {
    const next = typeof newValue === 'function' ? newValue(hook.state) : newValue;
    if (next !== hook.state) {
      hook.state = next;
      scheduleUpdate(comp);
    }
  };
  return [hook.state, setState];
}

export function useReducer(reducer, initialArg, init) {
  const [state, setState] = useState(init ? init(initialArg) : initialArg);
  const dispatch = useCallback((action) => {
    setState((prevState) => reducer(prevState, action));
  }, [reducer]);
  return [state, dispatch];
}

export function useEffect(callback, deps) {
  const comp = currentComponent;
  const idx = hookIndex++;
  const oldHook = comp.hooks[idx];
  const hasChanged = !oldHook || !deps || deps.some((dep, i) => dep !== oldHook.deps[i]);
  if (hasChanged) {
    if (oldHook && oldHook.cleanup) oldHook.cleanup();
    comp.hooks[idx] = { deps, cleanup: null };
    queueMicrotask(() => {
      if (comp.hooks[idx]) {
        comp.hooks[idx].cleanup = callback();
      }
    });
  }
}

export function useLayoutEffect(callback, deps) {
  useEffect(callback, deps);
}

export function useCallback(callback, deps) {
  return useMemo(() => callback, deps);
}

export function useMemo(factory, deps) {
  const comp = currentComponent;
  const idx = hookIndex++;
  const oldHook = comp.hooks[idx];
  const hasChanged = !oldHook || !deps || deps.some((dep, i) => dep !== oldHook.deps[i]);
  if (hasChanged) {
    comp.hooks[idx] = { value: factory(), deps };
  }
  return comp.hooks[idx].value;
}

export function useRef(initialValue) {
  const comp = currentComponent;
  const idx = hookIndex++;
  if (!comp.hooks[idx]) {
    comp.hooks[idx] = { current: initialValue };
  }
  return comp.hooks[idx];
}

export function useId() {
  return useMemo(() => ':r' + (++idCounter).toString(36) + ':', []);
}

export function createContext(defaultValue) {
  const context = {
    _currentValue: defaultValue,
    Provider({ value, children }) {
      context._currentValue = value !== undefined ? value : defaultValue;
      return children;
    }
  };
  return context;
}

export function useContext(context) {
  return context ? context._currentValue : undefined;
}

export function forwardRef(renderFn) {
  return function ForwardRefComponent(props) {
    const { ref, ...rest } = props;
    return renderFn(rest, ref);
  };
}

export function useImperativeHandle(ref, createHandle, deps) {
  useEffect(() => {
    if (typeof ref === 'function') {
      ref(createHandle());
      return () => ref(null);
    } else if (ref && typeof ref === 'object') {
      ref.current = createHandle();
      return () => { ref.current = null; };
    }
  }, deps);
}

export function memo(Comp, areEqual) {
  const checkEqual = areEqual || ((prev, next) => {
    const prevKeys = Object.keys(prev);
    const nextKeys = Object.keys(next);
    if (prevKeys.length !== nextKeys.length) return false;
    for (let i = 0; i < prevKeys.length; i++) {
      const k = prevKeys[i];
      if (prev[k] !== next[k]) return false;
    }
    return true;
  });

  return function MemoComponent(props) {
    const prevPropsRef = useRef(null);
    const prevResultRef = useRef(null);

    if (prevPropsRef.current && checkEqual(prevPropsRef.current, props)) {
      return prevResultRef.current;
    }
    prevPropsRef.current = props;
    const result = Comp(props);
    prevResultRef.current = result;
    return result;
  };
}

export function useSyncExternalStore(subscribe, getSnapshot, _getServerSnapshot) {
  const [state, setState] = useState(() => getSnapshot());
  useEffect(() => {
    const handleStoreChange = () => {
      setState(getSnapshot());
    };
    return subscribe(handleStoreChange);
  }, [subscribe, getSnapshot]);
  return state;
}

export function startTransition(scope) {
  queueMicrotask(() => {
    scope();
  });
}

export function useTransition() {
  const [isPending, setIsPending] = useState(false);
  const start = useCallback((callback) => {
    setIsPending(true);
    queueMicrotask(() => {
      try {
        callback();
      } finally {
        setIsPending(false);
      }
    });
  }, []);
  return [isPending, start];
}

export function useDeferredValue(value) {
  const [deferred, setDeferred] = useState(value);
  useEffect(() => {
    const id = setTimeout(() => setDeferred(value), 0);
    return () => clearTimeout(id);
  }, [value]);
  return deferred;
}

export const Children = {
  map(children, fn) {
    if (!children) return [];
    return (Array.isArray(children) ? children : [children]).map(fn);
  },
  forEach(children, fn) {
    if (!children) return;
    (Array.isArray(children) ? children : [children]).forEach(fn);
  },
  count(children) {
    if (!children) return 0;
    return Array.isArray(children) ? children.length : 1;
  },
  toArray(children) {
    if (!children) return [];
    return Array.isArray(children) ? children : [children];
  },
  only(children) {
    if (Array.isArray(children)) {
      if (children.length !== 1) throw new Error('Children.only expected single child');
      return children[0];
    }
    if (!children) throw new Error('Children.only expected child');
    return children;
  }
};

export function createPortal(vnode, container) {
  return {
    _portal: true,
    vnode,
    container
  };
}

// ── Global Reactive Store ───────────────────────────────────────────

export function createStore(initialState) {
  let state = typeof initialState === 'function' ? initialState() : initialState;
  const listeners = new Set();
  return {
    getState: () => state,
    setState: (updater) => {
      const next = typeof updater === 'function' ? updater(state) : { ...state, ...updater };
      if (next !== state) {
        state = next;
        listeners.forEach(fn => fn(state));
      }
    },
    subscribe: (listener) => {
      listeners.add(listener);
      return () => listeners.delete(listener);
    }
  };
}

export function useStore(store, selector = s => s) {
  return useSyncExternalStore(
    store.subscribe,
    () => selector(store.getState())
  );
}

// ── Client-Side SPA Router ──────────────────────────────────────────

const routerListeners = new Set();
function notifyRouter() {
  routerListeners.forEach(listener => listener());
}
if (typeof window !== 'undefined') {
  window.addEventListener('popstate', notifyRouter);
}

export function useRouter() {
  const [, setTick] = useState(0);
  useEffect(() => {
    const listener = () => setTick(t => t + 1);
    routerListeners.add(listener);
    return () => routerListeners.delete(listener);
  }, []);

  const pathname = typeof window !== 'undefined' ? window.location.pathname : '/';
  const search = typeof window !== 'undefined' ? window.location.search : '';
  const hash = typeof window !== 'undefined' ? window.location.hash : '';

  const query = useMemo(() => {
    const params = {};
    if (typeof window !== 'undefined') {
      new URLSearchParams(window.location.search).forEach((v, k) => {
        params[k] = v;
      });
    }
    return params;
  }, [search]);

  return {
    pathname,
    search,
    hash,
    query,
    push(url) {
      if (typeof window !== 'undefined') {
        window.history.pushState({}, '', url);
        notifyRouter();
        window.scrollTo(0, 0);
      }
    },
    replace(url) {
      if (typeof window !== 'undefined') {
        window.history.replaceState({}, '', url);
        notifyRouter();
      }
    },
    back() {
      if (typeof window !== 'undefined') window.history.back();
    },
    forward() {
      if (typeof window !== 'undefined') window.history.forward();
    },
    reload() {
      if (typeof window !== 'undefined') window.location.reload();
    }
  };
}

export function usePathname() {
  const router = useRouter();
  return router.pathname;
}

export function useSearchParams() {
  const router = useRouter();
  return new URLSearchParams(router.search);
}

export function Link(props) {
  const { href, onClick, children, ...rest } = props;
  const router = useRouter();

  const handleClick = (e) => {
    if (onClick) onClick(e);
    if (
      !e.defaultPrevented &&
      e.button === 0 &&
      !e.metaKey &&
      !e.ctrlKey &&
      !e.altKey &&
      !e.shiftKey &&
      href &&
      (href.startsWith('/') || href.startsWith('#'))
    ) {
      e.preventDefault();
      router.push(href);
    }
  };

  return h('a', { ...rest, href, onClick: handleClick }, children);
}

// ── Data Fetching: useQuery & useMutation ───────────────────────────

const globalQueryCache = new Map();
const queryListeners = new Map();

export function useQuery(key, fetcher, options = {}) {
  const cacheKey = typeof key === 'function' ? key() : (Array.isArray(key) ? JSON.stringify(key) : String(key));
  const cached = globalQueryCache.get(cacheKey);

  const [state, setState] = useState(() => ({
    data: cached ? cached.data : undefined,
    error: cached ? cached.error : null,
    isLoading: !cached,
    isSuccess: !!cached && !cached.error,
    isError: !!(cached && cached.error),
  }));

  const execute = useCallback(async () => {
    setState(prev => ({ ...prev, isLoading: true }));
    try {
      const data = await fetcher();
      globalQueryCache.set(cacheKey, { data, error: null, time: Date.now() });
      setState({ data, error: null, isLoading: false, isSuccess: true, isError: false });
      const listeners = queryListeners.get(cacheKey) || [];
      listeners.forEach(cb => cb(data));
      if (options.onSuccess) options.onSuccess(data);
      return data;
    } catch (error) {
      globalQueryCache.set(cacheKey, { data: undefined, error, time: Date.now() });
      setState({ data: undefined, error, isLoading: false, isSuccess: false, isError: true });
      if (options.onError) options.onError(error);
      throw error;
    }
  }, [cacheKey]);

  useEffect(() => {
    if (!queryListeners.has(cacheKey)) {
      queryListeners.set(cacheKey, new Set());
    }
    const listeners = queryListeners.get(cacheKey);
    const onUpdate = (data) => setState(s => ({ ...s, data, isLoading: false, isSuccess: true }));
    listeners.add(onUpdate);

    if (!cached || options.refetchOnMount) {
      execute().catch(() => {});
    }

    return () => listeners.delete(onUpdate);
  }, [cacheKey]);

  return {
    ...state,
    refetch: execute,
    mutate: (newData) => {
      globalQueryCache.set(cacheKey, { data: newData, error: null, time: Date.now() });
      setState(s => ({ ...s, data: newData }));
    }
  };
}

export function useMutation(mutationFn, options = {}) {
  const [state, setState] = useState({
    data: null,
    error: null,
    isLoading: false,
    isSuccess: false,
    isError: false,
  });

  const mutate = useCallback(async (variables) => {
    setState({ data: null, error: null, isLoading: true, isSuccess: false, isError: false });
    try {
      const result = await mutationFn(variables);
      setState({ data: result, error: null, isLoading: false, isSuccess: true, isError: false });
      if (options.onSuccess) options.onSuccess(result, variables);
      return result;
    } catch (err) {
      setState({ data: null, error: err, isLoading: false, isSuccess: false, isError: true });
      if (options.onError) options.onError(err, variables);
      throw err;
    }
  }, [mutationFn]);

  const reset = useCallback(() => {
    setState({ data: null, error: null, isLoading: false, isSuccess: false, isError: false });
  }, []);

  return [mutate, { ...state, reset, isPending: state.isLoading }];
}

// ── DOM Renderer & Reconciler ───────────────────────────────────────

let updateQueue = new Set();
function scheduleUpdate(comp) {
  updateQueue.add(comp);
  if (updateQueue.size === 1) {
    requestAnimationFrame(() => {
      updateQueue.clear();
      if (rootContainer && rootVNode) {
        render(rootVNode, rootContainer);
      }
    });
  }
}

export function render(vnode, container) {
  rootContainer = container;
  rootVNode = vnode;

  // Preserve focused element & cursor position during state updates
  const activeEl = typeof document !== 'undefined' ? document.activeElement : null;
  let activeSelector = null;
  let selStart = 0;
  let selEnd = 0;
  if (activeEl && (activeEl.tagName === 'INPUT' || activeEl.tagName === 'TEXTAREA')) {
    if (activeEl.id) activeSelector = '#' + activeEl.id;
    else if (activeEl.name) activeSelector = `input[name="${activeEl.name}"]`;
    else if (activeEl.placeholder) activeSelector = `input[placeholder="${activeEl.placeholder}"]`;
    else activeSelector = activeEl.tagName.toLowerCase();
    selStart = activeEl.selectionStart;
    selEnd = activeEl.selectionEnd;
  }

  const dom = createDOM(vnode);
  container.innerHTML = '';
  container.appendChild(dom);

  if (activeSelector) {
    const newActive = container.querySelector(activeSelector);
    if (newActive) {
      newActive.focus();
      if (newActive.setSelectionRange) {
        try { newActive.setSelectionRange(selStart, selEnd); } catch (_) {}
      }
    }
  }
}

const SVG_TAGS = new Set([
  'svg', 'path', 'circle', 'rect', 'line', 'polyline', 'polygon',
  'g', 'text', 'defs', 'clipPath', 'linearGradient', 'radialGradient',
  'stop', 'mask', 'use', 'symbol', 'ellipse'
]);

const SVG_ATTR_MAP = {
  strokeWidth: 'stroke-width',
  strokeLinecap: 'stroke-linecap',
  strokeLinejoin: 'stroke-linejoin',
  strokeDasharray: 'stroke-dasharray',
  strokeDashoffset: 'stroke-dashoffset',
  strokeMiterlimit: 'stroke-miterlimit',
  strokeOpacity: 'stroke-opacity',
  fillRule: 'fill-rule',
  clipRule: 'clip-rule',
  fillOpacity: 'fill-opacity',
};

function createDOM(vnode) {
  if (vnode === null || vnode === undefined || typeof vnode === 'boolean') {
    return document.createTextNode('');
  }
  if (typeof vnode === 'string' || typeof vnode === 'number') {
    return document.createTextNode(String(vnode));
  }
  if (Array.isArray(vnode)) {
    const frag = document.createDocumentFragment();
    for (let i = 0; i < vnode.length; i++) {
      frag.appendChild(createDOM(vnode[i]));
    }
    return frag;
  }
  if (vnode._portal && vnode.container) {
    const portalDom = createDOM(vnode.vnode);
    vnode.container.appendChild(portalDom);
    return document.createTextNode('');
  }
  if (!vnode || !vnode.type) {
    return document.createTextNode('');
  }

  // Functional Component with Built-in Error Boundary
  if (typeof vnode.type === 'function') {
    const compInstance = {
      type: vnode.type,
      props: vnode.props,
      hooks: vnode._hooks || [],
    };
    currentComponent = compInstance;
    hookIndex = 0;
    let rendered;
    try {
      rendered = vnode.type(vnode.props);
    } catch (err) {
      console.error('Component Render Error:', err);
      rendered = h('div', {
        style: {
          padding: '16px',
          margin: '12px',
          background: '#fef2f2',
          border: '1px solid #f87171',
          borderRadius: '8px',
          color: '#991b1b',
          fontFamily: 'monospace'
        }
      }, h('strong', null, 'Component Error: ' + err.message));
    }
    vnode._hooks = compInstance.hooks;
    currentComponent = null;
    return createDOM(rendered);
  }

  // Native DOM or SVG Element
  const isSvg = SVG_TAGS.has(vnode.type);
  const el = isSvg
    ? document.createElementNS('http://www.w3.org/2000/svg', vnode.type)
    : document.createElement(vnode.type);

  // Set props / attributes / event listeners
  if (vnode.props) {
    for (const [key, val] of Object.entries(vnode.props)) {
      if (key === 'children') continue;

      if (key === 'ref') {
        if (typeof val === 'function') val(el);
        else if (val && typeof val === 'object') val.current = el;
        continue;
      }

      if (key === 'dangerouslySetInnerHTML' && val && val.__html !== undefined) {
        el.innerHTML = val.__html;
        continue;
      }

      if (key === 'htmlFor') {
        el.setAttribute('for', val);
        continue;
      }

      if (key === 'onChange' && typeof val === 'function') {
        el.addEventListener('input', val);
        el.addEventListener('change', val);
        continue;
      }

      if (key.startsWith('on') && typeof val === 'function') {
        const eventName = key.slice(2).toLowerCase();
        el.addEventListener(eventName, val);
      } else if (key === 'className' || key === 'class') {
        if (isSvg) el.setAttribute('class', val);
        else el.className = val;
      } else if (key === 'style') {
        if (typeof val === 'object' && val !== null) {
          Object.assign(el.style, val);
        } else if (typeof val === 'string') {
          el.style.cssText = val;
        }
      } else if (typeof val === 'boolean') {
        if (key in el) el[key] = val;
        if (key.startsWith('aria-') || key.startsWith('data-')) {
          el.setAttribute(key, String(val));
        } else if (val) {
          el.setAttribute(key, '');
        } else {
          el.removeAttribute(key);
        }
      } else if (val !== null && val !== undefined) {
        if (key in el && (key === 'value' || key === 'checked')) {
          el[key] = val;
        }
        const targetAttr = (isSvg && SVG_ATTR_MAP[key]) ? SVG_ATTR_MAP[key] : key;
        el.setAttribute(targetAttr, val);
      }
    }
  }

  // Render children
  const children = vnode.props.children || [];
  for (let i = 0; i < children.length; i++) {
    el.appendChild(createDOM(children[i]));
  }

  return el;
}

// ── AI-Native Hooks & Primitives ────────────────────────────────────

export function useChat(options = {}) {
  const api = options.api || '/api/chat';
  const [messages, setMessages] = useState(options.initialMessages || []);
  const [input, setInput] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  const abortCtrlRef = useRef(null);

  const stop = () => {
    if (abortCtrlRef.current) {
      abortCtrlRef.current.abort();
      abortCtrlRef.current = null;
      setIsStreaming(false);
    }
  };

  const sendMessage = async (e) => {
    if (e && e.preventDefault) e.preventDefault();
    const query = input.trim();
    if (!query || isStreaming) return;

    const userMsg = { role: 'user', content: query };
    const nextMessages = [...messages, userMsg];
    setMessages(nextMessages);
    setInput('');
    setIsStreaming(true);

    const assistantMsg = { role: 'assistant', content: '' };
    setMessages([...nextMessages, assistantMsg]);

    const ctrl = new AbortController();
    abortCtrlRef.current = ctrl;

    try {
      const res = await fetch(api, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ messages: nextMessages }),
        signal: ctrl.signal,
      });

      if (!res.ok) {
        throw new Error(`HTTP ${res.status}: ${await res.text()}`);
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let accumulated = '';
      let streamBuffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        streamBuffer += decoder.decode(value, { stream: true });
        
        // Parse complete SSE lines, buffering any incomplete line across TCP packet boundaries
        const lines = streamBuffer.split('\n');
        streamBuffer = lines.pop() || '';

        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed) continue;
          if (trimmed.startsWith('data: ')) {
            const dataStr = trimmed.slice(6).trim();
            if (dataStr === '[DONE]') break;
            try {
              const data = JSON.parse(dataStr);
              if (data.delta) accumulated += data.delta;
              else if (data.text) accumulated += data.text;
              else if (data.content) accumulated += data.content;
            } catch (_) {
              accumulated += dataStr;
            }
          }
        }

        setMessages([...nextMessages, { role: 'assistant', content: accumulated }]);
      }
      if (options.onFinish) options.onFinish(accumulated);
    } catch (err) {
      if (err.name !== 'AbortError') {
        setMessages([
          ...nextMessages,
          { role: 'assistant', content: `[Error: ${err.message}]` },
        ]);
        if (options.onError) options.onError(err);
      }
    } finally {
      setIsStreaming(false);
      abortCtrlRef.current = null;
    }
  };

  return {
    messages,
    setMessages,
    input,
    setInput,
    sendMessage,
    isStreaming,
    stop,
  };
}

export function useCompletion(options = {}) {
  const api = options.api || '/api/completion';
  const [completion, setCompletion] = useState('');
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const abortCtrlRef = useRef(null);

  const complete = async (prompt, opts = {}) => {
    const text = prompt !== undefined ? prompt : input;
    if (!text || isLoading) return;
    setIsLoading(true);
    setCompletion('');
    const ctrl = new AbortController();
    abortCtrlRef.current = ctrl;

    try {
      const res = await fetch(api, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt: text, ...opts }),
        signal: ctrl.signal,
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let accumulated = '';
      let streamBuffer = '';
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        streamBuffer += decoder.decode(value, { stream: true });
        const lines = streamBuffer.split('\n');
        streamBuffer = lines.pop() || '';
        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed) continue;
          if (trimmed.startsWith('data: ')) {
            const dataStr = trimmed.slice(6).trim();
            if (dataStr === '[DONE]') break;
            try {
              const d = JSON.parse(dataStr);
              if (d.delta) accumulated += d.delta;
              else if (d.text) accumulated += d.text;
              else if (d.content) accumulated += d.content;
            } catch (_) { accumulated += dataStr; }
          }
        }
        setCompletion(accumulated);
      }
      if (options.onFinish) options.onFinish(accumulated);
    } catch (err) {
      if (err.name !== 'AbortError' && options.onError) options.onError(err);
    } finally {
      setIsLoading(false);
      abortCtrlRef.current = null;
    }
  };

  const stop = () => {
    if (abortCtrlRef.current) {
      abortCtrlRef.current.abort();
      abortCtrlRef.current = null;
      setIsLoading(false);
    }
  };

  return { completion, input, setInput, complete, isLoading, stop };
}

// ── UI Components ───────────────────────────────────────────────────

export function ThinkingBlock({ thinking, isStreaming, defaultOpen = false }) {
  const [isOpen, setIsOpen] = useState(defaultOpen);
  if (!thinking && !isStreaming) return null;

  return h('div', {
    style: {
      margin: '8px 0 12px 0',
      borderRadius: '8px',
      border: '1px solid #e5e5e5',
      background: '#f9f9f9',
      overflow: 'hidden',
      fontSize: '13px',
      color: '#666666'
    }
  },
    h('button', {
      type: 'button',
      onClick: () => setIsOpen(!isOpen),
      style: {
        width: '100%',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '8px 12px',
        background: 'none',
        border: 'none',
        cursor: 'pointer',
        color: '#555555',
        fontFamily: 'inherit',
        fontSize: '13px',
        fontWeight: '500'
      }
    },
      h('div', { style: { display: 'flex', alignItems: 'center', gap: '8px' } },
        isStreaming && h('span', {
          style: {
            width: '8px',
            height: '8px',
            borderRadius: '50%',
            background: '#3b82f6',
            display: 'inline-block',
            animation: 'sofuu-pulse 1.2s infinite ease-in-out'
          }
        }),
        h('span', null, isStreaming ? 'Thinking...' : 'Reasoning process')
      ),
      h('span', { style: { transform: isOpen ? 'rotate(180deg)' : 'rotate(0deg)', transition: 'transform 0.2s', fontSize: '11px' } }, '▼')
    ),
    isOpen && h('div', {
      style: {
        padding: '8px 12px 12px 12px',
        borderTop: '1px solid #eeeeee',
        whiteSpace: 'pre-wrap',
        fontFamily: 'monospace',
        fontSize: '12px',
        lineHeight: '1.5',
        color: '#444444',
        background: '#ffffff'
      }
    }, thinking)
  );
}

export function Markdown({ text }) {
  if (!text) return h('div', { className: 'sofuu-markdown' });
  
  const lines = text.split('\n');
  const elements = [];
  let inCodeBlock = false;
  let codeBuffer = [];

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    if (line.startsWith('```')) {
      if (inCodeBlock) {
        elements.push(h(CodeBlock, { code: codeBuffer.join('\n') }));
        codeBuffer = [];
        inCodeBlock = false;
      } else {
        inCodeBlock = true;
      }
      continue;
    }
    if (inCodeBlock) {
      codeBuffer.push(line);
      continue;
    }

    if (line.startsWith('# ')) {
      elements.push(h('h1', null, line.slice(2)));
    } else if (line.startsWith('## ')) {
      elements.push(h('h2', null, line.slice(3)));
    } else if (line.startsWith('### ')) {
      elements.push(h('h3', null, line.slice(4)));
    } else if (line.startsWith('- ')) {
      elements.push(h('li', null, line.slice(2)));
    } else if (line.trim()) {
      elements.push(h('p', null, line));
    }
  }

  return h('div', { className: 'sofuu-markdown' }, ...elements);
}

export function CodeBlock({ code, language }) {
  const [copied, setCopied] = useState(false);
  const handleCopy = () => {
    navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return h('div', { className: 'sofuu-code-block', style: { position: 'relative', background: '#1e1e1e', color: '#d4d4d4', padding: '1rem', borderRadius: '6px', fontFamily: 'monospace' } },
    h('button', {
      onClick: handleCopy,
      style: { position: 'absolute', right: '8px', top: '8px', background: '#333333', color: '#ffffff', border: 'none', borderRadius: '4px', padding: '4px 8px', cursor: 'pointer', fontSize: '12px' }
    }, copied ? 'Copied!' : 'Copy'),
    h('pre', { style: { margin: 0, overflowX: 'auto' } }, h('code', null, code))
  );
}

// ── Utilities ───────────────────────────────────────────────────────

export function clsx(...args) {
  const classes = [];
  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    if (!arg) continue;
    if (typeof arg === 'string' || typeof arg === 'number') {
      classes.push(arg);
    } else if (Array.isArray(arg)) {
      const inner = clsx(...arg);
      if (inner) classes.push(inner);
    } else if (typeof arg === 'object') {
      for (const key of Object.keys(arg)) {
        if (arg[key]) classes.push(key);
      }
    }
  }
  return classes.join(' ');
}

export const cn = clsx;

// ── React Compatibility Layer ───────────────────────────────────────

export const React = {
  createElement: h,
  Fragment,
  useState,
  useReducer,
  useEffect,
  useLayoutEffect,
  useCallback,
  useMemo,
  useRef,
  useId,
  createContext,
  useContext,
  forwardRef,
  useImperativeHandle,
  memo,
  useSyncExternalStore,
  startTransition,
  useTransition,
  useDeferredValue,
  createPortal,
  Children,
};

export default React;

export const ReactDOM = {
  render,
  createPortal,
  createRoot(container) {
    return {
      render(vnode) {
        render(vnode, container);
      },
      unmount() {
        if (container) container.innerHTML = '';
      }
    };
  }
};

export const createRoot = ReactDOM.createRoot;
export const jsx = (type, props, key) => h(type, { ...props, key });
export const jsxs = jsx;
export const jsxDEV = jsx;
"#;

pub const HTML_SHELL: &str = r#"<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>ChatGPT</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    html, body {
      width: 100%;
      height: 100%;
      overflow: hidden;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background: #ffffff;
      color: #0d0d0d;
      -webkit-font-smoothing: antialiased;
    }
    #root {
      width: 100vw;
      height: 100vh;
      display: flex;
      overflow: hidden;
    }
    @keyframes sofuu-pulse {
      0%, 100% { opacity: 1; transform: scale(1); }
      50% { opacity: 0.35; transform: scale(0.85); }
    }
  </style>
</head>
<body>
  <div id="root"></div>
  <script type="module">
    import App from "{{ENTRY_ROUTE}}";
    import { h, render } from "/_sofuu/ui.js";
    render(h(App), document.getElementById("root"));
  </script>
  <script>
    (() => {
      if (location.hostname === 'localhost' || location.hostname === '127.0.0.1') {
        const es = new EventSource('/_sofuu/live');
        es.onmessage = (e) => {
          if (e.data === 'reload') {
            console.log('[sofuu] Changes detected, reloading...');
            location.reload();
          }
        };
      }
    })();
  </script>
</body>
</html>
"#;
