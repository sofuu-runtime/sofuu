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

pub const UI_JS_GZ: &[u8] = include_bytes!("ui.js.gz");

pub fn get_ui_js() -> String {
    use std::io::Read;
    let mut decoder = flate2::read::GzDecoder::new(UI_JS_GZ);
    let mut s = String::new();
    let _ = decoder.read_to_string(&mut s);
    s
}

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
