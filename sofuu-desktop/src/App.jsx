import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Send, Settings, TerminalSquare, Brain, Plus, Trash2, MessageSquare } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import SettingsModal from './components/SettingsModal';

// ─── Helpers ───────────────────────────────────────────────────────────────

function loadState() {
  try {
    const sessions = JSON.parse(localStorage.getItem('sofuu_sessions') || 'null');
    const config   = JSON.parse(localStorage.getItem('sofuu_config')   || 'null');
    const activeId = Number(localStorage.getItem('sofuu_active_session') || '0');
    return { sessions, config, activeId };
  } catch {
    return { sessions: null, config: null, activeId: 0 };
  }
}

function mkSession() {
  return { id: Date.now(), title: 'New Chat', messages: [] };
}

function parseThink(content) {
  if (!content) return { thinkText: null, bodyText: '' };
  const m = content.match(/<think>([\s\S]*?)<\/think>/);
  if (m) {
    return { thinkText: m[1], bodyText: content.replace(/<think>[\s\S]*?<\/think>/, '').trim() };
  }
  // mid-stream: <think> opened but not closed yet
  if (content.includes('<think>')) {
    return { thinkText: content.replace('<think>', ''), bodyText: '' };
  }
  return { thinkText: null, bodyText: content };
}

// ─── App ───────────────────────────────────────────────────────────────────

export default function App() {
  const initial = loadState();

  const [sessions, setSessions] = useState(
    initial.sessions?.length ? initial.sessions : [mkSession()]
  );
  const [activeId, setActiveId] = useState(
    initial.activeId || initial.sessions?.[0]?.id || sessions[0].id
  );
  const [config, setConfig] = useState(initial.config || {
    provider : 'openrouter',
    model    : 'nvidia/nemotron-3-super-120b-a12b:free',
    apiKey   : '',
    think    : true,
    brainId  : '',
  });

  const [inputValue,    setInputValue]    = useState('');
  const [isLoading,     setIsLoading]     = useState(false);
  const [streamBuf,     setStreamBuf]     = useState('');   // live streaming text
  const [settingsOpen,  setSettingsOpen]  = useState(false);

  const messagesEndRef = useRef(null);
  // Keep a ref to activeId so IPC callbacks always see the current value
  const activeIdRef = useRef(activeId);
  useEffect(() => { activeIdRef.current = activeId; }, [activeId]);

  // ── Persist ──────────────────────────────────────────────────────────
  useEffect(() => {
    localStorage.setItem('sofuu_sessions', JSON.stringify(sessions));
  }, [sessions]);
  useEffect(() => {
    localStorage.setItem('sofuu_active_session', String(activeId));
  }, [activeId]);
  useEffect(() => {
    localStorage.setItem('sofuu_config', JSON.stringify(config));
  }, [config]);

  // ── Scroll ───────────────────────────────────────────────────────────
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [sessions, streamBuf]);

  // ── IPC wiring (mount once, use refs for mutable state) ──────────────
  useEffect(() => {
    if (!window.api) return;

    // Accumulate streaming text into session messages
    window.api.onChatChunk((chunk) => {
      setStreamBuf(prev => prev + chunk);
    });

    window.api.onChatDone((stats) => {
      // Capture the final streamed text synchronously via functional update
      setStreamBuf(prev => {
        const finalText = prev;
        setSessions(prevSessions => prevSessions.map(s => {
          if (s.id !== activeIdRef.current) return s;
          const msgs = s.messages.map((m, i, arr) => {
            if (i === arr.length - 1 && m.role === 'assistant') {
              return { ...m, content: m.content + finalText, stats };
            }
            return m;
          });
          return { ...s, messages: msgs };
        }));
        return '';  // clear stream buffer
      });
      setIsLoading(false);
    });

    window.api.onChatError((error) => {
      setStreamBuf(prev => {
        setSessions(prevSessions => prevSessions.map(s => {
          if (s.id !== activeIdRef.current) return s;
          const msgs = s.messages.map((m, i, arr) => {
            if (i === arr.length - 1 && m.role === 'assistant') {
              return { ...m, error };
            }
            return m;
          });
          return { ...s, messages: msgs };
        }));
        return '';
      });
      setIsLoading(false);
    });

    return () => window.api.removeAllListeners();
  }, []); // mount ONCE — use activeIdRef for current session

  // ── Session helpers ───────────────────────────────────────────────────
  const activeSession = sessions.find(s => s.id === activeId) || sessions[0];
  const messages      = activeSession?.messages ?? [];

  const pushMessages = (id, newMsgs) => {
    setSessions(prev => prev.map(s => {
      if (s.id !== id) return s;
      let title = s.title;
      if (title === 'New Chat' && newMsgs.length >= 1) {
        title = newMsgs[0].content.slice(0, 28) + (newMsgs[0].content.length > 28 ? '…' : '');
      }
      return { ...s, title, messages: newMsgs };
    }));
  };

  const newSession = () => {
    const s = mkSession();
    setSessions(prev => [s, ...prev]);
    setActiveId(s.id);
  };

  const deleteSession = (e, id) => {
    e.stopPropagation();
    setSessions(prev => {
      const next = prev.filter(s => s.id !== id);
      if (!next.length) next.push(mkSession());
      if (id === activeIdRef.current) setActiveId(next[0].id);
      return next;
    });
  };

  const clearAll = () => {
    if (confirm('Delete all sessions?')) {
      const s = mkSession();
      setSessions([s]);
      setActiveId(s.id);
    }
  };

  // ── Send ──────────────────────────────────────────────────────────────
  const handleSend = useCallback(() => {
    const text = inputValue.trim();
    if (!text || isLoading) return;

    const currId = activeIdRef.current;

    if (!config.apiKey && config.provider !== 'ollama') {
      const placeholder = { role: 'assistant', content: '',
        error: `No API key set. Open Configuration and add your ${config.provider} key.` };
      pushMessages(currId, [...messages, { role: 'user', content: text }, placeholder]);
      setSettingsOpen(true);
      return;
    }

    const userMsg = { role: 'user', content: text };
    const newMsgs = [...messages, userMsg, { role: 'assistant', content: '' }];
    pushMessages(currId, newMsgs);

    setInputValue('');
    setIsLoading(true);
    setStreamBuf('');

    const cleanHistory = [...messages, userMsg].map(m => ({ role: m.role, content: m.content }));

    if (window.api) {
      window.api.sendChat({ ...config, messages: cleanHistory });
    }
  }, [inputValue, isLoading, config, messages]);

  const handleKey = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSend(); }
  };

  // ── Render ────────────────────────────────────────────────────────────
  return (
    <div className="app-container">

      {/* ── Sidebar ── */}
      <div className="sidebar glass-sidebar drag-region">
        <div style={{ display:'flex', flexDirection:'column', height:'100%' }}>

          {/* Logo */}
          <div style={{ display:'flex', alignItems:'center', gap:'0.5rem', marginBottom:'1.25rem' }}>
            <TerminalSquare size={24} style={{ color:'var(--accent-color)' }} />
            <span style={{ fontWeight:700, fontSize:'1.1rem' }}>Sofuu Desktop</span>
          </div>

          {/* New Chat */}
          <button
            className="no-drag"
            onClick={newSession}
            style={{
              display:'flex', alignItems:'center', justifyContent:'center', gap:'0.4rem',
              padding:'0.65rem', marginBottom:'0.75rem',
              background:'var(--accent-color)', color:'#fff',
              border:'none', borderRadius:'8px', fontWeight:600, cursor:'pointer',
              fontSize:'0.9rem', transition:'opacity 0.15s'
            }}
            onMouseOver={e => e.currentTarget.style.opacity='0.85'}
            onMouseOut={e  => e.currentTarget.style.opacity='1'}
          >
            <Plus size={15}/> New Chat
          </button>

          {/* Session list */}
          <div className="no-drag" style={{ flex:1, overflowY:'auto', display:'flex', flexDirection:'column', gap:'2px' }}>
            <div style={{ fontSize:'0.7rem', fontWeight:700, letterSpacing:'0.8px',
              textTransform:'uppercase', color:'var(--text-secondary)', marginBottom:'0.4rem' }}>
              Sessions
            </div>

            {sessions.map(s => {
              const active = s.id === activeId;
              return (
                <div
                  key={s.id}
                  onClick={() => setActiveId(s.id)}
                  style={{
                    display:'flex', alignItems:'center', gap:'0.4rem',
                    padding:'0.6rem 0.5rem', borderRadius:'7px', cursor:'pointer',
                    background: active ? 'rgba(37,99,235,0.08)' : 'transparent',
                    color: active ? 'var(--accent-color)' : 'var(--text-primary)',
                    fontWeight: active ? 600 : 400, fontSize:'0.88rem',
                    transition:'background 0.15s'
                  }}
                  onMouseOver={e => { if(!active) e.currentTarget.style.background='var(--border-color)'; }}
                  onMouseOut={e  => { if(!active) e.currentTarget.style.background='transparent'; }}
                >
                  <MessageSquare size={13} style={{ flexShrink:0, opacity:0.6 }}/>
                  <div style={{ flex:1, overflow:'hidden', textOverflow:'ellipsis', whiteSpace:'nowrap' }}>
                    {s.title}
                  </div>
                  <button
                    onClick={e => deleteSession(e, s.id)}
                    style={{ background:'none', border:'none', color:'var(--text-secondary)',
                      cursor:'pointer', padding:'0', opacity:0.6, display:'flex' }}
                  >
                    <Trash2 size={12}/>
                  </button>
                </div>
              );
            })}
          </div>

          {/* Footer actions */}
          <div style={{ paddingTop:'0.75rem', borderTop:'1px solid var(--border-color)', display:'flex', flexDirection:'column', gap:'0.5rem' }}>
            <button
              className="no-drag"
              onClick={clearAll}
              style={{ background:'none', border:'none', color:'var(--text-secondary)',
                fontSize:'0.78rem', cursor:'pointer', textAlign:'left', padding:'0.25rem 0' }}
            >
              Clear All Sessions
            </button>

            <button
              className="no-drag"
              onClick={() => setSettingsOpen(true)}
              style={{
                display:'flex', alignItems:'center', gap:'0.5rem',
                background:'none', border:'1px solid var(--border-color)',
                color:'var(--text-primary)', padding:'0.65rem', borderRadius:'8px',
                cursor:'pointer', width:'100%', fontSize:'0.9rem', transition:'background 0.15s'
              }}
              onMouseOver={e => e.currentTarget.style.background='var(--border-color)'}
              onMouseOut={e  => e.currentTarget.style.background='none'}
            >
              <Settings size={15}/> Configuration
            </button>
          </div>
        </div>
      </div>

      {/* ── Main chat ── */}
      <div className="main-chat">
        <div className="messages-container no-drag">

          {/* Empty state */}
          {messages.length === 0 && (
            <div style={{ margin:'auto', textAlign:'center', color:'var(--text-secondary)',
              display:'flex', flexDirection:'column', alignItems:'center', gap:'1rem' }}>
              <Brain size={44} style={{ opacity:0.35 }}/>
              <div style={{ fontWeight:600, fontSize:'1.1rem', color:'var(--text-primary)' }}>
                Start a conversation
              </div>
              <div style={{ fontSize:'0.85rem', background:'var(--bg-card)',
                border:'1px solid var(--border-color)', padding:'0.4rem 1rem',
                borderRadius:'20px', display:'flex', gap:'0.75rem' }}>
                <span><strong>{config.provider}</strong></span>
                <span>·</span>
                <span style={{ opacity:0.7 }}>{config.model.split('/').pop()}</span>
                {config.brainId && <><span>·</span><span>🧠 {config.brainId}</span></>}
                {config.think   && <><span>·</span><span>💭 think</span></>}
              </div>
            </div>
          )}

          {/* Messages */}
          {messages.map((msg, idx) => {
            const isAI   = msg.role === 'assistant';
            const isLast = idx === messages.length - 1;
            // Append live streamBuf to the last assistant placeholder
            const raw   = isAI && isLast ? msg.content + streamBuf : msg.content;
            const { thinkText, bodyText } = parseThink(raw);

            return (
              <div key={idx} className={`message-bubble ${isAI ? 'assistant' : 'user'}`}>

                {/* Think block */}
                {thinkText && (
                  <details className="think-block">
                    <summary>Deep Think Process</summary>
                    <div className="think-content">{thinkText}</div>
                  </details>
                )}

                {/* Body */}
                {isAI ? (
                  <ReactMarkdown remarkPlugins={[remarkGfm]} className="markdown-body">
                    {bodyText || (isLoading && isLast ? '▍' : '')}
                  </ReactMarkdown>
                ) : (
                  <span>{raw}</span>
                )}

                {/* Error */}
                {msg.error && (
                  <div style={{ marginTop:'0.5rem', padding:'0.5rem 0.75rem', borderRadius:'6px',
                    background:'rgba(220,38,38,0.08)', color:'#dc2626', fontSize:'0.85rem' }}>
                    ⚠ {msg.error}
                  </div>
                )}

                {/* Stats footer */}
                {msg.stats && (
                  <div className="message-stats">
                    [QTSQ Memory: {msg.stats.memoryRecords ?? 0} records
                    {' | '}Context: {msg.stats.promptTokens ?? 0} tk
                    {' | '}Output: {msg.stats.completionTokens ?? 0} tk]
                  </div>
                )}
              </div>
            );
          })}

          <div ref={messagesEndRef}/>
        </div>

        {/* ── Input ── */}
        <div className="input-area no-drag">
          <div className="chat-input-wrapper glass">
            <textarea
              className="chat-input"
              value={inputValue}
              onChange={e => setInputValue(e.target.value)}
              onKeyDown={handleKey}
              placeholder="Message Sofuu…"
              rows={1}
              style={{ resize:'none' }}
            />
            <button
              className="send-button"
              onClick={handleSend}
              disabled={!inputValue.trim() || isLoading}
            >
              <Send size={16}/>
            </button>
          </div>
        </div>
      </div>

      {/* ── Settings modal ── */}
      <SettingsModal
        isOpen={settingsOpen}
        onClose={() => setSettingsOpen(false)}
        config={config}
        setConfig={setConfig}
      />
    </div>
  );
}
