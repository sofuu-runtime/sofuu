import React, { useState, useEffect } from 'react';
import { Settings, X } from 'lucide-react';

export default function SettingsModal({ isOpen, onClose, config, setConfig }) {
  const [localConfig, setLocalConfig] = useState(config);

  // Sync whenever modal opens so we show the latest saved config
  useEffect(() => { if (isOpen) setLocalConfig(config); }, [isOpen]);

  if (!isOpen) return null;

  const handleChange = (e) => {
    const { name, value } = e.target;
    setLocalConfig({ ...localConfig, [name]: value });
  };

  const handleSave = () => {
    setConfig(localConfig);
    onClose();
  };

  return (
    <div className="modal-overlay">
      <div className="modal-content glass">
        <div className="modal-header">
          <h2 className="modal-title">Settings</h2>
          <button className="close-btn" onClick={onClose}><X size={20} /></button>
        </div>

        <div className="form-group">
          <label className="form-label">Provider</label>
          <select 
            name="provider" 
            className="form-select" 
            value={localConfig.provider} 
            onChange={handleChange}
          >
            <option value="openrouter">OpenRouter</option>
            <option value="openai">OpenAI</option>
            <option value="anthropic">Anthropic</option>
            <option value="gemini">Gemini</option>
            <option value="ollama">Ollama (Local)</option>
          </select>
        </div>

        <div className="form-group">
          <label className="form-label">Model Name</label>
          <input 
            type="text" 
            name="model" 
            className="form-input" 
            value={localConfig.model} 
            onChange={handleChange} 
            placeholder="e.g. nvidia/nemotron-3-super..."
          />
        </div>

        <div className="form-group">
          <label className="form-label">API Key</label>
          <input 
            type="password" 
            name="apiKey" 
            className="form-input" 
            value={localConfig.apiKey} 
            onChange={handleChange} 
            placeholder="sk-..."
          />
        </div>

        <div className="form-group">
          <label className="form-label">QTSQ Brain ID (Memory)</label>
          <input 
            type="text" 
            name="brainId" 
            className="form-input" 
            value={localConfig.brainId} 
            onChange={handleChange} 
            placeholder="e.g. extreme_test"
          />
        </div>

        <div className="form-group" style={{ flexDirection: 'row', alignItems: 'center', gap: '0.5rem', marginTop: '0.5rem' }}>
          <input 
            type="checkbox" 
            name="think" 
            id="think_toggle"
            checked={localConfig.think} 
            onChange={(e) => setLocalConfig({ ...localConfig, think: e.target.checked })} 
          />
          <label className="form-label" htmlFor="think_toggle" style={{ margin: 0 }}>Enable Think Mode</label>
        </div>

        <button className="save-btn" onClick={handleSave}>Save Configuration</button>
      </div>
    </div>
  );
}
