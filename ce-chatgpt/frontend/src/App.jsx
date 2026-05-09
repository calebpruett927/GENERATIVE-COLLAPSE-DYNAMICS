import React, { useState } from 'react';
import './App.css';

const CE_MODES = [
  { value: 'silent', label: 'Silent' },
  { value: 'lightweight', label: 'Lightweight' },
  { value: 'full', label: 'Full Spine' },
];

function App() {
  const [message, setMessage] = useState('');
  const [ceMode, setCeMode] = useState('lightweight');
  const [response, setResponse] = useState(null);
  const [loading, setLoading] = useState(false);

  const sendMessage = async () => {
    setLoading(true);
    const res = await fetch('http://localhost:8000/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message, ce_mode: ceMode }),
    });
    const data = await res.json();
    setResponse(data);
    setLoading(false);
  };

  return (
    <div className="app-container">
      <h1>Cognitive Equalizer ChatGPT</h1>
      <div className="input-row">
        <textarea
          value={message}
          onChange={e => setMessage(e.target.value)}
          placeholder="Type your message..."
        />
        <select value={ceMode} onChange={e => setCeMode(e.target.value)}>
          {CE_MODES.map(m => (
            <option key={m.value} value={m.value}>{m.label}</option>
          ))}
        </select>
        <button onClick={sendMessage} disabled={loading || !message}>
          {loading ? 'Sending...' : 'Send'}
        </button>
      </div>
      {response && (
        <div className="response-block">
          <div className="gpt-answer"><strong>GPT:</strong> {response.answer}</div>
          <div className="ce-audit">
            <strong>CE Audit:</strong>
            <pre>{JSON.stringify(response.ce_audit, null, 2)}</pre>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
