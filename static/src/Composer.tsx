import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import QSphere from './QSphere';
import './Composer.css';

const Composer: React.FC = () => {
  const navigate = useNavigate();
  const [noiseEnabled, setNoiseEnabled] = useState(false);
  const [optimizationEnabled, setOptimizationEnabled] = useState(false);
  const [qubits] = useState(4);

  const handleBack = () => {
    navigate('/');
  };

  return (
    <div className="composer-container">
      {/* Header */}
      <header className="composer-header">
        <div className="header-left">
          <button className="back-button" onClick={handleBack}>
            ← Back
          </button>
          <h1 className="header-title">Untitled circuit</h1>
        </div>
        <div className="header-right">
          <button className="save-button">Save file</button>
          <button className="run-button">⚡ Set up and run</button>
        </div>
      </header>

      <div className="composer-content">
        {/* Left Sidebar - Operations */}
        <aside className="operations-panel">
          <div className="operations-header">
            <h2>Operations</h2>
            <button className="search-button" title="Search">🔍</button>
          </div>

          <div className="operations-grid">
            {/* Row 1 */}
            <button className="op-button op-h">H</button>
            <button className="op-button op-x">X</button>
            <button className="op-button op-y">Y</button>
            <button className="op-button op-z">Z</button>
            <button className="op-button op-s">S</button>
            <button className="op-button op-t">T</button>

            {/* Row 2 */}
            <button className="op-button op-t-dag">T†</button>
            <button className="op-button op-s-dag">S†</button>
            <button className="op-button op-z">Z</button>
            <button className="op-button op-t-prime">T'</button>
            <button className="op-button op-s-prime">S'</button>
            <button className="op-button op-p">P</button>

            {/* Row 3 */}
            <button className="op-button op-rz">RZ</button>
            <button className="op-button op-rx">RX</button>
            <button className="op-button op-ry">RY</button>
            <button className="op-button op-rx">RX</button>
            <button className="op-button op-ry">RY</button>
            <button className="op-button op-rxx">RXX</button>

            {/* Row 4 */}
            <button className="op-button op-rzz">RZZ</button>
            <button className="op-button op-u">U</button>
            <button className="op-button op-rccx">RCCX</button>
            <button className="op-button op-rc3x">RC3X</button>
            <button className="op-button op-measure">⊙</button>
          </div>
        </aside>

        {/* Main Content */}
        <main className="main-content">
          {/* Toolbar */}
          <div className="toolbar">
            <button className="toolbar-button" title="Undo">↶</button>
            <button className="toolbar-button" title="Redo">↷</button>
            <button className="toolbar-button" title="Delete">🗑</button>

            <div className="toolbar-divider"></div>

            {/* Left alignment */}
            <button className="toolbar-button" title="Left alignment">⬅</button>

            {/* Inspect toggle */}
            <label className="toolbar-toggle">
              <input type="checkbox" defaultChecked={false} />
              <span>Inspect</span>
            </label>

            <div className="toolbar-divider"></div>

            {/* Noise Toggle */}
            <label className="toolbar-toggle noise-toggle">
              <input 
                type="checkbox" 
                checked={noiseEnabled}
                onChange={(e) => setNoiseEnabled(e.target.checked)}
              />
              <span className="toggle-label">🌀 Noise</span>
            </label>

            {/* Optimization Toggle */}
            <label className="toolbar-toggle optimization-toggle">
              <input 
                type="checkbox" 
                checked={optimizationEnabled}
                onChange={(e) => setOptimizationEnabled(e.target.checked)}
              />
              <span className="toggle-label">⚙ Optimization</span>
            </label>
          </div>

          {/* Circuit Editor Area */}
          <div className="circuit-editor">
            <div className="circuit-info">
              <div className="circuit-metadata">
                <h3>Circuit Metadata</h3>
                <div className="metadata-content">
                  <p><strong>Qubits:</strong> {qubits}</p>
                  <p><strong>Classical bits:</strong> 0</p>
                  <p><strong>Depth:</strong> 0</p>
                </div>
              </div>
            </div>

            {/* Circuit Canvas */}
            <div className="circuit-canvas">
              <div className="circuit-qubits">
                {Array.from({ length: qubits }).map((_, i) => (
                  <div key={i} className="qubit-line">
                    <span className="qubit-label">q[{i}]</span>
                    <div className="qubit-wire"></div>
                  </div>
                ))}
                <div className="classical-line">
                  <span className="classical-label">c4</span>
                  <div className="classical-wire"></div>
                </div>
              </div>
            </div>
          </div>
        </main>

        {/* Right Sidebar - Visualization */}
        <aside className="visualization-panel">
          <div className="viz-header">
            <h3>OpenQASM 2.0</h3>
            <button className="refresh-button">🔄</button>
          </div>

          <div className="viz-content">
            <div className="qasm-code">
              <pre>{`OPENQASM 2.0;
include "qelib1.inc";

qreg q[4];
creg c[4];

// Circuit definition
// Add gates here`}</pre>
            </div>
          </div>

          <div className="viz-section">
            <h4>Q-Sphere</h4>
            <div className="q-sphere-placeholder">
              <QSphere 
                size={220}
                state={{ 
                  theta: Math.PI / 4, 
                  phi: Math.PI / 4,
                  label: '|0000⟩'
                }} 
              />
            </div>
          </div>

          <div className="viz-section">
            <h4>Probabilities</h4>
            <div className="probabilities-chart">
              <div className="prob-bar">
                <div className="prob-label">|0000⟩</div>
                <div className="prob-value" style={{ width: '100%' }}>100%</div>
              </div>
            </div>
          </div>
        </aside>
      </div>
    </div>
  );
};

export default Composer;
