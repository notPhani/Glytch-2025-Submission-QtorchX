import React, { useState } from 'react';
import QSphere from './QSphere';
import './ComposerNew.css';

const ComposerNew: React.FC = () => {
  const [inspectEnabled, setInspectEnabled] = useState(false);

  // Quantum gates with IBM colors
  const gates = [
    { name: 'H', color: '#ff5555' },
    { name: 'X', color: '#4dd0e1' },
    { name: 'Y', color: '#ff3366' },
    { name: 'Z', color: '#4dd0e1' },
    { name: 'S', color: '#b0bec5' },
    { name: 'T', color: '#b0bec5' },
    { name: 'S†', color: '#b0bec5' },
    { name: 'T†', color: '#b0bec5' },
    { name: 'P', color: '#ffa726' },
    { name: 'RZ', color: '#ffa726' },
    { name: 'RX', color: '#ff3366' },
    { name: 'RY', color: '#ff3366' },
    { name: 'RXX', color: '#ff3366' },
    { name: 'RZZ', color: '#ff3366' },
    { name: 'U', color: '#b0bec5' },
    { name: 'RCCX', color: '#ff3366' },
    { name: 'RC3X', color: '#ff3366' },
    { name: 'Measure', color: '#4dd0e1' },
  ];

  return (
    <div className="composer-container">
      {/* Top IBM Header */}
      <header className="ibm-header">
        <div className="ibm-logo">
          <button className="menu-toggle">☰</button>
          <span className="ibm-text">IBM Quantum Platform</span>
        </div>
        <div className="header-right">
          <button className="header-icon">🔍</button>
          <button className="region-btn">us-east ▼</button>
          <button className="signin-btn">Sign in</button>
        </div>
      </header>

      {/* Circuit Header */}
      <div className="circuit-header">
        <div className="circuit-header-left">
          <h2 className="circuit-title">Untitled circuit</h2>
          <div className="header-menu">
            <button className="header-menu-btn">File</button>
            <button className="header-menu-btn">Edit</button>
            <button className="header-menu-btn">View</button>
            <button className="header-menu-btn">Help</button>
          </div>
        </div>
        <div className="circuit-header-right">
          <button className="header-btn save-btn">Save file ↓</button>
          <button className="header-btn run-btn">Set up and run</button>
          <button className="header-btn code-format-btn">Code format ▼</button>
          <button className="header-btn openqasm-btn">OpenQASM 2.0 ▼</button>
        </div>
      </div>

      {/* Toolbar */}
      <div className="toolbar">
        <div className="toolbar-left">
          <button className="toolbar-icon">🔍</button>
          <button className="toolbar-icon">≡</button>
          <button className="toolbar-icon">⊞</button>
          <div className="toolbar-sep"></div>
          <button className="toolbar-icon">↶</button>
          <button className="toolbar-icon">↷</button>
          <div className="toolbar-sep"></div>
          <select className="alignment-dropdown">
            <option>Left alignment</option>
          </select>
          <div className="toolbar-spacer"></div>
          <label className="inspect-toggle">
            <input
              type="checkbox"
              checked={inspectEnabled}
              onChange={(e) => setInspectEnabled(e.target.checked)}
            />
            <span className="toggle-circle"></span>
          </label>
          <span className="toggle-text">Inspect</span>
        </div>
      </div>

      {/* Main Layout */}
      <div className="main-layout">
        {/* Left: Operations */}
        <aside className="ops-panel">
          <div className="ops-header">
            <h3>Operations</h3>
            <button className="ops-search">🔍</button>
          </div>
          <div className="gates-grid">
            {gates.map((gate) => (
              <button
                key={gate.name}
                className="gate-btn"
                style={{ backgroundColor: gate.color }}
              >
                {gate.name}
              </button>
            ))}
          </div>
        </aside>

        {/* Center: Circuit */}
        <main className="circuit-center">
          <div className="circuit-qubit-labels">
            <div className="qubit-label">q[0]</div>
            <div className="qubit-label">q[1]</div>
            <div className="qubit-label">q[2]</div>
            <div className="qubit-label">q[3]</div>
            <div className="classical-label">c4</div>
          </div>

          <div className="circuit-workspace">
            <div className="qubit-row"><div className="qubit-wire"></div></div>
            <div className="qubit-row"><div className="qubit-wire"></div></div>
            <div className="qubit-row"><div className="qubit-wire"></div></div>
            <div className="qubit-row"><div className="qubit-wire"></div></div>
            <div className="classical-row"><div className="classical-wire"></div></div>
          </div>
        </main>

        {/* Right: Code & Visualization */}
        <aside className="viz-panel">
          {/* OpenQASM Section */}
          <div className="viz-section code-viz">
            <div className="viz-header">
              <h4>OpenQASM 2.0</h4>
              <button className="viz-menu">⋮</button>
            </div>
            <div className="code-display">
              <pre>{`OPENQASM 2.0;
include "qelib1.inc";

qreg q[4];
creg c[4];`}</pre>
            </div>
          </div>

          {/* Q-Sphere Section */}
          <div className="viz-section sphere-viz">
            <div className="viz-header">
              <h4>Q-sphere</h4>
              <button className="viz-menu">⋮</button>
            </div>
            <div className="sphere-container">
              <QSphere 
                size={160}
                state={{ 
                  theta: Math.PI / 4, 
                  phi: Math.PI / 4,
                  label: '|0⟩'
                }} 
              />
            </div>
            <div className="sphere-legend">
              <label><input type="checkbox" defaultChecked /> State</label>
              <label><input type="checkbox" /> Phase angle</label>
            </div>
          </div>

          {/* Probabilities Section */}
          <div className="viz-section prob-viz">
            <div className="viz-header">
              <h4>Probabilities</h4>
              <button className="viz-menu">⋮</button>
            </div>
            <div className="prob-chart">
              <div className="prob-bar">
                <span className="prob-label">|0000⟩</span>
                <div className="prob-bar-bg">
                  <div className="prob-bar-fill" style={{ width: '100%' }}></div>
                </div>
                <span className="prob-val">100%</span>
              </div>
            </div>
            <div className="prob-axis">
              <span>0</span><span>20</span><span>40</span><span>60</span><span>80</span><span>100</span>
            </div>
            <div className="prob-title">Computational basis states</div>
          </div>
        </aside>
      </div>
    </div>
  );
};

export default ComposerNew;
