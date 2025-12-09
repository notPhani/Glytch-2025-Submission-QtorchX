import React, { useState } from "react";
import "./ComposerPage.css";
import QSphere from "./QSphere";

// Gate library matching backend's GateLibrary structure
const GATES = [
  ["H", "X", "Y"],
  ["Z", "S", "T"],
  ["S†", "T†", "P"],
  ["RZ", "RX", "RY"],
  ["RXX", "RYY", "RZZ"],
  ["CNOT", "CZ", "SWAP"],
  ["TOFFOLI", "FREDKIN", "M"],
];

// Gate metadata matching backend's burst weights
const GATE_METADATA: Record<string, { numQubits: number; burstWeight: number; parameterized: boolean }> = {
  // Single-qubit gates
  'I': { numQubits: 1, burstWeight: 0.05, parameterized: false },
  'H': { numQubits: 1, burstWeight: 0.5, parameterized: false },
  'X': { numQubits: 1, burstWeight: 0.4, parameterized: false },
  'Y': { numQubits: 1, burstWeight: 0.4, parameterized: false },
  'Z': { numQubits: 1, burstWeight: 0.3, parameterized: false },
  'S': { numQubits: 1, burstWeight: 0.3, parameterized: false },
  'S†': { numQubits: 1, burstWeight: 0.3, parameterized: false },
  'T': { numQubits: 1, burstWeight: 0.35, parameterized: false },
  'T†': { numQubits: 1, burstWeight: 0.35, parameterized: false },
  'P': { numQubits: 1, burstWeight: 0.35, parameterized: true },
  'RX': { numQubits: 1, burstWeight: 0.5, parameterized: true },
  'RY': { numQubits: 1, burstWeight: 0.5, parameterized: true },
  'RZ': { numQubits: 1, burstWeight: 0.4, parameterized: true },
  // Two-qubit gates
  'CNOT': { numQubits: 2, burstWeight: 2.5, parameterized: false },
  'CZ': { numQubits: 2, burstWeight: 2.3, parameterized: false },
  'SWAP': { numQubits: 2, burstWeight: 3.0, parameterized: false },
  'RXX': { numQubits: 2, burstWeight: 2.8, parameterized: true },
  'RYY': { numQubits: 2, burstWeight: 2.8, parameterized: true },
  'RZZ': { numQubits: 2, burstWeight: 2.5, parameterized: true },
  // Three-qubit gates
  'TOFFOLI': { numQubits: 3, burstWeight: 8.0, parameterized: false },
  'FREDKIN': { numQubits: 3, burstWeight: 9.0, parameterized: false },
  // Measurement
  'M': { numQubits: 1, burstWeight: 1.0, parameterized: false },
};

// Gate structure matching backend's Gate dataclass
interface Gate {
  id: string;
  name: string;
  qubits: number[];
  params: number[];
  t: number | null; // Time step (auto-assigned by scheduler)
  label: string;
  metadata: {
    burstWeight: number;
    numQubits: number;
    gateType: 'static' | 'parameterized';
  };
}

const NUM_QUBITS = 4;
const MAX_DEPTH = 50; // Maximum circuit depth

export const ComposerPage: React.FC = () => {
  const qubits = Array.from({ length: NUM_QUBITS }, (_, i) => `q[${i}]`);
  const classical = `c[${NUM_QUBITS}]`;
  const [gates, setGates] = useState<Gate[]>([]);
  const [draggedGateName, setDraggedGateName] = useState<string | null>(null);
  const [selectedQubits, setSelectedQubits] = useState<number[]>([]);
  const [circuitDepth, setCircuitDepth] = useState<number>(0);
  const [noiseEnabled, setNoiseEnabled] = useState<boolean>(false);
  const [optimizationEnabled, setOptimizationEnabled] = useState<boolean>(false);

  // Circuit scheduler matching backend's Circuit.add() logic
  const scheduleGate = (gate: Gate): number => {
    const targetQubits = gate.qubits;
    
    // Find latest occupied time across target qubits
    let lastTime = -1;
    const grid: (Gate | null)[][] = Array.from({ length: NUM_QUBITS }, () => 
      Array(MAX_DEPTH).fill(null)
    );
    
    // Populate grid with existing gates
    gates.forEach(g => {
      if (g.t !== null) {
        g.qubits.forEach(q => {
          grid[q][g.t!] = g;
        });
      }
    });
    
    // Get last occupied time for target qubits
    targetQubits.forEach(q => {
      for (let t = MAX_DEPTH - 1; t >= 0; t--) {
        if (grid[q][t] !== null) {
          lastTime = Math.max(lastTime, t);
          break;
        }
      }
    });
    
    // For multi-qubit gates, block ALL qubits in span
    if (targetQubits.length > 1) {
      const minQubit = Math.min(...targetQubits);
      const maxQubit = Math.max(...targetQubits);
      
      for (let q = minQubit; q <= maxQubit; q++) {
        for (let t = MAX_DEPTH - 1; t >= 0; t--) {
          if (grid[q][t] !== null) {
            lastTime = Math.max(lastTime, t);
            break;
          }
        }
      }
    }
    
    // Find first available slot
    let timeSlot = lastTime + 1;
    while (timeSlot < MAX_DEPTH) {
      let conflict = false;
      
      // Check target qubits
      for (const q of targetQubits) {
        if (grid[q][timeSlot] !== null) {
          conflict = true;
          break;
        }
      }
      
      // Check span for multi-qubit gates
      if (!conflict && targetQubits.length > 1) {
        const minQubit = Math.min(...targetQubits);
        const maxQubit = Math.max(...targetQubits);
        
        for (let q = minQubit; q <= maxQubit; q++) {
          if (grid[q][timeSlot] !== null) {
            const existing = grid[q][timeSlot];
            // Allow single-qubit gates on non-target qubits
            if (targetQubits.includes(q) || existing!.qubits.length > 1) {
              conflict = true;
              break;
            }
          }
        }
      }
      
      if (!conflict) break;
      timeSlot++;
    }
    
    return timeSlot;
  };

  const handleDragStart = (e: React.DragEvent, gateName: string) => {
    e.dataTransfer.effectAllowed = 'copy';
    e.dataTransfer.setData('gateName', gateName);
    setDraggedGateName(gateName);
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = 'copy';
  };

  const handleDrop = (e: React.DragEvent, qubitIdx: number) => {
    e.preventDefault();
    const gateName = e.dataTransfer.getData('gateName');
    
    if (!gateName) return;
    
    const metadata = GATE_METADATA[gateName];
    if (!metadata) return;
    
    // For single-qubit gates, place immediately
    if (metadata.numQubits === 1) {
      addGate(gateName, [qubitIdx]);
      setDraggedGateName(null);
      return;
    }
    
    // For multi-qubit gates, collect qubits
    const newSelected = [...selectedQubits, qubitIdx];
    
    if (newSelected.length === metadata.numQubits) {
      addGate(gateName, newSelected);
      setSelectedQubits([]);
      setDraggedGateName(null);
    } else {
      setSelectedQubits(newSelected);
    }
  };

  const handleQubitClick = (qubitIdx: number) => {
    if (!draggedGateName) return;
    
    const metadata = GATE_METADATA[draggedGateName];
    if (!metadata) return;
    
    // For single-qubit gates, place immediately
    if (metadata.numQubits === 1) {
      addGate(draggedGateName, [qubitIdx]);
      setDraggedGateName(null);
      return;
    }
    
    // For multi-qubit gates, collect qubits
    const newSelected = [...selectedQubits, qubitIdx];
    
    if (newSelected.length === metadata.numQubits) {
      addGate(draggedGateName, newSelected);
      setSelectedQubits([]);
      setDraggedGateName(null);
    } else {
      setSelectedQubits(newSelected);
    }
  };

  const addGate = (gateName: string, qubits: number[], params: number[] = []) => {
    const metadata = GATE_METADATA[gateName];
    if (!metadata) return;
    
    // Get params for parameterized gates
    let gateParams = params;
    if (metadata.parameterized && params.length === 0) {
      const paramValue = prompt(`Enter parameter for ${gateName} (angle in radians):`, '1.57');
      if (paramValue === null) return;
      gateParams = [parseFloat(paramValue) || 0];
    }
    
    const newGate: Gate = {
      id: `${gateName}-${qubits.join(',')}-${Date.now()}`,
      name: gateName,
      qubits: qubits,
      params: gateParams,
      t: null,
      label: `Gate${gateName}${qubits.join('')}#${gates.length}`,
      metadata: {
        burstWeight: metadata.burstWeight,
        numQubits: metadata.numQubits,
        gateType: metadata.parameterized ? 'parameterized' : 'static',
      },
    };
    
    // Schedule the gate (auto-assign time step)
    newGate.t = scheduleGate(newGate);
    
    const updatedGates = [...gates, newGate];
    setGates(updatedGates);
    setCircuitDepth(Math.max(circuitDepth, newGate.t + 1));
  };

  const handleRemoveGate = (id: string) => {
    const updatedGates = gates.filter((g) => g.id !== id);
    setGates(updatedGates);
    
    // Recalculate depth
    const maxDepth = updatedGates.reduce((max, g) => Math.max(max, g.t || 0), 0);
    setCircuitDepth(maxDepth + 1);
  };

  const generateOpenQASM = () => {
    let code = `OPENQASM 2.0;
include "qelib1.inc";

qreg q[${NUM_QUBITS}];
creg c[${NUM_QUBITS}];

`;
    
    // Sort gates by time step, then by qubit
    const sortedGates = [...gates].sort((a, b) => {
      if (a.t !== b.t) return (a.t || 0) - (b.t || 0);
      return Math.min(...a.qubits) - Math.min(...b.qubits);
    });
    
    sortedGates.forEach((g) => {
      const gateName = g.name.toLowerCase().replace('†', 'dg');
      
      if (g.name === 'M') {
        // Measurement
        code += `measure q[${g.qubits[0]}] -> c[${g.qubits[0]}];\n`;
      } else if (g.qubits.length === 1) {
        // Single-qubit gate
        if (g.params.length > 0) {
          code += `${gateName}(${g.params.map(p => p.toFixed(4)).join(', ')}) q[${g.qubits[0]}];\n`;
        } else {
          code += `${gateName} q[${g.qubits[0]}];\n`;
        }
      } else if (g.qubits.length === 2) {
        // Two-qubit gate
        if (g.params.length > 0) {
          code += `${gateName}(${g.params.map(p => p.toFixed(4)).join(', ')}) q[${g.qubits[0]}], q[${g.qubits[1]}];\n`;
        } else {
          code += `${gateName} q[${g.qubits[0]}], q[${g.qubits[1]}];\n`;
        }
      } else {
        // Three-qubit gate
        code += `${gateName} q[${g.qubits.join(']}, q[')}];\n`;
      }
    });
    
    return code;
  };

  const visualizeCircuit = (): string => {
    let lines: string[] = [];
    
    for (let q = 0; q < NUM_QUBITS; q++) {
      let line = `q${q}: |0⟩─`;
      for (let t = 0; t < circuitDepth; t++) {
        const gateHere = gates.find(g => g.t === t && g.qubits.includes(q));
        
        if (gateHere) {
          if (q === Math.min(...gateHere.qubits)) {
            const params = gateHere.params.length > 0 ? `(${gateHere.params[0].toFixed(2)})` : '';
            line += `[${gateHere.name}${params}]─`;
          } else {
            line += '──●──';
          }
        } else {
          line += '─────';
        }
      }
      lines.push(line);
    }
    
    return lines.join('\n');
  };

  return (
    <div className="composer-root">
      {/* Top bar */}
      <header className="composer-topbar">
        <div className="topbar-left">
          <span className="logo-dot" />
          <span className="logo-text">QtorchX Composer</span>
          <span className="topbar-title">Untitled circuit</span>
        </div>
        <div className="topbar-right">
          <button 
            className="topbar-btn"
            onClick={() => {
              setGates([]);
              setCircuitDepth(0);
              setSelectedQubits([]);
              setDraggedGateName(null);
            }}
          >
            Clear Circuit
          </button>
          <button 
            className="topbar-btn primary"
            onClick={() => {
              const backendConfig = {
                simulate_with_noise: noiseEnabled,
                fusion_optimizations: optimizationEnabled,
                persistant_data: true,
              };
              console.log('Circuit:', { gates, depth: circuitDepth });
              console.log('Backend Config:', backendConfig);
              console.log('OpenQASM:', generateOpenQASM());
              alert(`Circuit ready!\nGates: ${gates.length}\nDepth: ${circuitDepth}\nNoise: ${noiseEnabled ? 'ON' : 'OFF'}\nOptimization: ${optimizationEnabled ? 'ON' : 'OFF'}\n\nCheck console for full circuit data.`);
            }}
          >
            Set up and run
          </button>
        </div>
      </header>

      <div className="composer-main">
        {/* LEFT: gate library */}
        <aside className="composer-sidebar">
          <div className="sidebar-header">
            Operations
            {draggedGateName && (
              <div style={{ fontSize: '10px', color: '#888', marginTop: '4px' }}>
                {GATE_METADATA[draggedGateName]?.numQubits === 1 
                  ? 'Click qubit to place'
                  : `Select ${GATE_METADATA[draggedGateName]?.numQubits} qubits (${selectedQubits.length} selected)`
                }
              </div>
            )}
          </div>
          <div className="gate-grid">
            {GATES.map((row, r) =>
              row.map((g, i) => (
                <button
                  key={`${g}-${r}-${i}`}
                  className={`gate-btn gate-${g} ${draggedGateName === g ? 'gate-active' : ''}`}
                  draggable
                  onDragStart={(e) => handleDragStart(e, g)}
                  onDragEnd={() => setDraggedGateName(null)}
                  onClick={() => {
                    setDraggedGateName(g);
                    setSelectedQubits([]);
                  }}
                  title={`${g} gate (${GATE_METADATA[g]?.numQubits || 1} qubit${(GATE_METADATA[g]?.numQubits || 1) > 1 ? 's' : ''})`}
                >
                  {g}
                </button>
              ))
            )}
          </div>
          <div style={{ marginTop: '12px', fontSize: '11px', color: '#888' }}>
            <div><strong>Circuit Stats:</strong></div>
            <div>Gates: {gates.length}</div>
            <div>Depth: {circuitDepth}</div>
          </div>
        </aside>

        {/* CENTER: circuit canvas */}
        <section className="composer-center">
          <canvas 
            className="circuit-canvas"
            width={1600}
            height={600}
            ref={(canvas) => {
              if (!canvas) return;
              const ctx = canvas.getContext('2d');
              if (!ctx) return;

              // Clear canvas
              ctx.fillStyle = '#1a1d29';
              ctx.fillRect(0, 0, canvas.width, canvas.height);

              const padding = 60;
              const wireSpacing = 100;
              const gateSize = 40;
              const timeStepWidth = 80;

              // Draw qubit labels and wires
              qubits.forEach((q, idx) => {
                const y = padding + idx * wireSpacing;
                
                // Qubit label
                ctx.fillStyle = '#888a98';
                ctx.font = '14px IBM Plex Mono, monospace';
                ctx.fillText(q, 10, y + 5);

                // Wire line
                ctx.strokeStyle = '#3a5a7f';
                ctx.lineWidth = 2;
                ctx.beginPath();
                ctx.moveTo(padding, y);
                ctx.lineTo(canvas.width - padding, y);
                ctx.stroke();
              });

              // Draw classical bit wire
              const classicalY = padding + NUM_QUBITS * wireSpacing;
              ctx.fillStyle = '#f0a020';
              ctx.fillText(classical, 10, classicalY + 5);
              ctx.strokeStyle = '#f0a020';
              ctx.lineWidth = 2;
              ctx.setLineDash([5, 5]);
              ctx.beginPath();
              ctx.moveTo(padding, classicalY);
              ctx.lineTo(canvas.width - padding, classicalY);
              ctx.stroke();
              ctx.setLineDash([]);

              // Draw time step markers
              ctx.fillStyle = '#555';
              ctx.font = '11px IBM Plex Mono, monospace';
              const maxSteps = Math.max(circuitDepth, 10);
              for (let t = 0; t < maxSteps; t++) {
                const x = padding + 40 + t * timeStepWidth;
                ctx.fillText(`t${t}`, x - 8, padding - 30);
              }

              // Draw gates
              gates.forEach((gate) => {
                if (gate.t === null) return;

                const minQubit = Math.min(...gate.qubits);
                const maxQubit = Math.max(...gate.qubits);
                const x = padding + 40 + gate.t * timeStepWidth;

                // Multi-qubit connection line
                if (gate.qubits.length > 1) {
                  const y1 = padding + minQubit * wireSpacing;
                  const y2 = padding + maxQubit * wireSpacing;
                  ctx.strokeStyle = '#0f62fe';
                  ctx.lineWidth = 2;
                  ctx.beginPath();
                  ctx.moveTo(x, y1);
                  ctx.lineTo(x, y2);
                  ctx.stroke();
                }

                // Draw gate boxes
                gate.qubits.forEach((q) => {
                  const y = padding + q * wireSpacing;

                  // Gate color based on type (matching CSS palette)
                  let gateColor = '#ff5050'; // Default: H, X, Y (red)
                  if (['Z', 'S', 'T', 'S†', 'T†', 'P', 'I'].includes(gate.name)) gateColor = '#7fd5ff'; // Light cyan
                  if (['RZ', 'RX', 'RY', 'RXX', 'RYY', 'RZZ', 'U', 'TOFFOLI', 'FREDKIN', 'RCCX', 'RC3X'].includes(gate.name)) gateColor = '#ff7fbf'; // Pink/Magenta
                  if (['CNOT', 'CZ', 'SWAP', 'M'].includes(gate.name)) gateColor = '#4da6ff'; // Blue

                  // Control/target markers for multi-qubit gates
                  if (gate.qubits.length > 1 && q !== minQubit) {
                    ctx.fillStyle = '#0f62fe';
                    ctx.beginPath();
                    ctx.arc(x, y, 8, 0, 2 * Math.PI);
                    ctx.fill();
                  } else {
                    // Gate box
                    ctx.fillStyle = gateColor;
                    ctx.fillRect(x - gateSize / 2, y - gateSize / 2, gateSize, gateSize);
                    ctx.strokeStyle = '#1a1d29';
                    ctx.lineWidth = 2;
                    ctx.strokeRect(x - gateSize / 2, y - gateSize / 2, gateSize, gateSize);

                    // Gate label
                    ctx.fillStyle = '#000';
                    ctx.font = 'bold 12px IBM Plex Sans, sans-serif';
                    ctx.textAlign = 'center';
                    ctx.textBaseline = 'middle';
                    ctx.fillText(gate.name, x, y);
                  }
                });
              });

              ctx.textAlign = 'left';
              ctx.textBaseline = 'alphabetic';
            }}
            onDragOver={handleDragOver}
            onClick={(e) => {
              if (!draggedGateName) return;
              const canvas = e.currentTarget;
              const rect = canvas.getBoundingClientRect();
              const y = e.clientY - rect.top;
              
              const padding = 60;
              const wireSpacing = 100;
              const qubitIdx = Math.floor((y - padding + wireSpacing / 2) / wireSpacing);
              
              if (qubitIdx >= 0 && qubitIdx < NUM_QUBITS) {
                handleQubitClick(qubitIdx);
              }
            }}
          />
        </section>

        {/* Hidden div for drag-drop compatibility */}
        <div style={{ display: 'none' }}>
          <div className="circuit-grid">
            {qubits.map((q, qubitIdx) => (
              <div key={q} className="circuit-row">
                <span 
                  className={`row-label ${selectedQubits.includes(qubitIdx) ? 'row-label-selected' : ''}`}
                  onClick={() => handleQubitClick(qubitIdx)}
                  onDragOver={handleDragOver}
                  onDrop={(e) => handleDrop(e, qubitIdx)}
                  style={{ cursor: draggedGateName ? 'pointer' : 'default' }}
                  title={draggedGateName ? `Drop ${draggedGateName} on ${q}` : q}
                >
                  {q}
                </span>
                <div 
                  className="row-wire-container"
                  onDragOver={handleDragOver}
                  onDrop={(e) => handleDrop(e, qubitIdx)}
                >
                  <div className="row-wire" />
                  <div className="gate-slots">
                    {Array.from({ length: circuitDepth || 10 }).map((_, timeStep) => {
                      const gateHere = gates.find(g => g.t === timeStep && g.qubits.includes(qubitIdx));
                      
                      return (
                        <div
                          key={timeStep}
                          className="gate-slot"
                        >
                          {gateHere && (
                            // Only render gate on its control qubit
                            qubitIdx === Math.min(...gateHere.qubits) ? (
                              <button
                                className={`placed-gate gate-${gateHere.name}`}
                                onClick={() => handleRemoveGate(gateHere.id)}
                                title={`${gateHere.name} on q[${gateHere.qubits.join(',')}] at t=${gateHere.t}\nClick to remove`}
                              >
                                {gateHere.name}
                                {gateHere.params.length > 0 && (
                                  <span style={{ fontSize: '8px', display: 'block' }}>
                                    ({gateHere.params[0].toFixed(2)})
                                  </span>
                                )}
                              </button>
                            ) : (
                              // Target qubit marker for multi-qubit gates
                              <div className="gate-target-marker">●</div>
                            )
                          )}
                        </div>
                      );
                    })}
                  </div>
                </div>
              </div>
            ))}
            <div className="circuit-row">
              <span className="row-label classical">{classical}</span>
              <div className="row-wire-container">
                <div className="row-wire classical-wire" />
              </div>
            </div>
          </div>
        </div>

        {/* RIGHT: Q-sphere + settings + probabilities */}
        <aside className="composer-right">
          {/* Circuit Visualization */}
          <div className="panel panel-code" style={{ flex: '0.6' }}>
            <div className="panel-header">Circuit Diagram</div>
            <pre className="code-block" style={{ fontSize: '10px', lineHeight: '1.2' }}>
              {visualizeCircuit()}
            </pre>
          </div>

          {/* Settings Panel */}
          <div className="panel panel-settings">
            <div className="panel-header">Settings</div>
            <div className="settings-content">
              <div className="setting-row">
                <div className="setting-info">
                  <div className="setting-label">Noise Simulation</div>
                  <div className="setting-description">Enable realistic quantum noise</div>
                </div>
                <label className="toggle-switch">
                  <input 
                    type="checkbox" 
                    checked={noiseEnabled}
                    onChange={(e) => setNoiseEnabled(e.target.checked)}
                  />
                  <span className="toggle-slider"></span>
                </label>
              </div>
              <div className="setting-row">
                <div className="setting-info">
                  <div className="setting-label">Circuit Optimization</div>
                  <div className="setting-description">Optimize gate sequences</div>
                </div>
                <label className="toggle-switch">
                  <input 
                    type="checkbox" 
                    checked={optimizationEnabled}
                    onChange={(e) => setOptimizationEnabled(e.target.checked)}
                  />
                  <span className="toggle-slider"></span>
                </label>
              </div>
            </div>
          </div>

          {/* Q-sphere with Three.js */}
          <div className="panel panel-qsphere">
            <div className="panel-header">Q-sphere</div>
            <div className="qsphere-canvas">
              <QSphere 
                state={{ 
                  theta: Math.PI / 4, 
                  phi: Math.PI / 4, 
                  label: '|ψ⟩' 
                }} 
                size={220} 
              />
            </div>
            <div className="qsphere-footer">
              <label>
                <input type="checkbox" defaultChecked /> State
              </label>
              <label>
                <input type="checkbox" /> Phase angle
              </label>
            </div>
          </div>

          {/* Probabilities / histogram */}
          <div className="panel panel-prob">
            <div className="panel-header">Probabilities</div>
            <div className="prob-row">
              <span className="prob-label">|0000⟩</span>
              <div className="prob-bar-track">
                <div className="prob-bar-fill" style={{ width: "100%" }} />
              </div>
              <span className="prob-pct">100%</span>
            </div>
            <div className="prob-axis">
              <span>0</span>
              <span>20</span>
              <span>40</span>
              <span>60</span>
              <span>80</span>
              <span>100</span>
            </div>
            <div className="prob-caption">
              Computational basis states
            </div>
          </div>
        </aside>
      </div>

      {/* Bottom Section: 50% Histogram + 50% Bar Graph */}
      <div className="composer-bottom">
        {/* State Histogram - Left 50% */}
        <div className="panel panel-histogram">
          <div className="panel-header">State Probability Distribution</div>
          <div className="histogram-container">
            {['|0000⟩', '|0001⟩', '|0010⟩', '|0011⟩', '|0100⟩', '|0101⟩', '|0110⟩', '|0111⟩'].map((state, idx) => (
              <div key={state} className="histogram-bar-wrapper">
                <div 
                  className="histogram-bar" 
                  style={{ 
                    height: idx === 0 ? '100%' : '0%',
                    background: 'linear-gradient(180deg, #0f62fe, #3dd9ff)'
                  }}
                >
                  {idx === 0 && <span className="histogram-value">1.00</span>}
                </div>
                <span className="histogram-label">{state}</span>
              </div>
            ))}
          </div>
          <div className="histogram-y-axis">
            <span>1.0</span>
            <span>0.75</span>
            <span>0.50</span>
            <span>0.25</span>
            <span>0.0</span>
          </div>
        </div>

        {/* Measurement Statistics - Right 50% */}
        <div className="panel panel-bargraph">
          <div className="panel-header">Measurement Statistics</div>
          <div className="bargraph-container">
            <div className="bargraph-row">
              <span className="bargraph-label">Qubit 0</span>
              <div className="bargraph-track">
                <div className="bargraph-fill bargraph-zero" style={{ width: '50%' }}>
                  <span className="bargraph-value">0.50</span>
                </div>
                <div className="bargraph-fill bargraph-one" style={{ width: '50%' }}>
                  <span className="bargraph-value">0.50</span>
                </div>
              </div>
              <div className="bargraph-legend">
                <span className="legend-zero">|0⟩</span>
                <span className="legend-one">|1⟩</span>
              </div>
            </div>
            <div className="bargraph-row">
              <span className="bargraph-label">Qubit 1</span>
              <div className="bargraph-track">
                <div className="bargraph-fill bargraph-zero" style={{ width: '50%' }}>
                  <span className="bargraph-value">0.50</span>
                </div>
                <div className="bargraph-fill bargraph-one" style={{ width: '50%' }}>
                  <span className="bargraph-value">0.50</span>
                </div>
              </div>
              <div className="bargraph-legend">
                <span className="legend-zero">|0⟩</span>
                <span className="legend-one">|1⟩</span>
              </div>
            </div>
            <div className="bargraph-row">
              <span className="bargraph-label">Qubit 2</span>
              <div className="bargraph-track">
                <div className="bargraph-fill bargraph-zero" style={{ width: '50%' }}>
                  <span className="bargraph-value">0.50</span>
                </div>
                <div className="bargraph-fill bargraph-one" style={{ width: '50%' }}>
                  <span className="bargraph-value">0.50</span>
                </div>
              </div>
              <div className="bargraph-legend">
                <span className="legend-zero">|0⟩</span>
                <span className="legend-one">|1⟩</span>
              </div>
            </div>
            <div className="bargraph-row">
              <span className="bargraph-label">Qubit 3</span>
              <div className="bargraph-track">
                <div className="bargraph-fill bargraph-zero" style={{ width: '50%' }}>
                  <span className="bargraph-value">0.50</span>
                </div>
                <div className="bargraph-fill bargraph-one" style={{ width: '50%' }}>
                  <span className="bargraph-value">0.50</span>
                </div>
              </div>
              <div className="bargraph-legend">
                <span className="legend-zero">|0⟩</span>
                <span className="legend-one">|1⟩</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ComposerPage;
