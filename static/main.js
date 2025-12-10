// ========================================================================
// QTORCHX QUANTUM CIRCUIT SIMULATOR - FRONTEND
// ========================================================================
// Features:
// - High-DPI canvas rendering (Retina/4K support)
// - Drag-and-drop gate placement
// - Real-time circuit visualization
// - Phi manifold heatmap with bilinear interpolation
// - Histogram display (ideal vs noisy)
// - Bloch sphere integration
// - Backend API integration
// ========================================================================

// ---- CONFIG ----
const N_QUBITS = 4;
const HAS_CLASSICAL = true;
const N_STEPS = 15;
const MAX_GATES = 15;
// ========================================================================
// GATE COLOR SCHEME
// ========================================================================

const GATE_COLORS = {
  // Basic gates (Pastel Blue/Purple family)
  'H': { fill: '#9b59b6', stroke: '#8e44ad', text: '#fff' },      // Purple
  'X': { fill: '#3498db', stroke: '#2980b9', text: '#fff' },      // Blue
  'Y': { fill: '#5dade2', stroke: '#3498db', text: '#fff' },      // Light Blue
  'Z': { fill: '#a29bfe', stroke: '#6c5ce7', text: '#fff' },      // Periwinkle
  'I': { fill: '#bdc3c7', stroke: '#95a5a6', text: '#2c3e50' },   // Gray (identity)
  
  // Phase gates (Pastel Green/Teal family)
  'S': { fill: '#1abc9c', stroke: '#16a085', text: '#fff' },      // Teal
  'T': { fill: '#2ecc71', stroke: '#27ae60', text: '#fff' },      // Green
  'SDG': { fill: '#48c9b0', stroke: '#1abc9c', text: '#fff' },    // Aqua
  'TDG': { fill: '#58d68d', stroke: '#2ecc71', text: '#fff' },    // Mint
  
  // Rotation gates (Pastel Orange/Yellow family)
  'RX': { fill: '#f39c12', stroke: '#e67e22', text: '#fff' },     // Orange
  'RY': { fill: '#f1c40f', stroke: '#f39c12', text: '#2c3e50' },  // Yellow
  'RZ': { fill: '#e67e22', stroke: '#d35400', text: '#fff' },     // Dark Orange
  'Rx': { fill: '#f39c12', stroke: '#e67e22', text: '#fff' },     // Alias
  'Ry': { fill: '#f1c40f', stroke: '#f39c12', text: '#2c3e50' },  // Alias
  'Rz': { fill: '#e67e22', stroke: '#d35400', text: '#fff' },     // Alias
  
  // Two-qubit gates (Darker, contrasting colors)
  'CNOT': { fill: '#e74c3c', stroke: '#c0392b', text: '#fff' },   // Red
  'CX': { fill: '#e74c3c', stroke: '#c0392b', text: '#fff' },     // Red (alias)
  'CZ': { fill: '#c0392b', stroke: '#922b21', text: '#fff' },     // Dark Red
  'SWAP': { fill: '#8e44ad', stroke: '#6c3483', text: '#fff' },   // Purple
  'CY': { fill: '#d91e48', stroke: '#a93226', text: '#fff' },     // Crimson
  
  // Three-qubit gates (Very dark, high contrast)
  'TOFFOLI': { fill: '#1a1a2e', stroke: '#16213e', text: '#fff' }, // Near Black
  'CCNOT': { fill: '#1a1a2e', stroke: '#16213e', text: '#fff' },   // Near Black
  'FREDKIN': { fill: '#0f3460', stroke: '#16213e', text: '#fff' }, // Navy
  
  // Special gates
  'U': { fill: '#ff6b6b', stroke: '#ee5a6f', text: '#fff' },      // Coral
  'U1': { fill: '#ff6b6b', stroke: '#ee5a6f', text: '#fff' },     
  'U2': { fill: '#ff8787', stroke: '#ff6b6b', text: '#fff' },     
  'U3': { fill: '#ffa07a', stroke: '#ff8787', text: '#fff' },     
  
  // Measurement (Special - Blue)
  'M': { fill: '#2980b9', stroke: '#1f618d', text: '#fff' },      // Measurement Blue
  
  // Default fallback
  'DEFAULT': { fill: '#34495e', stroke: '#2c3e50', text: '#fff' }
};

function getGateColor(gateName) {
  return GATE_COLORS[gateName.toUpperCase()] || GATE_COLORS['DEFAULT'];
}


// Circuit data structure: circuit[q][t] = gate object | null
const circuit = Array.from({ length: N_QUBITS }, () =>
  Array.from({ length: N_STEPS }, () => null)
);

let gateList = []; // List of all placed gates

// Canvas + geometry
const canvas = document.getElementById('circuitCanvas');
const ctx = canvas.getContext('2d');
const rows = N_QUBITS + (HAS_CLASSICAL ? 1 : 0);
let W, H, cellW, cellH;

// ========================================================================
// HIGH-DPI CANVAS INITIALIZATION
// ========================================================================

function syncGeometry() {
  const container = document.getElementById('circuit-inner');
  const rect = container.getBoundingClientRect();
  const padding = 4;
  
  // CSS dimensions (what user sees)
  const displayWidth = rect.width - 2 * padding;
  const displayHeight = rect.height - 2 * padding;
  
  // Device pixel ratio (2x on Retina, 1x on standard)
  const dpr = window.devicePixelRatio || 1;
  
  // Set actual canvas size (high resolution)
  canvas.width = displayWidth * dpr;
  canvas.height = displayHeight * dpr;
  
  // Set CSS size (what gets displayed)
  canvas.style.width = displayWidth + 'px';
  canvas.style.height = displayHeight + 'px';
  
  // Scale context to match
  ctx.scale(dpr, dpr);
  
  // Use CSS dimensions for geometry calculations
  W = displayWidth;
  H = displayHeight;
  cellW = W / N_STEPS;
  cellH = H / rows;
  
  console.log(`📐 Circuit canvas: ${canvas.width}×${canvas.height} (${dpr}x DPR), Display: ${W}×${H}`);
}

syncGeometry();

window.addEventListener('resize', () => {
  syncGeometry();
  drawCircuit();
  syncHistogramCanvas();
});

// ========================================================================
// GATE FILTERING & SEARCH
// ========================================================================

let selectedGateType = null;
let dragging = false;
let dragGate = null;
let dragX = 0, dragY = 0;
let activeFilters = new Set();
let searchQuery = '';

const gateCategories = {
  'H': 'basic',
  'X': 'basic',
  'Y': 'basic',
  'Z': 'basic',
  'I': 'basic',
  'S': 'rotation',
  'T': 'rotation',
  'Rx': 'rotation',
  'Ry': 'rotation',
  'Rz': 'rotation',
  'CNOT': 'multi',
  'SWAP': 'multi',
  'CZ': 'multi'
};

function updateGateVisibility() {
  const gateButtons = document.querySelectorAll('.gate-btn');
  gateButtons.forEach(btn => {
    const gateName = btn.dataset.gate;
    if (!gateName) return;
    
    let shouldShow = true;
    
    // Check search query
    if (searchQuery && !gateName.toLowerCase().includes(searchQuery.toLowerCase())) {
      shouldShow = false;
    }
    
    // Check filters
    if (activeFilters.size > 0) {
      const category = gateCategories[gateName];
      if (!category || !activeFilters.has(category)) {
        shouldShow = false;
      }
    }
    
    btn.style.display = shouldShow ? 'block' : 'none';
  });
}

// Search functionality
const searchInput = document.getElementById('gateSearch');
if (searchInput) {
  searchInput.addEventListener('input', (e) => {
    searchQuery = e.target.value.trim();
    updateGateVisibility();
  });
}

// Filter functionality
document.querySelectorAll('.filter-btn').forEach(filterBtn => {
  filterBtn.addEventListener('click', () => {
    const filter = filterBtn.dataset.filter;
    if (activeFilters.has(filter)) {
      activeFilters.delete(filter);
      filterBtn.classList.remove('active');
    } else {
      activeFilters.add(filter);
      filterBtn.classList.add('active');
    }
    updateGateVisibility();
  });
});

// ========================================================================
// GATE LIBRARY - SELECTION & DRAG
// ========================================================================

document.querySelectorAll('.gate-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    selectedGateType = btn.dataset.gate;
    document.querySelectorAll('.gate-btn')
      .forEach(b => b.classList.remove('active-gate'));
    btn.classList.add('active-gate');
  });
  
  btn.addEventListener('mousedown', e => {
    const gateName = btn.dataset.gate;
    if (!gateName) return;
    
    dragging = true;
    dragGate = { name: gateName };
    
    const rect = canvas.getBoundingClientRect();
    dragX = rect.width * 0.15;
    dragY = rect.height * 0.25;
    
    drawCircuit();
    e.preventDefault();
  });
});

// ========================================================================
// HELPER FUNCTIONS
// ========================================================================

function getCellFromCoords(x, y) {
  const t = Math.floor(x / cellW);
  const r = Math.floor(y / cellH);
  
  if (t < 0 || t >= N_STEPS || r < 0 || r >= rows) {
    return { q: null, t: null };
  }
  
  return { q: r, t };
}

function gateCount() {
  let count = 0;
  for (let q = 0; q < N_QUBITS; q++) {
    for (let t = 0; t < N_STEPS; t++) {
      if (circuit[q][t]) count++;
    }
  }
  return count;
}

function getCnotQubits(dropQ) {
  const target = dropQ;
  const control = dropQ + 1 < N_QUBITS ? dropQ + 1 : dropQ;
  if (control === target) return null;
  return [control, target];
}

function arraysEqual(a, b) {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}

function placeGate(name, q, t) {
  if (q == null || t == null) return false;
  if (q < 0 || q >= N_QUBITS) return false;
  if (t === N_STEPS - 1 && name === 'M') return false;
  
  let qubits;
  if (name === 'CNOT') {
    const pair = getCnotQubits(q);
    if (!pair) return false;
    qubits = pair;
  } else {
    qubits = [q];
  }
  
  // Check if same gate already exists (toggle delete)
  let existingGate = null;
  let fullyOccupied = true;
  
  for (const qq of qubits) {
    const g = circuit[qq][t];
    if (!g) {
      fullyOccupied = false;
      break;
    }
    if (!existingGate) existingGate = g;
    else if (existingGate !== g) {
      fullyOccupied = false;
      break;
    }
  }
  
  // Toggle delete
  if (fullyOccupied && existingGate && existingGate.name === name) {
    gateList = gateList.filter(
      g => !(g.t === t && arraysEqual(g.qubits, existingGate.qubits) && g.name === existingGate.name)
    );
    for (const qq of existingGate.qubits) {
      circuit[qq][t] = null;
    }
    return true;
  }
  
  // Check for conflicts
  for (const qq of qubits) {
    const g = circuit[qq][t];
    if (g && g !== existingGate) {
      const warn = document.getElementById('warning');
      if (warn) warn.textContent = 'Time step already occupied.';
      return false;
    }
  }
  
  // Check gate limit
  if (!existingGate && gateCount() >= MAX_GATES) {
    const warn = document.getElementById('warning');
    if (warn) warn.textContent = `Gate limit reached (${MAX_GATES}).`;
    return false;
  }
  
  const warn = document.getElementById('warning');
  if (warn) warn.textContent = '';
  
  const gateObj = { name, qubits: qubits.slice(), t };
  gateList = gateList.filter(
    g => !(g.t === t && arraysEqual(g.qubits, gateObj.qubits) && g.name === gateObj.name)
  );
  gateList.push(gateObj);
  
  for (const qq of qubits) {
    circuit[qq][t] = gateObj;
  }
  
  return true;
}

// ========================================================================
// CIRCUIT DRAWING
// ========================================================================

function drawCircuit() {
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = '#10131b';
  ctx.fillRect(0, 0, W, H);
  
  ctx.font = '12px "IBM Plex Mono", monospace';
  ctx.textAlign = 'left';
  ctx.textBaseline = 'middle';

  // 1. Draw grid lines + labels
  for (let r = 0; r < rows; r++) {
    const y = (r + 0.5) * cellH;
    ctx.strokeStyle = '#333';
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(W, y);
    ctx.stroke();
    
    ctx.fillStyle = '#9ba0b5';
    if (r < N_QUBITS) ctx.fillText(`q[${r}]`, 4, y);
    else ctx.fillText(`c[${r}]`, 4, y);
  }

  ctx.strokeStyle = '#222';
  for (let t = 0; t < N_STEPS; t++) {
    const x = t * cellW;
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, H);
    ctx.stroke();
  }

  // 2. PHI MANIFOLD HEATMAP (LAYER 1 - underneath everything)
  drawPhiManifoldSmooth();

  // 3. Draw user gates (LAYER 2 - on top of heatmap)
  const drawn = new Set();
  for (let q = 0; q < N_QUBITS; q++) {
    for (let t = 0; t < N_STEPS; t++) {
      const g = circuit[q][t];
      if (!g) continue;
      if (g.name === 'M') continue;
      const key = g.name + ':' + g.qubits.join(',') + ':' + g.t;
      if (drawn.has(key)) continue;
      drawn.add(key);
      drawGateGlyph(g);
    }
  }

  // 4. Fixed M column
  drawMeasurements();

  // 5. Ghost gate (dragging)
  if (dragging && dragGate) drawDraggingGate();
}
function drawGateGlyph(gate) {
  const { name, qubits, t } = gate;
  const pad = 4;
  
  if (name === 'CNOT' && qubits.length === 2) {
    const qA = qubits[0];
    const qB = qubits[1];
    const control = Math.max(qA, qB);
    const target = Math.min(qA, qB);
    
    const xCenter = t * cellW + cellW / 2;
    const yControl = control * cellH + cellH / 2;
    const yTarget = target * cellH + cellH / 2;
    
    const colors = getGateColor('CNOT');
    
    // Connecting line
    ctx.strokeStyle = colors.fill;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(xCenter, yTarget);
    ctx.lineTo(xCenter, yControl);
    ctx.stroke();
    
    // Control dot (filled)
    ctx.fillStyle = colors.fill;
    ctx.beginPath();
    ctx.arc(xCenter, yControl, Math.min(cellH, cellW) * 0.18, 0, 2 * Math.PI);
    ctx.fill();
    
    // Stroke around control
    ctx.strokeStyle = colors.stroke;
    ctx.lineWidth = 2;
    ctx.stroke();
    
    // Target (⊕)
    ctx.strokeStyle = colors.fill;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(xCenter, yTarget, Math.min(cellH, cellW) * 0.18, 0, 2 * Math.PI);
    ctx.stroke();
    
    ctx.beginPath();
    ctx.moveTo(xCenter - 6, yTarget);
    ctx.lineTo(xCenter + 6, yTarget);
    ctx.moveTo(xCenter, yTarget - 6);
    ctx.lineTo(xCenter, yTarget + 6);
    ctx.stroke();
    
    return;
  }
  
  // Single-qubit gate (or other multi-qubit gates drawn as boxes)
  const q = qubits[0];
  const x = t * cellW;
  const y = q * cellH;
  const w = cellW - 2 * pad;
  const h = cellH - 2 * pad;
  
  const colors = getGateColor(name);
  
  // Draw gate box with gradient
  const gradient = ctx.createLinearGradient(x + pad, y + pad, x + pad, y + pad + h);
  gradient.addColorStop(0, colors.fill);
  gradient.addColorStop(1, colors.stroke);
  
  ctx.fillStyle = gradient;
  ctx.fillRect(x + pad, y + pad, w, h);
  
  // Border
  ctx.strokeStyle = colors.stroke;
  ctx.lineWidth = 2;
  ctx.strokeRect(x + pad, y + pad, w, h);
  
  // Gate label
  ctx.fillStyle = colors.text;
  ctx.font = 'bold 12px "IBM Plex Mono", monospace';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(name, x + cellW / 2, y + cellH / 2);
}
function drawMeasurements() {
  const t = N_STEPS - 1;
  const pad = 4;
  
  const colors = getGateColor('M');
  
  for (let q = 0; q < N_QUBITS; q++) {
    const x = t * cellW;
    const y = q * cellH;
    const w = cellW - 2 * pad;
    const h = cellH - 2 * pad;
    
    // Gradient fill
    const gradient = ctx.createLinearGradient(x + pad, y + pad, x + pad, y + pad + h);
    gradient.addColorStop(0, colors.fill);
    gradient.addColorStop(1, colors.stroke);
    
    ctx.fillStyle = gradient;
    ctx.fillRect(x + pad, y + pad, w, h);
    
    ctx.strokeStyle = colors.stroke;
    ctx.lineWidth = 2;
    ctx.strokeRect(x + pad, y + pad, w, h);
    
    ctx.fillStyle = colors.text;
    ctx.font = 'bold 12px "IBM Plex Mono", monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('M', x + cellW / 2, y + cellH / 2);
  }
}

function drawDraggingGate() {
  const pad = 4;
  const { q, t } = getCellFromCoords(dragX, dragY);
  
  if (q !== null && t !== null && q < N_QUBITS && t !== N_STEPS - 1) {
    const x = t * cellW, y = q * cellH;
    const colors = getGateColor(dragGate.name);
    
    // Highlight target cell with gate color
    ctx.fillStyle = colors.fill + '30'; // 30 = ~20% opacity in hex
    ctx.fillRect(x + 1, y + 1, cellW - 2, cellH - 2);
  }
  
  const w = cellW - 2 * pad;
  const h = cellH - 2 * pad;
  
  const colors = getGateColor(dragGate.name);
  
  // Draw ghost gate with gradient
  const gradient = ctx.createLinearGradient(
    dragX - w / 2, 
    dragY - h / 2, 
    dragX - w / 2, 
    dragY + h / 2
  );
  gradient.addColorStop(0, colors.fill);
  gradient.addColorStop(1, colors.stroke);
  
  ctx.fillStyle = gradient;
  ctx.fillRect(dragX - w / 2, dragY - h / 2, w, h);
  
  ctx.strokeStyle = colors.stroke;
  ctx.lineWidth = 2;
  ctx.strokeRect(dragX - w / 2, dragY - h / 2, w, h);
  
  ctx.fillStyle = colors.text;
  ctx.font = 'bold 12px "IBM Plex Mono", monospace';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(dragGate.name, dragX, dragY);
}

// ========================================================================
// DRAG EVENTS
// ========================================================================

window.addEventListener('mousemove', e => {
  if (!dragging || !dragGate) return;
  
  const rect = canvas.getBoundingClientRect();
  dragX = e.clientX - rect.left;
  dragY = e.clientY - rect.top;
  
  drawCircuit();
});

window.addEventListener('mouseup', e => {
  if (!dragging || !dragGate) return;
  
  dragging = false;
  const rect = canvas.getBoundingClientRect();
  const x = e.clientX - rect.left;
  const y = e.clientY - rect.top;
  const { q, t } = getCellFromCoords(x, y);
  
  if (q !== null && t !== null && q < N_QUBITS && t !== N_STEPS - 1) {
    placeGate(dragGate.name, q, t);
  }
  
  dragGate = null;
  drawCircuit();
});

canvas.addEventListener('click', e => {
  if (dragging || dragGate) return;
  
  const rect = canvas.getBoundingClientRect();
  const x = e.clientX - rect.left;
  const y = e.clientY - rect.top;
  const { q, t } = getCellFromCoords(x, y);
  
  if (q == null || t == null || q >= N_QUBITS) return;
  if (t === N_STEPS - 1) return;
  
  const g = circuit[q][t];
  if (!g) return;
  
  placeGate(g.name, q, g.t);
  drawCircuit();
});

// ========================================================================
// HISTOGRAM DRAWING
// ========================================================================

let histogramData = null;

function syncHistogramCanvas() {
  const histCanvas = document.getElementById('histCanvas');
  if (!histCanvas) return;
  
  const container = document.getElementById('bottom-left');
  if (!container) return;
  
  const rect = container.getBoundingClientRect();
  
  // CSS dimensions
  const displayWidth = rect.width - 16;
  const displayHeight = rect.height - 40;
  
  // Device pixel ratio
  const dpr = window.devicePixelRatio || 1;
  
  // Set canvas resolution
  histCanvas.width = displayWidth * dpr;
  histCanvas.height = displayHeight * dpr;
  
  // Set CSS size
  histCanvas.style.width = displayWidth + 'px';
  histCanvas.style.height = displayHeight + 'px';
  
  // Scale context
  const histCtx = histCanvas.getContext('2d');
  histCtx.scale(dpr, dpr);
  
  console.log(`📊 Histogram canvas: ${histCanvas.width}×${histCanvas.height} (${dpr}x DPR)`);
  
  drawHistogram();
}

function drawHistogram() {
  const histCanvas = document.getElementById('histCanvas');
  if (!histCanvas) return;
  
  const ctx = histCanvas.getContext('2d');
  
  // Use logical dimensions (CSS size)
  const w = parseInt(histCanvas.style.width) || (histCanvas.width / (window.devicePixelRatio || 1));
  const h = parseInt(histCanvas.style.height) || (histCanvas.height / (window.devicePixelRatio || 1));
  
  // Clear canvas
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = '#10131b';
  ctx.fillRect(0, 0, w, h);

  // Define margins and plotting area
  const margin = { left: 60, right: 20, top: 40, bottom: 100 };
  const plotWidth = w - margin.left - margin.right;
  const plotHeight = h - margin.top - margin.bottom;

  // Generate state labels
  const numStates = Math.pow(2, N_QUBITS);
  const stateLabels = [];
  for (let i = 0; i < numStates; i++) {
    stateLabels.push('|' + i.toString(2).padStart(N_QUBITS, '0') + '⟩');
  }

  // Draw axes
  ctx.strokeStyle = '#9ba0b5';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(margin.left, margin.top);
  ctx.lineTo(margin.left, margin.top + plotHeight);
  ctx.lineTo(margin.left + plotWidth, margin.top + plotHeight);
  ctx.stroke();

  // Y-axis labels
  ctx.fillStyle = '#9ba0b5';
  ctx.font = '11px "IBM Plex Mono", monospace';
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';

  for (let i = 0; i <= 5; i++) {
    const prob = i / 5;
    const y = margin.top + plotHeight - (prob * plotHeight);
    ctx.fillText(prob.toFixed(1), margin.left - 10, y);

    // Grid lines
    ctx.strokeStyle = '#2b2f3a';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(margin.left, y);
    ctx.lineTo(margin.left + plotWidth, y);
    ctx.stroke();
  }

  // Y-axis label
  ctx.save();
  ctx.translate(15, margin.top + plotHeight / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.textAlign = 'center';
  ctx.fillStyle = '#e0e3ff';
  ctx.font = '12px "IBM Plex Mono", monospace';
  ctx.fillText('Probability', 0, 0);
  ctx.restore();

  // X-axis labels
  ctx.fillStyle = '#9ba0b5';
  ctx.font = '10px "IBM Plex Mono", monospace';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';

  const barWidth = plotWidth / numStates;
  for (let i = 0; i < numStates; i++) {
    const x = margin.left + (i + 0.5) * barWidth;
    const y = margin.top + plotHeight + 5;
    ctx.fillText(stateLabels[i], x, y);
  }

  // X-axis label
  ctx.fillStyle = '#e0e3ff';
  ctx.font = '12px "IBM Plex Mono", monospace';
  ctx.textAlign = 'center';
  ctx.fillText('Quantum States', margin.left + plotWidth / 2, h - 10);

  // Legend
  const legendX = margin.left + 10;
  const legendY = margin.top + 10;

  ctx.fillStyle = '#3498db';
  ctx.fillRect(legendX, legendY, 15, 10);
  ctx.fillStyle = '#e0e3ff';
  ctx.font = '11px sans-serif';
  ctx.textAlign = 'left';
  ctx.fillText('Ideal', legendX + 20, legendY + 8);

  ctx.fillStyle = '#e67e22';
  ctx.fillRect(legendX + 80, legendY, 15, 10);
  ctx.fillStyle = '#e0e3ff';
  ctx.fillText('Noisy', legendX + 100, legendY + 8);

  // Draw bars
  if (histogramData) {
    const { ideal, noisy } = histogramData;
    const barPadding = 2;
    const barGroupWidth = barWidth - barPadding;
    const singleBarWidth = barGroupWidth / 2 - 1;

    for (let i = 0; i < numStates; i++) {
      const state = i.toString(2).padStart(N_QUBITS, '0');
      const idealProb = ideal[state] || 0;
      const noisyProb = noisy[state] || 0;
      const baseX = margin.left + i * barWidth;

      // Ideal bar
      if (idealProb > 0) {
        const barHeight = idealProb * plotHeight;
        ctx.fillStyle = '#3498db';
        ctx.fillRect(
          baseX + barPadding,
          margin.top + plotHeight - barHeight,
          singleBarWidth,
          barHeight
        );
      }

      // Noisy bar
      if (noisyProb > 0) {
        const barHeight = noisyProb * plotHeight;
        ctx.fillStyle = '#e67e22';
        ctx.fillRect(
          baseX + singleBarWidth + barPadding + 1,
          margin.top + plotHeight - barHeight,
          singleBarWidth,
          barHeight
        );
      }
    }
  }
}

// ========================================================================
// PHI MANIFOLD HEATMAP - Smooth Bilinear Interpolation
// ========================================================================

function drawPhiManifoldSmooth() {
  if (!window.lastPhiManifold || !document.getElementById('inspectToggle').checked) {
    return;
  }

  const phiData = window.lastPhiManifold;
  
  // Normalize phi values
  let minPhi = Infinity, maxPhi = -Infinity;
  for (const qubitData of phiData) {
    for (const val of qubitData) {
      minPhi = Math.min(minPhi, val);
      maxPhi = Math.max(maxPhi, val);
    }
  }
  
  const range = maxPhi - minPhi;
  if (range === 0) return;

  // Interpolation helpers
  function getPhi(q, t) {
    q = Math.max(0, Math.min(N_QUBITS - 1, Math.floor(q)));
    t = Math.max(0, Math.min(phiData[0].length - 1, Math.floor(t)));
    return phiData[q][t];
  }

  function bilinearInterp(qCont, tCont) {
    const q0 = Math.floor(qCont);
    const q1 = Math.min(q0 + 1, N_QUBITS - 1);
    const t0 = Math.floor(tCont);
    const t1 = Math.min(t0 + 1, phiData[0].length - 1);
    
    const fq = qCont - q0;
    const ft = tCont - t0;
    
    const v00 = getPhi(q0, t0);
    const v10 = getPhi(q1, t0);
    const v01 = getPhi(q0, t1);
    const v11 = getPhi(q1, t1);
    
    const v0 = v00 * (1 - ft) + v01 * ft;
    const v1 = v10 * (1 - ft) + v11 * ft;
    return v0 * (1 - fq) + v1 * fq;
  }

  // Draw smooth heatmap
  const subsamples = 8;
  
  for (let q = 0; q < N_QUBITS; q++) {
    for (let t = 0; t < N_STEPS; t++) {
      const x = t * cellW;
      const y = q * cellH;
      const w = cellW - 4;
      const h = cellH - 4;
      
      const cellCanvas = document.createElement('canvas');
      cellCanvas.width = subsamples;
      cellCanvas.height = subsamples;
      const cellCtx = cellCanvas.getContext('2d');
      const imageData = cellCtx.createImageData(subsamples, subsamples);
      
      for (let py = 0; py < subsamples; py++) {
        for (let px = 0; px < subsamples; px++) {
          const qSample = q + (py / subsamples);
          const tSample = t + (px / subsamples);
          
          const phi = bilinearInterp(qSample, tSample);
          const normalized = (phi - minPhi) / range;
          const rgba = getPhiColorRGBA(normalized);
          
          const idx = (py * subsamples + px) * 4;
          imageData.data[idx + 0] = rgba.r;
          imageData.data[idx + 1] = rgba.g;
          imageData.data[idx + 2] = rgba.b;
          imageData.data[idx + 3] = rgba.a;
        }
      }
      
      cellCtx.putImageData(imageData, 0, 0);
      ctx.imageSmoothingEnabled = true;
      ctx.imageSmoothingQuality = 'high';
      ctx.drawImage(cellCanvas, x + 2, y + 2, w, h);
    }
  }
}

function getPhiColorRGBA(normalized) {
  normalized = Math.max(0, Math.min(1, normalized));
  
  let r, g, b, a;
  
  if (normalized < 0.33) {
    // Black to Red (0.0 → 0.33)
    const t = normalized / 0.33; // 0 to 1
    r = Math.floor(t * 255);      // 0 → 255
    g = 0;
    b = 0;
    a = Math.floor((0.4 + t * 0.2) * 255); // 40% → 60% opacity
  } else if (normalized < 0.66) {
    // Red to Orange (0.33 → 0.66)
    const t = (normalized - 0.33) / 0.33; // 0 to 1
    r = 255;
    g = Math.floor(t * 165);      // 0 → 165 (orange)
    b = 0;
    a = Math.floor((0.6 + t * 0.15) * 255); // 60% → 75% opacity
  } else {
    // Orange to White (0.66 → 1.0)
    const t = (normalized - 0.66) / 0.34; // 0 to 1
    r = 255;
    g = Math.floor(165 + t * (255 - 165)); // 165 → 255
    b = Math.floor(t * 255);      // 0 → 255
    a = Math.floor((0.75 + t * 0.25) * 255); // 75% → 100% opacity
  }
  
  return { r, g, b, a };
}


function drawPhiLegend() {
  if (!window.lastPhiManifold || !document.getElementById('inspectToggle').checked) {
    return;
  }

  const legendX = W - 120;
  const legendY = 10;
  const legendWidth = 100;
  const legendHeight = 15;

  const gradient = ctx.createLinearGradient(legendX, 0, legendX + legendWidth, 0);
  gradient.addColorStop(0, 'rgba(52, 152, 219, 0.4)');
  gradient.addColorStop(0.5, 'rgba(148, 0, 133, 0.6)');
  gradient.addColorStop(1, 'rgba(231, 76, 60, 0.8)');

  ctx.fillStyle = gradient;
  ctx.fillRect(legendX, legendY, legendWidth, legendHeight);

  ctx.strokeStyle = '#9ba0b5';
  ctx.lineWidth = 1;
  ctx.strokeRect(legendX, legendY, legendWidth, legendHeight);

  ctx.fillStyle = '#e0e3ff';
  ctx.font = '9px "IBM Plex Mono", monospace';
  ctx.textAlign = 'center';
  ctx.fillText('Low', legendX + 15, legendY + legendHeight + 10);
  ctx.fillText('Phi Intensity', legendX + legendWidth / 2, legendY + legendHeight + 10);
  ctx.fillText('High', legendX + legendWidth - 15, legendY + legendHeight + 10);
}

// ========================================================================
// LOADING SCREEN WITH ROTATING MESSAGES
// ========================================================================

const LOADING_MESSAGES = [
  "🔨 Building quantum circuit...",
  "⚛️  Initializing state vectors...",
  "🌀 Applying quantum gates...",
  "📊 Computing observables...",
  "🚀 Finalizing simulation..."
];

let loadingMessageInterval = null;
let currentMessageIndex = 0;

function showLoading() {
  const overlay = document.getElementById('loadingOverlay');
  const messageEl = document.getElementById('loadingMessage');
  
  overlay.classList.remove('hidden');
  currentMessageIndex = 0;
  messageEl.textContent = LOADING_MESSAGES[0];
  
  // Rotate messages every 1.5 seconds
  loadingMessageInterval = setInterval(() => {
    currentMessageIndex = (currentMessageIndex + 1) % LOADING_MESSAGES.length;
    messageEl.textContent = LOADING_MESSAGES[currentMessageIndex];
  }, 1500);
  
  console.log('🎬 Loading screen shown');
}

function hideLoading() {
  const overlay = document.getElementById('loadingOverlay');
  const messageEl = document.getElementById('loadingMessage');
  
  // Clear interval
  if (loadingMessageInterval) {
    clearInterval(loadingMessageInterval);
    loadingMessageInterval = null;
  }
  
  // Show completion message briefly
  messageEl.textContent = "✅ Simulation complete!";
  
  // Fade out after 500ms
  setTimeout(() => {
    overlay.classList.add('hidden');
  }, 650);
  
  console.log('🎬 Loading screen hidden');
}

// ========================================================================
// BACKEND INTEGRATION
// ========================================================================
// ========================================================================
// BACKEND INTEGRATION WITH LOADING
// ========================================================================

document.getElementById('runBtn').addEventListener('click', async () => {
  const noise = document.getElementById('noiseToggle').checked;
  const inspect = document.getElementById('inspectToggle').checked;
  const persistent = document.getElementById('persistToggle').checked;
  const shots = 100;

  const payloadGates = gateList
    .slice()
    .sort((a, b) => a.t - b.t || a.qubits[0] - b.qubits[0]);

  const measT = N_STEPS - 1;
  for (let q = 0; q < N_QUBITS; q++) {
    payloadGates.push({
      name: 'M',
      qubits: [q],
      t: measT
    });
  }

  const body = {
    num_qubits: N_QUBITS,
    shots,
    noise_enabled: noise,
    persistent_mode: persistent,
    show_phi: inspect,
    gates: payloadGates
  };

  console.log('📤 Sending to backend:', body);

  // Show loading screen
  showLoading();
  
  try {
    const response = await fetch('http://127.0.0.1:8000/simulate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    const result = await response.json();
    console.log('✅ Backend response:', result);

    // Update histogram
    histogramData = {
      ideal: result.histogram_ideal,
      noisy: result.histogram_noisy || {}
    };
    drawHistogram();

    // Update Bloch spheres
    if (typeof updateBlochSpheres !== 'undefined' && blochInitialized) {
      const blochStates = result.bloch_states.map(state => ({
        theta: state.theta,
        phi: state.phi,
        probability: state.probability,
        label: state.state
      }));
      updateBlochSpheres(blochStates);
      
      if (typeof updateLegend !== 'undefined') {
        updateLegend();
      }
    }

    // Update metadata box
    const metaBox = document.getElementById('metaBox');
    if (metaBox) {
      const meta = result.metadata;
      const timing = meta.timing;
      const cache = meta.cache_stats || {};
      const lru = cache.lru_cache || { hits: 0, misses: 0, hit_rate: 0, current_size: 0, max_size: 0 };

      metaBox.value = `
CIRCUIT INFO
├─ Qubits:        ${N_QUBITS}
├─ Circuit Depth: ${meta.circuit_depth}
├─ Total Gates:   ${meta.circuit_size}
└─ Shots:         ${meta.shots}

TIMING (seconds)
├─ Total:         ${timing.total_seconds.toFixed(4)}s
├─ Circuit Build: ${timing.circuit_build_seconds.toFixed(4)}s
├─ Ideal Sim:     ${timing.ideal_simulation_seconds.toFixed(4)}s
├─ Noisy Sim:     ${timing.noisy_simulation_seconds.toFixed(4)}s
└─ Phi Extraction: ${timing.phi_extraction_seconds.toFixed(4)}s

BACKEND
├─ Device:        ${meta.device.toUpperCase()}
├─ Noise:         ${meta.noise_enabled ? 'ENABLED' : 'DISABLED'}
└─ Persistent:    ${meta.persistent_mode ? 'ON' : 'OFF'}

CACHE STATS
├─ Fixed Gates:   ${cache.fixed_cache_size || 0}
├─ LRU Hits:      ${lru.hits}
├─ LRU Misses:    ${lru.misses}
├─ Hit Rate:      ${(lru.hit_rate * 100).toFixed(1)}%
└─ Cache Size:    ${lru.current_size}/${lru.max_size}

RESULTS
├─ Statevector:   ${result.statevector.length} amplitudes
├─ Bloch States:  ${result.bloch_states.length} significant
└─ Phi Manifold:  ${result.phi_manifold ? `${result.phi_manifold.length} × ${result.phi_manifold[0].length}` : 'N/A'}

${noise ? `
NOISE ANALYSIS
Ideal vs Noisy Fidelity: ${calculateFidelity(histogramData.ideal, histogramData.noisy).toFixed(3)}
` : ''}
`.trim();
    }

    window.lastPhiManifold = result.phi_manifold;
    drawCircuit();

    console.log('✨ UI updated successfully');
    
    // Hide loading screen
    hideLoading();

  } catch (error) {
    console.error('❌ Simulation failed:', error);
    
    // Hide loading
    hideLoading();
    
    // Show error after brief delay
    setTimeout(() => {
      alert(`Simulation Error: ${error.message}\n\nCheck console for details.`);
    }, 600);
    
    const metaBox = document.getElementById('metaBox');
    if (metaBox) {
      metaBox.value = `ERROR: ${error.message}\n\nMake sure backend is running:\nuvicorn api:app --reload`;
    }
  }
});

function calculateFidelity(ideal, noisy) {
  let fidelity = 0;
  for (const state in ideal) {
    const p_ideal = ideal[state] || 0;
    const p_noisy = noisy[state] || 0;
    fidelity += Math.sqrt(p_ideal * p_noisy);
  }
  return fidelity;
}

// ========================================================================
// INITIALIZATION
// ========================================================================

window.addEventListener('load', () => {
  syncHistogramCanvas();
});

drawCircuit();

console.log('✅ QtorchX Circuit Simulator loaded');
