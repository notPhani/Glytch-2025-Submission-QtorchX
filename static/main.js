
// ---- CONFIG ----
const N_QUBITS = 4;
const HAS_CLASSICAL = true;
const N_STEPS = 15;
const MAX_GATES = 15;

// circuit[q][t] = user gate | null (M is visual only; added to payload on run)
const circuit = Array.from({ length: N_QUBITS }, () =>
  Array.from({ length: N_STEPS }, () => null)
);
let gateList = [];

// Canvas + geometry
const canvas = document.getElementById('circuitCanvas');
const ctx = canvas.getContext('2d');
const rows = N_QUBITS + (HAS_CLASSICAL ? 1 : 0);
let W, H, cellW, cellH;

function syncGeometry() {
  const container = document.getElementById('circuit-inner');
  const rect = container.getBoundingClientRect();
  const padding = 4; // matches #circuit-inner padding
  canvas.width = rect.width - 2 * padding;
  canvas.height = rect.height - 2* padding;
  W = canvas.width;
  H = canvas.height;
  cellW = W / N_STEPS;
  cellH = H / rows;
}

syncGeometry();
window.addEventListener('resize', () => {
  syncGeometry();
  drawCircuit();
  syncHistogramCanvas();
});

// Drag state
let selectedGateType = null;
let dragging = false;
let dragGate = null; // { name }
let dragX = 0, dragY = 0;

// ---------- Gate filtering state ----------
let activeFilters = new Set(); // e.g., 'basic', 'rotation', 'multi'
let searchQuery = '';

// Gate categories mapping
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

// Filter gate buttons based on active filters and search
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

    // Show or hide the button
    btn.style.display = shouldShow ? 'block' : 'none';
  });
}

// ---------- Search functionality ----------
const searchInput = document.getElementById('gateSearch');
if (searchInput) {
  searchInput.addEventListener('input', (e) => {
    searchQuery = e.target.value.trim();
    updateGateVisibility();
  });
}

// ---------- Filter functionality ----------
document.querySelectorAll('.filter-btn').forEach(filterBtn => {
  filterBtn.addEventListener('click', () => {
    const filter = filterBtn.dataset.filter;

    if (activeFilters.has(filter)) {
      // Remove filter
      activeFilters.delete(filter);
      filterBtn.classList.remove('active');
    } else {
      // Add filter
      activeFilters.add(filter);
      filterBtn.classList.add('active');
    }

    updateGateVisibility();
  });
});

// ---------- Gate library (selection + drag ghost) ----------
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

// ---------- Helpers ----------
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

// CNOT: control above, target below
function getCnotQubits(dropQ) {
  const target = dropQ;
  const control = dropQ + 1 < N_QUBITS ? dropQ + 1 : dropQ;
  if (control === target) return null;
  return [control, target]; // order not important; draw decides
}

function arraysEqual(a, b) {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}

// Place or toggle‑remove non‑measurement gate
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

  // same gate already here?
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

  // toggle delete
  if (fullyOccupied && existingGate && existingGate.name === name) {
    gateList = gateList.filter(
      g => !(g.t === t && arraysEqual(g.qubits, existingGate.qubits) && g.name === existingGate.name)
    );
    for (const qq of existingGate.qubits) {
      circuit[qq][t] = null;
    }
    return true;
  }

  // conflicting gate?
  for (const qq of qubits) {
    const g = circuit[qq][t];
    if (g && g !== existingGate) {
      const warn = document.getElementById('warning');
      if (warn) warn.textContent = 'Time step already occupied.';
      return false;
    }
  }

  // gate limit
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

function drawCircuit() {
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = '#10131b';
  ctx.fillRect(0, 0, W, H);
  
  ctx.font = '12px monospace';
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

  // 2. *** PHI MANIFOLD HEATMAP (LAYER 1) ***
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
    const control = Math.max(qA, qB); // upper
    const target = Math.min(qA, qB); // lower
    const xCenter = t * cellW + cellW / 2;
    const yControl = control * cellH + cellH / 2;
    const yTarget = target * cellH + cellH / 2;

    ctx.strokeStyle = '#e74c3c';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(xCenter, yTarget);
    ctx.lineTo(xCenter, yControl);
    ctx.stroke();

    // control (filled) on top
    ctx.fillStyle = '#e74c3c';
    ctx.beginPath();
    ctx.arc(xCenter, yControl, Math.min(cellH, cellW) * 0.18, 0, 2 * Math.PI);
    ctx.fill();

    // target (⊕) below
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

  // single‑qubit gate
  const q = qubits[0];
  const x = t * cellW;
  const y = q * cellH;
  ctx.strokeStyle = '#222';
  ctx.fillStyle = '#e74c3c';
  const w = cellW - 2 * pad;
  const h = cellH - 2 * pad;
  ctx.fillRect(x + pad, y + pad, w, h);
  ctx.strokeRect(x + pad, y + pad, w, h);
  ctx.fillStyle = '#fff';
  ctx.font = '12px monospace';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(name, x + cellW / 2, y + cellH / 2);
}

// fixed blue M gates at last column (visual only)
function drawMeasurements() {
  const t = N_STEPS - 1;
  const pad = 4;
  for (let q = 0; q < N_QUBITS; q++) {
    const x = t * cellW;
    const y = q * cellH;
    const w = cellW - 2 * pad;
    const h = cellH - 2 * pad;
    ctx.strokeStyle = '#222';
    ctx.fillStyle = '#2980b9';
    ctx.fillRect(x + pad, y + pad, w, h);
    ctx.strokeRect(x + pad, y + pad, w, h);
    ctx.fillStyle = '#fff';
    ctx.font = '12px monospace';
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
    ctx.fillStyle = 'rgba(231, 126, 35, 0.3)';
    ctx.fillRect(x + 1, y + 1, cellW - 2, cellH - 2);
  }
  const w = cellW - 2 * pad;
  const h = cellH - 2 * pad;
  ctx.fillStyle = '#e74c3c';
  ctx.fillRect(dragX - w / 2, dragY - h / 2, w, h);
  ctx.strokeStyle = '#222';
  ctx.strokeRect(dragX - w / 2, dragY - h / 2, w, h);
  ctx.fillStyle = '#fff';
  ctx.font = '12px monospace';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(dragGate.name, dragX, dragY);
}

// ---------- Drag events ----------
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

// click to delete existing gate (not M)
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

  // FIX: Use the clicked qubit position 'q' instead of g.qubits[0]
  // This ensures CNOT gates delete correctly no matter which part is clicked
  placeGate(g.name, q, g.t);
  drawCircuit();
});

// ---------- Histogram Drawing ----------
let histogramData = null; // Will store { ideal: {...}, noisy: {...} }

function drawHistogram() {
  const histCanvas = document.getElementById('histCanvas');
  if (!histCanvas) return;

  const ctx = histCanvas.getContext('2d');
  const w = histCanvas.width;
  const h = histCanvas.height;

  // Clear canvas
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = '#10131b';
  ctx.fillRect(0, 0, w, h);

  // Define margins and plotting area
  const margin = { left: 60, right: 20, top: 40, bottom: 60 };
  const plotWidth = w - margin.left - margin.right;
  const plotHeight = h - margin.top - margin.bottom;

  // Generate state labels for N_QUBITS
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

  // Y-axis labels (0 to 1)
  ctx.fillStyle = '#9ba0b5';
  ctx.font = '11px monospace';
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
  ctx.font = '12px monospace';
  ctx.fillText('Probability', 0, 0);
  ctx.restore();

  // X-axis labels (states)
  ctx.fillStyle = '#9ba0b5';
  ctx.font = '10px monospace';
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
  ctx.font = '12px monospace';
  ctx.textAlign = 'center';
  ctx.fillText('Quantum States', margin.left + plotWidth / 2, h - 10);

  // Legend
  const legendX = margin.left + 10;
  const legendY = margin.top + 10;

  // Ideal (Blue)
  ctx.fillStyle = '#3498db';
  ctx.fillRect(legendX, legendY, 15, 10);
  ctx.fillStyle = '#e0e3ff';
  ctx.font = '11px sans-serif';
  ctx.textAlign = 'left';
  ctx.fillText('Ideal', legendX + 20, legendY + 9);

  // Noisy (Orange)
  ctx.fillStyle = '#e67e22';
  ctx.fillRect(legendX + 80, legendY, 15, 10);
  ctx.fillStyle = '#e0e3ff';
  ctx.fillText('Noisy', legendX + 100, legendY + 9);

  // Draw bars if data exists
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

      // Ideal bar (blue)
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

      // Noisy bar (orange)
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

// Sync histogram canvas geometry
function syncHistogramCanvas() {
  const histCanvas = document.getElementById('histCanvas');
  if (!histCanvas) return;

  const container = document.getElementById('bottom-left');
  if (!container) return;

  const rect = container.getBoundingClientRect();
  histCanvas.width = rect.width - 16; // Account for padding
  histCanvas.height = rect.height - 40; // Account for title and padding

  drawHistogram();
}
// ========================================================================
// PHI MANIFOLD HEATMAP - Smooth Bilinear Interpolation
// ========================================================================

function drawPhiManifoldSmooth() {
  if (!window.lastPhiManifold || !document.getElementById('inspectToggle').checked) {
    return; // Only draw if phi data exists and inspect is enabled
  }

  const phiData = window.lastPhiManifold;
  
  // ---------- Normalize phi values ----------
  let minPhi = Infinity, maxPhi = -Infinity;
  for (const qubitData of phiData) {
    for (const val of qubitData) {
      minPhi = Math.min(minPhi, val);
      maxPhi = Math.max(maxPhi, val);
    }
  }
  
  const range = maxPhi - minPhi;
  if (range === 0) return; // No variation

  // ---------- Interpolation Helpers ----------
  function getPhi(q, t) {
    // Clamp to valid indices
    q = Math.max(0, Math.min(N_QUBITS - 1, Math.floor(q)));
    t = Math.max(0, Math.min(phiData[0].length - 1, Math.floor(t)));
    return phiData[q][t];
  }

  function bilinearInterp(qCont, tCont) {
    // qCont and tCont are continuous coordinates (can be fractional)
    const q0 = Math.floor(qCont);
    const q1 = Math.min(q0 + 1, N_QUBITS - 1);
    const t0 = Math.floor(tCont);
    const t1 = Math.min(t0 + 1, phiData[0].length - 1);
    
    const fq = qCont - q0; // fractional part (0 to 1)
    const ft = tCont - t0;
    
    // Get 4 corner values
    const v00 = getPhi(q0, t0);
    const v10 = getPhi(q1, t0);
    const v01 = getPhi(q0, t1);
    const v11 = getPhi(q1, t1);
    
    // Bilinear interpolation formula
    const v0 = v00 * (1 - ft) + v01 * ft;
    const v1 = v10 * (1 - ft) + v11 * ft;
    const v = v0 * (1 - fq) + v1 * fq;
    
    return v;
  }

  // ---------- Draw Smooth Heatmap ----------
  const subsamples = 8; // Samples per cell edge for smooth gradients
  
  for (let q = 0; q < N_QUBITS; q++) {
    for (let t = 0; t < N_STEPS; t++) {
      const x = t * cellW;
      const y = q * cellH;
      const w = cellW - 4; // Padding
      const h = cellH - 4;
      
      // Create ImageData for this cell at high resolution
      const cellCanvas = document.createElement('canvas');
      cellCanvas.width = subsamples;
      cellCanvas.height = subsamples;
      const cellCtx = cellCanvas.getContext('2d');
      const imageData = cellCtx.createImageData(subsamples, subsamples);
      
      // Sample phi at each sub-pixel
      for (let py = 0; py < subsamples; py++) {
        for (let px = 0; px < subsamples; px++) {
          // Map pixel to continuous coordinates
          const qSample = q + (py / subsamples);
          const tSample = t + (px / subsamples);
          
          // Interpolate phi value
          const phi = bilinearInterp(qSample, tSample);
          const normalized = (phi - minPhi) / range;
          
          // Get RGBA color
          const rgba = getPhiColorRGBA(normalized);
          
          // Set pixel
          const idx = (py * subsamples + px) * 4;
          imageData.data[idx + 0] = rgba.r;
          imageData.data[idx + 1] = rgba.g;
          imageData.data[idx + 2] = rgba.b;
          imageData.data[idx + 3] = rgba.a;
        }
      }
      
      // Put ImageData to offscreen canvas
      cellCtx.putImageData(imageData, 0, 0);
      
      // Draw smoothed cell to main canvas
      ctx.imageSmoothingEnabled = true;
      ctx.imageSmoothingQuality = 'high';
      ctx.drawImage(cellCanvas, x + 2, y + 2, w, h);
    }
  }
}

// Color gradient: Blue → Purple → Red (returns RGBA object)
function getPhiColorRGBA(normalized) {
  normalized = Math.max(0, Math.min(1, normalized));
  
  let r, g, b, a;
  
  if (normalized < 0.5) {
    // Blue to Purple (0.0 → 0.5)
    const t = normalized * 2;
    r = Math.floor(52 + t * (148 - 52));    // #3498db to #940085
    g = Math.floor(152 - t * 152);
    b = Math.floor(219 + t * (133 - 219));
    a = Math.floor((0.3 + t * 0.2) * 255);  // 30% to 50% opacity
  } else {
    // Purple to Red (0.5 → 1.0)
    const t = (normalized - 0.5) * 2;
    r = Math.floor(148 + t * (231 - 148));  // #940085 to #e74c3c
    g = 0;
    b = Math.floor(133 - t * 133);
    a = Math.floor((0.5 + t * 0.3) * 255);  // 50% to 80% opacity
  }
  
  return { r, g, b, a };
}

// Original string version for legend
function getPhiColor(normalized) {
  const rgba = getPhiColorRGBA(normalized);
  return `rgba(${rgba.r}, ${rgba.g}, ${rgba.b}, ${rgba.a / 255})`;
}

// ---------- Phi Legend ----------
function drawPhiLegend() {
  if (!window.lastPhiManifold || !document.getElementById('inspectToggle').checked) {
    return;
  }

  const legendX = W - 120;
  const legendY = 10;
  const legendWidth = 100;
  const legendHeight = 15;

  // Draw gradient bar
  const gradient = ctx.createLinearGradient(legendX, 0, legendX + legendWidth, 0);
  gradient.addColorStop(0, 'rgba(52, 152, 219, 0.4)');   // Blue
  gradient.addColorStop(0.5, 'rgba(148, 0, 133, 0.6)');  // Purple
  gradient.addColorStop(1, 'rgba(231, 76, 60, 0.8)');    // Red

  ctx.fillStyle = gradient;
  ctx.fillRect(legendX, legendY, legendWidth, legendHeight);

  // Border
  ctx.strokeStyle = '#9ba0b5';
  ctx.lineWidth = 1;
  ctx.strokeRect(legendX, legendY, legendWidth, legendHeight);

  // Labels
  ctx.fillStyle = '#e0e3ff';
  ctx.font = '9px monospace';
  ctx.textAlign = 'center';
  ctx.fillText('Low', legendX + 15, legendY + legendHeight + 10);
  ctx.fillText('Phi Intensity', legendX + legendWidth / 2, legendY + legendHeight + 10);
  ctx.fillText('High', legendX + legendWidth - 15, legendY + legendHeight + 10);
}

// Initialize histogram canvas
window.addEventListener('load', () => {
  syncHistogramCanvas();
});
// ---------- Run circuit: include Ms logically ----------
document.getElementById('runBtn').addEventListener('click', async () => {
  const noise      = document.getElementById('noiseToggle').checked;
  const inspect     = document.getElementById('inspectToggle').checked;
  const persistent = document.getElementById('persistToggle').checked;
  const shots      = 1024;

  // Sort gates: by time, then by first qubit
  const payloadGates = gateList
    .slice()
    .sort((a, b) => a.t - b.t || a.qubits[0] - b.qubits[0]);

  // Append measurements at final time step
  const measT = N_STEPS - 1;
  for (let q = 0; q < N_QUBITS; q++) {
    payloadGates.push({
      name:  'M',
      qubits: [q],
      t:     measT
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

  // --- BACKEND CALL ---
  try {
    const response = await fetch('http://localhost:8000/simulate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    const result = await response.json();
    console.log('✅ Backend response:', result);

    // --- UPDATE HISTOGRAM ---
    histogramData = {
      ideal: result.histogram_ideal,
      noisy: result.histogram_noisy || {}
    };
    drawHistogram();

    // --- UPDATE BLOCH SPHERES ---
    if (typeof updateBlochSpheres !== 'undefined' && blochInitialized) {
    const blochStates = result.bloch_states.map(state => ({
        theta: state.theta,
        phi: state.phi,
        probability: state.probability,
        label: state.state  // "0000", "1000"
    }));
    updateBlochSpheres(blochStates);
    
    // Update legend to show which states are displayed
    if (typeof updateLegend !== 'undefined') {
        updateLegend();
    }
    }


    // --- UPDATE METADATA BOX ---
    const metaBox = document.getElementById('metaBox');
if (metaBox) {
  const meta = result.metadata;
  const timing = meta.timing;
  const cache = meta.cache_stats || {};  // ← Safe default
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


    // --- STORE PHI MANIFOLD (handle later) ---
    window.lastPhiManifold = result.phi_manifold;
    
    // Redraw circuit with updated state
    drawCircuit();

    console.log('✨ UI updated successfully');

  } catch (error) {
    console.error('❌ Simulation failed:', error);
    alert(`Simulation Error: ${error.message}\n\nCheck console for details.`);
    
    const metaBox = document.getElementById('metaBox');
    if (metaBox) {
      metaBox.value = `ERROR: ${error.message}\n\nMake sure backend is running:\nuvicorn api:app --reload`;
    }
  }
});

// --- HELPER: Calculate Fidelity ---
function calculateFidelity(ideal, noisy) {
  let fidelity = 0;
  for (const state in ideal) {
    const p_ideal = ideal[state] || 0;
    const p_noisy = noisy[state] || 0;
    fidelity += Math.sqrt(p_ideal * p_noisy);
  }
  return fidelity;
}

// ---------- Init ----------
drawCircuit();
