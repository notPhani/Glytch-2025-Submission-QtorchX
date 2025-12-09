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

// ---------- Drawing ----------
function drawCircuit() {
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = '#10131b';
  ctx.fillRect(0, 0, W, H);
  ctx.font = '12px monospace';
  ctx.textAlign = 'left';
  ctx.textBaseline = 'middle';

  // horizontal lines + labels
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

  // vertical grid
  ctx.strokeStyle = '#222';
  for (let t = 0; t < N_STEPS; t++) {
    const x = t * cellW;
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, H);
    ctx.stroke();
  }

  // user gates
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

  // fixed M column
  drawMeasurements();

  // ghost
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
  placeGate(g.name, q, g.t);

  drawCircuit();
});

// ---------- Run circuit: include Ms logically ----------
document.getElementById('runBtn').addEventListener('click', async () => {
  const noise = document.getElementById('noiseToggle').checked;
  const persistent = document.getElementById('persistToggle').checked;

  const payloadGates = gateList
    .slice()
    .sort((a, b) => a.t - b.t || a.qubits[0] - b.qubits[0]);

  const measT = N_STEPS - 1;
  for (let q = 0; q < N_QUBITS; q++) {
    payloadGates.push({ name: 'M', qubits: [q], t: measT });
  }

  const body = {
    num_qubits: N_QUBITS,
    gates: payloadGates,
    noise,
    persistent,
    shots: 1024
  };

  const metaBox = document.getElementById('metaBox');
  if (metaBox) {
    metaBox.value = '# Circuit payload\n' + JSON.stringify(body, null, 2);
  }

  // TODO: call backend /simulate here
  drawCircuit();
});

// ---------- Init ----------
drawCircuit();