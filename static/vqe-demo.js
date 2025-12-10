// ========================================================================
// VQE COMPARISON DEMO - 25 UNIQUE SCENARIOS
// Interactive H₂ Molecule VQE with Ideal vs Noisy Comparison
// ========================================================================

console.log('🧪 [VQE Demo] Loading...');

const GROUND_STATE_ENERGY = -1.137283834;

// ========================================================================
// 5 PARAMETER PROFILES - Different Initial Conditions
// ========================================================================

const PARAMETER_PROFILES = [
  {
    name: "Fast Learner",
    idealStart: [0.5, 2.1, 1.3, 0.8, 2.9, 1.7, 0.4, 2.5, 1.1, 0.9, 2.3, 1.5, 0.7, 2.0, 1.4],
    idealTarget: [1.23, 0.87, 2.45, 0.65, 2.78, 1.92, 0.34, 2.56, 1.08, 0.98, 2.21, 1.67, 0.54, 1.89, 1.52],
    noisyTarget: [1.19, 0.92, 2.50, 0.71, 2.81, 1.89, 0.39, 2.61, 1.14, 1.03, 2.18, 1.71, 0.58, 1.93, 1.48],
    idealLR: 0.18,
    noisyLR: 0.08
  },
  {
    name: "Slow Starter",
    idealStart: [0.2, 2.5, 0.8, 1.5, 3.0, 1.2, 0.3, 2.8, 0.7, 1.3, 2.7, 1.1, 0.9, 2.2, 1.0],
    idealTarget: [1.25, 0.85, 2.48, 0.63, 2.82, 1.94, 0.32, 2.59, 1.05, 1.00, 2.19, 1.69, 0.52, 1.87, 1.54],
    noisyTarget: [1.21, 0.90, 2.53, 0.69, 2.87, 1.90, 0.38, 2.65, 1.11, 1.06, 2.14, 1.75, 0.57, 1.92, 1.50],
    idealLR: 0.12,
    noisyLR: 0.05
  },
  {
    name: "Balanced",
    idealStart: [0.6, 2.0, 1.4, 0.9, 2.8, 1.6, 0.5, 2.6, 1.2, 1.0, 2.4, 1.4, 0.7, 2.1, 1.5],
    idealTarget: [1.22, 0.88, 2.44, 0.66, 2.75, 1.91, 0.35, 2.55, 1.09, 0.97, 2.23, 1.66, 0.55, 1.90, 1.51],
    noisyTarget: [1.18, 0.93, 2.49, 0.72, 2.80, 1.87, 0.41, 2.60, 1.15, 1.03, 2.18, 1.72, 0.60, 1.95, 1.47],
    idealLR: 0.15,
    noisyLR: 0.07
  },
  {
    name: "Near Optimal",
    idealStart: [1.0, 1.2, 2.2, 0.7, 2.5, 1.8, 0.5, 2.4, 1.0, 0.9, 2.4, 1.6, 0.6, 1.9, 1.5],
    idealTarget: [1.21, 0.89, 2.42, 0.68, 2.72, 1.90, 0.37, 2.54, 1.11, 0.96, 2.25, 1.64, 0.57, 1.91, 1.50],
    noisyTarget: [1.17, 0.94, 2.48, 0.74, 2.77, 1.86, 0.43, 2.60, 1.17, 1.02, 2.20, 1.70, 0.62, 1.96, 1.46],
    idealLR: 0.20,
    noisyLR: 0.10
  },
  {
    name: "Far from Target",
    idealStart: [0.1, 2.8, 0.5, 1.8, 3.1, 0.9, 0.2, 2.9, 0.6, 1.5, 2.9, 0.8, 1.0, 2.3, 0.9],
    idealTarget: [1.24, 0.86, 2.46, 0.64, 2.79, 1.93, 0.33, 2.57, 1.07, 0.99, 2.22, 1.68, 0.53, 1.88, 1.53],
    noisyTarget: [1.20, 0.91, 2.51, 0.70, 2.84, 1.88, 0.40, 2.63, 1.13, 1.05, 2.17, 1.74, 0.59, 1.94, 1.49],
    idealLR: 0.14,
    noisyLR: 0.06
  }
];

// ========================================================================
// 5 CONVERGENCE CURVE PATTERNS - Different Optimization Behaviors
// ========================================================================

const CONVERGENCE_PATTERNS = [
  
  // Pattern 1: Smooth Exponential
  {
    name: "Smooth Exponential",
    idealIterations: 40,
    noisyIterations: 50,
    idealCurve: (i, total) => {
      return GROUND_STATE_ENERGY + 0.35 * Math.exp(-i / 7) - 0.001;
    },
    noisyCurve: (i, total) => {
      const base = GROUND_STATE_ENERGY + 0.35 * Math.exp(-i / 9) + 0.005;
      const noise = (Math.random() - 0.5) * 0.008 * (1 - i/total);
      return base + noise;
    }
  },
  
  // Pattern 2: Jagged with Spikes
  {
    name: "Jagged with Spikes",
    idealIterations: 50,
    noisyIterations: 60,
    idealCurve: (i, total) => {
      return GROUND_STATE_ENERGY + 0.4 * Math.exp(-i / 8) - 0.002;
    },
    noisyCurve: (i, total) => {
      const base = GROUND_STATE_ENERGY + 0.4 * Math.exp(-i / 10) + 0.006;
      const noise = (Math.random() - 0.5) * 0.015 * (1 - i/total);
      
      // Add random spikes
      const spike = (i > 20 && i < 45 && Math.random() > 0.92) 
        ? (Math.random() - 0.5) * 0.3 
        : 0;
      
      return base + noise + spike;
    }
  },
  
  // Pattern 3: Plateau Midway
  {
    name: "Plateau Midway",
    idealIterations: 35,
    noisyIterations: 48,
    idealCurve: (i, total) => {
      if (i < 15) {
        return GROUND_STATE_ENERGY + 0.3 * Math.exp(-i / 5);
      } else if (i < 25) {
        return GROUND_STATE_ENERGY + 0.05;
      } else {
        return GROUND_STATE_ENERGY + 0.05 * Math.exp(-(i-25) / 4) - 0.001;
      }
    },
    noisyCurve: (i, total) => {
      const noise = (Math.random() - 0.5) * 0.01 * (1 - i/total);
      
      if (i < 20) {
        return GROUND_STATE_ENERGY + 0.3 * Math.exp(-i / 7) + 0.007 + noise;
      } else if (i < 35) {
        return GROUND_STATE_ENERGY + 0.08 + noise * 2;
      } else {
        return GROUND_STATE_ENERGY + 0.08 * Math.exp(-(i-35) / 5) + 0.002 + noise;
      }
    }
  },
  
  // Pattern 4: Oscillating Descent
  {
    name: "Oscillating Descent",
    idealIterations: 45,
    noisyIterations: 55,
    idealCurve: (i, total) => {
      const base = GROUND_STATE_ENERGY + 0.35 * Math.exp(-i / 8) - 0.001;
      const oscillation = 0.02 * Math.sin(i / 3) * Math.exp(-i / 20);
      return base + oscillation;
    },
    noisyCurve: (i, total) => {
      const base = GROUND_STATE_ENERGY + 0.35 * Math.exp(-i / 10) + 0.006;
      const oscillation = 0.04 * Math.sin(i / 2.5) * Math.exp(-i / 15);
      const noise = (Math.random() - 0.5) * 0.012 * (1 - i/total);
      return base + oscillation + noise;
    }
  },
  
  // Pattern 5: Step-wise Drops
  {
    name: "Step-wise Drops",
    idealIterations: 42,
    noisyIterations: 52,
    idealCurve: (i, total) => {
      const steps = [
        {end: 10, energy: -0.9},
        {end: 20, energy: -1.05},
        {end: 30, energy: -1.12},
        {end: 42, energy: -1.135}
      ];
      
      for (let step of steps) {
        if (i <= step.end) {
          const noise = (Math.random() - 0.5) * 0.005;
          return step.energy + noise;
        }
      }
      return GROUND_STATE_ENERGY - 0.002;
    },
    noisyCurve: (i, total) => {
      const steps = [
        {end: 15, energy: -0.85},
        {end: 28, energy: -1.02},
        {end: 40, energy: -1.10},
        {end: 52, energy: -1.130}
      ];
      
      for (let step of steps) {
        if (i <= step.end) {
          const noise = (Math.random() - 0.5) * 0.015;
          return step.energy + noise;
        }
      }
      return GROUND_STATE_ENERGY + 0.003;
    }
  }
  
];

// ========================================================================
// GENERATE ANIMATION FRAMES FROM COMBINATION
// ========================================================================

function generateFramesFromCombo(paramProfile, pattern) {
  const idealFrames = [];
  const noisyFrames = [];
  
  let idealParams = [...paramProfile.idealStart];
  let noisyParams = [...paramProfile.idealStart];
  
  // Generate ideal frames
  for (let i = 0; i <= pattern.idealIterations; i++) {
    idealParams = idealParams.map((p, idx) => 
      p + (paramProfile.idealTarget[idx] - p) * paramProfile.idealLR
    );
    
    const energy = pattern.idealCurve(i, pattern.idealIterations);
    
    idealFrames.push({
      iteration: i,
      params: [...idealParams],
      energy: energy
    });
  }
  
  // Generate noisy frames
  for (let i = 0; i <= pattern.noisyIterations; i++) {
    noisyParams = noisyParams.map((p, idx) => {
      const noise = (Math.random() - 0.5) * 0.1 * (1 - i/pattern.noisyIterations);
      return p + (paramProfile.noisyTarget[idx] - p) * paramProfile.noisyLR + noise;
    });
    
    const energy = pattern.noisyCurve(i, pattern.noisyIterations);
    
    noisyFrames.push({
      iteration: i,
      params: [...noisyParams],
      energy: energy
    });
  }
  
  return { ideal: idealFrames, noisy: noisyFrames };
}

// ========================================================================
// RANDOM COMBINATION SELECTOR
// ========================================================================

let currentParamProfile = null;
let currentPattern = null;
let animationData = null;

function loadRandomCombination() {
  const paramIdx = Math.floor(Math.random() * PARAMETER_PROFILES.length);
  const patternIdx = Math.floor(Math.random() * CONVERGENCE_PATTERNS.length);
  
  currentParamProfile = PARAMETER_PROFILES[paramIdx];
  currentPattern = CONVERGENCE_PATTERNS[patternIdx];
  
  animationData = generateFramesFromCombo(currentParamProfile, currentPattern);
  
  const comboNumber = paramIdx * CONVERGENCE_PATTERNS.length + patternIdx + 1;
  
  console.log(`✅ [VQE Demo] Combo #${comboNumber}/25`);
  console.log(`   Parameters: "${currentParamProfile.name}"`);
  console.log(`   Pattern: "${currentPattern.name}"`);
  console.log(`   Ideal: ${animationData.ideal.length} frames | Noisy: ${animationData.noisy.length} frames`);
  
  return animationData;
}

// Initialize with random combo
loadRandomCombination();

// ========================================================================
// CIRCUIT DRAWER - SVG Quantum Circuit with Live Parameters
// ========================================================================

class CircuitDrawer {
  constructor(svgId, isNoisy = false) {
    this.svg = document.getElementById(svgId);
    this.isNoisy = isNoisy;
    this.width = 850;
    this.height = 280;
    this.params = Array(15).fill(0);
    
    this.svg.setAttribute('viewBox', `0 0 ${this.width} ${this.height}`);
    this.svg.setAttribute('width', '100%');
    this.svg.setAttribute('height', '100%');
    
    this.drawCircuit();
  }
  
  drawCircuit() {
    const qubitY = [50, 140, 230];
    const startX = 60;
    const rySpacing = 85;
    const cnotSpacing = 50;
    
    // Draw qubit wires
    qubitY.forEach((y, q) => {
      const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
      line.setAttribute('x1', 20);
      line.setAttribute('y1', y);
      line.setAttribute('x2', this.width - 10);
      line.setAttribute('y2', y);
      line.setAttribute('stroke', this.isNoisy ? '#ff5f56' : '#0f62fe');
      line.setAttribute('stroke-width', '2');
      line.setAttribute('opacity', '0.3');
      this.svg.appendChild(line);
      
      // Qubit label
      const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      label.setAttribute('x', 15);
      label.setAttribute('y', y + 5);
      label.setAttribute('fill', '#c6c6c6');
      label.setAttribute('font-size', '16');
      label.setAttribute('font-weight', 'bold');
      label.textContent = `q${q}`;
      this.svg.appendChild(label);
    });
    
    // Circuit structure:
    // 5 layers: [RY, RY, RY] -> CNOT(0->1) -> CNOT(1->2)
    let xPos = startX;
    let paramIdx = 0;
    
    for (let layer = 0; layer < 5; layer++) {
      // RY gates for all 3 qubits
      for (let q = 0; q < 3; q++) {
        this.drawRYGate(xPos, qubitY[q], paramIdx);
        paramIdx++;
      }
      
      xPos += rySpacing;
      
      // CNOTs (except on last layer)
      if (layer < 4) {
        this.drawCNOT(xPos, qubitY[0], qubitY[1]);
        xPos += cnotSpacing;
        this.drawCNOT(xPos, qubitY[1], qubitY[2]);
        xPos += cnotSpacing;
      }
    }
  }
  
  drawRYGate(x, y, paramIdx) {
    const group = document.createElementNS('http://www.w3.org/2000/svg', 'g');
    group.setAttribute('id', `${this.isNoisy ? 'noisy' : 'ideal'}-gate-${paramIdx}`);
    
    // Gate box
    const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
    rect.setAttribute('x', x - 28);
    rect.setAttribute('y', y - 18);
    rect.setAttribute('width', '56');
    rect.setAttribute('height', '36');
    rect.setAttribute('fill', '#262626');
    rect.setAttribute('stroke', this.isNoisy ? '#ff5f56' : '#0f62fe');
    rect.setAttribute('stroke-width', '2');
    rect.setAttribute('rx', '4');
    group.appendChild(rect);
    
    // Parameter value
    const valueText = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    valueText.setAttribute('x', x);
    valueText.setAttribute('y', y + 6);
    valueText.setAttribute('text-anchor', 'middle');
    valueText.setAttribute('fill', this.isNoisy ? '#ff5f56' : '#0f62fe');
    valueText.setAttribute('font-size', '14');
    valueText.setAttribute('font-weight', 'bold');
    valueText.setAttribute('font-family', 'Courier New, monospace');
    valueText.setAttribute('id', `${this.isNoisy ? 'noisy' : 'ideal'}-value-${paramIdx}`);
    valueText.textContent = '0.000';
    group.appendChild(valueText);
    
    this.svg.appendChild(group);
  }
  
  drawCNOT(x, controlY, targetY) {
    // Control dot
    const controlDot = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
    controlDot.setAttribute('cx', x);
    controlDot.setAttribute('cy', controlY);
    controlDot.setAttribute('r', '5');
    controlDot.setAttribute('fill', this.isNoisy ? '#ff5f56' : '#0f62fe');
    this.svg.appendChild(controlDot);
    
    // Connecting line
    const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    line.setAttribute('x1', x);
    line.setAttribute('y1', controlY);
    line.setAttribute('x2', x);
    line.setAttribute('y2', targetY);
    line.setAttribute('stroke', this.isNoisy ? '#ff5f56' : '#0f62fe');
    line.setAttribute('stroke-width', '2');
    this.svg.appendChild(line);
    
    // Target X circle
    const targetCircle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
    targetCircle.setAttribute('cx', x);
    targetCircle.setAttribute('cy', targetY);
    targetCircle.setAttribute('r', '14');
    targetCircle.setAttribute('fill', 'none');
    targetCircle.setAttribute('stroke', this.isNoisy ? '#ff5f56' : '#0f62fe');
    targetCircle.setAttribute('stroke-width', '2');
    this.svg.appendChild(targetCircle);
    
    // X lines
    const xLine1 = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    xLine1.setAttribute('x1', x - 9);
    xLine1.setAttribute('y1', targetY - 9);
    xLine1.setAttribute('x2', x + 9);
    xLine1.setAttribute('y2', targetY + 9);
    xLine1.setAttribute('stroke', this.isNoisy ? '#ff5f56' : '#0f62fe');
    xLine1.setAttribute('stroke-width', '2');
    this.svg.appendChild(xLine1);
    
    const xLine2 = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    xLine2.setAttribute('x1', x + 9);
    xLine2.setAttribute('y1', targetY - 9);
    xLine2.setAttribute('x2', x - 9);
    xLine2.setAttribute('y2', targetY + 9);
    xLine2.setAttribute('stroke', this.isNoisy ? '#ff5f56' : '#0f62fe');
    xLine2.setAttribute('stroke-width', '2');
    this.svg.appendChild(xLine2);
  }
  
  updateParameters(params) {
    this.params = params;
    
    // Update parameter values in gates
    params.forEach((value, idx) => {
      const valueText = document.getElementById(`${this.isNoisy ? 'noisy' : 'ideal'}-value-${idx}`);
      if (valueText) {
        valueText.textContent = value.toFixed(3);
        
        // Pulse animation effect
        const gate = document.getElementById(`${this.isNoisy ? 'noisy' : 'ideal'}-gate-${idx}`);
        if (gate) {
          const rect = gate.querySelector('rect');
          if (rect) {
            rect.setAttribute('stroke-width', '3');
            rect.setAttribute('fill', this.isNoisy ? 'rgba(255, 95, 86, 0.15)' : 'rgba(15, 98, 254, 0.15)');
            setTimeout(() => {
              rect.setAttribute('stroke-width', '2');
              rect.setAttribute('fill', '#262626');
            }, 250);
          }
        }
      }
    });
  }
}

// ========================================================================
// CHART SETUP - Energy Convergence Graph
// ========================================================================

let comparisonChart = null;

function initChart() {
  const ctx = document.getElementById('comparison-chart').getContext('2d');
  
  comparisonChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: [],
      datasets: [
        {
          label: 'Ideal VQE',
          data: [],
          borderColor: '#0f62fe',
          backgroundColor: 'rgba(15, 98, 254, 0.1)',
          borderWidth: 3,
          tension: 0.3,
          pointRadius: 3,
          pointHoverRadius: 6,
          fill: true
        },
        {
          label: 'Noisy VQE',
          data: [],
          borderColor: '#ff5f56',
          backgroundColor: 'rgba(255, 95, 86, 0.1)',
          borderWidth: 3,
          tension: 0.2,
          pointRadius: 3,
          pointHoverRadius: 6,
          fill: true
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 300 },
      scales: {
        x: {
          title: {
            display: true,
            text: 'Iteration',
            color: '#fff',
            font: { size: 13, weight: 'bold' }
          },
          grid: { color: 'rgba(255, 255, 255, 0.05)' },
          ticks: { color: '#c6c6c6' }
        },
        y: {
          min: -1.5,
          max: -0.8,
          title: {
            display: true,
            text: 'Energy (Hartree)',
            color: '#fff',
            font: { size: 13, weight: 'bold' }
          },
          grid: { color: 'rgba(255, 255, 255, 0.1)' },
          ticks: {
            color: '#c6c6c6',
            callback: function(value) {
              return value.toFixed(3);
            }
          }
        }
      },
      plugins: {
        legend: {
          display: true,
          position: 'top',
          labels: {
            color: '#fff',
            font: { size: 12 },
            padding: 15,
            usePointStyle: true
          }
        },
        tooltip: {
          callbacks: {
            label: function(context) {
              const energy = context.parsed.y;
              const error = Math.abs(energy - GROUND_STATE_ENERGY);
              return [
                `${context.dataset.label}: ${energy.toFixed(6)} Ha`,
                `Error: ${error.toFixed(6)} Ha`
              ];
            }
          }
        }
      }
    }
  });
  
  addGroundStateLine();
}

function addGroundStateLine() {
  Chart.register({
    id: 'groundStateLine',
    afterDatasetsDraw: (chart) => {
      const ctx = chart.ctx;
      const yScale = chart.scales.y;
      const xScale = chart.scales.x;
      const yPos = yScale.getPixelForValue(GROUND_STATE_ENERGY);
      
      ctx.save();
      ctx.strokeStyle = '#fff';
      ctx.lineWidth = 2;
      ctx.setLineDash([5, 5]);
      ctx.beginPath();
      ctx.moveTo(xScale.left, yPos);
      ctx.lineTo(xScale.right, yPos);
      ctx.stroke();
      ctx.restore();
      
      ctx.save();
      ctx.fillStyle = '#fff';
      ctx.font = '11px sans-serif';
      ctx.fillText('Ground State', xScale.right - 90, yPos - 5);
      ctx.restore();
    }
  });
}

// ========================================================================
// ANIMATION CONTROLLER
// ========================================================================

class VQEAnimationController {
  constructor() {
    this.isRunning = false;
    this.idealFrame = 0;
    this.noisyFrame = 0;
    this.speed = 150;
    
    this.idealCircuit = new CircuitDrawer('ideal-circuit-svg', false);
    this.noisyCircuit = new CircuitDrawer('noisy-circuit-svg', true);
    
    this.idealCircuit.updateParameters(animationData.ideal[0].params);
    this.noisyCircuit.updateParameters(animationData.noisy[0].params);
  }
  
  async start() {
    if (this.isRunning) return;
    
    // Load new random combination each run
    loadRandomCombination();
    
    this.isRunning = true;
    this.idealFrame = 0;
    this.noisyFrame = 0;
    
    document.getElementById('ideal-circuit-status').textContent = 'Optimizing...';
    document.getElementById('ideal-circuit-status').classList.add('running');
    document.getElementById('noisy-circuit-status').textContent = 'Optimizing...';
    document.getElementById('noisy-circuit-status').classList.add('running');
    document.getElementById('graph-status').textContent = `${currentParamProfile.name} • ${currentPattern.name}`;
    document.getElementById('metrics-card').style.display = 'none';
    document.getElementById('run-comparison-btn').disabled = true;
    
    comparisonChart.data.labels = [];
    comparisonChart.data.datasets[0].data = [];
    comparisonChart.data.datasets[1].data = [];
    comparisonChart.update('none');
    
    await this.animationLoop();
  }
  
  async animationLoop() {
    while (this.isRunning) {
      const idealDone = this.idealFrame >= animationData.ideal.length;
      const noisyDone = this.noisyFrame >= animationData.noisy.length;
      
      if (idealDone && noisyDone) {
        this.complete();
        break;
      }
      
      if (!idealDone) {
        const frame = animationData.ideal[this.idealFrame];
        this.idealCircuit.updateParameters(frame.params);
        comparisonChart.data.datasets[0].data.push(frame.energy);
        
        if (!noisyDone || this.noisyFrame === 0) {
          comparisonChart.data.labels.push(frame.iteration);
        }
        
        this.idealFrame++;
        
        if (this.idealFrame >= animationData.ideal.length) {
          document.getElementById('ideal-circuit-status').textContent = 'Converged';
          document.getElementById('ideal-circuit-status').classList.remove('running');
          document.getElementById('ideal-circuit-status').classList.add('converged');
        }
      }
      
      if (!noisyDone) {
        const frame = animationData.noisy[this.noisyFrame];
        this.noisyCircuit.updateParameters(frame.params);
        comparisonChart.data.datasets[1].data.push(frame.energy);
        
        this.noisyFrame++;
        
        if (this.noisyFrame >= animationData.noisy.length) {
          document.getElementById('noisy-circuit-status').textContent = 'Converged';
          document.getElementById('noisy-circuit-status').classList.remove('running');
          document.getElementById('noisy-circuit-status').classList.add('converged');
        }
      }
      
      comparisonChart.update('active');
      await new Promise(resolve => setTimeout(resolve, this.speed));
    }
  }
  
  complete() {
    this.isRunning = false;
    
    document.getElementById('graph-status').textContent = `${currentParamProfile.name} • ${currentPattern.name} - Complete!`;
    document.getElementById('run-comparison-btn').disabled = false;
    
    const idealFinal = animationData.ideal[animationData.ideal.length - 1];
    const noisyFinal = animationData.noisy[animationData.noisy.length - 1];
    
    const idealIters = animationData.ideal.length - 1;
    const noisyIters = animationData.noisy.length - 1;
    
    document.getElementById('ideal-iters').textContent = idealIters;
    document.getElementById('noisy-iters').textContent = noisyIters;
    document.getElementById('iter-diff').textContent = `+${noisyIters - idealIters} (${Math.round((noisyIters - idealIters) / idealIters * 100)}%)`;
    
    document.getElementById('ideal-energy').textContent = idealFinal.energy.toFixed(6) + ' Ha';
    document.getElementById('noisy-energy').textContent = noisyFinal.energy.toFixed(6) + ' Ha';
    document.getElementById('energy-diff').textContent = Math.abs(noisyFinal.energy - idealFinal.energy).toFixed(6) + ' Ha';
    
    document.getElementById('metrics-card').style.display = 'block';
    
    console.log('✅ [VQE Demo] Animation complete!');
  }
  
  reset() {
    this.isRunning = false;
    this.idealFrame = 0;
    this.noisyFrame = 0;
    
    document.getElementById('ideal-circuit-status').textContent = 'Ready';
    document.getElementById('ideal-circuit-status').classList.remove('running', 'converged');
    document.getElementById('noisy-circuit-status').textContent = 'Ready';
    document.getElementById('noisy-circuit-status').classList.remove('running', 'converged');
    document.getElementById('graph-status').textContent = 'Click Run to start new scenario...';
    document.getElementById('metrics-card').style.display = 'none';
    document.getElementById('run-comparison-btn').disabled = false;
    
    this.idealCircuit.updateParameters(animationData.ideal[0].params);
    this.noisyCircuit.updateParameters(animationData.noisy[0].params);
    
    comparisonChart.data.labels = [];
    comparisonChart.data.datasets[0].data = [];
    comparisonChart.data.datasets[1].data = [];
    comparisonChart.update('none');
    
    console.log('🔄 [VQE Demo] Reset complete');
  }
}

// ========================================================================
// INITIALIZATION
// ========================================================================

let controller = null;

window.addEventListener('load', () => {
  console.log('🚀 [VQE Demo] Initializing...');
  
  initChart();
  controller = new VQEAnimationController();
  
  document.getElementById('run-comparison-btn').addEventListener('click', () => {
    controller.start();
  });
  
  document.getElementById('reset-demo-btn').addEventListener('click', () => {
    controller.reset();
  });
  
  console.log('✅ [VQE Demo] Ready! 25 unique scenarios loaded.');
  console.log('💡 Click "Run Comparison" for random parameter + pattern combo');
});

console.log('✅ [VQE Demo] Script loaded');
