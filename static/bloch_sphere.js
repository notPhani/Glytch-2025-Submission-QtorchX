
// ========== BLOCH SPHERE - SINGLE SPHERE WITH COLORED ARROWS ==========

let blochScene, blochCamera, blochRenderer;
let qubitArrows = []; // One arrow per qubit
let blochInitialized = false;
let rotationSpeed = 0.003;

// Color for each qubit
const qubitColors = [
  0x3498db, // q[0] - Blue
  0xe74c3c, // q[1] - Red
  0x2ecc71, // q[2] - Green
  0xf39c12  // q[3] - Orange
];

// Initialize Three.js scene
function initBlochSphere() {
  console.log('🔵 [Bloch] Initializing single Bloch sphere...');

  const canvas = document.getElementById('blochCanvas');
  if (!canvas) {
    console.error('❌ [Bloch] Canvas element #blochCanvas not found!');
    return;
  }

  const container = document.getElementById('bottom-right');
  if (!container) {
    console.error('❌ [Bloch] Container #bottom-right not found!');
    return;
  }

  const rect = container.getBoundingClientRect();

  // Check if THREE is defined
  if (typeof THREE === 'undefined') {
    console.error('❌ [Bloch] THREE.js not loaded!');
    return;
  }
  console.log('✓ [Bloch] THREE.js loaded, version:', THREE.REVISION);

  // Scene
  blochScene = new THREE.Scene();
  blochScene.background = new THREE.Color(0x10131b);

  // Camera
  blochCamera = new THREE.PerspectiveCamera(50, rect.width / rect.height, 0.1, 1000);
  blochCamera.position.set(2.5, 2.5, 4);
  blochCamera.lookAt(0, 0, 0);

  // Renderer
  blochRenderer = new THREE.WebGLRenderer({ canvas: canvas, antialias: true });
  blochRenderer.setSize(rect.width - 16, rect.height - 40);
  blochRenderer.setPixelRatio(window.devicePixelRatio);

  // Lighting
  const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
  blochScene.add(ambientLight);

  const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
  directionalLight.position.set(5, 5, 5);
  blochScene.add(directionalLight);

  // Create the single Bloch sphere
  createBlochSphere();

  // Create arrows for each qubit
  createQubitArrows();

  blochInitialized = true;
  console.log('✅ [Bloch] Initialization complete!');
  animateBloch();
}

// Create single Bloch sphere with rings and axes
function createBlochSphere() {
  console.log('🔵 [Bloch] Creating Bloch sphere...');

  const sphereGroup = new THREE.Group();

  // Horizontal rings (latitude lines)
  const ringGeometry = new THREE.TorusGeometry(1, 0.012, 8, 32);
  const ringMaterial = new THREE.MeshBasicMaterial({ color: 0x3a4154 });

  for (let lat = -60; lat <= 60; lat += 30) {
    if (lat === 0) continue;
    const ring = new THREE.Mesh(ringGeometry, ringMaterial);
    const scale = Math.cos(lat * Math.PI / 180);
    ring.scale.set(scale, scale, 1);
    ring.position.y = Math.sin(lat * Math.PI / 180);
    ring.rotation.x = Math.PI / 2;
    sphereGroup.add(ring);
  }

  // Equator (brighter)
  const equator = new THREE.Mesh(
    ringGeometry,
    new THREE.MeshBasicMaterial({ color: 0x5a6174 })
  );
  equator.rotation.x = Math.PI / 2;
  sphereGroup.add(equator);

  // Axes (X, Y, Z)
  const axisLength = 1.4;
  const axesMaterial = new THREE.LineBasicMaterial({ color: 0x5a6174 });

  // X axis (red tint)
  const xGeometry = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(-axisLength, 0, 0),
    new THREE.Vector3(axisLength, 0, 0)
  ]);
  const xAxis = new THREE.Line(xGeometry, new THREE.LineBasicMaterial({ color: 0x774444 }));
  sphereGroup.add(xAxis);

  // Y axis (green tint) - |0⟩ to |1⟩
  const yGeometry = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(0, -axisLength, 0),
    new THREE.Vector3(0, axisLength, 0)
  ]);
  const yAxis = new THREE.Line(yGeometry, new THREE.LineBasicMaterial({ color: 0x447744 }));
  sphereGroup.add(yAxis);

  // Z axis (blue tint)
  const zGeometry = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(0, 0, -axisLength),
    new THREE.Vector3(0, 0, axisLength)
  ]);
  const zAxis = new THREE.Line(zGeometry, new THREE.LineBasicMaterial({ color: 0x444477 }));
  sphereGroup.add(zAxis);

  // Add state labels at poles
  // |0⟩ at top
  const canvas0 = document.createElement('canvas');
  const ctx0 = canvas0.getContext('2d');
  canvas0.width = 64;
  canvas0.height = 64;
  ctx0.fillStyle = '#e0e3ff';
  ctx0.font = 'bold 32px monospace';
  ctx0.textAlign = 'center';
  ctx0.fillText('|0⟩', 32, 40);
  const texture0 = new THREE.CanvasTexture(canvas0);
  const sprite0 = new THREE.Sprite(new THREE.SpriteMaterial({ map: texture0 }));
  sprite0.position.set(0, 1.6, 0);
  sprite0.scale.set(0.5, 0.5, 1);
  sphereGroup.add(sprite0);

  // |1⟩ at bottom
  const canvas1 = document.createElement('canvas');
  const ctx1 = canvas1.getContext('2d');
  canvas1.width = 64;
  canvas1.height = 64;
  ctx1.fillStyle = '#e0e3ff';
  ctx1.font = 'bold 32px monospace';
  ctx1.textAlign = 'center';
  ctx1.fillText('|1⟩', 32, 40);
  const texture1 = new THREE.CanvasTexture(canvas1);
  const sprite1 = new THREE.Sprite(new THREE.SpriteMaterial({ map: texture1 }));
  sprite1.position.set(0, -1.6, 0);
  sprite1.scale.set(0.5, 0.5, 1);
  sphereGroup.add(sprite1);

  blochScene.add(sphereGroup);
  console.log('✓ [Bloch] Sphere created');
}

// Create colored arrows for each qubit
function createQubitArrows() {
  console.log('🔵 [Bloch] Creating qubit arrows...');

  for (let i = 0; i < N_QUBITS; i++) {
    const arrowDir = new THREE.Vector3(0, 1, 0);
    const arrowOrigin = new THREE.Vector3(0, 0, 0);
    const arrowLength = 1.0;
    const arrowColor = qubitColors[i];

    const arrow = new THREE.ArrowHelper(
      arrowDir,
      arrowOrigin,
      arrowLength,
      arrowColor,
      0.2,
      0.15
    );

    arrow.visible = false; // Hidden until state is set
    blochScene.add(arrow);

    qubitArrows.push({
      arrow: arrow,
      visible: false,
      probability: 0,
      qubitIndex: i
    });

    console.log(`✓ [Bloch] Created arrow for q[${i}] (color: ${arrowColor.toString(16)})`);
  }

  // Add legend
  updateLegend();
}
// Create legend showing active states (dynamic)
let legendGroup = null;

function updateLegend() {
  // Remove old legend
  if (legendGroup) {
    blochScene.remove(legendGroup);
  }

  legendGroup = new THREE.Group();
  legendGroup.position.set(1.8, 1.2, 0);

  let activeStates = qubitArrows.filter(a => a.visible);
  
  for (let i = 0; i < activeStates.length; i++) {
    const arrow = activeStates[i];
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    canvas.width = 160;
    canvas.height = 32;

    // Draw colored square
    ctx.fillStyle = '#' + qubitColors[arrow.qubitIndex].toString(16).padStart(6, '0');
    ctx.fillRect(0, 8, 16, 16);

    // Draw state label
    ctx.fillStyle = '#e0e3ff';
    ctx.font = '14px monospace';
    ctx.textAlign = 'left';
    ctx.fillText(`|${arrow.stateLabel}⟩ ${(arrow.probability * 100).toFixed(1)}%`, 22, 22);

    const texture = new THREE.CanvasTexture(canvas);
    const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: texture }));
    sprite.position.set(0, -i * 0.3, 0);
    sprite.scale.set(0.8, 0.2, 1);
    legendGroup.add(sprite);
  }

  blochScene.add(legendGroup);
}


// Update arrow states - MODIFIED FOR STATES
function updateBlochSpheres(stateData) {
  console.log('🔵 [Bloch] Updating quantum states:', stateData);
  
  if (!blochInitialized) {
    console.warn('⚠️ [Bloch] Not initialized yet!');
    return;
  }

  // Hide all arrows first
  qubitArrows.forEach(arrow => {
    arrow.arrow.visible = false;
    arrow.visible = false;
  });

  // Show arrows for each significant state
  for (let i = 0; i < stateData.length && i < qubitArrows.length; i++) {
    const state = stateData[i];
    const arrow = qubitArrows[i];

    // Hide if probability too low
    if (state.probability < 0.01) {
      console.log(`[Bloch] Hiding state ${state.label} (probability ${state.probability})`);
      continue;
    }

    arrow.arrow.visible = true;
    arrow.visible = true;
    arrow.probability = state.probability;
    arrow.stateLabel = state.label;  // Store state label like "0000", "1000"

    // Convert spherical to Cartesian
    const x = Math.sin(state.theta) * Math.cos(state.phi);
    const y = Math.cos(state.theta);
    const z = Math.sin(state.theta) * Math.sin(state.phi);

    // Update arrow direction
    const direction = new THREE.Vector3(x, y, z).normalize();
    arrow.arrow.setDirection(direction);
    
    // Scale arrow by probability (bigger = more probable)
    const length = 0.7 + state.probability * 0.3;  // 0.7 to 1.0
    arrow.arrow.setLength(length, 0.2, 0.15);

    console.log(`✓ [Bloch] Updated state |${state.label}⟩: theta=${state.theta.toFixed(2)}, phi=${state.phi.toFixed(2)}, prob=${state.probability.toFixed(3)}`);
  }

  console.log('✅ [Bloch] Update complete');
}


// Animation loop with auto-rotation
function animateBloch() {
  if (!blochInitialized) return;

  requestAnimationFrame(animateBloch);

  // Auto-rotate entire scene
  blochScene.rotation.y += rotationSpeed;

  blochRenderer.render(blochScene, blochCamera);
}

// Handle window resize
function resizeBlochCanvas() {
  if (!blochInitialized) return;

  const container = document.getElementById('bottom-right');
  const rect = container.getBoundingClientRect();

  blochCamera.aspect = rect.width / rect.height;
  blochCamera.updateProjectionMatrix();
  blochRenderer.setSize(rect.width - 16, rect.height - 40);

  console.log('[Bloch] Resized');
}

// Initialize on load
window.addEventListener('load', () => {
  console.log('🔵 [Bloch] Window loaded, initializing...');
  setTimeout(() => {
    initBlochSphere();
  }, 100);
});

window.addEventListener('resize', () => {
  resizeBlochCanvas();
});

console.log('✅ [Bloch] Script loaded');