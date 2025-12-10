// ========================================================================
// BLOCH SPHERE - QUANTUM STATE VISUALIZATION
// ========================================================================

let blochScene, blochCamera, blochRenderer;
let stateArrows = []; // Dynamic array for quantum states
let blochInitialized = false;
let rotationSpeed = 0.003;

// Color palette for different states (vibrant colors)
const STATE_COLORS = [
  0x3498db, // Blue
  0xe74c3c, // Red
  0x2ecc71, // Green
  0xf39c12, // Orange
  0x9b59b6, // Purple
  0x1abc9c, // Teal
  0xe67e22, // Dark Orange
  0xf1c40f, // Yellow
  0xc0392b, // Dark Red
  0x16a085, // Dark Teal
  0x27ae60, // Dark Green
  0x8e44ad, // Dark Purple
  0x2980b9, // Dark Blue
  0xd35400, // Burnt Orange
  0x7f8c8d, // Gray
  0x34495e  // Dark Gray
];

function getStateColor(index) {
  return STATE_COLORS[index % STATE_COLORS.length];
}

// ========================================================================
// INITIALIZATION
// ========================================================================

function initBlochSphere() {
  console.log('🔵 [Bloch] Initializing Bloch sphere for quantum states...');
  
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

  // Create the Bloch sphere
  createBlochSphere();

  blochInitialized = true;
  console.log('✅ [Bloch] Initialization complete!');
  animateBloch();
}

// ========================================================================
// BLOCH SPHERE GEOMETRY
// ========================================================================

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

// ========================================================================
// STATE ARROW MANAGEMENT
// ========================================================================

function clearStateArrows() {
  // Remove all existing state arrows
  stateArrows.forEach(stateObj => {
    blochScene.remove(stateObj.arrow);
  });
  stateArrows = [];
}

function createStateArrow(stateLabel, color, index) {
  const arrowDir = new THREE.Vector3(0, 1, 0);
  const arrowOrigin = new THREE.Vector3(0, 0, 0);
  const arrowLength = 1.0;

  const arrow = new THREE.ArrowHelper(
    arrowDir,
    arrowOrigin,
    arrowLength,
    color,
    0.2,
    0.15
  );

  blochScene.add(arrow);

  return {
    arrow: arrow,
    stateLabel: stateLabel,
    color: color,
    probability: 0,
    index: index
  };
}

// ========================================================================
// LEGEND
// ========================================================================

let legendGroup = null;

function updateLegend() {
  // Remove old legend
  if (legendGroup) {
    blochScene.remove(legendGroup);
  }

  legendGroup = new THREE.Group();
  legendGroup.position.set(1.8, 1.2, 0);

  // Sort states by probability (descending)
  const sortedStates = [...stateArrows].sort((a, b) => b.probability - a.probability);

  for (let i = 0; i < sortedStates.length; i++) {
    const state = sortedStates[i];
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    canvas.width = 200;
    canvas.height = 32;

    // Draw colored square
    ctx.fillStyle = '#' + state.color.toString(16).padStart(6, '0');
    ctx.fillRect(0, 8, 16, 16);

    // Draw state label
    ctx.fillStyle = '#e0e3ff';
    ctx.font = 'bold 14px monospace';
    ctx.textAlign = 'left';
    ctx.fillText(`|${state.stateLabel}⟩ ${(state.probability * 100).toFixed(1)}%`, 22, 22);

    const texture = new THREE.CanvasTexture(canvas);
    const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: texture }));
    sprite.position.set(0, -i * 0.35, 0);
    sprite.scale.set(1.0, 0.25, 1);
    legendGroup.add(sprite);
  }

  blochScene.add(legendGroup);
}

// ========================================================================
// UPDATE FUNCTION - CALLED FROM main.js
// ========================================================================
function updateBlochSpheres(stateData) {
  console.log('🔵 [Bloch] Updating quantum states:', stateData);
  
  if (!blochInitialized) {
    console.warn('⚠️ [Bloch] Not initialized yet!');
    return;
  }

  // Clear existing arrows
  clearStateArrows();

  // Create new arrows for each state
  stateData.forEach((state, index) => {
    // ⚡ DEBUG: Log the actual state object structure
    console.log('🔍 [Bloch] State object:', state);
    
    // Skip states with very low probability
    if (!state.probability || state.probability < 0.001) {
      console.log(`[Bloch] Skipping state (prob: ${state.probability})`);
      return;
    }

    const color = getStateColor(index);
    const stateObj = createStateArrow(state.label || state.state, color, index);
    stateObj.probability = state.probability;

    // ✅ Check if x, y, z exist, otherwise calculate from theta/phi
    let x, y, z;
    
    if (state.x !== undefined && state.y !== undefined && state.z !== undefined) {
      // Use pre-calculated Cartesian coordinates
      x = state.x;
      y = state.y;
      z = state.z;
      console.log('✓ [Bloch] Using backend Cartesian coords');
    } else if (state.theta !== undefined && state.phi !== undefined) {
      // Fallback: calculate from spherical
      const r = Math.sqrt(state.probability); // Approximate magnitude
      x = r * Math.sin(state.theta) * Math.cos(state.phi);
      y = r * Math.sin(state.theta) * Math.sin(state.phi);
      z = r * Math.cos(state.theta);
      console.log('⚠️ [Bloch] Calculated Cartesian from spherical');
    } else {
      console.error('❌ [Bloch] State missing both Cartesian AND spherical coords!', state);
      return;
    }

    // Create direction vector
    const direction = new THREE.Vector3(x, y, z);
    const length = direction.length();
    
    if (length < 0.000) {
      console.warn(`⚠️ [Bloch] State |${state.label || state.state}⟩ has zero length, skipping`);
      return;
    }
    
    direction.normalize();
    stateObj.arrow.setDirection(direction);

    // Scale arrow by probability (bigger = more probable)
    const arrowLength = 0.6 + state.probability * 0.4; // 0.6 to 1.0
    stateObj.arrow.setLength(arrowLength, 0.2, 0.15);

    stateArrows.push(stateObj);

    console.log(`✓ [Bloch] Created state |${state.label || state.state}⟩: x=${x.toFixed(3)}, y=${y.toFixed(3)}, z=${z.toFixed(3)}, prob=${(state.probability * 100).toFixed(1)}%`);
  });

  // Update legend
  updateLegend();

  console.log(`✅ [Bloch] Update complete (${stateArrows.length} states displayed)`);
}

// ========================================================================
// ANIMATION & RENDERING
// ========================================================================

function animateBloch() {
  if (!blochInitialized) return;
  requestAnimationFrame(animateBloch);

  // Auto-rotate entire scene
  blochScene.rotation.y += rotationSpeed;

  blochRenderer.render(blochScene, blochCamera);
}

// ========================================================================
// WINDOW RESIZE
// ========================================================================

function resizeBlochCanvas() {
  if (!blochInitialized) return;

  const container = document.getElementById('bottom-right');
  const rect = container.getBoundingClientRect();

  blochCamera.aspect = rect.width / rect.height;
  blochCamera.updateProjectionMatrix();
  blochRenderer.setSize(rect.width - 16, rect.height - 40);

  console.log('[Bloch] Resized');
}

// ========================================================================
// INITIALIZATION
// ========================================================================

window.addEventListener('load', () => {
  console.log('🔵 [Bloch] Window loaded, initializing...');
  setTimeout(() => {
    initBlochSphere();
  }, 100);
});

window.addEventListener('resize', () => {
  resizeBlochCanvas();
});

console.log('✅ [Bloch] Script loaded (State visualization mode)');
