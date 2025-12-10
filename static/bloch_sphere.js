// ========================================================================
// Q-SPHERE - VECTOR-BASED QUANTUM STATE VISUALIZATION
// ========================================================================
// Displays all computational basis states as vectors/arrows on a sphere
// Position determined by binary representation
// Length = probability, Color = phase
// ========================================================================

let qsphereScene, qsphereCamera, qsphereRenderer;
let stateVectors = []; // Array of {arrow, label, probability}
let qsphereInitialized = false;
let rotationSpeed = 0.003;

// Color palette for phase (vibrant hue-based)
function getPhaseColor(phi) {
  // phi ranges from -π to π
  // Map to hue: -π = red (0°), 0 = cyan (180°), π = magenta (300°)
  const hue = ((phi + Math.PI) / (2 * Math.PI)) * 360;
  return new THREE.Color(`hsl(${hue}, 85%, 65%)`);
}

// ========================================================================
// COORDINATE CALCULATION FOR Q-SPHERE
// ========================================================================

function getQSpherePosition(stateLabel, numQubits) {
  /**
   * Map binary state to Q-sphere coordinates
   * States organized by Hamming weight (number of 1s)
   */
  
  // Count number of 1s (Hamming weight)
  const hammingWeight = (stateLabel.match(/1/g) || []).length;
  
  // Theta: vertical position based on Hamming weight
  const theta = (hammingWeight / numQubits) * Math.PI;
  
  // Phi: horizontal position - distribute states evenly around ring
  const stateIndex = parseInt(stateLabel, 2);
  const statesAtLevel = binomialCoefficient(numQubits, hammingWeight);
  
  // Find position among states with same Hamming weight
  let positionInRing = 0;
  for (let i = 0; i < stateIndex; i++) {
    const binaryStr = i.toString(2).padStart(numQubits, '0');
    const weight = (binaryStr.match(/1/g) || []).length;
    if (weight === hammingWeight) positionInRing++;
  }
  
  const phi = (positionInRing / statesAtLevel) * 2 * Math.PI;
  
  // Convert to Cartesian (unit sphere)
  const x = Math.sin(theta) * Math.cos(phi);
  const y = Math.cos(theta);
  const z = Math.sin(theta) * Math.sin(phi);
  
  return new THREE.Vector3(x, y, z);
}

function binomialCoefficient(n, k) {
  if (k < 0 || k > n) return 0;
  if (k === 0 || k === n) return 1;
  
  let result = 1;
  for (let i = 1; i <= k; i++) {
    result *= (n - i + 1) / i;
  }
  return Math.round(result);
}

// ========================================================================
// INITIALIZATION
// ========================================================================

function initQSphere() {
  console.log('🌍 [Q-Sphere] Initializing vector-based quantum visualization...');
  
  const canvas = document.getElementById('blochCanvas');
  if (!canvas) {
    console.error('❌ [Q-Sphere] Canvas not found!');
    return;
  }

  const container = document.getElementById('bottom-right');
  if (!container) {
    console.error('❌ [Q-Sphere] Container not found!');
    return;
  }

  const rect = container.getBoundingClientRect();

  if (typeof THREE === 'undefined') {
    console.error('❌ [Q-Sphere] THREE.js not loaded!');
    return;
  }

  // Scene
  qsphereScene = new THREE.Scene();
  qsphereScene.background = new THREE.Color(0x10131b);

  // Camera
  qsphereCamera = new THREE.PerspectiveCamera(50, rect.width / rect.height, 0.1, 1000);
  qsphereCamera.position.set(3.5, 3.5, 5);
  qsphereCamera.lookAt(0, 0, 0);

  // Renderer
  qsphereRenderer = new THREE.WebGLRenderer({ canvas: canvas, antialias: true });
  qsphereRenderer.setSize(rect.width - 16, rect.height - 40);
  qsphereRenderer.setPixelRatio(window.devicePixelRatio);

  // Lighting
  const ambientLight = new THREE.AmbientLight(0xffffff, 0.7);
  qsphereScene.add(ambientLight);

  const directionalLight = new THREE.DirectionalLight(0xffffff, 0.6);
  directionalLight.position.set(5, 5, 5);
  qsphereScene.add(directionalLight);

  // Create wire sphere reference
  createWireSphere();

  // Add axis labels
  addAxisLabels();

  qsphereInitialized = true;
  console.log('✅ [Q-Sphere] Initialization complete!');
  animateQSphere();
}

// ========================================================================
// SPHERE GEOMETRY
// ========================================================================

function createWireSphere() {
  // Wireframe sphere (slightly larger than unit sphere)
  const sphereGeometry = new THREE.SphereGeometry(1.5, 32, 16);
  const sphereMaterial = new THREE.MeshBasicMaterial({
    color: 0x2c3e50,
    wireframe: true,
    transparent: true,
    opacity: 0.15
  });
  const wireSphere = new THREE.Mesh(sphereGeometry, sphereMaterial);
  qsphereScene.add(wireSphere);

  // Equator ring (brighter)
  const ringGeometry = new THREE.TorusGeometry(1.5, 0.018, 8, 64);
  const ringMaterial = new THREE.MeshBasicMaterial({ 
    color: 0x5a6174,
    transparent: true,
    opacity: 0.6
  });
  const equator = new THREE.Mesh(ringGeometry, ringMaterial);
  equator.rotation.x = Math.PI / 2;
  qsphereScene.add(equator);

  // Add subtle meridian lines
  for (let i = 0; i < 4; i++) {
    const meridian = new THREE.Mesh(ringGeometry, ringMaterial);
    meridian.rotation.y = (i * Math.PI) / 4;
    qsphereScene.add(meridian);
  }
}

function addAxisLabels() {
  // |0...0⟩ at top (North pole)
  const canvas0 = document.createElement('canvas');
  const ctx0 = canvas0.getContext('2d');
  canvas0.width = 128;
  canvas0.height = 64;
  ctx0.fillStyle = '#e0e3ff';
  ctx0.font = 'bold 32px monospace';
  ctx0.textAlign = 'center';
  ctx0.fillText('|0...0⟩', 64, 42);
  
  const texture0 = new THREE.CanvasTexture(canvas0);
  const sprite0 = new THREE.Sprite(new THREE.SpriteMaterial({ map: texture0 }));
  sprite0.position.set(0, 2.0, 0);
  sprite0.scale.set(1.2, 0.6, 1);
  qsphereScene.add(sprite0);

  // |1...1⟩ at bottom (South pole)
  const canvas1 = document.createElement('canvas');
  const ctx1 = canvas1.getContext('2d');
  canvas1.width = 128;
  canvas1.height = 64;
  ctx1.fillStyle = '#e0e3ff';
  ctx1.font = 'bold 32px monospace';
  ctx1.textAlign = 'center';
  ctx1.fillText('|1...1⟩', 64, 42);
  
  const texture1 = new THREE.CanvasTexture(canvas1);
  const sprite1 = new THREE.Sprite(new THREE.SpriteMaterial({ map: texture1 }));
  sprite1.position.set(0, -2.0, 0);
  sprite1.scale.set(1.2, 0.6, 1);
  qsphereScene.add(sprite1);
}

// ========================================================================
// VECTOR VISUALIZATION
// ========================================================================

function clearStateVectors() {
  stateVectors.forEach(item => {
    qsphereScene.remove(item.arrow);
    if (item.label) qsphereScene.remove(item.label);
  });
  stateVectors = [];
}

function updateQSphere(stateData) {
  console.log('🌍 [Q-Sphere] Updating state vectors:', stateData);
  
  if (!qsphereInitialized) {
    console.warn('⚠️ [Q-Sphere] Not initialized yet!');
    return;
  }

  // Clear existing vectors
  clearStateVectors();

  // Determine number of qubits from first state label
  const numQubits = stateData[0]?.label?.length || 4;

  // Sort by probability (largest first for z-ordering)
  const sortedStates = [...stateData].sort((a, b) => b.probability - a.probability);

  sortedStates.forEach((state, index) => {
    if (state.probability < 0.001) return; // Skip very small probabilities

    const stateLabel = state.label || state.state || '0000';
    const direction = getQSpherePosition(stateLabel, numQubits);

    // Arrow length based on probability (scaled to sphere radius)
    const length = (0.3 + state.probability * 1.2)*1.5; // 0.3 to 1.5

    // Color based on phase
    const color = getPhaseColor(state.phi || 0);

    // Create arrow from origin to position on sphere
    const origin = new THREE.Vector3(0, 0, 0);
    const arrowHelper = new THREE.ArrowHelper(
      direction,           // Direction (normalized)
      origin,              // Origin (center of sphere)
      length,              // Length (probability-based)
      color.getHex(),      // Color (phase-based)
      0.15,                // Head length
      0.10                 // Head width
    );

    // Make arrow lines thicker for visibility
    arrowHelper.line.material.linewidth = 4;

    qsphereScene.add(arrowHelper);

    // Add label for significant states (prob > 5%)
    let labelSprite = null;
    if (state.probability > 0.05) {
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d');
      canvas.width = 160;
      canvas.height = 56;
      
      // State label
      ctx.fillStyle = color.getStyle();
      ctx.font = 'bold 24px monospace';
      ctx.textAlign = 'center';
      ctx.fillText(`|${stateLabel}⟩`, 80, 24);
      
      // Probability
      ctx.font = 'bold 16px monospace';
      ctx.fillStyle = '#ffffff';
      ctx.fillText(`${(state.probability * 100).toFixed(1)}%`, 80, 44);
      
      const texture = new THREE.CanvasTexture(canvas);
      labelSprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: texture }));
      
      // Position label at arrow tip
      const labelPos = direction.clone().multiplyScalar(length + 0.3);
      labelSprite.position.copy(labelPos);
      labelSprite.scale.set(0.8, 0.28, 1);
      qsphereScene.add(labelSprite);
    }

    stateVectors.push({
      arrow: arrowHelper,
      label: labelSprite,
      stateLabel: stateLabel,
      probability: state.probability
    });

    console.log(
      `✓ [Q-Sphere] |${stateLabel}⟩: ` +
      `dir=(${direction.x.toFixed(2)}, ${direction.y.toFixed(2)}, ${direction.z.toFixed(2)}), ` +
      `len=${length.toFixed(2)}, prob=${(state.probability * 100).toFixed(1)}%`
    );
  });

  console.log(`✅ [Q-Sphere] ${stateVectors.length} vectors displayed`);
}

// ========================================================================
// ANIMATION
// ========================================================================

function animateQSphere() {
  if (!qsphereInitialized) return;
  requestAnimationFrame(animateQSphere);

  // Auto-rotate entire scene
  qsphereScene.rotation.y += rotationSpeed;

  qsphereRenderer.render(qsphereScene, qsphereCamera);
}

// ========================================================================
// RESIZE HANDLER
// ========================================================================

function resizeQSphere() {
  if (!qsphereInitialized) return;

  const container = document.getElementById('bottom-right');
  const rect = container.getBoundingClientRect();

  qsphereCamera.aspect = rect.width / rect.height;
  qsphereCamera.updateProjectionMatrix();
  qsphereRenderer.setSize(rect.width - 16, rect.height - 40);

  console.log('[Q-Sphere] Resized');
}

// ========================================================================
// INITIALIZE ON LOAD
// ========================================================================

window.addEventListener('load', () => {
  console.log('🌍 [Q-Sphere] Window loaded, initializing...');
  setTimeout(() => {
    initQSphere();
  }, 100);
});

window.addEventListener('resize', () => {
  resizeQSphere();
});

// Alias for compatibility with main.js
const updateBlochSpheres = updateQSphere;
const blochInitialized = () => qsphereInitialized;

console.log('✅ [Q-Sphere] Vector visualization script loaded');
