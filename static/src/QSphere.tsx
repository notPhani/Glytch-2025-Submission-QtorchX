import React, { useEffect, useRef } from 'react';
import * as THREE from 'three';

interface QuantumState {
  theta: number;
  phi: number;
  label?: string;
}

interface QSphereProps {
  state?: QuantumState;
  size?: number;
}

const QSphere: React.FC<QSphereProps> = ({ 
  state = { theta: Math.PI / 4, phi: Math.PI / 4, label: '|0⟩' },
  size = 250 
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const cameraRef = useRef<THREE.Camera | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const sphereRef = useRef<THREE.Mesh | null>(null);
  const arrowRef = useRef<THREE.ArrowHelper | null>(null);
  const mouseDown = useRef(false);
  const mouseX = useRef(0);
  const mouseY = useRef(0);
  const targetRotationX = useRef(0);
  const targetRotationY = useRef(0);

  useEffect(() => {
    if (!containerRef.current) return;

    // Scene setup
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0f0f0f);
    sceneRef.current = scene;

    // Camera setup
    const camera = new THREE.PerspectiveCamera(
      75,
      size / size,
      0.1,
      1000
    );
    camera.position.z = 1.8;
    cameraRef.current = camera;

    // Renderer setup
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(size, size);
    renderer.setPixelRatio(window.devicePixelRatio);
    containerRef.current.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    // Lighting
    const light1 = new THREE.DirectionalLight(0xffffff, 0.8);
    light1.position.set(5, 5, 5);
    scene.add(light1);

    const light2 = new THREE.DirectionalLight(0xffffff, 0.4);
    light2.position.set(-5, -5, 5);
    scene.add(light2);

    const ambientLight = new THREE.AmbientLight(0xffffff, 0.3);
    scene.add(ambientLight);

    // Bloch sphere (wireframe)
    const sphereGeometry = new THREE.IcosahedronGeometry(1, 4);
    const sphereMaterial = new THREE.MeshBasicMaterial({
      color: 0x0ea5e9,
      wireframe: true,
      opacity: 0.3,
      transparent: true,
    });
    const sphere = new THREE.Mesh(sphereGeometry, sphereMaterial);
    scene.add(sphere);
    sphereRef.current = sphere;

    // Coordinate axes
    const axesSize = 1.3;
    
    // X axis (red)
    const xAxisArrow = new THREE.ArrowHelper(
      new THREE.Vector3(1, 0, 0),
      new THREE.Vector3(0, 0, 0),
      axesSize,
      0xff6b6b,
      0.3,
      0.2
    );
    scene.add(xAxisArrow);

    // Y axis (green)
    const yAxisArrow = new THREE.ArrowHelper(
      new THREE.Vector3(0, 1, 0),
      new THREE.Vector3(0, 0, 0),
      axesSize,
      0x51cf66,
      0.3,
      0.2
    );
    scene.add(yAxisArrow);

    // Z axis (blue)
    const zAxisArrow = new THREE.ArrowHelper(
      new THREE.Vector3(0, 0, 1),
      new THREE.Vector3(0, 0, 0),
      axesSize,
      0x0ea5e9,
      0.3,
      0.2
    );
    scene.add(zAxisArrow);

    // State vector arrow
    const stateDirection = new THREE.Vector3(
      Math.sin(state.theta) * Math.cos(state.phi),
      Math.sin(state.theta) * Math.sin(state.phi),
      Math.cos(state.theta)
    );

    const stateArrow = new THREE.ArrowHelper(
      stateDirection,
      new THREE.Vector3(0, 0, 0),
      1,
      0xffc75f,
      0.4,
      0.3
    );
    scene.add(stateArrow);
    arrowRef.current = stateArrow;

    // Add state point
    const pointGeometry = new THREE.SphereGeometry(0.08, 16, 16);
    const pointMaterial = new THREE.MeshBasicMaterial({ color: 0xffc75f });
    const statePoint = new THREE.Mesh(pointGeometry, pointMaterial);
    statePoint.position.copy(stateDirection);
    scene.add(statePoint);

    // Mouse interaction
    const onMouseDown = (e: MouseEvent) => {
      mouseDown.current = true;
      mouseX.current = e.clientX;
      mouseY.current = e.clientY;
    };

    const onMouseMove = (e: MouseEvent) => {
      if (!mouseDown.current) return;

      const deltaX = e.clientX - mouseX.current;
      const deltaY = e.clientY - mouseY.current;

      targetRotationY.current += deltaX * 0.01;
      targetRotationX.current += deltaY * 0.01;

      // Clamp X rotation
      targetRotationX.current = Math.max(
        -Math.PI / 2,
        Math.min(Math.PI / 2, targetRotationX.current)
      );

      mouseX.current = e.clientX;
      mouseY.current = e.clientY;
    };

    const onMouseUp = () => {
      mouseDown.current = false;
    };

    const onTouchStart = (e: TouchEvent) => {
      mouseDown.current = true;
      mouseX.current = e.touches[0].clientX;
      mouseY.current = e.touches[0].clientY;
    };

    const onTouchMove = (e: TouchEvent) => {
      if (!mouseDown.current) return;

      const deltaX = e.touches[0].clientX - mouseX.current;
      const deltaY = e.touches[0].clientY - mouseY.current;

      targetRotationY.current += deltaX * 0.01;
      targetRotationX.current += deltaY * 0.01;

      targetRotationX.current = Math.max(
        -Math.PI / 2,
        Math.min(Math.PI / 2, targetRotationX.current)
      );

      mouseX.current = e.touches[0].clientX;
      mouseY.current = e.touches[0].clientY;
    };

    const onTouchEnd = () => {
      mouseDown.current = false;
    };

    renderer.domElement.addEventListener('mousedown', onMouseDown);
    renderer.domElement.addEventListener('mousemove', onMouseMove);
    renderer.domElement.addEventListener('mouseup', onMouseUp);
    renderer.domElement.addEventListener('mouseleave', onMouseUp);
    renderer.domElement.addEventListener('touchstart', onTouchStart);
    renderer.domElement.addEventListener('touchmove', onTouchMove);
    renderer.domElement.addEventListener('touchend', onTouchEnd);

    // Animation loop
    const animate = () => {
      requestAnimationFrame(animate);

      // Smooth rotation
      if (sphereRef.current) {
        sphereRef.current.rotation.x += (targetRotationX.current - sphereRef.current.rotation.x) * 0.1;
        sphereRef.current.rotation.y += (targetRotationY.current - sphereRef.current.rotation.y) * 0.1;
      }

      renderer.render(scene, camera);
    };

    animate();

    // Handle window resize
    const handleResize = () => {
      if (containerRef.current && rendererRef.current) {
        const newSize = Math.min(containerRef.current.clientWidth, 250);
        camera.aspect = 1;
        camera.updateProjectionMatrix();
        rendererRef.current.setSize(newSize, newSize);
      }
    };

    window.addEventListener('resize', handleResize);

    // Cleanup
    return () => {
      window.removeEventListener('resize', handleResize);
      renderer.domElement.removeEventListener('mousedown', onMouseDown);
      renderer.domElement.removeEventListener('mousemove', onMouseMove);
      renderer.domElement.removeEventListener('mouseup', onMouseUp);
      renderer.domElement.removeEventListener('mouseleave', onMouseUp);
      renderer.domElement.removeEventListener('touchstart', onTouchStart);
      renderer.domElement.removeEventListener('touchmove', onTouchMove);
      renderer.domElement.removeEventListener('touchend', onTouchEnd);
      
      if (containerRef.current && renderer.domElement.parentElement === containerRef.current) {
        containerRef.current.removeChild(renderer.domElement);
      }
      sphereGeometry.dispose();
      sphereMaterial.dispose();
      renderer.dispose();
    };
  }, [size, state]);

  return (
    <div
      ref={containerRef}
      style={{
        width: `${size}px`,
        height: `${size}px`,
        position: 'relative',
        cursor: 'grab',
        userSelect: 'none',
      }}
    />
  );
};

export default QSphere;
