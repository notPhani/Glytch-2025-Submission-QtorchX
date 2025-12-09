import React, { useEffect, useRef, useState } from 'react';
import { motion } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import './Landing.css';

interface Particle {
  id: number;
  x: number;
  y: number;
  size: number;
  duration: number;
  delay: number;
  color: string;
}

const Landing: React.FC = () => {
  const navigate = useNavigate();
  const containerRef = useRef<HTMLDivElement>(null);
  const [particles, setParticles] = useState<Particle[]>([]);

  // Generate quantum particles
  useEffect(() => {
    if (!containerRef.current) return;

    const colors = ['#0ea5e9', '#06b6d4', '#0891b2', '#0d9488']; // various blues and cyans
    const newParticles: Particle[] = [];

    for (let i = 0; i < 300; i++) {
      newParticles.push({
        id: i,
        x: Math.random() * 100,
        y: Math.random() * 100,
        size: Math.random() * 5 + 2,
        duration: Math.random() * 8 + 12,
        delay: 0,
        color: colors[Math.floor(Math.random() * colors.length)],
      });
    }

    setParticles(newParticles);
  }, []);

  const handleNavigate = () => {
    navigate('/composer');
  };

  // Icon components
  const QuantumIcon = () => (
    <svg viewBox="0 0 24 24" width="60" height="60" fill="none" stroke="currentColor" strokeWidth="1.2">
      {/* Outer orbital ring */}
      <circle cx="12" cy="12" r="10" strokeDasharray="5,3" opacity="0.8" />
      {/* Middle orbital ring */}
      <circle cx="12" cy="12" r="7" strokeDasharray="4,3" opacity="0.6" />
      {/* Core electron */}
      <circle cx="12" cy="5" r="1.5" fill="currentColor" />
      {/* Electrons */}
      <circle cx="18" cy="14" r="1.2" fill="currentColor" />
      <circle cx="8" cy="16" r="1.2" fill="currentColor" />
      {/* Center nucleus */}
      <circle cx="12" cy="12" r="2" fill="currentColor" opacity="0.9" />
    </svg>
  );

  const AlgorithmIcon = () => (
    <svg viewBox="0 0 24 24" width="48" height="48" fill="none" stroke="currentColor" strokeWidth="1.5">
      <path d="M3 6h18M3 12h18M3 18h18" />
      <circle cx="6" cy="6" r="2" fill="currentColor" />
      <circle cx="18" cy="12" r="2" fill="currentColor" />
      <circle cx="6" cy="18" r="2" fill="currentColor" />
    </svg>
  );

  const NoiseIcon = () => (
    <svg viewBox="0 0 24 24" width="48" height="48" fill="none" stroke="currentColor" strokeWidth="1.5">
      <path d="M3 12c0-3.314 2.686-6 6-6s6 2.686 6 6m-12 0c0 3.314 2.686 6 6 6s6-2.686 6-6m-12 0h12M12 6v12M6 9l-3-3m12 0l3-3M9 18c0 1.657 1.343 3 3 3s3-1.343 3-3" />
    </svg>
  );

  const VisualizationIcon = () => (
    <svg viewBox="0 0 24 24" width="48" height="48" fill="none" stroke="currentColor" strokeWidth="1.5">
      <rect x="3" y="3" width="7" height="7" />
      <rect x="14" y="3" width="7" height="7" />
      <rect x="3" y="14" width="7" height="7" />
      <rect x="14" y="14" width="7" height="7" />
      <circle cx="6.5" cy="6.5" r="2" fill="currentColor" />
      <circle cx="17.5" cy="6.5" r="2" fill="currentColor" />
      <circle cx="6.5" cy="17.5" r="2" fill="currentColor" />
      <circle cx="17.5" cy="17.5" r="2" fill="currentColor" />
    </svg>
  );

  return (
    <div ref={containerRef} className="landing-container">
      {/* Particle Layer */}
      <div className="q-dots-layer">
        {particles.map((particle) => (
          <div
            key={particle.id}
            className="q-dot"
            style={{
              left: `${particle.x}%`,
              top: `${particle.y}%`,
              width: `${particle.size}px`,
              height: `${particle.size}px`,
              backgroundColor: particle.color,
              animation: `q-float ${particle.duration}s ease-in-out infinite`,
              boxShadow: `0 0 ${particle.size * 2}px ${particle.color}`,
            }}
          />
        ))}
      </div>

      {/* Main Content */}
      <div className="landing-content">
        {/* Hero Section */}
        <motion.div
          className="hero-section"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, ease: 'easeOut' }}
        >
          <motion.h1
            className="hero-title"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.1, ease: 'easeOut' }}
          >
            <div className="hero-title-icon">
              <QuantumIcon />
            </div>
            QtorchX
          </motion.h1>

          <motion.p
            className="hero-subtitle"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.2, ease: 'easeOut' }}
          >
            NOISE-AWARE QUANTUM CIRCUIT & PHI-FIELD VISUALIZER
          </motion.p>

          <motion.p
            className="hero-description"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.3, ease: 'easeOut' }}
          >
            Powered by the <span className="highlight-cyan">QtorchX Engine</span> and guided by our quantum simulator,
            demonstrating how modern variational algorithms solve real-time quantum circuits with noise resilience.
          </motion.p>

          <motion.div
            className="hero-cta"
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.8, delay: 0.3, ease: 'easeOut' }}
          >
            <button
              className="q-btn-quantum"
              onClick={handleNavigate}
              onMouseEnter={(e) => {
                e.currentTarget.style.transform = 'scale(1.05)';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.transform = 'scale(1)';
              }}
            >
              ⚡ Test A Circuit Yourself
            </button>
          </motion.div>
        </motion.div>

        {/* Feature Cards Section */}
        <motion.div
          className="features-section"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.8, delay: 0.5 }}
        >
          {/* Card 1: VQE Optimization */}
          <motion.div
            className="q-glass-card q-box-cyan"
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            viewport={{ once: true, margin: '-100px' }}
            whileHover={{ y: -8 }}
          >
            <div className="card-icon">
              <AlgorithmIcon />
            </div>
            <h3>VQE Optimization</h3>
            <p>Advanced quantum algorithms that adapt to real-time circuit noise</p>
          </motion.div>

          {/* Card 2: Noise-Aware Simulation */}
          <motion.div
            className="q-glass-card q-box-blue"
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.1 }}
            viewport={{ once: true, margin: '-100px' }}
            whileHover={{ y: -8 }}
          >
            <div className="card-icon">
              <NoiseIcon />
            </div>
            <h3>Noise-Aware Simulation</h3>
            <p>Realistic quantum noise modeling with mitigation strategies</p>
          </motion.div>

          {/* Card 3: 6-Channel Phi Field */}
          <motion.div
            className="q-glass-card q-box-cyan-light"
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.2 }}
            viewport={{ once: true, margin: '-100px' }}
            whileHover={{ y: -8 }}
          >
            <div className="card-icon">
              <VisualizationIcon />
            </div>
            <h3>6-Channel Phi Field</h3>
            <p>Multi-dimensional quantum state evolution visualization</p>
          </motion.div>
        </motion.div>

        {/* CTA Text */}
        <motion.p
          className="cta-text"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 1, delay: 0.8 }}
        >
          Build quantum circuits · Simulate with noise · Visualize phi fields
        </motion.p>
      </div>
    </div>
  );
};

export default Landing;
