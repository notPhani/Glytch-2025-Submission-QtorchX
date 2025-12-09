import React, { useEffect, useRef } from 'react';
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
  const particlesRef = useRef<Particle[]>([]);

  // Generate quantum particles
  useEffect(() => {
    if (!containerRef.current) return;

    const colors = ['#22d3ee', '#a855f7', '#bef264']; // cyan, violet, lime
    const particles: Particle[] = [];

    for (let i = 0; i < 175; i++) {
      particles.push({
        id: i,
        x: Math.random() * 100,
        y: Math.random() * 100,
        size: Math.random() * 4 + 1,
        duration: Math.random() * 20 + 20,
        delay: Math.random() * 5,
        color: colors[Math.floor(Math.random() * colors.length)],
      });
    }

    particlesRef.current = particles;
  }, []);

  const handleNavigate = () => {
    navigate('/composer');
  };

  return (
    <div ref={containerRef} className="landing-container">
      {/* Particle Layer */}
      <div className="q-dots-layer">
        {particlesRef.current.map((particle) => (
          <div
            key={particle.id}
            className="q-dot"
            style={{
              left: `${particle.x}%`,
              top: `${particle.y}%`,
              width: `${particle.size}px`,
              height: `${particle.size}px`,
              backgroundColor: particle.color,
              animation: `q-float ${particle.duration}s ease-in-out ${particle.delay}s infinite`,
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
            QtorchX
          </motion.h1>

          <motion.p
            className="hero-subtitle"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.2, ease: 'easeOut' }}
          >
            Noise-Aware Quantum Circuit & Phi-Field Visualizer
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
              Test A Circuit Yourself
            </button>
          </motion.div>
        </motion.div>

        {/* Feature Cards Section */}
        <motion.div
          className="features-section"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.8, delay: 0.4 }}
        >
          {/* Card 1: Real-World VQE */}
          <motion.div
            className="q-glass-card q-box-cyan"
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            viewport={{ once: true, margin: '-100px' }}
            whileHover={{ y: -10, boxShadow: '0 20px 60px rgba(34, 211, 238, 0.4)' }}
          >
            <div className="card-icon q-icon-cyan">⚛️</div>
            <h3>Real-World VQE on H₂</h3>
            <p>
              Experience variational quantum eigensolver simulations on hydrogen molecules with realistic noise models.
            </p>
          </motion.div>

          {/* Card 2: Why Noise Matters */}
          <motion.div
            className="q-glass-card q-box-violet"
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.1 }}
            viewport={{ once: true, margin: '-100px' }}
            whileHover={{ y: -10, boxShadow: '0 20px 60px rgba(168, 85, 247, 0.4)' }}
          >
            <div className="card-icon q-icon-violet">🌀</div>
            <h3>Why Noise Matters</h3>
            <p>
              Understand how quantum noise impacts circuit fidelity and learn mitigation strategies for NISQ devices.
            </p>
          </motion.div>

          {/* Card 3: 6-Channel Phi Field */}
          <motion.div
            className="q-glass-card q-box-lime"
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.2 }}
            viewport={{ once: true, margin: '-100px' }}
            whileHover={{ y: -10, boxShadow: '0 20px 60px rgba(190, 242, 100, 0.4)' }}
          >
            <div className="card-icon q-icon-lime">🔬</div>
            <h3>The 6-Channel Phi Field</h3>
            <p>
              Visualize multi-dimensional quantum state evolution across six channels in real-time.
            </p>
          </motion.div>
        </motion.div>
      </div>
    </div>
  );
};

export default Landing;
