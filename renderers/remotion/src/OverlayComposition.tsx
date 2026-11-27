import React from "react";
import { AbsoluteFill, interpolate, spring, useCurrentFrame, useVideoConfig } from "remotion";

export interface MotionLabOverlayProps {
  effectId: string;
  width: number;
  height: number;
  fps: number;
  durationInFrames: number;
  intensity: number;
  opacity: number;
  marginRatio: number;
  seed: number;
  regionCx: number;
  regionCy: number;
  barCount: number;
  particleCount: number;
  strokePx: number;
  audioEnvelope: number[];
}

export const DEFAULT_OVERLAY_PROPS: MotionLabOverlayProps = {
  effectId: "edge_lines",
  width: 1080,
  height: 1920,
  fps: 30,
  durationInFrames: 60,
  intensity: 0.8,
  opacity: 0.65,
  marginRatio: 0.045,
  seed: 42,
  regionCx: 0.84,
  regionCy: 0.86,
  barCount: 16,
  particleCount: 12,
  strokePx: 3,
  audioEnvelope: [],
};

function clamp01(v: number): number {
  return Math.max(0, Math.min(1, v));
}

function easeOutCubic(t: number): number {
  const x = clamp01(t);
  return 1 - Math.pow(1 - x, 3);
}

function smoothEnvelope(t: number, attack = 0.2, release = 0.22): number {
  const x = clamp01(t);
  if (x < attack) return easeOutCubic(x / attack);
  if (x > 1 - release) return easeOutCubic((1 - x) / release);
  return 1;
}

function seededUnit(seed: number, index: number): number {
  // Deterministic Mulberry32 step (no Math.random)
  let a = (seed + Math.imul(index + 1, 0x6d2b79f5)) >>> 0;
  a = Math.imul(a ^ (a >>> 15), a | 1);
  a ^= a + Math.imul(a ^ (a >>> 7), a | 61);
  return ((a ^ (a >>> 14)) >>> 0) / 4294967296;
}

const EdgeAccent: React.FC<{ props: MotionLabOverlayProps; t: number; env: number }> = ({
  props,
  t,
  env,
}) => {
  const { width, height, fps } = useVideoConfig();
  const frame = useCurrentFrame();
  const enter = spring({ frame, fps, config: { damping: 18, stiffness: 120, mass: 0.9 } });
  const mx = Math.round(width * props.marginRatio);
  const my = Math.round(height * props.marginRatio);
  const lineLen = Math.round(width * 0.24 * clamp01(enter));
  const stroke = Math.max(2, Math.round(Math.min(width, height) * 0.0042));
  const op = clamp01(props.opacity * props.intensity * env);

  return (
    <svg width={width} height={height} style={{ position: "absolute", inset: 0 }}>
      <line
        x1={mx}
        y1={my}
        x2={mx + lineLen}
        y2={my}
        stroke={`rgba(255,255,255,${op.toFixed(3)})`}
        strokeWidth={stroke}
        strokeLinecap="round"
      />
      <line
        x1={mx}
        y1={my + stroke * 3}
        x2={mx + Math.round(lineLen * 0.56)}
        y2={my + stroke * 3}
        stroke={`rgba(255,198,118,${(op * 0.85).toFixed(3)})`}
        strokeWidth={Math.max(1, stroke - 1)}
        strokeLinecap="round"
      />
      <line
        x1={width - mx - lineLen}
        y1={height - my}
        x2={width - mx}
        y2={height - my}
        stroke={`rgba(255,255,255,${op.toFixed(3)})`}
        strokeWidth={stroke}
        strokeLinecap="round"
      />
    </svg>
  );
};

const ForegroundFrameAccent: React.FC<{ props: MotionLabOverlayProps; t: number; env: number }> = ({
  props,
  t,
  env,
}) => {
  const { width, height } = useVideoConfig();
  const mx = Math.round(width * props.marginRatio);
  const my = Math.round(height * props.marginRatio);
  const arm = Math.round(Math.min(width, height) * 0.085 * easeOutCubic(Math.min(1, t / 0.35)));
  const stroke = Math.max(2, props.strokePx || 3);
  const op = clamp01(0.8 * props.intensity * env);
  const col = `rgba(246,247,252,${op.toFixed(3)})`;

  return (
    <svg width={width} height={height} style={{ position: "absolute", inset: 0 }}>
      <path d={`M ${mx} ${my + arm} L ${mx} ${my} L ${mx + arm} ${my}`} fill="none" stroke={col} strokeWidth={stroke} strokeLinecap="round" />
      <path d={`M ${width - mx - arm} ${my} L ${width - mx} ${my} L ${width - mx} ${my + arm}`} fill="none" stroke={col} strokeWidth={stroke} strokeLinecap="round" />
      <path d={`M ${mx} ${height - my - arm} L ${mx} ${height - my} L ${mx + arm} ${height - my}`} fill="none" stroke={col} strokeWidth={stroke} strokeLinecap="round" />
      <path d={`M ${width - mx - arm} ${height - my} L ${width - mx} ${height - my} L ${width - mx} ${height - my - arm}`} fill="none" stroke={col} strokeWidth={stroke} strokeLinecap="round" />
    </svg>
  );
};

const RingAccent: React.FC<{ props: MotionLabOverlayProps; t: number; env: number }> = ({
  props,
  t,
  env,
}) => {
  const { width, height } = useVideoConfig();
  let cx = props.regionCx * width;
  let cy = props.regionCy * height;
  if (props.regionCx > 0.3 && props.regionCx < 0.7 && props.regionCy > 0.25 && props.regionCy < 0.75) {
    cx = width * 0.85;
    cy = height * 0.86;
  }
  const baseR = Math.min(width, height) * 0.11;
  const r1 = baseR * (0.25 + 0.75 * easeOutCubic(t));
  const r2 = baseR * (0.14 + 0.56 * easeOutCubic(Math.max(0, (t - 0.12) / 0.88)));
  const stroke = Math.max(2, Math.round(Math.min(width, height) * 0.004));
  const op = clamp01(props.opacity * props.intensity * env);

  return (
    <svg width={width} height={height} style={{ position: "absolute", inset: 0 }}>
      <circle cx={cx} cy={cy} r={r1} fill="none" stroke={`rgba(255,255,255,${op.toFixed(3)})`} strokeWidth={stroke} />
      <circle cx={cx} cy={cy} r={r2} fill="none" stroke={`rgba(162,222,255,${(op * 0.78).toFixed(3)})`} strokeWidth={Math.max(1, stroke - 1)} />
    </svg>
  );
};

const DirectionalShapes: React.FC<{ props: MotionLabOverlayProps; t: number; env: number }> = ({
  props,
  t,
  env,
}) => {
  const { width, height } = useVideoConfig();
  const shift = interpolate(t, [0, 1], [width * 0.07, width * 0.15]);
  const baseY = height * 0.9;
  const size = Math.max(8, Math.round(Math.min(width, height) * 0.018));
  const stroke = Math.max(2, Math.round(Math.min(width, height) * 0.004));
  const op = clamp01(props.opacity * props.intensity * env);

  return (
    <svg width={width} height={height} style={{ position: "absolute", inset: 0 }}>
      {[0, 1, 2].map((i) => {
        const ox = shift + i * size * 1.45;
        const a = op * (0.5 + 0.5 * ((i + 1) / 3));
        return (
          <polyline
            key={i}
            points={`${ox - size / 2},${baseY - size} ${ox + size / 2},${baseY} ${ox - size / 2},${baseY + size}`}
            fill="none"
            stroke={`rgba(255,255,255,${a.toFixed(3)})`}
            strokeWidth={stroke}
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        );
      })}
    </svg>
  );
};

const AmbientParticles: React.FC<{ props: MotionLabOverlayProps; t: number; env: number }> = ({
  props,
  t,
  env,
}) => {
  const { width, height } = useVideoConfig();
  const count = Math.min(18, Math.max(6, props.particleCount || 12));
  const op = clamp01(props.opacity * props.intensity * env);
  const items = [];
  for (let i = 0; i < count; i++) {
    const side = i % 4;
    const u = 0.05 + seededUnit(props.seed, i * 3) * 0.9;
    const v = 0.03 + seededUnit(props.seed, i * 3 + 1) * 0.1;
    const speed = 0.6 + seededUnit(props.seed, i * 3 + 2) * 0.7;
    let px = u;
    let py = v;
    if (side === 1) {
      px = u;
      py = 1 - v;
    } else if (side === 2) {
      px = v;
      py = u;
    } else if (side === 3) {
      px = 1 - v;
      py = u;
    }
    const cx = clamp01(px + 0.014 * Math.sin(t * 2.5 * Math.PI + i)) * width;
    const cy = clamp01(py - 0.04 * t * speed) * height;
    const rad = Math.max(2, Math.min(width, height) * (0.003 + seededUnit(props.seed, i * 7 + 5) * 0.0045));
    const a = op * (0.5 + 0.5 * Math.pow(Math.sin(Math.PI * t + i * 0.7), 2));
    items.push(<circle key={i} cx={cx} cy={cy} r={rad} fill={`rgba(255,248,232,${a.toFixed(3)})`} />);
  }
  return (
    <svg width={width} height={height} style={{ position: "absolute", inset: 0 }}>
      {items}
    </svg>
  );
};

const AudioEnvelopeAccent: React.FC<{ props: MotionLabOverlayProps; t: number; env: number }> = ({
  props,
  t,
  env,
}) => {
  const { width, height } = useVideoConfig();
  const frame = useCurrentFrame();
  const op = clamp01(props.opacity * props.intensity * env);
  const levels = props.audioEnvelope || [];

  const levelAt = (idx: number): number => {
    if (!levels.length) return 0.25;
    const clamped = Math.max(0, Math.min(levels.length - 1, idx));
    return clamp01(levels[clamped]);
  };

  if (props.effectId === "pulse_border") {
    const energy =
      (levelAt(frame - 2) + levelAt(frame - 1) + levelAt(frame) + levelAt(frame + 1) + levelAt(frame + 2)) / 5;
    const modOp = clamp01(op * (0.45 + 0.55 * energy));
    const pad = Math.max(6, Math.round(Math.min(width, height) * 0.018));
    const rad = Math.max(8, Math.round(Math.min(width, height) * 0.025));
    return (
      <svg width={width} height={height} style={{ position: "absolute", inset: 0 }}>
        <rect
          x={pad}
          y={pad}
          width={width - pad * 2}
          height={height - pad * 2}
          rx={rad}
          ry={rad}
          fill="none"
          stroke={`rgba(255,240,210,${modOp.toFixed(3)})`}
          strokeWidth={Math.max(2, props.strokePx || 4)}
        />
      </svg>
    );
  }

  // waveform_accent
  const bars = Math.min(24, Math.max(8, props.barCount || 16));
  const areaW = width * 0.42;
  const xStart = (width - areaW) / 2;
  const yBase = height * 0.92;
  const maxBarH = Math.max(12, height * 0.042);
  const step = areaW / bars;
  const barW = Math.max(2, step * 0.52);

  return (
    <svg width={width} height={height} style={{ position: "absolute", inset: 0 }}>
      {Array.from({ length: bars }).map((_, i) => {
        const offset = Math.round((i - bars / 2) * 1.2);
        const lvl = levelAt(frame + offset);
        const taper = Math.sin(((i + 0.5) * Math.PI) / bars);
        const bh = Math.max(3, maxBarH * (0.12 + 0.88 * lvl * taper));
        const bx = xStart + i * step;
        return (
          <rect
            key={i}
            x={bx}
            y={yBase - bh}
            width={barW}
            height={bh}
            rx={barW / 2}
            ry={barW / 2}
            fill={`rgba(245,248,255,${op.toFixed(3)})`}
          />
        );
      })}
    </svg>
  );
};

export const MotionLabOverlay: React.FC<MotionLabOverlayProps> = (props) => {
  const frame = useCurrentFrame();
  const { durationInFrames } = useVideoConfig();
  const t = clamp01(frame / Math.max(1, durationInFrames - 1));
  const env = smoothEnvelope(t, 0.18, 0.22);

  if (props.intensity <= 1e-6) {
    return <AbsoluteFill style={{ backgroundColor: "transparent" }} />;
  }

  return (
    <AbsoluteFill style={{ backgroundColor: "transparent" }}>
      {props.effectId === "edge_lines" && <EdgeAccent props={props} t={t} env={env} />}
      {props.effectId === "foreground_frame_accent" && (
        <ForegroundFrameAccent props={props} t={t} env={env} />
      )}
      {props.effectId === "ring_accent" && <RingAccent props={props} t={t} env={env} />}
      {props.effectId === "directional_shapes" && <DirectionalShapes props={props} t={t} env={env} />}
      {props.effectId === "ambient_particles" && <AmbientParticles props={props} t={t} env={env} />}
      {(props.effectId === "waveform_accent" || props.effectId === "pulse_border") && (
        <AudioEnvelopeAccent props={props} t={t} env={env} />
      )}
    </AbsoluteFill>
  );
};
