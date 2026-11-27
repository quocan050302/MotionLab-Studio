// Imported from GTKottman/remotion-transitions (MIT) — transitions/07-morph/shape/frame-morph/index.tsx
// Adapted as a timeline-preserving (non-shortening) continuous clipPath frame-morph bridge.
import React from "react";
import { AbsoluteFill, useVideoConfig } from "remotion";

const clamp01 = (v: number) => Math.max(0, Math.min(1, v));
const mix = (a: number, b: number, t: number) => a + (b - a) * t;
const seg = (t: number, a: number, b: number) => clamp01((t - a) / Math.max(1e-5, b - a));
const inOutCubic = (t: number) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);

const Morphing: React.FC<{ k: number; size: number; children: React.ReactNode }> = ({ k, size, children }) => {
  const { width: W, height: H } = useVideoConfig();
  const D = size * Math.min(W, H);
  const w = mix(W, D, k);
  const h = mix(H, D, k);
  const r = mix(0, D / 2, Math.pow(k, 0.6));
  const s = mix(1, (D * 1.15) / Math.min(W, H), k);
  return (
    <AbsoluteFill style={{ clipPath: `inset(${(H - h) / 2}px ${(W - w) / 2}px round ${r}px)` }}>
      <AbsoluteFill style={{ transform: `scale(${s.toFixed(4)})` }}>{children}</AbsoluteFill>
    </AbsoluteFill>
  );
};

export interface FrameMorphBridgeProps {
  progress: number;
  background?: string;
  size?: number;
  exiting: React.ReactNode;
  entering: React.ReactNode;
}

export const FrameMorphBridge: React.FC<FrameMorphBridgeProps> = ({
  progress,
  background = "#292822",
  size = 0.62,
  exiting,
  entering,
}) => {
  const p = clamp01(progress);
  return (
    <AbsoluteFill style={{ background }}>
      {p < 0.6 ? <Morphing k={inOutCubic(seg(p, 0, 0.45))} size={size}>{exiting}</Morphing> : null}
      {p >= 0.4 ? (
        <AbsoluteFill style={{ opacity: inOutCubic(seg(p, 0.4, 0.6)) }}>
          <Morphing k={1 - inOutCubic(seg(p, 0.55, 1))} size={size}>{entering}</Morphing>
        </AbsoluteFill>
      ) : null}
    </AbsoluteFill>
  );
};
