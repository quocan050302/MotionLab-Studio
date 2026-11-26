import React from "react";
import { AbsoluteFill, Easing, interpolate, spring, useCurrentFrame, useVideoConfig } from "remotion";

export interface OverlayProps {
  effectId: string;
  width: number;
  height: number;
  fps: number;
  durationInFrames: number;
  parameters: Record<string, unknown>;
  audioEnvelope?: number[];
}

function bellOpacity(frame: number, total: number, maxOpacity: number): number {
  const ramp = Math.max(2, Math.round(total * 0.22));
  const fadeIn = interpolate(frame, [0, ramp], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.inOut(Easing.cubic),
  });
  const fadeOut = interpolate(frame, [Math.max(ramp, total - ramp), Math.max(ramp + 1, total - 1)], [1, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.inOut(Easing.cubic),
  });
  return Math.max(0, Math.min(1, fadeIn * fadeOut)) * maxOpacity;
}

const EdgeAccent: React.FC<{ width: number; height: number; accent: string; marginFrac: number; opacity: number; progress: number }> = ({
  width,
  height,
  accent,
  marginFrac,
  opacity,
  progress,
}) => {
  const mx = Math.max(8, Math.round(width * marginFrac));
  const my = Math.max(8, Math.round(height * marginFrac));
  const segW = Math.round((width - 2 * mx) * 0.30 * progress);
  const tick = Math.max(8, Math.round(Math.min(width, height) * 0.028 * progress));
  const strokeW = Math.max(2, Math.round(Math.min(width, height) * 0.004));

  return (
    <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} style={{ opacity }}>
      <line x1={mx} y1={my} x2={mx + segW} y2={my} stroke={accent} strokeWidth={strokeW} strokeLinecap="round" />
      <line x1={mx} y1={my} x2={mx} y2={my + tick} stroke={accent} strokeWidth={strokeW} strokeLinecap="round" />
      <line x1={width - mx - segW} y1={height - my} x2={width - mx} y2={height - my} stroke={accent} strokeWidth={strokeW} strokeLinecap="round" />
      <line x1={width - mx} y1={height - my - tick} x2={width - mx} y2={height - my} stroke={accent} strokeWidth={strokeW} strokeLinecap="round" />
    </svg>
  );
};

const RingAccent: React.FC<{ width: number; height: number; accent: string; cxFrac: number; cyFrac: number; radFrac: number; opacity: number; springVal: number }> = ({
  width,
  height,
  accent,
  cxFrac,
  cyFrac,
  radFrac,
  opacity,
  springVal,
}) => {
  const cx = Math.round(width * cxFrac);
  const cy = Math.round(height * cyFrac);
  const baseR = Math.max(14, Math.round(Math.min(width, height) * radFrac));
  const r1 = Math.max(4, Math.round(baseR * (0.45 + 0.65 * springVal)));
  const r2 = Math.max(3, Math.round(baseR * (0.25 + 0.45 * springVal)));
  const strokeW = Math.max(2, Math.round(Math.min(width, height) * 0.004));
  const circ = 2 * Math.PI * r1;

  return (
    <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} style={{ opacity }}>
      <circle
        cx={cx}
        cy={cy}
        r={r1}
        fill="none"
        stroke={accent}
        strokeWidth={strokeW}
        strokeDasharray={circ}
        strokeDashoffset={circ * (1 - Math.min(1, springVal * 0.9))}
        strokeLinecap="round"
      />
      <circle cx={cx} cy={cy} r={r2} fill="none" stroke={accent} strokeWidth={Math.max(1, strokeW - 1)} opacity={0.7} />
    </svg>
  );
};

const AudioEnvelopeAccent: React.FC<{ width: number; height: number; accent: string; bars: number; opacity: number; frame: number; envelope: number[] }> = ({
  width,
  height,
  accent,
  bars,
  opacity,
  frame,
  envelope,
}) => {
  const totalW = Math.round(width * 0.44);
  const xStart = Math.round((width - totalW) / 2);
  const baseY = Math.round(height * 0.935);
  const maxBarH = Math.max(10, Math.round(height * 0.04));
  const stepX = Math.max(4, Math.floor(totalW / Math.max(1, bars)));
  const barW = Math.max(2, Math.round(stepX * 0.54));
  const half = Math.floor(bars / 2);

  return (
    <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} style={{ opacity }}>
      {Array.from({ length: bars }).map((_, b) => {
        const idx = Math.max(0, Math.min(Math.max(0, envelope.length - 1), frame + (b - half)));
        const lvl = envelope.length > 0 ? Number(envelope[idx] || 0) : 0.35;
        const taper = Math.pow(Math.sin((Math.PI * (b + 0.5)) / bars), 0.7);
        const bh = Math.max(3, Math.round((0.16 + 0.84 * lvl) * maxBarH * taper));
        const bx = xStart + b * stepX;
        return <rect key={b} x={bx} y={baseY - bh / 2} width={barW} height={bh} rx={barW / 2} fill={accent} />;
      })}
    </svg>
  );
};

export const MotionLabOverlay: React.FC<OverlayProps> = ({
  effectId,
  parameters,
  audioEnvelope = [],
}) => {
  const frame = useCurrentFrame();
  const { width, height, fps, durationInFrames } = useVideoConfig();

  const intensity = Number(parameters.intensity ?? 0.85);
  const baseOpacity = Number(parameters.opacity ?? 0.62) * intensity;
  const accent = String(parameters.accent_hex ?? "#7CE0FF");
  const opacity = bellOpacity(frame, durationInFrames, baseOpacity);

  const sVal = spring({
    frame,
    fps,
    config: { damping: 16, stiffness: 120, mass: 0.9 },
  });

  return (
    <AbsoluteFill style={{ backgroundColor: "transparent" }}>
      {effectId === "G02" ? (
        <RingAccent
          width={width}
          height={height}
          accent={accent}
          cxFrac={Number(parameters.cx ?? 0.84)}
          cyFrac={Number(parameters.cy ?? 0.16)}
          radFrac={Number(parameters.radius_frac ?? 0.055)}
          opacity={opacity}
          springVal={sVal}
        />
      ) : effectId === "G05" ? (
        <AudioEnvelopeAccent
          width={width}
          height={height}
          accent={accent}
          bars={Number(parameters.bars ?? 20)}
          opacity={opacity}
          frame={frame}
          envelope={audioEnvelope}
        />
      ) : (
        <EdgeAccent
          width={width}
          height={height}
          accent={accent}
          marginFrac={Number(parameters.margin_frac ?? 0.04)}
          opacity={opacity}
          progress={Math.min(1, sVal)}
        />
      )}
    </AbsoluteFill>
  );
};
