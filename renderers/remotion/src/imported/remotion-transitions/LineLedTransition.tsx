// Imported from GTKottman/remotion-transitions (MIT) — transitions/08-carry/line-led/line-led/index.tsx
// Adapted as a timeline-preserving (non-shortening) center-line draw + split aperture reveal bridge.
import React from "react";
import { AbsoluteFill, useVideoConfig } from "remotion";

const clamp01 = (v: number) => Math.max(0, Math.min(1, v));
const seg = (t: number, a: number, b: number) => clamp01((t - a) / Math.max(1e-5, b - a));
const inOutCubic = (t: number) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);

export interface LineLedBridgeProps {
  progress: number;
  color?: string;
  thickness?: number;
  orientation?: "horizontal" | "vertical";
  exiting: React.ReactNode;
  entering: React.ReactNode;
}

export const LineLedBridge: React.FC<LineLedBridgeProps> = ({
  progress,
  color = "#C6A46A",
  thickness = 5,
  orientation = "horizontal",
  exiting,
  entering,
}) => {
  const { width: W, height: H } = useVideoConfig();
  const p = clamp01(progress);
  const horiz = orientation === "horizontal";
  const span = horiz ? H : W;
  const draw = inOutCubic(seg(p, 0, 0.35));
  const open = inOutCubic(seg(p, 0.35, 1));
  const half = open * (span / 2 + thickness * 2);
  const c = span / 2;
  const band: React.CSSProperties = horiz
    ? { clipPath: `inset(${Math.max(0, c - half)}px 0 ${Math.max(0, c - half)}px 0)` }
    : { clipPath: `inset(0 ${Math.max(0, c - half)}px 0 ${Math.max(0, c - half)}px)` };
  const line = (pos: number, key: string) => (
    <div
      key={key}
      style={
        horiz
          ? { position: "absolute", left: 0, top: pos - thickness / 2, width: W * (open > 0 ? 1 : draw), height: thickness, background: color }
          : { position: "absolute", top: 0, left: pos - thickness / 2, height: H * (open > 0 ? 1 : draw), width: thickness, background: color }
      }
    />
  );
  return (
    <AbsoluteFill>
      {exiting}
      {open > 0 ? <AbsoluteFill style={band}>{entering}</AbsoluteFill> : null}
      {p > 0 && p < 1 ? line(c - half, "a") : null}
      {open > 0 && p < 1 ? line(c + half, "b") : null}
    </AbsoluteFill>
  );
};
