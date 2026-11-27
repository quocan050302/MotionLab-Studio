/**
 * Phase R2-A2/A3 (ML-01.R5.R2): Deterministic Continuous Story World Runtime & Shared-Element Bridges.
 *
 * Mirrors `motionlab/composition/continuous_timeline.py::evaluate_shared_state` identically so
 * Python reference coordinates and Remotion rendered frames match within <= 2px across every boundary.
 * Also provides physically grounded `PaperFoldExitBridge` & `EditorialSheetSweepInBridge` to eliminate
 * jarring hard cuts between warm-ivory cutaways and live kitchen footage (fixes D06!).
 */

import React from "react";
import { AbsoluteFill, Easing, interpolate } from "remotion";

export interface ElementKeyframeTS {
  global_frame: number;
  x: number;
  y: number;
  w: number;
  h: number;
  scale?: number;
  rotation_deg?: number;
  opacity?: number;
  z_order?: number;
  easing?: string;
}

export interface SharedElementTrackTS {
  element_id: string;
  visual_role: string;
  logical_state_id: string;
  continuity_group: string;
  start_frame: number;
  end_frame_exclusive: number;
  color: string;
  keyframes: ElementKeyframeTS[];
}

export interface EvaluatedElementState {
  x: number;
  y: number;
  w: number;
  h: number;
  scale: number;
  rotation_deg: number;
  opacity: number;
  z_order: number;
}

function easeVal(u: number, easing?: string): number {
  const t = Math.max(0, Math.min(1, u));
  if (easing === "linear") return t;
  if (easing === "ease_in_out_cubic") {
    return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
  }
  if (easing === "smoothstep") {
    return t * t * (3 - 2 * t);
  }
  return 1 - Math.pow(1 - t, 3);
}

export function evaluateSharedStateTS(
  track: SharedElementTrackTS,
  globalFrame: number
): EvaluatedElementState {
  const kfs = [...(track.keyframes || [])].sort((a, b) => a.global_frame - b.global_frame);
  if (kfs.length === 0) {
    return { x: 0.067, y: 0.086, w: 0.788, h: 0.18, scale: 1, rotation_deg: 0, opacity: 0, z_order: 25 };
  }
  if (globalFrame <= kfs[0].global_frame) {
    const k = kfs[0];
    return {
      x: k.x,
      y: k.y,
      w: k.w,
      h: k.h,
      scale: k.scale ?? 1,
      rotation_deg: k.rotation_deg ?? 0,
      opacity: k.opacity ?? 1,
      z_order: k.z_order ?? 25,
    };
  }
  const last = kfs[kfs.length - 1];
  if (globalFrame >= last.global_frame) {
    return {
      x: last.x,
      y: last.y,
      w: last.w,
      h: last.h,
      scale: last.scale ?? 1,
      rotation_deg: last.rotation_deg ?? 0,
      opacity: last.opacity ?? 1,
      z_order: last.z_order ?? 25,
    };
  }
  for (let i = 0; i < kfs.length - 1; i++) {
    const a = kfs[i];
    const b = kfs[i + 1];
    if (a.global_frame <= globalFrame && globalFrame <= b.global_frame) {
      const u =
        b.global_frame === a.global_frame
          ? 1
          : easeVal((globalFrame - a.global_frame) / (b.global_frame - a.global_frame), b.easing);
      const lerp = (va: number, vb: number) => va + (vb - va) * u;
      return {
        x: lerp(a.x, b.x),
        y: lerp(a.y, b.y),
        w: lerp(a.w, b.w),
        h: lerp(a.h, b.h),
        scale: lerp(a.scale ?? 1, b.scale ?? 1),
        rotation_deg: lerp(a.rotation_deg ?? 0, b.rotation_deg ?? 0),
        opacity: lerp(a.opacity ?? 1, b.opacity ?? 1),
        z_order: a.z_order ?? 25,
      };
    }
  }
  return {
    x: last.x,
    y: last.y,
    w: last.w,
    h: last.h,
    scale: last.scale ?? 1,
    rotation_deg: last.rotation_deg ?? 0,
    opacity: last.opacity ?? 1,
    z_order: last.z_order ?? 25,
  };
}

export interface EvaluatedMotionStateTS extends EvaluatedElementState {
  vx: number;
  vy: number;
  ax: number;
  ay: number;
  speed: number;
  pivot_x: number;
  pivot_y: number;
  depth_z: number;
  flex_deg: number;
  motion_model: "analytic_physically_inspired";
}

export function evaluateMotionStateTS(
  track: SharedElementTrackTS,
  globalFrame: number,
  fps = 24
): EvaluatedMotionStateTS {
  const cur = evaluateSharedStateTS(track, globalFrame);
  const prev = evaluateSharedStateTS(track, globalFrame - 1);
  const next = evaluateSharedStateTS(track, globalFrame + 1);
  const dt = 1 / Math.max(1, fps);
  const vx = (next.x - prev.x) / (2 * dt);
  const vy = (next.y - prev.y) / (2 * dt);
  const ax = (next.x - 2 * cur.x + prev.x) / (dt * dt);
  const ay = (next.y - 2 * cur.y + prev.y) / (dt * dt);
  const speed = Math.hypot(vx, vy);
  const span = Math.max(1, track.end_frame_exclusive - track.start_frame);
  const u = Math.max(0, Math.min(1, (globalFrame - track.start_frame) / span));
  const env = Math.sin(u * Math.PI);
  return {
    ...cur,
    vx,
    vy,
    ax,
    ay,
    speed,
    pivot_x: cur.x + cur.w * 0.5,
    pivot_y: cur.y + cur.h * 0.5,
    depth_z: 0.15 + 0.55 * env,
    flex_deg: 4.2 * Math.sin(u * Math.PI * 2.4) * env,
    motion_model: "analytic_physically_inspired",
  };
}

/**
 * Outgoing tactile paper-sheet fold & unveil bridge revealing the live speaker underneath
 * at the exact source frame clock — eliminates jarring cutaway-to-live hard cuts (D06!).
 */
export const PaperFoldExitBridge: React.FC<{
  progress: number; // 0 (full cutaway) -> 1 (full live plate)
  accentColor: string;
  liveUnderneath: React.ReactNode;
  cutawayContent: React.ReactNode;
}> = ({ progress, accentColor, liveUnderneath, cutawayContent }) => {
  const p = Math.max(0, Math.min(1, progress));
  const eased = Easing.inOut(Easing.cubic)(p);
  const clipTopPct = eased * 104;
  const translateY = -eased * 44;
  const scale = 1 - eased * 0.035;
  const edgeOpacity = interpolate(p, [0, 0.12, 0.85, 1], [0, 1, 1, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const liveScale = 1.02 - 0.02 * eased;

  return (
    <AbsoluteFill style={{ overflow: "hidden" }}>
      <AbsoluteFill style={{ transform: `scale(${liveScale.toFixed(4)})` }}>
        {liveUnderneath}
      </AbsoluteFill>
      <AbsoluteFill
        style={{
          clipPath: `inset(0% 0% ${clipTopPct.toFixed(2)}% 0%)`,
          transform: `translateY(${translateY.toFixed(1)}px) scale(${scale.toFixed(4)})`,
          transformOrigin: "50% 0%",
          filter: `drop-shadow(0 18px 28px rgba(24,20,16,${(0.32 * edgeOpacity).toFixed(3)}))`,
        }}
      >
        {cutawayContent}
      </AbsoluteFill>
      {/* Crisp antique-gold architectural paper edge gliding cleanly along the fold seam */}
      <div
        style={{
          position: "absolute",
          left: 0,
          right: 0,
          top: `${(100 - clipTopPct).toFixed(2)}%`,
          height: 5,
          backgroundColor: accentColor,
          opacity: edgeOpacity,
          boxShadow: "0 6px 18px rgba(20,16,12,0.34)",
          pointerEvents: "none",
        }}
      />
    </AbsoluteFill>
  );
};

/**
 * Incoming luminous warm-ivory architectural sheet sweep replacing jarring dark/white circular iris holes (D05/D06).
 */
export const EditorialSheetSweepInBridge: React.FC<{
  progress: number; // 0 (live plate) -> 1 (entering scene)
  accentColor: string;
  exiting: React.ReactNode;
  entering: React.ReactNode;
}> = ({ progress, accentColor, exiting, entering }) => {
  const p = Math.max(0, Math.min(1, progress));
  const eased = Easing.out(Easing.cubic)(p);
  const remainTop = (1 - eased) * 100;
  const edgeOpacity = interpolate(p, [0, 0.12, 0.88, 1], [0, 1, 1, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  return (
    <AbsoluteFill style={{ overflow: "hidden" }}>
      <AbsoluteFill style={{ transform: `scale(${(1 + 0.018 * eased).toFixed(4)})` }}>
        {exiting}
      </AbsoluteFill>
      <AbsoluteFill
        style={{
          clipPath: `inset(${remainTop.toFixed(2)}% 0% 0% 0%)`,
          filter: `drop-shadow(0 -14px 26px rgba(24,20,16,${(0.28 * edgeOpacity).toFixed(3)}))`,
        }}
      >
        {entering}
      </AbsoluteFill>
      <div
        style={{
          position: "absolute",
          left: 0,
          right: 0,
          top: `${remainTop.toFixed(2)}%`,
          height: 5,
          backgroundColor: accentColor,
          opacity: edgeOpacity,
          boxShadow: "0 -4px 16px rgba(20,16,12,0.28)",
          pointerEvents: "none",
        }}
      />
    </AbsoluteFill>
  );
};
