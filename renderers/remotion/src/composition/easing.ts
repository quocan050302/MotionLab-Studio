import { Easing, interpolate } from "remotion";
import type { AnchorPoint } from "./schema";

export function clamp01(v: number): number {
  return Math.max(0, Math.min(1, v));
}

export function easeInOutCubic(t: number): number {
  const u = clamp01(t);
  return u < 0.5 ? 4 * u * u * u : 1 - Math.pow(-2 * u + 2, 3) / 2;
}

/**
 * Smooth 0 -> 1 -> 0 envelope that starts and ends at EXACTLY 0.0 on frames 0 and total-1,
 * ensuring 100% seamless zero-snap boundaries when spliced into the canonical 30s timeline.
 */
export function sceneEnvelope(frame: number, totalFrames: number, entryFrames: number, exitFrames: number): number {
  const last = Math.max(1, totalFrames - 1);
  if (frame <= 0 || frame >= last) {
    return 0;
  }
  const inEnd = Math.max(2, Math.min(Math.floor(totalFrames * 0.42), entryFrames));
  const outStart = Math.min(last - 1, Math.max(inEnd, last - Math.max(2, exitFrames)));

  if (frame < inEnd) {
    return interpolate(frame, [0, inEnd], [0, 1], {
      extrapolateLeft: "clamp",
      extrapolateRight: "clamp",
      easing: Easing.out(Easing.cubic),
    });
  }
  if (frame > outStart) {
    return interpolate(frame, [outStart, last], [1, 0], {
      extrapolateLeft: "clamp",
      extrapolateRight: "clamp",
      easing: Easing.inOut(Easing.cubic),
    });
  }
  return 1;
}

export function sampleAnchorPath(path: AnchorPoint[] | undefined, u: number, fallback: AnchorPoint): AnchorPoint {
  if (!path || path.length === 0) {
    return fallback;
  }
  const t = clamp01(u);
  if (t <= path[0].t) return path[0];
  if (t >= path[path.length - 1].t) return path[path.length - 1];
  for (let i = 0; i < path.length - 1; i++) {
    const a = path[i];
    const b = path[i + 1];
    if (t >= a.t && t <= b.t) {
      const f = (t - a.t) / Math.max(1e-5, b.t - a.t);
      return {
        t,
        cx: a.cx + (b.cx - a.cx) * f,
        cy: a.cy + (b.cy - a.cy) * f,
        w: a.w + (b.w - a.w) * f,
        h: a.h + (b.h - a.h) * f,
        conf: a.conf + (b.conf - a.conf) * f,
      };
    }
  }
  return fallback;
}
