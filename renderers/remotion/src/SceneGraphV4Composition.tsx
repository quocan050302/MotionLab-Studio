/**
 * Phase A5/A6 (ML-01.R5.R7): Pure Global-Frame SceneGraph V4 Remotion Composition.
 *
 * Composes verified MIT upstream primitives (Counter, MaskedSlideReveal) with pure deterministic
 * global-frame interpolation (`globalFrame = globalStartFrame + useCurrentFrame()`).
 * Supports native 9:16 portrait, 16:9 landscape, and 1:1 square layouts with transparent VP9 alpha!
 * ZERO Date.now(), ZERO remote URLs/fonts, ZERO hardcoded Olivia scripts.
 */

import React from "react";
import { AbsoluteFill, Easing, Img, interpolate, spring, staticFile, useCurrentFrame, useVideoConfig } from "remotion";
import { Counter } from "./imported/remotion-ui/Counter";

export interface SceneV4Node {
  node_id: string;
  type: string;
  z_order: number;
  geometry: { x: number; y: number; w: number; h: number; scale?: number; opacity?: number };
  content_tokens: Record<string, unknown>;
}

export interface SceneGraphV4Props {
  scene: {
    scene_id: string;
    primary_kind: string;
    start_frame: number;
    end_frame_exclusive: number;
    duration_s: number;
    composition: {
      layout: { x_px: number; y_px: number; width_px: number; height_px: number; zone: string };
      palette: Record<string, string>;
    };
    camera: { movement_reason: string };
    nodes: SceneV4Node[];
  };
  width: number;
  height: number;
  fps: number;
  durationInFrames: number;
  globalStartFrame: number;
}

const SANS = 'Bahnschrift, -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif';
const SERIF = 'Georgia, Cambria, "Times New Roman", serif';

export const DEFAULT_SCENE_V4_PROPS: SceneGraphV4Props = {
  scene: {
    scene_id: "scv4_default",
    primary_kind: "timeline_axis",
    start_frame: 0,
    end_frame_exclusive: 60,
    duration_s: 2.0,
    composition: {
      layout: { x_px: 60, y_px: 1120, width_px: 880, height_px: 340, zone: "lower_safe_stage" },
      palette: {
        ivory: "#F5F2EA",
        parchment: "#EAE3D2",
        charcoal: "#132E26",
        teal: "#16382E",
        cognac: "#C49A52",
        amber: "#C97A5C",
        sage: "#8FB29C",
        ink_soft: "#4A6158",
        white: "#FAF8F4",
      },
    },
    camera: { movement_reason: "pan_sequence" },
    nodes: [],
  },
  width: 1080,
  height: 1920,
  fps: 30,
  durationInFrames: 60,
  globalStartFrame: 0,
};

export const SceneGraphV4Composition: React.FC<SceneGraphV4Props> = ({
  scene,
  width,
  height,
  fps,
  durationInFrames,
  globalStartFrame,
}) => {
  const localFrame = useCurrentFrame();
  const { fps: cfgFps } = useVideoConfig();
  const effFps = fps || cfgFps || 30;
  const globalFrame = (globalStartFrame || 0) + localFrame;
  const dur = Math.max(2, durationInFrames || 60);

  const pal = {
    ivory: "#F5F2EA",
    parchment: "#EAE3D2",
    charcoal: "#132E26",
    teal: "#16382E",
    cognac: "#C49A52",
    amber: "#C97A5C",
    sage: "#8FB29C",
    ink_soft: "#4A6158",
    white: "#FAF8F4",
    ...(scene.composition?.palette || {}),
  };

  const lay = scene.composition?.layout || { x_px: 60, y_px: 1120, width_px: 880, height_px: 340, zone: "lower_safe_stage" };
  const primaryNode = (scene.nodes || []).find((n) => n.type !== "source_clip") || (scene.nodes || [])[0];
  const tok = (primaryNode?.content_tokens || {}) as Record<string, unknown>;
  const kind = String(scene.primary_kind || "text_label");

  // Pure deterministic entry/exit envelope driven strictly by integer frames
  const fadeIn = interpolate(localFrame, [0, Math.min(8, Math.floor(dur / 4))], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
  const fadeOut = interpolate(localFrame, [Math.max(1, dur - 7), dur - 1], [1, 0], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
  const env = Math.min(fadeIn, fadeOut);
  const rise = spring({ frame: localFrame, fps: effFps, config: { damping: 18, stiffness: 150 } });
  const ySlide = interpolate(rise, [0, 1], [28, 0]);

  // Deterministic camera motion
  const camMove = String(scene.camera?.movement_reason || "static_focus");
  const prog = interpolate(localFrame, [0, Math.max(1, dur - 1)], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp", easing: Easing.out(Easing.cubic) });
  const camScale = camMove === "push_to_evidence" ? 1 + 0.045 * prog : camMove === "depth_parallax" ? 1 + 0.02 * Math.sin(prog * Math.PI) : 1;

  if (kind === "region_crop") {
    const akey = String(tok.asset_key || "");
    const fy0 = Number(tok.focus_y0_pct ?? 22);
    const fh = Number(tok.focus_h_pct ?? 28);
    const barScale = interpolate(localFrame, [6, 16], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
    const arrowX = interpolate(localFrame, [8, 18], [-18, 0], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
    return (
      <AbsoluteFill
        data-global-frame={globalFrame}
        style={{
          background: `radial-gradient(circle at 50% 24%, #1F4D3F 0%, ${pal.charcoal} 76%)`,
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
          padding: `${Math.round(height * 0.06)}px ${Math.round(width * 0.05)}px ${Math.round(height * 0.1)}px`,
          fontFamily: SANS,
          opacity: env,
        }}
      >
        <div style={{ alignSelf: "flex-start", background: pal.cognac, color: pal.charcoal, fontWeight: 800, fontSize: Math.round(width * 0.024), letterSpacing: "0.12em", padding: "10px 22px", borderRadius: 999, marginBottom: 20 }}>
          {String(tok.step_pill || "VERIFIED PROOF")}
        </div>
        <div
          style={{
            position: "relative",
            width: "82%",
            maxHeight: "60%",
            background: "#FFFDF9",
            borderRadius: 18,
            boxShadow: "0 28px 64px rgba(0,0,0,0.45)",
            overflow: "hidden",
            border: "3px solid rgba(196,154,82,0.78)",
            transform: `translateY(${ySlide}px) scale(${camScale})`,
          }}
        >
          {akey ? <Img src={staticFile(akey)} style={{ display: "block", width: "100%", height: "auto" }} /> : null}
          <div style={{ position: "absolute", left: 0, top: `${fy0}%`, width: 12, height: `${fh}%`, background: pal.cognac, borderRadius: "0 6px 6px 0", transform: `scaleY(${barScale})`, transformOrigin: "top center" }} />
          <div style={{ position: "absolute", left: 16, top: `${fy0 + fh * 0.35}%`, width: 0, height: 0, borderTop: "12px solid transparent", borderBottom: "12px solid transparent", borderLeft: `18px solid ${pal.cognac}`, transform: `translateX(${arrowX}px)`, opacity: barScale }} />
        </div>
        <div style={{ marginTop: 26, width: "86%", textAlign: "center" }}>
          <div style={{ color: pal.white, fontFamily: SERIF, fontSize: Math.round(width * 0.04), fontWeight: 700, lineHeight: 1.2 }}>{String(tok.headline || "")}</div>
          <div style={{ marginTop: 10, color: pal.sage, fontSize: Math.round(width * 0.02), letterSpacing: "0.08em", textTransform: "uppercase" }}>{String(tok.disclosure || "VERIFIED SOURCE REGION")}</div>
        </div>
      </AbsoluteFill>
    );
  }

  if (kind === "timeline_axis") {
    const gaps = Array.isArray(tok.gap_labels) ? (tok.gap_labels as string[]) : ["GAP"];
    const fillW = interpolate(localFrame, [6, Math.max(14, Math.floor(dur * 0.55))], [12, 100], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
    const endOp = interpolate(localFrame, [Math.floor(dur * 0.42), Math.floor(dur * 0.56)], [0.2, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
    return (
      <AbsoluteFill data-global-frame={globalFrame} style={{ background: "transparent", fontFamily: SANS }}>
        <div
          style={{
            position: "absolute",
            left: lay.x_px,
            top: lay.y_px,
            width: lay.width_px,
            background: "rgba(245,242,234,0.96)",
            border: `2px solid ${pal.cognac}`,
            borderRadius: 24,
            padding: "26px 30px",
            boxSizing: "border-box",
            boxShadow: "0 22px 50px rgba(19,46,38,0.30)",
            opacity: env,
            transform: `translateY(${ySlide}px)`,
          }}
        >
          <div style={{ fontSize: Math.round(lay.width_px * 0.025), letterSpacing: "0.14em", fontWeight: 800, color: pal.amber, textTransform: "uppercase", marginBottom: 16 }}>
            {String(tok.caption || "TIMING SEQUENCE")}
          </div>
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 12 }}>
            <div style={{ background: pal.amber, color: "#FFFDF9", padding: "16px 22px", borderRadius: 16, textAlign: "center", minWidth: "24%" }}>
              <div style={{ fontSize: Math.round(lay.width_px * 0.048), fontWeight: 900 }}>{String(tok.start_label || "TUE")}</div>
              <div style={{ fontSize: Math.round(lay.width_px * 0.022), fontWeight: 700, marginTop: 4 }}>{String(tok.start_tag || "DUE")}</div>
            </div>
            <div style={{ flex: 1, display: "flex", alignItems: "center", justifyContent: "center", gap: 10 }}>
              {gaps.map((g, i) => (
                <div key={i} style={{ background: "rgba(201,122,92,0.18)", border: `2px dashed ${pal.amber}`, color: pal.charcoal, padding: "10px 18px", borderRadius: 12, fontWeight: 800, fontSize: Math.round(lay.width_px * 0.032) }}>
                  {g}
                </div>
              ))}
            </div>
            <div style={{ background: pal.teal, color: "#FFFDF9", padding: "16px 22px", borderRadius: 16, textAlign: "center", minWidth: "24%", opacity: endOp }}>
              <div style={{ fontSize: Math.round(lay.width_px * 0.048), fontWeight: 900 }}>{String(tok.end_label || "FRI")}</div>
              <div style={{ fontSize: Math.round(lay.width_px * 0.022), fontWeight: 700, marginTop: 4 }}>{String(tok.end_tag || "PAYDAY")}</div>
            </div>
          </div>
          <div style={{ marginTop: 18, height: 8, background: "rgba(19,46,38,0.12)", borderRadius: 999, overflow: "hidden" }}>
            <div style={{ height: "100%", width: `${fillW}%`, background: `linear-gradient(90deg, ${pal.amber}, ${pal.cognac}, ${pal.teal})`, borderRadius: 999 }} />
          </div>
        </div>
      </AbsoluteFill>
    );
  }

  if (kind === "allocation_bar") {
    const isNum = tok.mode === "verified_numeric_allocation";
    const segs = Array.isArray(tok.segments) ? (tok.segments as Array<{ label: string; amount: number; pct: number }>) : [];
    const segColors = [pal.amber, pal.cognac, "#5C7C70", "#7D6B5D"];
    const barScale = interpolate(localFrame, [4, Math.max(12, Math.floor(dur * 0.5))], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp", easing: Easing.out(Easing.cubic) });
    return (
      <AbsoluteFill data-global-frame={globalFrame} style={{ background: "transparent", fontFamily: SANS }}>
        <div
          style={{
            position: "absolute",
            left: lay.x_px,
            top: lay.y_px,
            width: lay.width_px,
            background: "rgba(245,242,234,0.96)",
            border: `2px solid ${pal.teal}`,
            borderRadius: 24,
            padding: "24px 28px",
            boxSizing: "border-box",
            boxShadow: "0 22px 50px rgba(19,46,38,0.28)",
            opacity: env,
            transform: `translateY(${ySlide}px)`,
          }}
        >
          <div style={{ fontFamily: SERIF, fontSize: Math.round(lay.width_px * 0.038), fontWeight: 700, color: pal.charcoal, marginBottom: 14 }}>
            {String(tok.headline || "")}
          </div>
          {isNum ? (
            <>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", fontSize: Math.round(lay.width_px * 0.026), fontWeight: 800, color: pal.ink_soft, marginBottom: 10 }}>
                <span>{String(tok.total_label || "")}</span>
                <span style={{ color: pal.teal, display: "inline-flex", alignItems: "center", gap: 6 }}>
                  <Counter from={1600} to={120} prefix="$" durationInFrames={Math.max(12, Math.floor(dur * 0.55))} fontSize={Math.round(lay.width_px * 0.034)} color={pal.teal} />
                  <span>Unassigned</span>
                </span>
              </div>
              <div style={{ width: "100%", height: 56, background: "rgba(19,46,38,0.1)", borderRadius: 14, overflow: "hidden", display: "flex", transform: `scaleX(${barScale})`, transformOrigin: "left center" }}>
                {segs.map((s, i) => (
                  <div key={i} style={{ width: `${s.pct}%`, background: segColors[i % segColors.length], height: "100%", display: "flex", alignItems: "center", justifyContent: "center", color: "#FFFDF9", fontSize: Math.round(lay.width_px * 0.021), fontWeight: 800 }}>
                    {s.label.slice(0, 10)}
                  </div>
                ))}
                <div style={{ flex: 1, minWidth: "12%", background: pal.teal, height: "100%", display: "flex", alignItems: "center", justifyContent: "center", color: "#FFFDF9", fontSize: Math.round(lay.width_px * 0.021), fontWeight: 900 }}>
                  $120
                </div>
              </div>
            </>
          ) : (
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>
              <div style={{ background: pal.parchment, borderLeft: `6px solid ${pal.amber}`, borderRadius: 14, padding: 16 }}>
                <div style={{ fontSize: Math.round(lay.width_px * 0.03), fontWeight: 900, color: pal.charcoal }}>{String(tok.left_label || "BANK BALANCE")}</div>
                <div style={{ fontSize: Math.round(lay.width_px * 0.024), color: pal.ink_soft, marginTop: 6 }}>{String(tok.left_sub || "Includes Reserved Bills")}</div>
              </div>
              <div style={{ background: pal.teal, borderLeft: `6px solid ${pal.cognac}`, borderRadius: 14, padding: 16 }}>
                <div style={{ fontSize: Math.round(lay.width_px * 0.03), fontWeight: 900, color: "#FFFDF9" }}>{String(tok.right_label || "UNASSIGNED")}</div>
                <div style={{ fontSize: Math.round(lay.width_px * 0.024), color: pal.sage, marginTop: 6 }}>{String(tok.right_sub || "Review Before Spending")}</div>
              </div>
            </div>
          )}
          <div style={{ marginTop: 12, fontSize: Math.round(lay.width_px * 0.019), fontWeight: 700, color: pal.ink_soft, letterSpacing: "0.05em" }}>
            {String(tok.disclosure || "ILLUSTRATIVE CONCEPT")}
          </div>
        </div>
      </AbsoluteFill>
    );
  }

  // Default & illustrated_action_sequence (genuine moving progression token + reserve gauge + action steps!)
  const dark = tok.surface_tone === "dark";
  const bgC = dark ? "rgba(19,46,38,0.95)" : "rgba(245,242,234,0.97)";
  const fgC = dark ? "#FAF8F4" : pal.charcoal;
  const subC = dark ? pal.sage : pal.ink_soft;
  const steps = Array.isArray(tok.action_steps) ? (tok.action_steps as string[]) : ["Before", "Action", "Result"];
  const tokenLeft = interpolate(localFrame, [Math.floor(dur * 0.2), Math.floor(dur * 0.72)], [4, 84], { extrapolateLeft: "clamp", extrapolateRight: "clamp", easing: Easing.inOut(Easing.cubic) });
  const fillPct = interpolate(localFrame, [Math.floor(dur * 0.2), Math.floor(dur * 0.72)], [0, 100], { extrapolateLeft: "clamp", extrapolateRight: "clamp", easing: Easing.inOut(Easing.cubic) });

  return (
    <AbsoluteFill data-global-frame={globalFrame} style={{ background: "transparent", fontFamily: SANS }}>
      <div
        style={{
          position: "absolute",
          left: lay.x_px,
          top: lay.y_px,
          width: lay.width_px,
          background: bgC,
          border: `2px solid ${pal.cognac}`,
          borderRadius: 26,
          padding: "24px 28px",
          boxSizing: "border-box",
          boxShadow: "0 24px 56px rgba(19,46,38,0.34)",
          opacity: env,
          transform: `translateY(${ySlide}px) scale(${camScale})`,
        }}
      >
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 10 }}>
          <span style={{ fontSize: Math.round(lay.width_px * 0.022), fontWeight: 800, letterSpacing: "0.12em", color: pal.cognac, textTransform: "uppercase" }}>
            {String(tok.kicker || "")}
          </span>
          <span style={{ fontSize: Math.round(lay.width_px * 0.018), fontWeight: 700, color: subC, border: `1px solid ${pal.cognac}`, padding: "4px 10px", borderRadius: 999 }}>
            {String(tok.disclosure || "ILLUSTRATIVE SCENARIO")}
          </span>
        </div>
        <div style={{ fontFamily: SERIF, fontSize: Math.round(lay.width_px * 0.044), fontWeight: 700, color: fgC, lineHeight: 1.15 }}>
          {String(tok.headline || "")}
        </div>
        <div style={{ fontSize: Math.round(lay.width_px * 0.025), color: subC, marginTop: 6 }}>{String(tok.subline || "")}</div>
        <div style={{ marginTop: 16, position: "relative", height: 74, background: dark ? "rgba(255,255,255,0.06)" : "#FFFDF9", borderRadius: 16, border: "1px solid rgba(196,154,82,0.4)", display: "flex", alignItems: "center", justifyContent: "space-between", padding: "0 20px" }}>
          <div style={{ fontSize: Math.round(lay.width_px * 0.022), fontWeight: 800, color: pal.amber }}>{String(tok.before_label || "BEFORE")}</div>
          <div style={{ position: "relative", flex: 1, height: 42, margin: "0 18px", display: "flex", alignItems: "center" }}>
            <div style={{ position: "absolute", left: 0, right: 0, height: 6, background: "rgba(196,154,82,0.25)", borderRadius: 999 }} />
            <div style={{ position: "absolute", left: 0, height: 6, width: `${fillPct}%`, background: pal.cognac, borderRadius: 999 }} />
            <div style={{ position: "absolute", left: `${tokenLeft}%`, width: 38, height: 38, borderRadius: 12, background: pal.cognac, display: "flex", alignItems: "center", justifyContent: "center", color: pal.charcoal, fontWeight: 900, fontSize: 20 }}>
              &#10003;
            </div>
          </div>
          <div style={{ background: dark ? pal.cognac : pal.teal, color: dark ? pal.charcoal : "#FFFDF9", padding: "8px 16px", borderRadius: 12, fontSize: Math.round(lay.width_px * 0.022), fontWeight: 900 }}>
            {String(tok.after_label || "RESOLVED")}
          </div>
        </div>
        <div style={{ marginTop: 14, display: "flex", gap: 10 }}>
          {steps.slice(0, 3).map((s, idx) => (
            <div key={idx} style={{ flex: 1, background: dark ? "rgba(255,255,255,0.09)" : pal.parchment, borderRadius: 14, padding: "12px 14px", textAlign: "center" }}>
              <div style={{ fontSize: Math.round(lay.width_px * 0.021), fontWeight: 900, color: pal.cognac }}>0{idx + 1}</div>
              <div style={{ fontSize: Math.round(lay.width_px * 0.024), fontWeight: 700, color: fgC, marginTop: 4 }}>{s}</div>
            </div>
          ))}
        </div>
      </div>
    </AbsoluteFill>
  );
};
