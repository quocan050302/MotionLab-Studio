import React from "react";
import { AbsoluteFill, OffthreadVideo, spring, staticFile, useCurrentFrame, useVideoConfig } from "remotion";
import { easeInOutCubic, sampleAnchorPath, sceneEnvelope } from "./easing";
import type { CreativeSceneProps } from "./schema";

const DEFAULT_PAL = {
  primary: "#C8A26A",
  secondary: "#3B312A",
  tertiary: "#F7F1E6",
  sage: "#6E8B74",
};

function getPal(params: Record<string, unknown>) {
  const pal = (params.palette as Record<string, string> | undefined) || {};
  return {
    primary: pal.primary_hex || DEFAULT_PAL.primary,
    secondary: pal.secondary_hex || DEFAULT_PAL.secondary,
    tertiary: pal.tertiary_hex || DEFAULT_PAL.tertiary,
    sage: pal.accent_sage || DEFAULT_PAL.sage,
  };
}

/**
 * VFX01 SubjectDepthSetExtension & VFX04 TrackedActionFlow & VFX07 LayerPassOcclusionWipe
 * True 3-plane depth stage: softened real background -> HyperFrames semantic/flow/wipe graphics -> MODNet foreground occlusion.
 */
const VfxDepthOcclusionStage: React.FC<CreativeSceneProps> = ({
  sceneKind,
  sourceFile,
  fgAlphaFile,
  hfGraphicFile,
  entryFrames,
  exitFrames,
  parameters,
  excludeLayerRole,
}) => {
  const frame = useCurrentFrame();
  const { durationInFrames } = useVideoConfig();
  const u = frame / Math.max(1, durationInFrames - 1);
  const env = sceneEnvelope(frame, durationInFrames, entryFrames, exitFrames);

  const bgBlur = Math.min(6.0, Number(parameters.bg_blur_px ?? 5.2)) * env;
  const bgDim = 1 - (1 - Number(parameters.bg_dim ?? 0.85)) * env;
  const bgScale = 1 + (Number(parameters.bg_scale ?? 1.018) - 1) * env;
  const fgLift = 1 + (Number(parameters.fg_lift ?? 1.030) - 1) * env;
  const warmShift = sceneKind === "vfx_occlusion_wipe" ? Math.sin(u * Math.PI) * 8 * env : 0;

  const showBgGraphics = Boolean(hfGraphicFile) && excludeLayerRole !== "background_graphics";
  const showFgMatte = Boolean(fgAlphaFile) && excludeLayerRole !== "foreground_subject_matte";

  return (
    <AbsoluteFill style={{ backgroundColor: "#14110E", overflow: "hidden" }}>
      <AbsoluteFill
        style={{
          transform: `scale(${bgScale.toFixed(4)})`,
          transformOrigin: "50% 44%",
          filter:
            bgBlur > 0.1
              ? `blur(${bgBlur.toFixed(2)}px) brightness(${bgDim.toFixed(3)}) sepia(${(warmShift * 0.015).toFixed(3)})`
              : "none",
        }}
      >
        <OffthreadVideo src={staticFile(sourceFile)} muted style={{ width: "100%", height: "100%", objectFit: "cover" }} />
      </AbsoluteFill>

      {/* Subtle warm editorial depth vignette on background plane only */}
      <AbsoluteFill
        style={{
          background: `radial-gradient(ellipse at 50% 38%, rgba(20,16,12,0) 36%, rgba(20,15,10,${(0.28 * env).toFixed(3)}) 100%)`,
          pointerEvents: "none",
        }}
      />

      {showBgGraphics && hfGraphicFile && (
        <AbsoluteFill style={{ opacity: Math.min(1, env * 1.15) }}>
          <OffthreadVideo src={staticFile(hfGraphicFile)} transparent muted style={{ width: "100%", height: "100%", objectFit: "cover" }} />
        </AbsoluteFill>
      )}

      {showFgMatte && fgAlphaFile && (
        <AbsoluteFill
          style={{
            transform: `scale(${fgLift.toFixed(4)})`,
            transformOrigin: "50% 60%",
            filter: env > 0.05 ? `drop-shadow(0px 9px 18px rgba(22, 16, 11, ${(0.30 * env).toFixed(3)}))` : "none",
          }}
        >
          <OffthreadVideo src={staticFile(fgAlphaFile)} transparent muted style={{ width: "100%", height: "100%", objectFit: "cover" }} />
        </AbsoluteFill>
      )}
    </AbsoluteFill>
  );
};

/**
 * VFX02 PropPopOutLayeredObject
 * Isolates real tracked foreground prop via local GrabCut+flow alpha matte, applies physical anticipation,
 * spring 2.5D perspective lift, soft contact drop-shadow & subtle sheen, then settles cleanly back onto live motion!
 */
const VfxPropPopout: React.FC<CreativeSceneProps> = ({
  sourceFile,
  propAlphaFile,
  hfGraphicFile,
  anchors,
  entryFrames,
  exitFrames,
  parameters,
  excludeLayerRole,
}) => {
  const frame = useCurrentFrame();
  const { fps, durationInFrames } = useVideoConfig();
  const env = sceneEnvelope(frame, durationInFrames, entryFrames, exitFrames);

  const springUp = spring({ frame: Math.max(0, frame - 2), fps, config: { damping: 14, stiffness: 120, mass: 0.9 } });
  const exitMul = frame > durationInFrames - exitFrames ? Math.max(0, (durationInFrames - 1 - frame) / Math.max(1, exitFrames)) : 1;
  const liftEnv = Math.min(env, springUp) * easeInOutCubic(exitMul);

  const liftScale = 1 + (Number(parameters.lift_scale ?? 1.16) - 1) * liftEnv;
  const liftY = Number(parameters.lift_y_px ?? -24) * liftEnv;
  const tiltDeg = Number(parameters.tilt_deg ?? -3.5) * liftEnv;
  const bgDim = 1 - (1 - Number(parameters.bg_dim ?? 0.84)) * env;
  const bgBlur = Number(parameters.bg_blur_px ?? 3.2) * env;

  const taskMid = anchors.task_mid || { cx: 0.52, cy: 0.76 };
  const origX = `${Math.round(taskMid.cx * 100)}%`;
  const origY = `${Math.round(Math.min(0.88, taskMid.cy + 0.08) * 100)}%`;

  const showProp = Boolean(propAlphaFile) && excludeLayerRole !== "prop_cutout_matte";
  const showFgGraphics = Boolean(hfGraphicFile) && excludeLayerRole !== "foreground_graphics";

  return (
    <AbsoluteFill style={{ backgroundColor: "#14110E", overflow: "hidden" }}>
      {/* Layer 0: Gently softened & dimmed real source so lifted physical prop pops crisply */}
      <AbsoluteFill
        style={{
          filter: bgBlur > 0.1 ? `blur(${bgBlur.toFixed(2)}px) brightness(${bgDim.toFixed(3)})` : "none",
        }}
      >
        <OffthreadVideo src={staticFile(sourceFile)} muted style={{ width: "100%", height: "100%", objectFit: "cover" }} />
      </AbsoluteFill>

      {/* Layer 1: Real GrabCut-isolated foreground prop with 2.5D physical perspective lift + soft contact shadow */}
      {showProp && propAlphaFile && (
        <AbsoluteFill
          style={{
            transform: `perspective(920px) translateY(${liftY.toFixed(2)}px) scale(${liftScale.toFixed(4)}) rotateX(${(liftEnv * 5.5).toFixed(2)}deg) rotateZ(${tiltDeg.toFixed(2)}deg)`,
            transformOrigin: `${origX} ${origY}`,
            filter:
              liftEnv > 0.02
                ? `drop-shadow(0px ${(22 * liftEnv).toFixed(1)}px ${(26 * liftEnv).toFixed(1)}px rgba(18, 12, 7, ${(0.52 * liftEnv).toFixed(3)})) brightness(${(1 + 0.05 * liftEnv).toFixed(3)}) contrast(${(1 + 0.04 * liftEnv).toFixed(3)})`
                : "none",
          }}
        >
          <OffthreadVideo src={staticFile(propAlphaFile)} transparent muted style={{ width: "100%", height: "100%", objectFit: "cover" }} />
        </AbsoluteFill>
      )}

      {/* Layer 2: HyperFrames subtle tactile callout / grounded prop label */}
      {showFgGraphics && hfGraphicFile && (
        <AbsoluteFill style={{ opacity: env }}>
          <OffthreadVideo src={staticFile(hfGraphicFile)} transparent muted style={{ width: "100%", height: "100%", objectFit: "cover" }} />
        </AbsoluteFill>
      )}
    </AbsoluteFill>
  );
};

/**
 * VFX03 MacroDetailMatchAction
 * Large sweeping diagonal organic macro lens magnifying live tracked hands/props (not a tiny corner inset!).
 */
const VfxMacroAction: React.FC<CreativeSceneProps> = ({
  sceneId,
  sourceFile,
  hfGraphicFile,
  anchors,
  entryFrames,
  exitFrames,
  parameters,
  excludeLayerRole,
}) => {
  const frame = useCurrentFrame();
  const { width, height, durationInFrames } = useVideoConfig();
  const u = frame / Math.max(1, durationInFrames - 1);
  const env = sceneEnvelope(frame, durationInFrames, entryFrames, exitFrames);
  const pal = getPal(parameters);

  const taskPt = sampleAnchorPath(anchors.task_path, u, anchors.task_mid);
  const detailZoom = Number(parameters.detail_zoom ?? 1.56);
  const splitY = Math.round(height * (1 - (1 - Number(parameters.split_ratio ?? 0.58)) * easeInOutCubic(env)));
  const curveDip = Math.round(Number(parameters.curve_depth_px ?? 44) * env);

  const upperShiftY = -Math.round(height * 0.075 * env);
  const upperScale = 1 + 0.07 * env;

  const targetHandsY = Math.min(0.86, Math.max(0.75, taskPt.cy + 0.13));
  const scaledW = width * detailZoom;
  const scaledH = height * detailZoom;
  const paneCenterY = splitY + (height - splitY) * 0.50;
  const offX = Math.min(0, Math.max(width - scaledW, width * 0.5 - taskPt.cx * scaledW));
  const offY = Math.min(0, Math.max(height - scaledH, paneCenterY - targetHandsY * scaledH));

  const clipId = `vfxMacro_${sceneId.replace(/[^a-zA-Z0-9]/g, "")}`;
  const wavePath = `M 0 ${splitY + curveDip} Q ${Math.round(width * 0.36)} ${splitY - curveDip} ${Math.round(width * 0.72)} ${splitY + Math.round(curveDip * 0.4)} T ${width} ${splitY - Math.round(curveDip * 0.35)} L ${width} ${height} L 0 ${height} Z`;
  const strokePath = `M 0 ${splitY + curveDip} Q ${Math.round(width * 0.36)} ${splitY - curveDip} ${Math.round(width * 0.72)} ${splitY + Math.round(curveDip * 0.4)} T ${width} ${splitY - Math.round(curveDip * 0.35)}`;

  const showDetail = excludeLayerRole !== "source_detail";
  const showFgGraphics = Boolean(hfGraphicFile) && excludeLayerRole !== "foreground_graphics";

  return (
    <AbsoluteFill style={{ backgroundColor: "#14110E", overflow: "hidden" }}>
      <AbsoluteFill
        style={{
          transform: `translateY(${upperShiftY}px) scale(${upperScale.toFixed(4)})`,
          transformOrigin: "50% 34%",
        }}
      >
        <OffthreadVideo src={staticFile(sourceFile)} muted style={{ width: "100%", height: "100%", objectFit: "cover" }} />
      </AbsoluteFill>

      {showDetail && env > 0.02 && (
        <AbsoluteFill style={{ clipPath: `url(#${clipId})`, WebkitClipPath: `url(#${clipId})` }}>
          <div style={{ position: "absolute", left: offX, top: offY, width: scaledW, height: scaledH }}>
            <OffthreadVideo src={staticFile(sourceFile)} muted style={{ width: scaledW, height: scaledH, objectFit: "cover" }} />
          </div>
          {/* Soft warm depth shadow along top macro rim */}
          <div
            style={{
              position: "absolute",
              inset: 0,
              boxShadow: "inset 0 28px 36px -18px rgba(18,12,8,0.45)",
              pointerEvents: "none",
            }}
          />
        </AbsoluteFill>
      )}

      <svg width={width} height={height} style={{ position: "absolute", inset: 0, pointerEvents: "none" }}>
        <defs>
          <clipPath id={clipId}>
            <path d={wavePath} />
          </clipPath>
        </defs>
        {showDetail && env > 0.02 && (
          <path d={strokePath} fill="none" stroke={pal.tertiary} strokeWidth={3.5} opacity={env * 0.94} />
        )}
      </svg>

      {showFgGraphics && hfGraphicFile && (
        <AbsoluteFill style={{ opacity: Math.max(0, Math.min(1, (env - 0.35) / 0.65)) }}>
          <OffthreadVideo src={staticFile(hfGraphicFile)} transparent muted style={{ width: "100%", height: "100%", objectFit: "cover" }} />
        </AbsoluteFill>
      )}
    </AbsoluteFill>
  );
};

/**
 * VFX05 SemanticBenefitReveal & VFX11 PayoffParticleAccent
 * Balanced framing + subtle warm atmospheric headroom scrim so HyperFrames grounded benefit ledger / payoff glint
 * reads with luxury commercial clarity while keeping speaker face & hands 100% open.
 */
const VfxBenefitOrPayoffStage: React.FC<CreativeSceneProps> = ({
  sceneKind,
  sourceFile,
  hfGraphicFile,
  entryFrames,
  exitFrames,
  parameters,
  excludeLayerRole,
}) => {
  const frame = useCurrentFrame();
  const { durationInFrames } = useVideoConfig();
  const env = sceneEnvelope(frame, durationInFrames, entryFrames, exitFrames);

  const isPayoff = sceneKind === "vfx_payoff_accent";
  const shiftY = isPayoff ? 0 : Number(parameters.main_shift_y_px ?? 16) * env;
  const heroScale = 1 + (isPayoff ? 0.042 : 0.032) * env;
  const scrimAlpha = (isPayoff ? 0.24 : 0.30) * env;
  const showFgGraphics = Boolean(hfGraphicFile) && excludeLayerRole !== "foreground_graphics";

  return (
    <AbsoluteFill style={{ backgroundColor: "#14110E", overflow: "hidden" }}>
      <AbsoluteFill
        style={{
          transform: `translateY(${shiftY.toFixed(2)}px) scale(${heroScale.toFixed(4)})`,
          transformOrigin: "50% 42%",
          filter: isPayoff ? `contrast(${(1 + 0.04 * env).toFixed(3)}) saturate(${(1 + 0.05 * env).toFixed(3)})` : "none",
        }}
      >
        <OffthreadVideo src={staticFile(sourceFile)} muted style={{ width: "100%", height: "100%", objectFit: "cover" }} />
      </AbsoluteFill>

      {/* Soft top editorial gradient scrim so grounded upper-headroom benefit pills pop cleanly on bright interiors */}
      <AbsoluteFill
        style={{
          background: isPayoff
            ? `radial-gradient(ellipse at 50% 46%, rgba(18,14,10,0) 34%, rgba(18,14,10,${scrimAlpha.toFixed(3)}) 100%)`
            : `linear-gradient(180deg, rgba(22,17,13,${scrimAlpha.toFixed(3)}) 0%, rgba(22,17,13,0) 28%, rgba(22,17,13,0) 100%)`,
          pointerEvents: "none",
        }}
      />

      {showFgGraphics && hfGraphicFile && (
        <AbsoluteFill style={{ opacity: env }}>
          <OffthreadVideo src={staticFile(hfGraphicFile)} transparent muted style={{ width: "100%", height: "100%", objectFit: "cover" }} />
        </AbsoluteFill>
      )}
    </AbsoluteFill>
  );
};

/**
 * VFX06 ShapeMatchMaskMorph & VFX12 CinematicReframeReturn
 * Elevated organic squircle editorial stage over warm blurred ambient source + HyperFrames synchronized trim.
 */
const VfxMorphOrClosingStage: React.FC<CreativeSceneProps> = ({
  sourceFile,
  hfGraphicFile,
  entryFrames,
  exitFrames,
  parameters,
  excludeLayerRole,
}) => {
  const frame = useCurrentFrame();
  const { durationInFrames } = useVideoConfig();
  const u = frame / Math.max(1, durationInFrames - 1);
  const env = sceneEnvelope(frame, durationInFrames, entryFrames, exitFrames);
  const pal = getPal(parameters);

  const stageScale = 1 - (1 - Number(parameters.stage_scale ?? 0.89)) * env;
  const stageRadius = Math.round(Number(parameters.stage_radius_px ?? 38) * env);
  const bgBlur = Number(parameters.bg_blur_px ?? 16) * env;
  const bgDim = 1 - (1 - Number(parameters.bg_dim ?? 0.72)) * env;
  const driftY = Math.sin(u * Math.PI) * -5 * env;
  const showFgGraphics = Boolean(hfGraphicFile) && excludeLayerRole !== "foreground_graphics";

  return (
    <AbsoluteFill style={{ backgroundColor: "#171310", overflow: "hidden" }}>
      <AbsoluteFill
        style={{
          transform: `scale(${(1.06 + 0.05 * env).toFixed(4)})`,
          filter: bgBlur > 0.2 ? `blur(${bgBlur.toFixed(1)}px) brightness(${bgDim.toFixed(3)})` : "none",
        }}
      >
        <OffthreadVideo src={staticFile(sourceFile)} muted style={{ width: "100%", height: "100%", objectFit: "cover" }} />
      </AbsoluteFill>

      <AbsoluteFill
        style={{
          transform: `translateY(${driftY.toFixed(2)}px) scale(${stageScale.toFixed(4)})`,
          borderRadius: stageRadius,
          overflow: "hidden",
          boxShadow: env > 0.02 ? `0 26px 64px rgba(15, 11, 8, ${(0.50 * env).toFixed(3)}), 0 0 0 2px ${pal.primary}` : "none",
        }}
      >
        <OffthreadVideo src={staticFile(sourceFile)} muted style={{ width: "100%", height: "100%", objectFit: "cover" }} />
      </AbsoluteFill>

      {showFgGraphics && hfGraphicFile && (
        <AbsoluteFill style={{ opacity: env }}>
          <OffthreadVideo src={staticFile(hfGraphicFile)} transparent muted style={{ width: "100%", height: "100%", objectFit: "cover" }} />
        </AbsoluteFill>
      )}
    </AbsoluteFill>
  );
};

/**
 * VFX08 KineticEditorialCollage
 * Asymmetric editorial showcase combining gently balanced live speaker + crisp rounded live hands/prop tile +
 * HyperFrames layered tactile feature cards (strictly grounded from spoken words!).
 */
const VfxEditorialCollage: React.FC<CreativeSceneProps> = ({
  sourceFile,
  hfGraphicFile,
  anchors,
  entryFrames,
  exitFrames,
  parameters,
  excludeLayerRole,
}) => {
  const frame = useCurrentFrame();
  const { width, height, durationInFrames } = useVideoConfig();
  const u = frame / Math.max(1, durationInFrames - 1);
  const env = sceneEnvelope(frame, durationInFrames, entryFrames, exitFrames);
  const pal = getPal(parameters);

  const taskPt = sampleAnchorPath(anchors.task_path, u, anchors.task_mid);
  const detailZoom = Number(parameters.detail_zoom ?? 1.48);
  const slot = anchors.detail_slot || { cx: 0.25, cy: 0.115, w: 0.41, h: 0.165, side: "left" };

  const boxW = Math.round(width * 0.39);
  const boxH = Math.round(height * 0.16);
  const boxX = Math.round(width * (slot.side === "left" ? 0.05 : 0.56));
  const boxY = Math.round(height * 0.038);

  const targetY = Math.min(0.86, Math.max(0.76, taskPt.cy + 0.14));
  const scaledW = width * detailZoom;
  const scaledH = height * detailZoom;
  const offX = Math.min(0, Math.max(boxW - scaledW, boxW / 2 - taskPt.cx * scaledW));
  const offY = Math.min(0, Math.max(boxH - scaledH, boxH / 2 - targetY * scaledH));

  const showDetail = excludeLayerRole !== "source_detail";
  const showFgGraphics = Boolean(hfGraphicFile) && excludeLayerRole !== "foreground_graphics";

  return (
    <AbsoluteFill style={{ backgroundColor: "#14110E", overflow: "hidden" }}>
      <AbsoluteFill
        style={{
          transform: `translateY(${(height * 0.020 * env).toFixed(2)}px) scale(${(1 + 0.036 * env).toFixed(4)})`,
          transformOrigin: "50% 45%",
          filter: `brightness(${(1 - 0.07 * env).toFixed(3)})`,
        }}
      >
        <OffthreadVideo src={staticFile(sourceFile)} muted style={{ width: "100%", height: "100%", objectFit: "cover" }} />
      </AbsoluteFill>

      {showDetail && env > 0.02 && (
        <div
          style={{
            position: "absolute",
            left: boxX,
            top: boxY,
            width: boxW,
            height: boxH,
            borderRadius: 24,
            overflow: "hidden",
            opacity: env,
            transform: `translateY(${((1 - env) * -16).toFixed(2)}px) rotate(${(-1.2 * env).toFixed(2)}deg)`,
            boxShadow: `0 18px 40px rgba(18, 12, 8, ${(0.44 * env).toFixed(3)}), 0 0 0 2.5px ${pal.tertiary}`,
          }}
        >
          <div style={{ position: "absolute", left: offX, top: offY, width: scaledW, height: scaledH }}>
            <OffthreadVideo src={staticFile(sourceFile)} muted style={{ width: scaledW, height: scaledH, objectFit: "cover" }} />
          </div>
        </div>
      )}

      {showFgGraphics && hfGraphicFile && (
        <AbsoluteFill style={{ opacity: env }}>
          <OffthreadVideo src={staticFile(hfGraphicFile)} transparent muted style={{ width: "100%", height: "100%", objectFit: "cover" }} />
        </AbsoluteFill>
      )}
    </AbsoluteFill>
  );
};

/**
 * VFX09 SpatialCamera25D
 * Multi-plane 2.5D parallax perspective (explicitly 2.5D layered planes, not fake 3D scene reconstruction).
 */
const VfxSpatial25D: React.FC<CreativeSceneProps> = ({
  sourceFile,
  fgAlphaFile,
  hfGraphicFile,
  entryFrames,
  exitFrames,
  parameters,
  excludeLayerRole,
}) => {
  const frame = useCurrentFrame();
  const { durationInFrames } = useVideoConfig();
  const u = frame / Math.max(1, durationInFrames - 1);
  const env = sceneEnvelope(frame, durationInFrames, entryFrames, exitFrames);

  const swing = Math.sin(u * Math.PI * 2 - Math.PI / 2) * 0.5 + 0.5; // smooth 0 -> 1 -> 0
  const tiltY = (swing - 0.5) * 2 * Number(parameters.persp_tilt_deg ?? 2.4) * env;
  const bgX = (swing - 0.5) * -18 * env;
  const fgX = (swing - 0.5) * 10 * env;
  const bgBlur = Math.min(5.5, Number(parameters.bg_blur_px ?? 5.0)) * env;
  const bgDim = 1 - (1 - Number(parameters.bg_dim ?? 0.86)) * env;
  const fgLift = 1 + (Number(parameters.fg_lift ?? 1.036) - 1) * env;

  const showBgGraphics = Boolean(hfGraphicFile) && excludeLayerRole !== "background_graphics";
  const showFgMatte = Boolean(fgAlphaFile) && excludeLayerRole !== "foreground_subject_matte";

  return (
    <AbsoluteFill style={{ backgroundColor: "#14110E", overflow: "hidden" }}>
      <AbsoluteFill
        style={{
          transform: `perspective(1100px) translateX(${bgX.toFixed(2)}px) rotateY(${tiltY.toFixed(2)}deg) scale(${(1.045 * (1 + 0.01 * env)).toFixed(4)})`,
          transformOrigin: "50% 45%",
          filter: bgBlur > 0.1 ? `blur(${bgBlur.toFixed(2)}px) brightness(${bgDim.toFixed(3)})` : "none",
        }}
      >
        <OffthreadVideo src={staticFile(sourceFile)} muted style={{ width: "100%", height: "100%", objectFit: "cover" }} />
      </AbsoluteFill>

      {showBgGraphics && hfGraphicFile && (
        <AbsoluteFill
          style={{
            opacity: Math.min(1, env * 1.15),
            transform: `translateX(${(bgX * 0.45).toFixed(2)}px)`,
          }}
        >
          <OffthreadVideo src={staticFile(hfGraphicFile)} transparent muted style={{ width: "100%", height: "100%", objectFit: "cover" }} />
        </AbsoluteFill>
      )}

      {showFgMatte && fgAlphaFile && (
        <AbsoluteFill
          style={{
            transform: `translateX(${fgX.toFixed(2)}px) scale(${fgLift.toFixed(4)})`,
            transformOrigin: "50% 60%",
            filter: env > 0.05 ? `drop-shadow(0px 10px 20px rgba(20, 14, 10, ${(0.30 * env).toFixed(3)}))` : "none",
          }}
        >
          <OffthreadVideo src={staticFile(fgAlphaFile)} transparent muted style={{ width: "100%", height: "100%", objectFit: "cover" }} />
        </AbsoluteFill>
      )}
    </AbsoluteFill>
  );
};

/**
 * VFX10 MotionTrailGhostEcho
 * Subtle short-persistence alpha ghost echo strictly on tracked foreground prop matte (never smearing face!).
 */
const VfxMotionTrail: React.FC<CreativeSceneProps> = ({
  sourceFile,
  propAlphaFile,
  hfGraphicFile,
  entryFrames,
  exitFrames,
  parameters,
  excludeLayerRole,
}) => {
  const frame = useCurrentFrame();
  const { durationInFrames } = useVideoConfig();
  const env = sceneEnvelope(frame, durationInFrames, entryFrames, exitFrames);
  const shiftPx = Number(parameters.echo_shift_px ?? 14) * env;
  const echoOp = Number(parameters.echo_opacity ?? 0.36) * env;
  const showProp = Boolean(propAlphaFile) && excludeLayerRole !== "prop_cutout_matte";
  const showFgGraphics = Boolean(hfGraphicFile) && excludeLayerRole !== "foreground_graphics";

  return (
    <AbsoluteFill style={{ backgroundColor: "#14110E", overflow: "hidden" }}>
      <AbsoluteFill>
        <OffthreadVideo src={staticFile(sourceFile)} muted style={{ width: "100%", height: "100%", objectFit: "cover" }} />
      </AbsoluteFill>
      {showProp && propAlphaFile && env > 0.05 && (
        <>
          <AbsoluteFill style={{ opacity: echoOp * 0.55, transform: `translate(${(-shiftPx * 1.3).toFixed(1)}px, ${(-shiftPx * 0.4).toFixed(1)}px)`, filter: "blur(2.5px)" }}>
            <OffthreadVideo src={staticFile(propAlphaFile)} transparent muted style={{ width: "100%", height: "100%", objectFit: "cover" }} />
          </AbsoluteFill>
          <AbsoluteFill style={{ opacity: echoOp, transform: `translate(${(-shiftPx * 0.65).toFixed(1)}px, ${(-shiftPx * 0.2).toFixed(1)}px)`, filter: "blur(1.0px)" }}>
            <OffthreadVideo src={staticFile(propAlphaFile)} transparent muted style={{ width: "100%", height: "100%", objectFit: "cover" }} />
          </AbsoluteFill>
        </>
      )}
      {showFgGraphics && hfGraphicFile && (
        <AbsoluteFill style={{ opacity: env }}>
          <OffthreadVideo src={staticFile(hfGraphicFile)} transparent muted style={{ width: "100%", height: "100%", objectFit: "cover" }} />
        </AbsoluteFill>
      )}
    </AbsoluteFill>
  );
};

export const VfxSceneComposition: React.FC<CreativeSceneProps> = (props) => {
  switch (props.sceneKind) {
    case "vfx_depth_stage":
    case "vfx_action_flow":
    case "vfx_occlusion_wipe":
      return <VfxDepthOcclusionStage {...props} />;
    case "vfx_prop_popout":
      return <VfxPropPopout {...props} />;
    case "vfx_macro_action":
      return <VfxMacroAction {...props} />;
    case "vfx_benefit_stage":
    case "vfx_payoff_accent":
      return <VfxBenefitOrPayoffStage {...props} />;
    case "vfx_shape_morph":
    case "vfx_closing_return":
      return <VfxMorphOrClosingStage {...props} />;
    case "vfx_editorial_collage":
      return <VfxEditorialCollage {...props} />;
    case "vfx_spatial_25d":
      return <VfxSpatial25D {...props} />;
    case "vfx_motion_trail":
      return <VfxMotionTrail {...props} />;
    default:
      return <VfxDepthOcclusionStage {...props} />;
  }
};
