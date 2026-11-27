import React from "react";
import { AbsoluteFill, OffthreadVideo, staticFile, useCurrentFrame, useVideoConfig } from "remotion";
import { easeInOutCubic, sampleAnchorPath, sceneEnvelope } from "./composition/easing";
import type { CreativeSceneProps } from "./composition/schema";
import { VfxSceneComposition } from "./composition/VfxSceneComposition";

const DEFAULT_PALETTE = {
  primary_hex: "#C8A26A",
  secondary_hex: "#4A3F35",
  tertiary_hex: "#F4EBDD",
};

function getPalette(params: Record<string, unknown>) {
  const pal = (params.palette as Record<string, string> | undefined) || {};
  return {
    primary: pal.primary_hex || DEFAULT_PALETTE.primary_hex,
    secondary: pal.secondary_hex || DEFAULT_PALETTE.secondary_hex,
    tertiary: pal.tertiary_hex || DEFAULT_PALETTE.tertiary_hex,
  };
}

/**
 * RCP01 (SubjectDepthStage), RCP08 (DepthRevealAndRelease), RCP09 (AudioReactiveHeroGraphic)
 * True 4-layer depth compositing:
 *   L0: Softened real background plane with independent subtle parallax scale
 *   L1: HyperFrames GSAP architectural depth stage / halo / audio arcs BEHIND subject
 *   L2: MODNet continuous alpha subject matte IN FRONT OF graphics (true hair/shoulder occlusion!)
 */
const DepthStageScene: React.FC<CreativeSceneProps> = ({
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

  const bgBlur = Math.min(6.0, Number(parameters.bg_blur_px ?? 5.5)) * env;
  const bgScale = 1 + (Math.min(1.022, Number(parameters.bg_scale_drift ?? 1.020)) - 1) * env * (0.65 + 0.35 * Math.sin(u * Math.PI));
  const bgDim = 1 - (1 - Number(parameters.warm_dim ?? 0.86)) * env;
  const fgLift = 1 + Math.max(0.028, Number(parameters.fg_micro_lift ?? 1.030) - 1) * env;
  const showBgGraphics = Boolean(hfGraphicFile) && excludeLayerRole !== "background_graphics";
  const showFgMatte = Boolean(fgAlphaFile) && excludeLayerRole !== "foreground_subject_matte";

  return (
    <AbsoluteFill style={{ backgroundColor: "#14110E", overflow: "hidden" }}>
      {/* Layer 0: Softened real background plane with restrained depth blur (no light halo bleed!) */}
      <AbsoluteFill
        style={{
          transform: `scale(${bgScale.toFixed(4)})`,
          transformOrigin: "50% 44%",
          filter: bgBlur > 0.1 ? `blur(${bgBlur.toFixed(2)}px) brightness(${bgDim.toFixed(3)}) saturate(0.92)` : "none",
        }}
      >
        <OffthreadVideo src={staticFile(sourceFile)} muted style={{ width: "100%", height: "100%", objectFit: "cover" }} />
      </AbsoluteFill>

      {/* Layer 1: HyperFrames substantive graphic stage strictly BEHIND foreground subject */}
      {showBgGraphics && hfGraphicFile && (
        <AbsoluteFill style={{ opacity: Math.min(1, env * 1.15) }}>
          <OffthreadVideo
            src={staticFile(hfGraphicFile)}
            transparent
            muted
            style={{ width: "100%", height: "100%", objectFit: "cover" }}
          />
        </AbsoluteFill>
      )}

      {/* Layer 2: MODNet continuous alpha subject matte + tabletop guard IN FRONT OF graphics */}
      {showFgMatte && fgAlphaFile && (
        <AbsoluteFill
          style={{
            transform: `scale(${fgLift.toFixed(4)})`,
            transformOrigin: "50% 60%",
            filter: env > 0.05 ? `drop-shadow(0px 8px 16px rgba(24, 17, 11, ${(0.28 * env).toFixed(3)}))` : "none",
          }}
        >
          <OffthreadVideo
            src={staticFile(fgAlphaFile)}
            transparent
            muted
            style={{ width: "100%", height: "100%", objectFit: "cover" }}
          />
        </AbsoluteFill>
      )}
    </AbsoluteFill>
  );
};

/**
 * RCP02 (AsymmetricDetailReveal)
 * Main context footage shifts subtly while a curved editorial capsule aperture reveals a same-time
 * live detail crop of the tracked active task/hands ROI, connected by HyperFrames animated vector callout.
 */
const DetailApertureScene: React.FC<CreativeSceneProps> = ({
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
  const pal = getPalette(parameters);

  const slot = anchors.detail_slot || { cx: 0.27, cy: 0.19, w: 0.42, h: 0.26, side: "left" };
  const taskPt = sampleAnchorPath(anchors.task_path, u, anchors.task_mid);

  const detailZoom = Number(parameters.detail_zoom ?? 1.50);
  const radiusPx = Math.round(Number(parameters.capsule_radius_px ?? 30));
  const shiftDir = slot.side === "left" ? 1 : -1;
  const mainOverscan = 1 + 0.048 * env;
  const mainShiftPx = shiftDir * width * 0.028 * env;
  const mainShiftY = height * 0.022 * env;

  const boxW = Math.round(width * slot.w);
  const boxH = Math.round(height * slot.h);
  const boxX = Math.round(width * slot.cx - boxW / 2);
  const boxY = Math.round(height * slot.cy - boxH / 2);

  // Focus detail crop on the lower active action zone of tracked task ROI (hands/props rather than torso!)
  const targetTaskY = Math.min(0.86, taskPt.cy + 0.095);
  const scaledW = width * detailZoom;
  const scaledH = height * detailZoom;
  const rawOffX = boxW / 2 - taskPt.cx * scaledW;
  const rawOffY = boxH / 2 - targetTaskY * scaledH;
  const offX = Math.min(0, Math.max(boxW - scaledW, rawOffX));
  const offY = Math.min(0, Math.max(boxH - scaledH, rawOffY));

  const cardScale = 0.86 + 0.14 * easeInOutCubic(env);
  const cardSlideY = (1 - env) * -18;
  const showDetail = excludeLayerRole !== "source_detail";
  const showFgGraphics = Boolean(hfGraphicFile) && excludeLayerRole !== "foreground_graphics";

  return (
    <AbsoluteFill style={{ backgroundColor: "#14110E", overflow: "hidden" }}>
      {/* Layer 0: Main live context video gently balanced opposite & slightly below elevated detail aperture */}
      <AbsoluteFill
        style={{
          transform: `translate(${mainShiftPx.toFixed(2)}px, ${mainShiftY.toFixed(2)}px) scale(${mainOverscan.toFixed(4)})`,
          transformOrigin: "50% 45%",
          filter: `brightness(${(1 - 0.06 * env).toFixed(3)})`,
        }}
      >
        <OffthreadVideo src={staticFile(sourceFile)} muted style={{ width: "100%", height: "100%", objectFit: "cover" }} />
      </AbsoluteFill>

      {/* Layer 1: Same-time curved-capsule live detail crop of tracked hands/task ROI */}
      {showDetail && env > 0.01 && (
        <div
          style={{
            position: "absolute",
            left: boxX,
            top: boxY,
            width: boxW,
            height: boxH,
            borderRadius: radiusPx,
            overflow: "hidden",
            opacity: env,
            transform: `translateY(${cardSlideY.toFixed(2)}px) scale(${cardScale.toFixed(4)})`,
            boxShadow: `0 18px 42px rgba(24, 18, 13, ${(0.42 * env).toFixed(3)}), 0 0 0 2.5px ${pal.tertiary}`,
            backgroundColor: "#1E1712",
          }}
        >
          <div
            style={{
              position: "absolute",
              left: offX,
              top: offY,
              width: scaledW,
              height: scaledH,
            }}
          >
            <OffthreadVideo
              src={staticFile(sourceFile)}
              muted
              style={{ width: scaledW, height: scaledH, objectFit: "cover" }}
            />
          </div>
          {/* Soft warm inner rim sheen for tactile editorial lens feel */}
          <div
            style={{
              position: "absolute",
              inset: 0,
              borderRadius: radiusPx,
              boxShadow: `inset 0 0 0 1.5px ${pal.primary}`,
              pointerEvents: "none",
            }}
          />
        </div>
      )}

      {/* Layer 2: HyperFrames animated vector callout connecting tracked ROI -> detail capsule */}
      {showFgGraphics && hfGraphicFile && (
        <AbsoluteFill style={{ opacity: env }}>
          <OffthreadVideo
            src={staticFile(hfGraphicFile)}
            transparent
            muted
            style={{ width: "100%", height: "100%", objectFit: "cover" }}
          />
        </AbsoluteFill>
      )}
    </AbsoluteFill>
  );
};

/**
 * RCP03 (TrackedActionPath) & RCP04 (ForegroundPassThrough)
 * Hero HyperFrames GSAP/SVG curved trajectory or sweeping sculptural ribbon composited with TRUE z-order
 * occlusion behind the MODNet subject matte!
 */
const ActionPathScene: React.FC<CreativeSceneProps> = ({
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
  const env = sceneEnvelope(frame, durationInFrames, entryFrames, exitFrames);

  const bgBlur = Math.min(4.8, Number(parameters.bg_blur_px ?? 4.5)) * env;
  const bgDim = 1 - (1 - Number(parameters.bg_depth_dim ?? 0.90)) * env;
  const bgScale = 1 + 0.016 * env;
  const fgLift = 1 + 0.026 * env;
  const showBgGraphics = Boolean(hfGraphicFile) && excludeLayerRole !== "background_graphics";
  const showFgMatte = Boolean(fgAlphaFile) && excludeLayerRole !== "foreground_subject_matte";

  return (
    <AbsoluteFill style={{ backgroundColor: "#14110E", overflow: "hidden" }}>
      {/* Layer 0: Slightly softened source background for clean graphic readability */}
      <AbsoluteFill
        style={{
          transform: `scale(${bgScale.toFixed(4)})`,
          transformOrigin: "50% 45%",
          filter: bgBlur > 0.1 ? `blur(${bgBlur.toFixed(2)}px) brightness(${bgDim.toFixed(3)})` : "none",
        }}
      >
        <OffthreadVideo src={staticFile(sourceFile)} muted style={{ width: "100%", height: "100%", objectFit: "cover" }} />
      </AbsoluteFill>

      {/* Layer 1: Hero HyperFrames sweeping path/ribbon behind subject matte */}
      {showBgGraphics && hfGraphicFile && (
        <AbsoluteFill style={{ opacity: Math.min(1, env * 1.1) }}>
          <OffthreadVideo
            src={staticFile(hfGraphicFile)}
            transparent
            muted
            style={{ width: "100%", height: "100%", objectFit: "cover" }}
          />
        </AbsoluteFill>
      )}

      {/* Layer 2: MODNet foreground subject matte cleanly occluding passing graphics */}
      {showFgMatte && fgAlphaFile && (
        <AbsoluteFill
          style={{
            transform: `scale(${fgLift.toFixed(4)})`,
            transformOrigin: "50% 60%",
            filter: env > 0.05 ? `drop-shadow(0px 8px 18px rgba(24, 18, 13, ${(0.20 * env).toFixed(3)}))` : "none",
          }}
        >
          <OffthreadVideo
            src={staticFile(fgAlphaFile)}
            transparent
            muted
            style={{ width: "100%", height: "100%", objectFit: "cover" }}
          />
        </AbsoluteFill>
      )}
    </AbsoluteFill>
  );
};

/**
 * RCP05 (MatchShapeLayoutMorph)
 * Smooth organic morph from full-bleed -> elevated recessed editorial stage over warm blurred ambient plate
 * with independent layer parallax + HyperFrames curved frame flourish -> smooth morph back to 1.0x.
 */
const LayoutMorphScene: React.FC<CreativeSceneProps> = ({
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
  const pal = getPalette(parameters);

  const stageScale = 1 - (1 - Number(parameters.stage_scale ?? 0.88)) * env;
  const stageRadius = Math.round(Number(parameters.stage_radius_px ?? 36) * env);
  const bgBlur = Number(parameters.bg_blur_px ?? 18) * env;
  const bgDim = 1 - (1 - Number(parameters.bg_dim ?? 0.72)) * env;
  const driftY = Math.sin(u * Math.PI) * -6 * env;
  const showFgGraphics = Boolean(hfGraphicFile) && excludeLayerRole !== "foreground_graphics";

  return (
    <AbsoluteFill style={{ backgroundColor: "#1A1511", overflow: "hidden" }}>
      {/* Layer 0: Soft warm blurred ambient source plate with opposite slow zoom */}
      <AbsoluteFill
        style={{
          transform: `scale(${(1.06 + 0.06 * env).toFixed(4)})`,
          filter: bgBlur > 0.2 ? `blur(${bgBlur.toFixed(1)}px) brightness(${bgDim.toFixed(3)})` : "none",
        }}
      >
        <OffthreadVideo src={staticFile(sourceFile)} muted style={{ width: "100%", height: "100%", objectFit: "cover" }} />
      </AbsoluteFill>

      {/* Layer 1: Morphed elevated editorial stage */}
      <AbsoluteFill
        style={{
          transform: `translateY(${driftY.toFixed(2)}px) scale(${stageScale.toFixed(4)})`,
          borderRadius: stageRadius,
          overflow: "hidden",
          boxShadow: env > 0.02 ? `0 24px 60px rgba(15, 11, 8, ${(0.48 * env).toFixed(3)}), 0 0 0 2px ${pal.primary}` : "none",
        }}
      >
        <OffthreadVideo src={staticFile(sourceFile)} muted style={{ width: "100%", height: "100%", objectFit: "cover" }} />
      </AbsoluteFill>

      {/* Layer 2: HyperFrames curved morph flourish */}
      {showFgGraphics && hfGraphicFile && (
        <AbsoluteFill style={{ opacity: env }}>
          <OffthreadVideo
            src={staticFile(hfGraphicFile)}
            transparent
            muted
            style={{ width: "100%", height: "100%", objectFit: "cover" }}
          />
        </AbsoluteFill>
      )}
    </AbsoluteFill>
  );
};

/**
 * RCP06 (SplitFocusEditorial)
 * Asymmetric curved-wave editorial dual-focus layout: upper pane frames speaker cleanly while a sweeping
 * curved lower inset highlights the same-time tracked foreground action/hands with independent drift + HyperFrames divider.
 */
const SplitFocusScene: React.FC<CreativeSceneProps> = ({
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
  const pal = getPalette(parameters);

  const taskPt = sampleAnchorPath(anchors.task_path, u, anchors.task_mid);
  const detailZoom = Number(parameters.detail_zoom ?? 1.44);
  const splitY = Math.round(height * (1 - (1 - Number(parameters.split_ratio ?? 0.62)) * env));
  const curveDip = Math.round(Number(parameters.curve_depth_px ?? 42) * env);

  // Subtle upward framing shift for primary upper speaker pane so face sits generously centered above curve
  const upperShiftY = -Math.round(height * 0.070 * env);
  const upperScale = 1 + 0.075 * env;

  // Center lower split pane cleanly onto active hands/props band (lower half of tracked ROI, avoiding torso gap!)
  const targetHandsY = Math.min(0.86, Math.max(0.76, taskPt.cy + 0.14));
  const scaledW = width * detailZoom;
  const scaledH = height * detailZoom;
  const paneCenterY = splitY + (height - splitY) * 0.50;
  const rawOffX = width * 0.5 - taskPt.cx * scaledW;
  const rawOffY = paneCenterY - targetHandsY * scaledH;
  const offX = Math.min(0, Math.max(width - scaledW, rawOffX));
  const offY = Math.min(0, Math.max(height - scaledH, rawOffY));

  const clipId = `splitWave_${sceneId.replace(/[^a-zA-Z0-9]/g, "")}`;
  const wavePath = `M 0 ${splitY + curveDip} Q ${Math.round(width * 0.34)} ${splitY - curveDip} ${Math.round(width * 0.68)} ${splitY + Math.round(curveDip * 0.45)} T ${width} ${splitY - Math.round(curveDip * 0.35)} L ${width} ${height} L 0 ${height} Z`;
  const strokePath = `M 0 ${splitY + curveDip} Q ${Math.round(width * 0.34)} ${splitY - curveDip} ${Math.round(width * 0.68)} ${splitY + Math.round(curveDip * 0.45)} T ${width} ${splitY - Math.round(curveDip * 0.35)}`;

  const showDetail = excludeLayerRole !== "source_detail";
  const showFgGraphics = Boolean(hfGraphicFile) && excludeLayerRole !== "foreground_graphics";

  return (
    <AbsoluteFill style={{ backgroundColor: "#14110E", overflow: "hidden" }}>
      {/* Layer 0: Upper speaker context pane */}
      <AbsoluteFill
        style={{
          transform: `translateY(${upperShiftY}px) scale(${upperScale.toFixed(4)})`,
          transformOrigin: "50% 34%",
        }}
      >
        <OffthreadVideo src={staticFile(sourceFile)} muted style={{ width: "100%", height: "100%", objectFit: "cover" }} />
      </AbsoluteFill>

      {/* Layer 1: Curved-wave lower same-time detail pane centered on tracked hands/task ROI */}
      {showDetail && env > 0.02 && (
        <AbsoluteFill style={{ clipPath: `url(#${clipId})`, WebkitClipPath: `url(#${clipId})` }}>
          <div style={{ position: "absolute", left: offX, top: offY, width: scaledW, height: scaledH }}>
            <OffthreadVideo
              src={staticFile(sourceFile)}
              muted
              style={{ width: scaledW, height: scaledH, objectFit: "cover" }}
            />
          </div>
        </AbsoluteFill>
      )}

      {/* SVG clip path definition + crisp curved editorial seam */}
      <svg width={width} height={height} style={{ position: "absolute", inset: 0, pointerEvents: "none" }}>
        <defs>
          <clipPath id={clipId}>
            <path d={wavePath} />
          </clipPath>
        </defs>
        {showDetail && env > 0.02 && (
          <path
            d={strokePath}
            fill="none"
            stroke={pal.tertiary}
            strokeWidth={3.5}
            opacity={env * 0.92}
          />
        )}
      </svg>

      {/* Layer 2: HyperFrames animated divider accent — fades cleanly onto settled curved seam */}
      {showFgGraphics && hfGraphicFile && (
        <AbsoluteFill style={{ opacity: Math.max(0, Math.min(1, (env - 0.55) / 0.45)) }}>
          <OffthreadVideo
            src={staticFile(hfGraphicFile)}
            transparent
            muted
            style={{ width: "100%", height: "100%", objectFit: "cover" }}
          />
        </AbsoluteFill>
      )}
    </AbsoluteFill>
  );
};

/**
 * RCP07 (ContextualContourAccent)
 * Subtle dynamic radial editorial spotlight tracking the active hands/task ROI + HyperFrames tracked contour arcs.
 */
const ContourFocusScene: React.FC<CreativeSceneProps> = ({
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

  const taskPt = sampleAnchorPath(anchors.task_path, u, anchors.task_mid);
  const dim = Number(parameters.surround_dim ?? 0.16) * env;
  const cxPx = Math.round(taskPt.cx * width);
  const cyPx = Math.round(taskPt.cy * height);
  const showFgGraphics = Boolean(hfGraphicFile) && excludeLayerRole !== "foreground_graphics";

  return (
    <AbsoluteFill style={{ backgroundColor: "#14110E", overflow: "hidden" }}>
      <AbsoluteFill>
        <OffthreadVideo src={staticFile(sourceFile)} muted style={{ width: "100%", height: "100%", objectFit: "cover" }} />
      </AbsoluteFill>
      {/* Dynamic tracked editorial radial depth vignette focusing eye on active task zone */}
      <AbsoluteFill
        style={{
          background: `radial-gradient(circle at ${cxPx}px ${cyPx}px, rgba(20,16,12,0) 26%, rgba(20,16,12,${dim.toFixed(3)}) 78%)`,
          pointerEvents: "none",
        }}
      />
      {showFgGraphics && hfGraphicFile && (
        <AbsoluteFill style={{ opacity: env }}>
          <OffthreadVideo
            src={staticFile(hfGraphicFile)}
            transparent
            muted
            style={{ width: "100%", height: "100%", objectFit: "cover" }}
          />
        </AbsoluteFill>
      )}
    </AbsoluteFill>
  );
};

export const MotionLabCreativeScene: React.FC<CreativeSceneProps> = (props) => {
  if (props.sceneKind.startsWith("vfx_")) {
    return <VfxSceneComposition {...props} />;
  }
  switch (props.sceneKind) {
    case "depth_stage":
      return <DepthStageScene {...props} />;
    case "detail_aperture":
      return <DetailApertureScene {...props} />;
    case "action_path":
      return <ActionPathScene {...props} />;
    case "layout_morph":
      return <LayoutMorphScene {...props} />;
    case "split_focus":
      return <SplitFocusScene {...props} />;
    case "contour_focus":
      return <ContourFocusScene {...props} />;
    default:
      return <DepthStageScene {...props} />;
  }
};
