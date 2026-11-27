/**
 * Phase A2/A3 (ML-01.R5): Generative Full-Frame Remotion Composition Engine.
 *
 * Directly imports and composes 11 verified MIT upstream Remotion primitives from:
 *   - riaz37/remotion-ui (Counter, WaterfallChart, StrikethroughReplace, MaskedSlideReveal, TrackMatte, ParallaxLayers)
 *   - vivmagarwal/remotion-effects-library (AspectMorphCard, ParticleField)
 *   - RenderComp/free-remotion-templates (FlipPageTransition / FlipPage)
 *   - GTKottman/remotion-transitions (FrameMorphBridge, LineLedBridge)
 *
 * Zero hardcoded video scripts, zero remote URLs/fonts, 100% deterministic frame-clock rendering.
 */

import React from "react";
import {
  AbsoluteFill,
  Easing,
  OffthreadVideo,
  interpolate,
  spring,
  staticFile,
  useCurrentFrame,
  useVideoConfig,
} from "remotion";

import { Counter } from "./imported/remotion-ui/Counter";
import { WaterfallChart, type WaterfallStep } from "./imported/remotion-ui/WaterfallChart";
import { StrikethroughReplace } from "./imported/remotion-ui/StrikethroughReplace";
import { MaskedSlideReveal } from "./imported/remotion-ui/MaskedSlideReveal";
import { TrackMatte } from "./imported/remotion-ui/TrackMatte";
import { ParallaxLayers } from "./imported/remotion-ui/ParallaxLayers";
import { AspectMorphCard } from "./imported/remotion-effects-library/AspectMorphCard";
import { ParticleField } from "./imported/remotion-effects-library/ParticleField";
import { FlipPage, FlipPageTransition } from "./imported/free-remotion-templates/FlipPageTransition";
import { FrameMorphBridge } from "./imported/remotion-transitions/FrameMorphTransition";
import { LineLedBridge } from "./imported/remotion-transitions/LineLedTransition";
import { sceneEnvelope } from "./composition/easing";

const EDITORIAL_SANS = '-apple-system, BlinkMacSystemFont, "Segoe UI", "Helvetica Neue", Arial, sans-serif';
const EDITORIAL_SERIF = 'Georgia, "Times New Roman", serif';

export interface PaletteTokens {
  ivory: string;
  espresso: string;
  forest: string;
  sage: string;
  brass: string;
  clay: string;
}

const DEFAULT_PALETTE: PaletteTokens = {
  ivory: "#F5F0E6",
  espresso: "#292822",
  forest: "#365B49",
  sage: "#A4B39A",
  brass: "#C6A46A",
  clay: "#B85D43",
};

export interface GenerativePrimitiveNode {
  primitiveId: string;
  layerRole: "background_graphics" | "foreground_graphics" | "hero_stage";
  params: Record<string, unknown>;
}

export interface PersistentLedgerState {
  active: boolean;
  currencySymbol: string;
  initialCents: number;
  prevBalanceCents: number;
  nextBalanceCents: number;
  currentCategory?: string;
  currentDeltaCents?: number;
  steps?: Array<{
    label: string;
    value: number; // dollars or units
    isTotal?: boolean;
    color?: string;
  }>;
  provenanceNote?: string;
}

export interface GenerativeSceneProps {
  sceneId: string;
  stageMode:
    | "live_depth_matte"
    | "aspect_morph_editorial"
    | "full_graphic_cutaway"
    | "tracked_macro_split"
    | "live_clean_bridge";
  width: number;
  height: number;
  fps: number;
  durationInFrames: number;
  entryFrames: number;
  exitFrames: number;
  sourceFile: string | null;
  fgAlphaFile: string | null;
  propAlphaFile: string | null;
  hfGraphicFile: string | null;
  hfLayerRole: "background_graphics" | "foreground_graphics";
  transitionIn?: "none" | "frame_morph" | "line_led" | "flip_page";
  transitionInFrames?: number;
  palette?: Partial<PaletteTokens>;
  primitives: GenerativePrimitiveNode[];
  ledger?: PersistentLedgerState;
  taskAnchor?: { cx: number; cy: number; w: number; h: number };
  faceAnchor?: { cx: number; cy: number; w: number; h: number };
  parameters?: Record<string, unknown>;
  excludeLayerRole?: string | null;
}

export const DEFAULT_GENERATIVE_SCENE_PROPS: GenerativeSceneProps = {
  sceneId: "gen_default",
  stageMode: "full_graphic_cutaway",
  width: 720,
  height: 1280,
  fps: 24,
  durationInFrames: 72,
  entryFrames: 8,
  exitFrames: 8,
  sourceFile: null,
  fgAlphaFile: null,
  propAlphaFile: null,
  hfGraphicFile: null,
  hfLayerRole: "foreground_graphics",
  transitionIn: "none",
  transitionInFrames: 12,
  palette: DEFAULT_PALETTE,
  primitives: [],
  parameters: {},
  excludeLayerRole: null,
};

function mergePalette(p?: Partial<PaletteTokens>): PaletteTokens {
  return { ...DEFAULT_PALETTE, ...(p || {}) };
}

function fmtCurrencyDollars(val: number, sym: string): string {
  const sign = val < 0 ? "-" : "";
  const abs = Math.abs(Math.round(val));
  return `${sign}${sym}${abs.toLocaleString("en-US")}`;
}

/** Subtle deterministic warm paper-grain SVG texture (3% opacity) for editorial cutaways */
const PaperGrainOverlay: React.FC<{ opacity?: number }> = ({ opacity = 0.035 }) => (
  <AbsoluteFill
    style={{
      pointerEvents: "none",
      opacity,
      backgroundImage:
        "radial-gradient(rgba(41,40,34,0.45) 1px, transparent 0), radial-gradient(rgba(198,164,106,0.35) 1px, transparent 0)",
      backgroundSize: "14px 14px, 21px 21px",
      backgroundPosition: "0 0, 7px 9px",
    }}
  />
);

/** Render a single imported MIT Remotion primitive node cleanly with phone-scale editorial typography */
const RenderImportedPrimitive: React.FC<{
  node: GenerativePrimitiveNode;
  pal: PaletteTokens;
  width: number;
  height: number;
  durationInFrames: number;
  ledger?: PersistentLedgerState;
  sourceFile?: string | null;
  taskAnchor?: { cx: number; cy: number; w: number; h: number };
}> = ({ node, pal, width, height, durationInFrames, ledger, sourceFile, taskAnchor }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const p = node.params || {};

  switch (node.primitiveId) {
    case "rem_ru_counter": {
      const fromVal = Number(p.from_value ?? (ledger ? ledger.prevBalanceCents / 100 : 0));
      const toVal = Number(p.to_value ?? (ledger ? ledger.nextBalanceCents / 100 : 1200));
      const prefix = String(p.prefix ?? ledger?.currencySymbol ?? "$");
      const suffix = String(p.suffix ?? "");
      const eyebrow = String(p.eyebrow ?? "");
      const caption = String(p.caption ?? "");
      const roll = Boolean(p.roll ?? true);
      const fontSize = Number(p.font_size ?? 116);
      const yPct = Number(p.y_pct ?? 12);
      const darkCard = Boolean(p.dark_card ?? true);
      const accent = String(p.accent_color ?? pal.brass);
      const countDur = Math.max(8, Math.min(durationInFrames - 12, Number(p.count_frames ?? 22)));
      const delayF = Math.max(0, Number(p.delay_frames ?? 3));
      const exitAt = durationInFrames > 26 ? durationInFrames - 10 : undefined;
      const cardOpacity = interpolate(frame, [0, 7, durationInFrames - 8, durationInFrames - 1], [0, 1, 1, 0], {
        extrapolateLeft: "clamp",
        extrapolateRight: "clamp",
      });

      return (
        <div
          style={{
            position: "absolute",
            left: 48,
            right: 48,
            top: `${yPct}%`,
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            opacity: cardOpacity,
          }}
        >
          <div
            style={{
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              padding: darkCard ? "26px 40px 28px" : "12px 20px",
              borderRadius: 26,
              backgroundColor: darkCard ? "rgba(41, 40, 34, 0.90)" : "transparent",
              border: darkCard ? `2.5px solid ${accent}` : "none",
              boxShadow: darkCard ? "0 22px 54px rgba(18, 16, 12, 0.46)" : "none",
            }}
          >
            {eyebrow ? (
              <div
                style={{
                  fontFamily: EDITORIAL_SANS,
                  fontSize: 26,
                  fontWeight: 800,
                  letterSpacing: "0.16em",
                  textTransform: "uppercase",
                  color: accent,
                  marginBottom: 8,
                }}
              >
                {eyebrow}
              </div>
            ) : null}
            <Counter
              from={fromVal}
              to={toVal}
              prefix={prefix}
              suffix={suffix}
              roll={roll}
              settle={true}
              delayInFrames={delayF}
              durationInFrames={countDur}
              exitAtInFrames={exitAt}
              exitInFrames={8}
              fontSize={fontSize}
              fontWeight={900}
              fontFamily={EDITORIAL_SANS}
              color={String(p.color ?? (darkCard ? pal.ivory : pal.espresso))}
            />
            {caption ? (
              <div
                style={{
                  marginTop: 10,
                  fontFamily: EDITORIAL_SANS,
                  fontSize: 28,
                  fontWeight: 700,
                  color: darkCard ? pal.sage : pal.forest,
                  textAlign: "center",
                }}
              >
                {caption}
              </div>
            ) : null}
          </div>
        </div>
      );
    }

    case "rem_ru_waterfall": {
      const sym = String(p.currency_symbol ?? ledger?.currencySymbol ?? "$");
      const rawSteps = (p.steps as WaterfallStep[] | undefined) || ledger?.steps || [
        { label: "Balance", value: 1200, isTotal: true, color: pal.espresso },
        { label: "Bills", value: -800, color: pal.clay },
        { label: "Groceries", value: -200, color: pal.brass },
        { label: "Gas", value: -100, color: pal.sage },
        { label: "Left", value: 100, isTotal: true, color: pal.forest },
      ];
      const title = String(p.title ?? "PAYCHECK WATERFALL");
      const subtitle = String(p.subtitle ?? "");
      const chartW = Math.min(width - 72, Number(p.chart_width ?? 648));
      const chartH = Number(p.chart_height ?? 430);
      const yPct = Number(p.y_pct ?? 10);
      const darkSurface = Boolean(p.dark_surface ?? false);
      const exitAt = durationInFrames > 30 ? durationInFrames - 10 : undefined;

      return (
        <div
          style={{
            position: "absolute",
            left: 36,
            right: 36,
            top: `${yPct}%`,
            padding: "28px 20px 22px",
            borderRadius: 28,
            backgroundColor: darkSurface ? "rgba(41,40,34,0.92)" : pal.ivory,
            border: `2px solid ${pal.brass}`,
            boxShadow: "0 24px 56px rgba(20,18,14,0.28)",
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
          }}
        >
          <div style={{ width: "100%", paddingLeft: 16, paddingRight: 16, marginBottom: 12, display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
            <span style={{ fontFamily: EDITORIAL_SANS, fontSize: 26, fontWeight: 800, letterSpacing: "0.14em", color: darkSurface ? pal.brass : pal.forest, textTransform: "uppercase" }}>
              {title}
            </span>
            {subtitle ? (
              <span style={{ fontFamily: EDITORIAL_SANS, fontSize: 24, fontWeight: 700, color: darkSurface ? pal.ivory : pal.espresso, fontVariantNumeric: "tabular-nums" }}>
                {subtitle}
              </span>
            ) : null}
          </div>
          <WaterfallChart
            steps={rawSteps}
            width={chartW}
            height={chartH}
            upColor={pal.forest}
            downColor={pal.clay}
            totalColor={darkSurface ? pal.brass : pal.espresso}
            gridColor={darkSurface ? "rgba(245,240,230,0.16)" : "rgba(41,40,34,0.14)"}
            labelColor={darkSurface ? "rgba(245,240,230,0.88)" : pal.espresso}
            inkColor={darkSurface ? pal.ivory : pal.espresso}
            showConnectors={true}
            showValues={true}
            showAxis={true}
            valueFormatter={(v: number) => fmtCurrencyDollars(v, sym)}
            durationInFrames={Math.max(8, Number(p.bar_frames ?? 14))}
            staggerInFrames={Math.max(3, Number(p.stagger_frames ?? 8))}
            delayInFrames={Number(p.delay_frames ?? 2)}
            exitAtInFrames={exitAt}
            exitInFrames={8}
          />
        </div>
      );
    }

    case "rem_ru_strikethrough": {
      const fromText = String(p.from_text ?? "$1,200 TO SPEND");
      const toText = String(p.to_text ?? "BALANCE ≠ SPENDABLE");
      const eyebrow = String(p.eyebrow ?? "DON'T CONFUSE THEM");
      const fontSize = Number(p.font_size ?? 54);
      const yPct = Number(p.y_pct ?? 11);
      const darkCard = Boolean(p.dark_card ?? true);
      const op = interpolate(frame, [0, 6, durationInFrames - 8, durationInFrames - 1], [0, 1, 1, 0], {
        extrapolateLeft: "clamp",
        extrapolateRight: "clamp",
      });
      return (
        <div
          style={{
            position: "absolute",
            left: 44,
            right: 44,
            top: `${yPct}%`,
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            opacity: op,
          }}
        >
          <div
            style={{
              width: "100%",
              padding: "28px 26px 32px",
              borderRadius: 26,
              backgroundColor: darkCard ? "rgba(41,40,34,0.92)" : pal.ivory,
              border: `2.5px solid ${pal.brass}`,
              boxShadow: "0 22px 54px rgba(18,16,12,0.42)",
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              gap: 12,
            }}
          >
            {eyebrow ? (
              <div style={{ fontFamily: EDITORIAL_SANS, fontSize: 24, fontWeight: 800, letterSpacing: "0.16em", color: pal.brass, textTransform: "uppercase" }}>
                {eyebrow}
              </div>
            ) : null}
            <StrikethroughReplace
              from={fromText}
              to={toText}
              delayInFrames={Number(p.delay_frames ?? 6)}
              strikeDurationInFrames={Number(p.strike_frames ?? 12)}
              holdInFrames={Number(p.hold_frames ?? 4)}
              departDurationInFrames={10}
              replaceDelayInFrames={8}
              replaceDurationInFrames={14}
              color={darkCard ? pal.ivory : pal.espresso}
              strikeColor={String(p.strike_color ?? pal.clay)}
              replaceColor={String(p.replace_color ?? pal.brass)}
              fontSize={fontSize}
              fontWeight={800}
              fontFamily={EDITORIAL_SANS}
            />
          </div>
        </div>
      );
    }

    case "rem_ru_masked_slide": {
      const lines = (p.lines as string[] | undefined) || [String(p.text ?? "Give every dollar a purpose")];
      const eyebrow = String(p.eyebrow ?? "");
      const subline = String(p.subline ?? "");
      const fontSize = Number(p.font_size ?? 58);
      const yPct = Number(p.y_pct ?? 12);
      const darkCard = Boolean(p.dark_card ?? true);
      const op = interpolate(frame, [0, 6, durationInFrames - 8, durationInFrames - 1], [0, 1, 1, 0], {
        extrapolateLeft: "clamp",
        extrapolateRight: "clamp",
      });
      return (
        <div style={{ position: "absolute", left: 48, right: 48, top: `${yPct}%`, opacity: op }}>
          <div
            style={{
              padding: darkCard ? "28px 32px" : "8px 0",
              borderRadius: 24,
              backgroundColor: darkCard ? "rgba(41,40,34,0.90)" : "transparent",
              borderLeft: darkCard ? `6px solid ${pal.brass}` : "none",
              boxShadow: darkCard ? "0 20px 50px rgba(18,16,12,0.44)" : "none",
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              gap: 12,
            }}
          >
            {eyebrow ? (
              <div style={{ fontFamily: EDITORIAL_SANS, fontSize: 24, fontWeight: 800, letterSpacing: "0.16em", color: pal.brass, textTransform: "uppercase" }}>
                {eyebrow}
              </div>
            ) : null}
            <MaskedSlideReveal
              lines={lines}
              staggerInFrames={Number(p.stagger_frames ?? 5)}
              durationInFrames={Number(p.reveal_frames ?? 14)}
              delayInFrames={Number(p.delay_frames ?? 3)}
              fontSize={fontSize}
              fontWeight={800}
              fontFamily={Boolean(p.serif ?? true) ? EDITORIAL_SERIF : EDITORIAL_SANS}
              color={String(p.color ?? (darkCard ? pal.ivory : pal.espresso))}
              textAlign="center"
            />
            {subline ? (
              <div style={{ marginTop: 6, fontFamily: EDITORIAL_SANS, fontSize: 28, fontWeight: 700, letterSpacing: "0.08em", color: pal.brass, textTransform: "uppercase" }}>
                {subline}
              </div>
            ) : null}
          </div>
        </div>
      );
    }

    case "rem_viv_particle_field": {
      const count = Math.max(12, Math.min(110, Number(p.count ?? 48)));
      const speed = Number(p.speed ?? 0.16);
      const connect = Boolean(p.connect ?? false);
      const colors = (p.colors as string[] | undefined) || [pal.brass, pal.ivory, pal.sage];
      const op = interpolate(frame, [0, 8, durationInFrames - 8, durationInFrames - 1], [0, 0.72, 0.72, 0], {
        extrapolateLeft: "clamp",
        extrapolateRight: "clamp",
      });
      return (
        <AbsoluteFill style={{ opacity: op, pointerEvents: "none" }}>
          <ParticleField
            count={count}
            speed={speed}
            connect={connect}
            colors={colors}
            linkColor={pal.brass}
            backgroundColor="transparent"
            showVignette={false}
            title=""
            subtitle=""
          />
        </AbsoluteFill>
      );
    }

    case "rem_rc_flip_page": {
      // Real 3D perspective book page flip from RenderComp/free-remotion-templates (MIT)
      const labelBefore = String(p.label_before ?? "The Next Paycheck Plan");
      const subBefore = String(p.sub_before ?? "CONCEPTUAL PRODUCT OVERVIEW");
      const labelAfter = String(p.label_after ?? "56-page workbook");
      const subAfter = String(p.sub_after ?? "+ Notion dashboard · Organize your money");
      const flipDelay = Number(p.flip_delay ?? Math.max(10, Math.round(durationInFrames * 0.28)));
      const flipDur = Number(p.flip_duration ?? Math.max(16, Math.round(durationInFrames * 0.34)));
      const boxW = Math.round(width * 0.84);
      const boxH = Math.round(height * 0.44);
      const yTop = Math.round(height * Number(p.y_frac ?? 0.085));
      const op = interpolate(frame, [0, 7, durationInFrames - 8, durationInFrames - 1], [0, 1, 1, 0], {
        extrapolateLeft: "clamp",
        extrapolateRight: "clamp",
      });

      const beforePage = (
        <AbsoluteFill
          style={{
            background: `linear-gradient(155deg, ${pal.ivory} 0%, #EBE3D3 100%)`,
            border: `3px solid ${pal.brass}`,
            borderRadius: 24,
            padding: "42px 36px",
            display: "flex",
            flexDirection: "column",
            justifyContent: "space-between",
          }}
        >
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <span style={{ fontFamily: EDITORIAL_SANS, fontSize: 20, fontWeight: 800, letterSpacing: "0.14em", color: pal.forest, textTransform: "uppercase" }}>
              {subBefore}
            </span>
            <span style={{ width: 16, height: 16, borderRadius: 8, backgroundColor: pal.brass }} />
          </div>
          <div>
            <div style={{ fontFamily: EDITORIAL_SERIF, fontSize: 54, fontWeight: 700, lineHeight: 1.1, color: pal.espresso }}>
              {labelBefore}
            </div>
            <div style={{ marginTop: 20, height: 5, width: 140, borderRadius: 3, backgroundColor: pal.brass }} />
          </div>
          <div style={{ display: "flex", gap: 14 }}>
            <div style={{ flex: 1, height: 14, borderRadius: 7, backgroundColor: "rgba(41,40,34,0.14)" }} />
            <div style={{ flex: 2, height: 14, borderRadius: 7, backgroundColor: "rgba(54,91,73,0.24)" }} />
          </div>
        </AbsoluteFill>
      );

      const afterPage = (
        <AbsoluteFill
          style={{
            background: `linear-gradient(155deg, ${pal.espresso} 0%, #1F1E19 100%)`,
            border: `3px solid ${pal.brass}`,
            borderRadius: 24,
            padding: "38px 36px",
            display: "flex",
            flexDirection: "column",
            justifyContent: "space-between",
          }}
        >
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <span style={{ fontFamily: EDITORIAL_SANS, fontSize: 20, fontWeight: 800, letterSpacing: "0.14em", color: pal.brass, textTransform: "uppercase" }}>
              CONCEPTUAL SCHEMATIC
            </span>
            <span style={{ fontFamily: EDITORIAL_SANS, fontSize: 22, fontWeight: 800, color: pal.sage }}>01 / 56</span>
          </div>
          <div>
            <div style={{ fontFamily: EDITORIAL_SERIF, fontSize: 50, fontWeight: 700, lineHeight: 1.12, color: pal.ivory }}>
              {labelAfter}
            </div>
            <div style={{ marginTop: 14, fontFamily: EDITORIAL_SANS, fontSize: 28, fontWeight: 600, color: pal.sage }}>
              {subAfter}
            </div>
          </div>
          {/* Clean abstract conceptual ledger rows (explicitly NOT fake Notion UI!) */}
          <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
            {[0.82, 0.54, 0.36].map((frac, idx) => (
              <div key={idx} style={{ display: "flex", alignItems: "center", gap: 12 }}>
                <div style={{ width: 16, height: 16, borderRadius: 4, backgroundColor: idx === 0 ? pal.brass : pal.sage }} />
                <div style={{ flex: 1, height: 12, borderRadius: 6, backgroundColor: "rgba(245,240,230,0.16)", overflow: "hidden" }}>
                  <div style={{ width: `${Math.round(frac * 100)}%`, height: "100%", backgroundColor: idx === 0 ? pal.brass : pal.forest }} />
                </div>
              </div>
            ))}
          </div>
        </AbsoluteFill>
      );

      return (
        <div
          style={{
            position: "absolute",
            left: Math.round((width - boxW) / 2),
            top: yTop,
            width: boxW,
            height: boxH,
            opacity: op,
            borderRadius: 24,
            overflow: "hidden",
            boxShadow: "0 28px 64px rgba(15,13,10,0.48)",
          }}
        >
          <FlipPageTransition
            colorBefore={pal.ivory}
            colorAfter={pal.espresso}
            labelBefore={labelBefore}
            labelAfter={labelAfter}
            textColor={pal.espresso}
            flipDelay={flipDelay}
            flipDuration={flipDur}
            beforeContent={beforePage}
            afterContent={afterPage}
          />
        </div>
      );
    }

    case "rem_ru_track_matte": {
      // Real SVG luminance/alpha TrackMatte from riaz37/remotion-ui (MIT) revealing a live macro crop!
      if (!sourceFile) return null;
      const boxW = Math.round(width * 0.44);
      const boxH = Math.round(height * 0.23);
      const xPos = Math.round(width * Number(p.x_frac ?? 0.51));
      const yPos = Math.round(height * Number(p.y_frac ?? 0.50));
      const zoom = Number(p.detail_zoom ?? 1.6);
      const cx = taskAnchor?.cx ?? 0.5;
      const cy = Math.min(0.86, (taskAnchor?.cy ?? 0.72) + 0.06);
      const scaledW = width * zoom;
      const scaledH = height * zoom;
      const offX = Math.min(0, Math.max(boxW - scaledW, boxW / 2 - cx * scaledW));
      const offY = Math.min(0, Math.max(boxH - scaledH, boxH / 2 - cy * scaledH));
      const revealP = spring({ frame, fps, config: { damping: 18, stiffness: 150 }, durationInFrames: 16 });
      const radius = Math.round(Math.min(boxW, boxH) * 0.48 * revealP);
      const label = String(p.label ?? "LIVE DETAIL");

      return (
        <div
          style={{
            position: "absolute",
            left: xPos,
            top: yPos,
            width: boxW,
            height: boxH,
            filter: "drop-shadow(0 18px 34px rgba(18,16,12,0.48))",
          }}
        >
          <TrackMatte
            mode="alpha"
            width={boxW}
            height={boxH}
            matte={
              <rect
                x={Math.round((boxW - boxW * revealP) / 2)}
                y={Math.round((boxH - boxH * revealP) / 2)}
                width={Math.max(4, Math.round(boxW * revealP))}
                height={Math.max(4, Math.round(boxH * revealP))}
                rx={Math.min(34, radius)}
                fill="white"
              />
            }
          >
            <div style={{ position: "relative", width: boxW, height: boxH, overflow: "hidden", backgroundColor: pal.espresso }}>
              <div style={{ position: "absolute", left: offX, top: offY, width: scaledW, height: scaledH }}>
                <OffthreadVideo src={staticFile(sourceFile)} muted style={{ width: scaledW, height: scaledH, objectFit: "cover" }} />
              </div>
              <div style={{ position: "absolute", inset: 0, borderRadius: 34, border: `3px solid ${pal.brass}`, pointerEvents: "none" }} />
              {label ? (
                <div
                  style={{
                    position: "absolute",
                    left: 14,
                    bottom: 14,
                    padding: "6px 14px",
                    borderRadius: 999,
                    backgroundColor: "rgba(41,40,34,0.88)",
                    color: pal.ivory,
                    fontFamily: EDITORIAL_SANS,
                    fontSize: 20,
                    fontWeight: 800,
                    letterSpacing: "0.08em",
                  }}
                >
                  {label}
                </div>
              ) : null}
            </div>
          </TrackMatte>
        </div>
      );
    }

    default:
      return null;
  }
};

/** Persistent bottom-or-top compact running balance ribbon bridging consecutive ledger scenes */
const PersistentLedgerRibbon: React.FC<{ ledger: PersistentLedgerState; pal: PaletteTokens; durationInFrames: number }> = ({
  ledger,
  pal,
  durationInFrames,
}) => {
  const frame = useCurrentFrame();
  if (!ledger.active || ledger.initialCents <= 0) return null;
  const pctPrev = Math.max(0, Math.min(100, (ledger.prevBalanceCents / ledger.initialCents) * 100));
  const pctNext = Math.max(0, Math.min(100, (ledger.nextBalanceCents / ledger.initialCents) * 100));
  const prog = interpolate(frame, [4, Math.min(24, durationInFrames - 6)], [0, 1], {
    easing: Easing.out(Easing.cubic),
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const curPct = pctPrev + (pctNext - pctPrev) * prog;
  const op = interpolate(frame, [0, 6, durationInFrames - 6, durationInFrames - 1], [0, 1, 1, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  return (
    <div
      style={{
        position: "absolute",
        left: 48,
        right: 48,
        bottom: 205,
        opacity: op,
        padding: "16px 22px",
        borderRadius: 18,
        backgroundColor: "rgba(41,40,34,0.90)",
        border: `2px solid ${pal.brass}`,
        boxShadow: "0 14px 36px rgba(15,13,10,0.42)",
        fontFamily: EDITORIAL_SANS,
      }}
    >
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 10 }}>
        <span style={{ fontSize: 22, fontWeight: 800, letterSpacing: "0.1em", color: pal.brass, textTransform: "uppercase" }}>
          {ledger.currentCategory ? `AFTER ${ledger.currentCategory}` : "REMAINING BALANCE"}
        </span>
        <Counter
          from={ledger.prevBalanceCents / 100}
          to={ledger.nextBalanceCents / 100}
          prefix={ledger.currencySymbol}
          roll={true}
          settle={true}
          delayInFrames={4}
          durationInFrames={Math.min(20, Math.max(8, durationInFrames - 12))}
          fontSize={42}
          fontWeight={900}
          fontFamily={EDITORIAL_SANS}
          color={pal.ivory}
        />
      </div>
      <div style={{ width: "100%", height: 12, borderRadius: 6, backgroundColor: "rgba(245,240,230,0.18)", overflow: "hidden" }}>
        <div
          style={{
            width: `${curPct.toFixed(2)}%`,
            height: "100%",
            borderRadius: 6,
            backgroundColor: curPct <= 15 ? pal.clay : pal.forest,
          }}
        />
      </div>
    </div>
  );
};

export const MotionLabGenerativeScene: React.FC<GenerativeSceneProps> = (props) => {
  const frame = useCurrentFrame();
  const { width, height, durationInFrames } = useVideoConfig();
  const pal = mergePalette(props.palette);
  const env = sceneEnvelope(frame, durationInFrames, props.entryFrames, props.exitFrames);
  const params = props.parameters || {};
  const ex = props.excludeLayerRole || null;

  const bgNodes = (props.primitives || []).filter((n) => n.layerRole === "background_graphics");
  const fgNodes = (props.primitives || []).filter((n) => n.layerRole !== "background_graphics");

  const showHfBg = Boolean(props.hfGraphicFile) && props.hfLayerRole === "background_graphics" && ex !== "background_graphics";
  const showHfFg = Boolean(props.hfGraphicFile) && props.hfLayerRole === "foreground_graphics" && ex !== "foreground_graphics";
  const showFgMatte = Boolean(props.fgAlphaFile) && ex !== "foreground_subject_matte";
  const showProp = Boolean(props.propAlphaFile) && ex !== "prop_cutout_matte";
  const showRemPrims = ex !== "remotion_primitive";

  let sceneContent: React.ReactNode;

  if (props.stageMode === "aspect_morph_editorial" && props.sourceFile) {
    // REAL imported AspectMorphCard (vivmagarwal/remotion-effects-library, MIT)!
    // Live footage smoothly morphs its layout rectangle downward with objectFit: cover (zero squash, zero duplicate blurry card!)
    const kicker = String(params.kicker ?? "");
    const lines = (params.lines as string[] | undefined) || [];
    const restRect = { x: 0, y: 0, w: width, h: height };
    const shiftedRect = {
      x: 42,
      y: Math.round(height * Number(params.shifted_y_frac ?? 0.50)),
      w: width - 84,
      h: Math.round(height * Number(params.shifted_h_frac ?? 0.40)),
    };
    const holdF = Math.max(12, durationInFrames - 22);
    sceneContent = (
      <AbsoluteFill style={{ backgroundColor: pal.ivory }}>
        <PaperGrainOverlay />
        <AspectMorphCard
          title={String(params.header_title ?? "")}
          paperColor={pal.ivory}
          accentColor={pal.forest}
          rest={restRect}
          shifted={shiftedRect}
          morphFrames={12}
          showReadout={false}
          chipLabel={String(params.chip_label ?? "")}
          beats={[{ at: 12, hold: holdF, kicker, lines }]}
          cardContent={
            <OffthreadVideo
              src={staticFile(props.sourceFile)}
              muted
              style={{ width: "100%", height: "100%", objectFit: "cover", objectPosition: "50% 24%" }}
            />
          }
        />
        {showRemPrims
          ? fgNodes.map((n, i) => (
              <RenderImportedPrimitive
                key={`${n.primitiveId}_${i}`}
                node={n}
                pal={pal}
                width={width}
                height={height}
                durationInFrames={durationInFrames}
                ledger={props.ledger}
                sourceFile={props.sourceFile}
                taskAnchor={props.taskAnchor}
              />
            ))
          : null}
        {showHfFg && props.hfGraphicFile ? (
          <AbsoluteFill style={{ pointerEvents: "none" }}>
            <OffthreadVideo src={staticFile(props.hfGraphicFile)} transparent muted style={{ width: "100%", height: "100%", objectFit: "cover" }} />
          </AbsoluteFill>
        ) : null}
      </AbsoluteFill>
    );
  } else if (props.stageMode === "full_graphic_cutaway") {
    const bgColor = String(params.bg_color ?? pal.ivory);
    sceneContent = (
      <AbsoluteFill style={{ backgroundColor: bgColor, overflow: "hidden" }}>
        <PaperGrainOverlay opacity={0.04} />
        {showRemPrims
          ? bgNodes.map((n, i) => (
              <RenderImportedPrimitive
                key={`bg_${n.primitiveId}_${i}`}
                node={n}
                pal={pal}
                width={width}
                height={height}
                durationInFrames={durationInFrames}
                ledger={props.ledger}
                sourceFile={props.sourceFile}
                taskAnchor={props.taskAnchor}
              />
            ))
          : null}
        {showHfBg && props.hfGraphicFile ? (
          <AbsoluteFill>
            <OffthreadVideo src={staticFile(props.hfGraphicFile)} transparent muted style={{ width: "100%", height: "100%", objectFit: "cover" }} />
          </AbsoluteFill>
        ) : null}
        {showRemPrims
          ? fgNodes.map((n, i) => (
              <RenderImportedPrimitive
                key={`fg_${n.primitiveId}_${i}`}
                node={n}
                pal={pal}
                width={width}
                height={height}
                durationInFrames={durationInFrames}
                ledger={props.ledger}
                sourceFile={props.sourceFile}
                taskAnchor={props.taskAnchor}
              />
            ))
          : null}
        {showHfFg && props.hfGraphicFile ? (
          <AbsoluteFill>
            <OffthreadVideo src={staticFile(props.hfGraphicFile)} transparent muted style={{ width: "100%", height: "100%", objectFit: "cover" }} />
          </AbsoluteFill>
        ) : null}
      </AbsoluteFill>
    );
  } else if (props.stageMode === "live_depth_matte" && props.sourceFile) {
    // REAL imported ParallaxLayers (riaz37/remotion-ui, MIT) with MODNet foreground occlusion!
    const bgDim = 1 - (1 - Number(params.bg_dim ?? 0.78)) * env;
    const bgBlur = Math.min(3.5, Number(params.bg_blur_px ?? 2.2)) * env;
    const travel = Number(params.parallax_travel_px ?? 26);

    const layers = [
      {
        depth: 0.25,
        blur: bgBlur,
        content: (
          <AbsoluteFill style={{ filter: `brightness(${bgDim.toFixed(3)}) saturate(0.94)` }}>
            <OffthreadVideo src={staticFile(props.sourceFile)} muted style={{ width: "100%", height: "100%", objectFit: "cover" }} />
          </AbsoluteFill>
        ),
      },
      ...(showRemPrims && bgNodes.length > 0
        ? [
            {
              depth: 0.55,
              content: (
                <AbsoluteFill>
                  {bgNodes.map((n, i) => (
                    <RenderImportedPrimitive
                      key={`bg_${n.primitiveId}_${i}`}
                      node={n}
                      pal={pal}
                      width={width}
                      height={height}
                      durationInFrames={durationInFrames}
                      ledger={props.ledger}
                      sourceFile={props.sourceFile}
                      taskAnchor={props.taskAnchor}
                    />
                  ))}
                </AbsoluteFill>
              ),
            },
          ]
        : []),
      ...(showHfBg && props.hfGraphicFile
        ? [
            {
              depth: 0.62,
              content: (
                <AbsoluteFill>
                  <OffthreadVideo src={staticFile(props.hfGraphicFile)} transparent muted style={{ width: "100%", height: "100%", objectFit: "cover" }} />
                </AbsoluteFill>
              ),
            },
          ]
        : []),
      ...(showFgMatte && props.fgAlphaFile
        ? [
            {
              depth: 0.95,
              content: (
                <AbsoluteFill style={{ filter: `drop-shadow(0 10px 22px rgba(20,16,12,${(0.32 * env).toFixed(3)}))` }}>
                  <OffthreadVideo src={staticFile(props.fgAlphaFile)} transparent muted style={{ width: "100%", height: "100%", objectFit: "cover" }} />
                </AbsoluteFill>
              ),
            },
          ]
        : []),
    ];

    sceneContent = (
      <AbsoluteFill style={{ backgroundColor: pal.espresso, overflow: "hidden" }}>
        <ParallaxLayers layers={layers} travel={travel} angle={12} zoom={0.032} backgroundColor={pal.espresso} />
        {showRemPrims
          ? fgNodes.map((n, i) => (
              <RenderImportedPrimitive
                key={`fg_${n.primitiveId}_${i}`}
                node={n}
                pal={pal}
                width={width}
                height={height}
                durationInFrames={durationInFrames}
                ledger={props.ledger}
                sourceFile={props.sourceFile}
                taskAnchor={props.taskAnchor}
              />
            ))
          : null}
        {showHfFg && props.hfGraphicFile ? (
          <AbsoluteFill style={{ pointerEvents: "none" }}>
            <OffthreadVideo src={staticFile(props.hfGraphicFile)} transparent muted style={{ width: "100%", height: "100%", objectFit: "cover" }} />
          </AbsoluteFill>
        ) : null}
        {props.ledger?.active && Boolean(params.show_persistent_ribbon) ? (
          <PersistentLedgerRibbon ledger={props.ledger} pal={pal} durationInFrames={durationInFrames} />
        ) : null}
      </AbsoluteFill>
    );
  } else {
    // tracked_macro_split OR live_clean_bridge
    const liftY = -22 * env;
    const liftScale = 1 + 0.14 * env;
    sceneContent = (
      <AbsoluteFill style={{ backgroundColor: pal.espresso, overflow: "hidden" }}>
        {props.sourceFile ? (
          <AbsoluteFill>
            <OffthreadVideo src={staticFile(props.sourceFile)} muted style={{ width: "100%", height: "100%", objectFit: "cover" }} />
          </AbsoluteFill>
        ) : null}
        {showProp && props.propAlphaFile ? (
          <AbsoluteFill
            style={{
              transform: `translateY(${liftY.toFixed(1)}px) scale(${liftScale.toFixed(4)})`,
              transformOrigin: `${Math.round((props.taskAnchor?.cx ?? 0.5) * 100)}% ${Math.round((props.taskAnchor?.cy ?? 0.72) * 100)}%`,
              filter: `drop-shadow(0 18px 28px rgba(16,13,10,${(0.45 * env).toFixed(3)}))`,
            }}
          >
            <OffthreadVideo src={staticFile(props.propAlphaFile)} transparent muted style={{ width: "100%", height: "100%", objectFit: "cover" }} />
          </AbsoluteFill>
        ) : null}
        {showRemPrims
          ? [...bgNodes, ...fgNodes].map((n, i) => (
              <RenderImportedPrimitive
                key={`n_${n.primitiveId}_${i}`}
                node={n}
                pal={pal}
                width={width}
                height={height}
                durationInFrames={durationInFrames}
                ledger={props.ledger}
                sourceFile={props.sourceFile}
                taskAnchor={props.taskAnchor}
              />
            ))
          : null}
        {showHfFg && props.hfGraphicFile ? (
          <AbsoluteFill style={{ pointerEvents: "none" }}>
            <OffthreadVideo src={staticFile(props.hfGraphicFile)} transparent muted style={{ width: "100%", height: "100%", objectFit: "cover" }} />
          </AbsoluteFill>
        ) : null}
        {props.ledger?.active && Boolean(params.show_persistent_ribbon) ? (
          <PersistentLedgerRibbon ledger={props.ledger} pal={pal} durationInFrames={durationInFrames} />
        ) : null}
      </AbsoluteFill>
    );
  }

  // Optional timeline-preserving entrance transition bridge (GTKottman FrameMorph / LineLed or FlipPage)
  const trFrames = Math.max(4, Math.min(18, Number(props.transitionInFrames ?? 12)));
  if (props.transitionIn && props.transitionIn !== "none" && frame < trFrames && props.sourceFile) {
    const trP = frame / Math.max(1, trFrames - 1);
    const livePrev = (
      <AbsoluteFill>
        <OffthreadVideo src={staticFile(props.sourceFile)} muted style={{ width: "100%", height: "100%", objectFit: "cover" }} />
      </AbsoluteFill>
    );
    if (props.transitionIn === "frame_morph") {
      return <FrameMorphBridge progress={trP} background={pal.espresso} size={0.62} exiting={livePrev} entering={sceneContent} />;
    }
    if (props.transitionIn === "line_led") {
      return <LineLedBridge progress={trP} color={pal.brass} thickness={6} orientation="horizontal" exiting={livePrev} entering={sceneContent} />;
    }
    if (props.transitionIn === "flip_page") {
      return (
        <AbsoluteFill>
          {sceneContent}
          <FlipPage mode="exit" direction="left" delay={0} duration={trFrames}>
            {livePrev}
          </FlipPage>
        </AbsoluteFill>
      );
    }
  }

  return <AbsoluteFill>{sceneContent}</AbsoluteFill>;
};

/** Upstream Smoke Test Composition for A2 proof of every single imported MIT Remotion mechanism */
export interface UpstreamSmokeProps {
  primitiveId: string;
  width: number;
  height: number;
  fps: number;
  durationInFrames: number;
}

export const MotionLabUpstreamSmoke: React.FC<UpstreamSmokeProps> = ({ primitiveId, width, height, durationInFrames }) => {
  const frame = useCurrentFrame();
  const p = frame / Math.max(1, durationInFrames - 1);
  const pal = DEFAULT_PALETTE;

  switch (primitiveId) {
    case "rem_ru_counter":
      return (
        <AbsoluteFill style={{ backgroundColor: pal.espresso, alignItems: "center", justifyContent: "center" }}>
          <Counter from={0} to={1200} prefix="$" roll={true} durationInFrames={28} fontSize={120} color={pal.ivory} fontFamily={EDITORIAL_SANS} />
        </AbsoluteFill>
      );
    case "rem_ru_waterfall":
      return (
        <AbsoluteFill style={{ backgroundColor: pal.ivory, alignItems: "center", justifyContent: "center", padding: 24 }}>
          <WaterfallChart
            width={width - 48}
            height={440}
            steps={[
              { label: "Start", value: 1200, isTotal: true, color: pal.espresso },
              { label: "Bills", value: -800, color: pal.clay },
              { label: "Food", value: -200, color: pal.brass },
              { label: "Gas", value: -100, color: pal.sage },
              { label: "Left", value: 100, isTotal: true, color: pal.forest },
            ]}
            upColor={pal.forest}
            downColor={pal.clay}
            totalColor={pal.espresso}
            labelColor={pal.espresso}
            inkColor={pal.espresso}
            valueFormatter={(v: number) => `$${Math.round(v).toLocaleString("en-US")}`}
            durationInFrames={14}
            staggerInFrames={7}
          />
        </AbsoluteFill>
      );
    case "rem_ru_strikethrough":
      return (
        <AbsoluteFill style={{ backgroundColor: pal.espresso, alignItems: "center", justifyContent: "center" }}>
          <StrikethroughReplace from="$1,200 TO SPEND" to="ONLY $100 LEFT" color={pal.ivory} strikeColor={pal.clay} replaceColor={pal.brass} fontSize={54} fontFamily={EDITORIAL_SANS} />
        </AbsoluteFill>
      );
    case "rem_ru_masked_slide":
      return (
        <AbsoluteFill style={{ backgroundColor: pal.ivory, alignItems: "center", justifyContent: "center", padding: 48 }}>
          <MaskedSlideReveal lines={["Give every dollar", "a purpose"]} fontSize={68} color={pal.espresso} fontFamily={EDITORIAL_SERIF} />
        </AbsoluteFill>
      );
    case "rem_ru_track_matte":
      return (
        <AbsoluteFill style={{ backgroundColor: pal.espresso, alignItems: "center", justifyContent: "center" }}>
          <TrackMatte mode="alpha" width={540} height={540} matte={<circle cx={270} cy={270} r={Math.round(60 + 190 * p)} fill="white" />}>
            <div style={{ width: 540, height: 540, background: `linear-gradient(135deg, ${pal.forest}, ${pal.brass})`, display: "grid", placeItems: "center", color: pal.ivory, fontSize: 56, fontWeight: 900, fontFamily: EDITORIAL_SANS }}>
              TRACK MATTE
            </div>
          </TrackMatte>
        </AbsoluteFill>
      );
    case "rem_ru_parallax":
      return (
        <ParallaxLayers
          travel={140}
          zoom={0.1}
          backgroundColor={pal.espresso}
          layers={[
            { depth: 0.2, content: <AbsoluteFill style={{ background: `radial-gradient(circle at 50% 45%, ${pal.forest}, ${pal.espresso})` }} /> },
            { depth: 0.6, content: <AbsoluteFill style={{ alignItems: "center", justifyContent: "center" }}><div style={{ width: 420, height: 420, borderRadius: 210, border: `6px solid ${pal.brass}`, opacity: 0.65 }} /></AbsoluteFill> },
            { depth: 1.0, content: <AbsoluteFill style={{ alignItems: "center", justifyContent: "center", color: pal.ivory, fontSize: 64, fontWeight: 800, fontFamily: EDITORIAL_SERIF }}>2.5D DEPTH</AbsoluteFill> },
          ]}
        />
      );
    case "rem_viv_aspect_morph":
      return (
        <AspectMorphCard
          title="EDITORIAL MORPH"
          paperColor={pal.ivory}
          accentColor={pal.forest}
          rest={{ x: 40, y: 240, w: width - 80, h: 720 }}
          shifted={{ x: 64, y: 740, w: width - 128, h: 360 }}
          beats={[{ at: 18, hold: 30, kicker: "REFRAME", lines: ["Layout rect", "morphs cleanly"] }]}
        />
      );
    case "rem_viv_particle_field":
      return <ParticleField count={72} colors={[pal.brass, pal.ivory, pal.sage]} backgroundColor={pal.espresso} title="DEPTH" subtitle="closed-form particles" />;
    case "rem_rc_flip_page":
      return <FlipPageTransition colorBefore={pal.ivory} colorAfter={pal.forest} labelBefore="PLAN" labelAfter="56p" textColor={pal.espresso} flipDelay={12} flipDuration={28} />;
    case "rem_gtk_frame_morph":
      return (
        <FrameMorphBridge
          progress={p}
          background={pal.espresso}
          exiting={<AbsoluteFill style={{ backgroundColor: pal.forest, alignItems: "center", justifyContent: "center", color: pal.ivory, fontSize: 64, fontWeight: 800, fontFamily: EDITORIAL_SANS }}>SCENE A</AbsoluteFill>}
          entering={<AbsoluteFill style={{ backgroundColor: pal.ivory, alignItems: "center", justifyContent: "center", color: pal.espresso, fontSize: 64, fontWeight: 800, fontFamily: EDITORIAL_SANS }}>SCENE B</AbsoluteFill>}
        />
      );
    case "rem_gtk_line_led":
      return (
        <LineLedBridge
          progress={p}
          color={pal.brass}
          thickness={8}
          exiting={<AbsoluteFill style={{ backgroundColor: pal.espresso, alignItems: "center", justifyContent: "center", color: pal.ivory, fontSize: 64, fontWeight: 800, fontFamily: EDITORIAL_SANS }}>BEFORE</AbsoluteFill>}
          entering={<AbsoluteFill style={{ backgroundColor: pal.forest, alignItems: "center", justifyContent: "center", color: pal.ivory, fontSize: 64, fontWeight: 800, fontFamily: EDITORIAL_SANS }}>AFTER</AbsoluteFill>}
        />
      );
    default:
      return <AbsoluteFill style={{ backgroundColor: pal.espresso }} />;
  }
};

