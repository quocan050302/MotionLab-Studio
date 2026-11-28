import React from "react";
import { Composition } from "remotion";
import {
  MotionLabOverlay,
  MotionLabOverlayProps,
  DEFAULT_OVERLAY_PROPS,
} from "./OverlayComposition";
import { MotionLabCreativeScene } from "./CreativeSceneComposition";
import type { CreativeSceneProps } from "./composition/schema";
import {
  MotionLabGenerativeScene,
  MotionLabUpstreamSmoke,
  DEFAULT_GENERATIVE_SCENE_PROPS,
  type GenerativeSceneProps,
  type UpstreamSmokeProps,
} from "./GenerativeSceneComposition";
import {
  SceneGraphV4Composition,
  DEFAULT_SCENE_V4_PROPS,
  type SceneGraphV4Props,
} from "./SceneGraphV4Composition";

const DEFAULT_SCENE_PROPS: CreativeSceneProps = {
  sceneId: "scene_default",
  recipeId: "RCP01",
  sceneKind: "depth_stage",
  width: 720,
  height: 1280,
  fps: 24,
  durationInFrames: 48,
  entryFrames: 10,
  exitFrames: 10,
  sourceFile: "scene_source.mp4",
  fgAlphaFile: null,
  propAlphaFile: null,
  hfGraphicFile: null,
  anchors: {
    start_frame: 0,
    end_frame_exclusive: 48,
    face_mid: { t: 0.5, cx: 0.5, cy: 0.34, w: 0.3, h: 0.2, conf: 0.9 },
    task_mid: { t: 0.5, cx: 0.5, cy: 0.68, w: 0.36, h: 0.22, conf: 0.85 },
    face_path: [],
    task_path: [],
    action_vector: { origin_x: 0.42, origin_y: 0.72, dest_x: 0.60, dest_y: 0.64, confidence: 0.8 },
    detail_slot: { side: "left", cx: 0.27, cy: 0.19, w: 0.42, h: 0.26 },
    behind_stage: { cx: 0.5, cy: 0.38, radius_w: 0.44, radius_h: 0.28 },
  },
  parameters: {},
  excludeLayerRole: null,
};

export const RemotionRoot: React.FC = () => {
  return (
    <>
      <Composition
        id="MotionLabOverlay"
        component={MotionLabOverlay as unknown as React.FC<Record<string, unknown>>}
        durationInFrames={60}
        fps={30}
        width={1080}
        height={1920}
        defaultProps={DEFAULT_OVERLAY_PROPS as unknown as Record<string, unknown>}
        calculateMetadata={({ props }) => {
          const p = props as unknown as MotionLabOverlayProps;
          return {
            durationInFrames: Math.max(2, Math.round(p.durationInFrames || 60)),
            fps: Math.max(1, Math.round(p.fps || 30)),
            width: Math.max(64, Math.round(p.width || 1080)),
            height: Math.max(64, Math.round(p.height || 1920)),
          };
        }}
      />
      <Composition
        id="MotionLabCreativeScene"
        component={MotionLabCreativeScene as unknown as React.FC<Record<string, unknown>>}
        durationInFrames={48}
        fps={24}
        width={720}
        height={1280}
        defaultProps={DEFAULT_SCENE_PROPS as unknown as Record<string, unknown>}
        calculateMetadata={({ props }) => {
          const p = props as unknown as CreativeSceneProps;
          return {
            durationInFrames: Math.max(2, Math.round(p.durationInFrames || 48)),
            fps: Math.max(1, Math.round(p.fps || 24)),
            width: Math.max(64, Math.round(p.width || 720)),
            height: Math.max(64, Math.round(p.height || 1280)),
          };
        }}
      />
      <Composition
        id="MotionLabGenerativeScene"
        component={MotionLabGenerativeScene as unknown as React.FC<Record<string, unknown>>}
        durationInFrames={72}
        fps={24}
        width={720}
        height={1280}
        defaultProps={DEFAULT_GENERATIVE_SCENE_PROPS as unknown as Record<string, unknown>}
        calculateMetadata={({ props }) => {
          const p = props as unknown as GenerativeSceneProps;
          return {
            durationInFrames: Math.max(2, Math.round(p.durationInFrames || 72)),
            fps: Math.max(1, Math.round(p.fps || 24)),
            width: Math.max(64, Math.round(p.width || 720)),
            height: Math.max(64, Math.round(p.height || 1280)),
          };
        }}
      />
      <Composition
        id="MotionLabUpstreamSmoke"
        component={MotionLabUpstreamSmoke as unknown as React.FC<Record<string, unknown>>}
        durationInFrames={48}
        fps={24}
        width={720}
        height={1280}
        defaultProps={{ primitiveId: "rem_ru_counter", width: 720, height: 1280, fps: 24, durationInFrames: 48 }}
        calculateMetadata={({ props }) => {
          const p = props as unknown as UpstreamSmokeProps;
          return {
            durationInFrames: Math.max(2, Math.round(p.durationInFrames || 48)),
            fps: Math.max(1, Math.round(p.fps || 24)),
            width: Math.max(64, Math.round(p.width || 720)),
            height: Math.max(64, Math.round(p.height || 1280)),
          };
        }}
      />
      <Composition
        id="SceneGraphV4Composition"
        component={SceneGraphV4Composition as unknown as React.FC<Record<string, unknown>>}
        durationInFrames={60}
        fps={30}
        width={1080}
        height={1920}
        defaultProps={DEFAULT_SCENE_V4_PROPS as unknown as Record<string, unknown>}
        calculateMetadata={({ props }) => {
          const p = props as unknown as SceneGraphV4Props;
          return {
            durationInFrames: Math.max(2, Math.round(p.durationInFrames || 60)),
            fps: Math.max(1, Math.round(p.fps || 30)),
            width: Math.max(64, Math.round(p.width || 1080)),
            height: Math.max(64, Math.round(p.height || 1920)),
          };
        }}
      />
    </>
  );
};
