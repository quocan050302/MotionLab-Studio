import React from "react";
import { Composition } from "remotion";
import {
  MotionLabOverlay,
  MotionLabOverlayProps,
  DEFAULT_OVERLAY_PROPS,
} from "./OverlayComposition";

export const RemotionRoot: React.FC = () => {
  return (
    <Composition
      id="MotionLabOverlay"
      component={MotionLabOverlay}
      durationInFrames={60}
      fps={30}
      width={1080}
      height={1920}
      defaultProps={DEFAULT_OVERLAY_PROPS}
      calculateMetadata={({ props }) => {
        const p = props as MotionLabOverlayProps;
        return {
          durationInFrames: Math.max(2, Math.round(p.durationInFrames || 60)),
          fps: Math.max(1, Math.round(p.fps || 30)),
          width: Math.max(64, Math.round(p.width || 1080)),
          height: Math.max(64, Math.round(p.height || 1920)),
        };
      }}
    />
  );
};
