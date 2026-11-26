import React from "react";
import { Composition } from "remotion";
import { MotionLabOverlay, OverlayProps } from "./OverlayComposition";

export const RemotionRoot: React.FC = () => {
  return (
    <Composition
      id="MotionLabOverlay"
      component={MotionLabOverlay}
      durationInFrames={60}
      fps={30}
      width={1080}
      height={1920}
      defaultProps={{
        effectId: "G01",
        width: 1080,
        height: 1920,
        fps: 30,
        durationInFrames: 60,
        parameters: {
          intensity: 0.85,
          opacity: 0.65,
          margin_frac: 0.04,
          accent_hex: "#7CE0FF",
        },
        audioEnvelope: [],
      }}
      calculateMetadata={({ props }: { props: OverlayProps }) => {
        return {
          durationInFrames: Math.max(2, Math.round(props.durationInFrames || 60)),
          fps: Math.max(1, Math.round(props.fps || 30)),
          width: Math.max(64, Math.round(props.width || 1080)),
          height: Math.max(64, Math.round(props.height || 1920)),
        };
      }}
    />
  );
};
