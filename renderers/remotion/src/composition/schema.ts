export interface AnchorPoint {
  t: number;
  cx: number;
  cy: number;
  w: number;
  h: number;
  conf: number;
}

export interface DetailSlot {
  side: string;
  cx: number;
  cy: number;
  w: number;
  h: number;
}

export interface BehindStageAnchor {
  cx: number;
  cy: number;
  radius_w: number;
  radius_h: number;
}

export interface ActionVector {
  origin_x: number;
  origin_y: number;
  dest_x: number;
  dest_y: number;
  confidence: number;
}

export interface AnchorBundle {
  start_frame: number;
  end_frame_exclusive: number;
  face_mid: AnchorPoint;
  task_mid: AnchorPoint;
  face_path: AnchorPoint[];
  task_path: AnchorPoint[];
  action_vector: ActionVector;
  detail_slot: DetailSlot;
  behind_stage: BehindStageAnchor;
}

export type SceneKind =
  | "depth_stage"
  | "detail_aperture"
  | "action_path"
  | "layout_morph"
  | "split_focus"
  | "contour_focus"
  | "vfx_depth_stage"
  | "vfx_prop_popout"
  | "vfx_macro_action"
  | "vfx_action_flow"
  | "vfx_benefit_stage"
  | "vfx_shape_morph"
  | "vfx_occlusion_wipe"
  | "vfx_editorial_collage"
  | "vfx_spatial_25d"
  | "vfx_motion_trail"
  | "vfx_payoff_accent"
  | "vfx_closing_return";

export interface CreativeSceneProps {
  sceneId: string;
  recipeId: string;
  sceneKind: SceneKind;
  width: number;
  height: number;
  fps: number;
  durationInFrames: number;
  entryFrames: number;
  exitFrames: number;
  sourceFile: string;
  fgAlphaFile: string | null;
  propAlphaFile?: string | null;
  hfGraphicFile: string | null;
  anchors: AnchorBundle;
  parameters: Record<string, unknown>;
  excludeLayerRole?: string | null;
}
