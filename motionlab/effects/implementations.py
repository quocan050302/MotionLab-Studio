"""Phase C — Deterministic Frame-Level Implementations for ALL 40 ML-01 Effects.

Each effect is evaluated purely from integer frame index `abs_frame`, normalized local
progress `local_t` in [0, 1], validated parameters, and bounded context (no wall clock,
no unbounded RAM, no random unseeded behavior).

Invariant: `intensity == 0` ALWAYS returns the input frame identically (verified in tests).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable

import cv2
import numpy as np
from PIL import Image, ImageDraw

from motionlab.effects.easing import bell_curve, clamp01, ease_in_out_cubic, ease_out_cubic, eval_easing, smooth_envelope, spring_damped
from motionlab.schemas import EffectEvent


@dataclass
class FrameRenderContext:
    width: int
    height: int
    fps: float
    seed: int
    audio_envelope_by_frame: dict[int, float] = field(default_factory=dict)
    subject_track_by_frame: dict[int, tuple[float, float]] = field(default_factory=dict)
    recent_frames: dict[int, np.ndarray] = field(default_factory=dict)  # small bounded rolling buffer (<= 16 frames)
    cut_anchor_frames: dict[str, np.ndarray] = field(default_factory=dict)  # pre-cut reference frame per transition event

    def audio_level(self, frame_idx: int) -> float:
        if not self.audio_envelope_by_frame:
            return 0.0
        if frame_idx in self.audio_envelope_by_frame:
            return self.audio_envelope_by_frame[frame_idx]
        # Nearest key lookup (audio envelope sampled every ~1-2 frames)
        keys = self.audio_envelope_by_frame
        for d in range(1, 5):
            if (frame_idx - d) in keys:
                return keys[frame_idx - d]
            if (frame_idx + d) in keys:
                return keys[frame_idx + d]
        return 0.0

    def subject_center(self, frame_idx: int, fallback: tuple[float, float]) -> tuple[float, float]:
        if not self.subject_track_by_frame:
            return fallback
        if frame_idx in self.subject_track_by_frame:
            return self.subject_track_by_frame[frame_idx]
        # Interpolate/nearest
        best_k = min(self.subject_track_by_frame.keys(), key=lambda k: abs(k - frame_idx))
        return self.subject_track_by_frame[best_k]


# ------------------------- Geometry & Backdrop Helpers -------------------------


def _warp_zoom_pan_rot(
    frame: np.ndarray,
    scale: float = 1.0,
    shift_x_px: float = 0.0,
    shift_y_px: float = 0.0,
    angle_deg: float = 0.0,
    anchor_x: float = 0.5,
    anchor_y: float = 0.5,
) -> np.ndarray:
    """Sub-pixel affine transform guaranteed free of black borders via safe shift clamping & edge replicate."""
    h, w = frame.shape[:2]
    scale = max(1.0, float(scale))
    ax = clamp01(anchor_x) * (w - 1)
    ay = clamp01(anchor_y) * (h - 1)

    # Guarantee overscan covers rotation corners
    if abs(angle_deg) > 1e-4:
        rad = math.radians(abs(angle_deg))
        aspect = max(w / max(1, h), h / max(1, w))
        min_rot_scale = math.cos(rad) + math.sin(rad) * aspect + 0.008
        scale = max(scale, min_rot_scale)

    # Clamp translation so corners stay strictly inside source bounds
    max_tx = max(0.0, (scale - 1.0) * w * 0.46)
    max_ty = max(0.0, (scale - 1.0) * h * 0.46)
    tx = max(-max_tx, min(max_tx, shift_x_px))
    ty = max(-max_ty, min(max_ty, shift_y_px))

    # Build rotation+scale matrix around (ax, ay), then adjust translation to keep anchor within bounds
    M = cv2.getRotationMatrix2D((ax, ay), angle_deg, scale)
    # Center-pull compensation so anchor stays safely framed without exposing black borders
    M[0, 2] += (w * 0.5 - ax) * (1.0 - 1.0 / scale) * 0.5 + tx
    M[1, 2] += (h * 0.5 - ay) * (1.0 - 1.0 / scale) * 0.5 + ty

    # Verify all 4 output corners map inside [0, w-1] x [0, h-1]; if not, clamp M[:,2]
    corners = np.array([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]], dtype=np.float64)
    A = M[:, :2]
    b = M[:, 2]
    # Inverse map: src = A^-1 (dst - b)
    det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
    if abs(det) > 1e-6:
        invA = np.linalg.inv(A)
        src_pts = (corners - b) @ invA.T
        min_x, min_y = src_pts[:, 0].min(), src_pts[:, 1].min()
        max_x, max_y = src_pts[:, 0].max(), src_pts[:, 1].max()
        dx_src = 0.0
        dy_src = 0.0
        if min_x < 0:
            dx_src = -min_x
        elif max_x > w - 1:
            dx_src = (w - 1) - max_x
        if min_y < 0:
            dy_src = -min_y
        elif max_y > h - 1:
            dy_src = (h - 1) - max_y
        if dx_src != 0.0 or dy_src != 0.0:
            shift_dst = A @ np.array([dx_src, dy_src])
            M[0, 2] -= shift_dst[0]
            M[1, 2] -= shift_dst[1]

    return cv2.warpAffine(
        frame,
        M,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def _soft_same_media_backdrop(frame: np.ndarray, dim: float = 0.55) -> np.ndarray:
    """Create a smooth blurred same-media backdrop (no artificial black bars)."""
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (max(32, w // 8), max(32, h // 8)), interpolation=cv2.INTER_AREA)
    blurred = cv2.GaussianBlur(small, (0, 0), sigmaX=4.0, sigmaY=4.0)
    up = cv2.resize(blurred, (w, h), interpolation=cv2.INTER_LINEAR)
    return np.clip(up.astype(np.float32) * dim, 0, 255).astype(np.uint8)


def _rounded_rect_mask(h: int, w: int, x0: int, y0: int, x1: int, y1: int, radius: int) -> np.ndarray:
    """Return float32 anti-aliased mask in [0, 1] of shape (h, w)."""
    img = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(img)
    r = max(0, min(radius, (x1 - x0) // 2, (y1 - y0) // 2))
    draw.rounded_rectangle([x0, y0, x1, y1], radius=r, fill=255)
    mask = np.asarray(img, dtype=np.float32) / 255.0
    mask = cv2.GaussianBlur(mask, (3, 3), 0.7)
    return mask


def _composite_card_on_backdrop(
    frame: np.ndarray,
    backdrop: np.ndarray,
    scale: float,
    corner_radius: int,
    shadow_offset_px: int = 10,
    shadow_alpha: float = 0.45,
) -> np.ndarray:
    """Composite a cleanly scaled rounded foreground card with soft drop shadow onto backdrop."""
    h, w = frame.shape[:2]
    cw = max(16, int(round(w * scale)))
    ch = max(16, int(round(h * scale)))
    x0 = (w - cw) // 2
    y0 = (h - ch) // 2
    x1 = x0 + cw
    y1 = y0 + ch

    out = backdrop.astype(np.float32)

    # Soft drop shadow
    if shadow_alpha > 0.01:
        smask = _rounded_rect_mask(
            h,
            w,
            min(w - 2, x0 + shadow_offset_px),
            min(h - 2, y0 + shadow_offset_px),
            min(w - 1, x1 + shadow_offset_px),
            min(h - 1, y1 + shadow_offset_px),
            corner_radius,
        )
        smask = cv2.GaussianBlur(smask, (0, 0), sigmaX=max(4.0, w * 0.015))
        out *= 1.0 - (smask[:, :, None] * shadow_alpha)

    fg_resized = cv2.resize(frame, (cw, ch), interpolation=cv2.INTER_CUBIC)
    fg_canvas = np.zeros_like(frame)
    fg_canvas[y0:y1, x0:x1] = fg_resized
    mask = _rounded_rect_mask(h, w, x0, y0, x1, y1, corner_radius)[:, :, None]
    out = fg_canvas.astype(np.float32) * mask + out * (1.0 - mask)
    return np.clip(out, 0, 255).astype(np.uint8)


def _alpha_composite_rgba(frame_rgb: np.ndarray, overlay_rgba: Image.Image) -> np.ndarray:
    ov = np.asarray(overlay_rgba, dtype=np.float32)
    alpha = (ov[:, :, 3:4] / 255.0)
    rgb = ov[:, :, :3]
    out = rgb * alpha + frame_rgb.astype(np.float32) * (1.0 - alpha)
    return np.clip(out, 0, 255).astype(np.uint8)


def _get_intensity(event: EffectEvent) -> float:
    return float(event.parameters.get("intensity", 0.75))


# ======================== CAMERA & COMPOSITION (C01–C10) ========================


def _fx_subtle_push_in(frame: np.ndarray, t: float, abs_f: int, ev: EffectEvent, ctx: FrameRenderContext) -> np.ndarray:
    inten = _get_intensity(ev)
    if inten <= 1e-6:
        return frame
    max_z = float(ev.parameters.get("max_zoom", 1.05))
    prog = eval_easing(ev.easing, t)
    # Smoothly ease in and softly ease back on final 15% to prevent hard cut snap
    env = 1.0 if t < 0.85 else ease_out_cubic((1.0 - t) / 0.15)
    z = 1.0 + (max_z - 1.0) * inten * prog * (0.25 + 0.75 * env)
    return _warp_zoom_pan_rot(frame, scale=z, anchor_x=ev.region.get("cx", 0.5), anchor_y=ev.region.get("cy", 0.5))


def _fx_eased_pull_out(frame: np.ndarray, t: float, abs_f: int, ev: EffectEvent, ctx: FrameRenderContext) -> np.ndarray:
    inten = _get_intensity(ev)
    if inten <= 1e-6:
        return frame
    start_z = float(ev.parameters.get("start_zoom", 1.05))
    prog = eval_easing(ev.easing, t)
    z = 1.0 + (start_z - 1.0) * inten * (1.0 - prog)
    return _warp_zoom_pan_rot(frame, scale=z, anchor_x=ev.region.get("cx", 0.5), anchor_y=ev.region.get("cy", 0.5))


def _fx_subject_punch_in(frame: np.ndarray, t: float, abs_f: int, ev: EffectEvent, ctx: FrameRenderContext) -> np.ndarray:
    inten = _get_intensity(ev)
    if inten <= 1e-6:
        return frame
    target_z = float(ev.parameters.get("target_zoom", 1.08))
    env = smooth_envelope(t, attack=0.22, release=0.28)
    z = 1.0 + (target_z - 1.0) * inten * env
    ax, ay = ctx.subject_center(abs_f, (ev.region.get("cx", 0.5), ev.region.get("cy", 0.5)))
    return _warp_zoom_pan_rot(frame, scale=z, anchor_x=ax, anchor_y=ay)


def _fx_subject_follow_pan(frame: np.ndarray, t: float, abs_f: int, ev: EffectEvent, ctx: FrameRenderContext) -> np.ndarray:
    inten = _get_intensity(ev)
    if inten <= 1e-6:
        return frame
    overscan = float(ev.parameters.get("overscan", 1.06))
    deadband = float(ev.parameters.get("deadband", 0.03))
    env = smooth_envelope(t, attack=0.20, release=0.20)
    z = 1.0 + (overscan - 1.0) * inten * env
    ax, ay = ctx.subject_center(abs_f, (ev.region.get("cx", 0.5), ev.region.get("cy", 0.5)))
    dx_norm = ax - 0.5
    dy_norm = ay - 0.5
    if abs(dx_norm) < deadband:
        dx_norm = 0.0
    if abs(dy_norm) < deadband:
        dy_norm = 0.0
    h, w = frame.shape[:2]
    tx = -dx_norm * w * 0.14 * inten * env
    ty = -dy_norm * h * 0.10 * inten * env
    return _warp_zoom_pan_rot(frame, scale=z, shift_x_px=tx, shift_y_px=ty, anchor_x=0.5, anchor_y=0.5)


def _fx_lateral_drift(frame: np.ndarray, t: float, abs_f: int, ev: EffectEvent, ctx: FrameRenderContext) -> np.ndarray:
    inten = _get_intensity(ev)
    if inten <= 1e-6:
        return frame
    overscan = float(ev.parameters.get("overscan", 1.05))
    direction = str(ev.parameters.get("direction", "right"))
    sign = 1.0 if direction == "right" else -1.0
    env = smooth_envelope(t, attack=0.18, release=0.18)
    z = 1.0 + (overscan - 1.0) * inten * env
    prog = eval_easing(ev.easing, t) - 0.5  # -0.5 .. +0.5
    w = frame.shape[1]
    tx = sign * prog * (overscan - 1.0) * w * 0.72 * inten * env
    return _warp_zoom_pan_rot(frame, scale=z, shift_x_px=tx)


def _fx_vertical_drift(frame: np.ndarray, t: float, abs_f: int, ev: EffectEvent, ctx: FrameRenderContext) -> np.ndarray:
    inten = _get_intensity(ev)
    if inten <= 1e-6:
        return frame
    overscan = float(ev.parameters.get("overscan", 1.05))
    direction = str(ev.parameters.get("direction", "up"))
    sign = -1.0 if direction == "up" else 1.0
    env = smooth_envelope(t, attack=0.18, release=0.18)
    z = 1.0 + (overscan - 1.0) * inten * env
    prog = eval_easing(ev.easing, t) - 0.5
    h = frame.shape[0]
    ty = sign * prog * (overscan - 1.0) * h * 0.65 * inten * env
    return _warp_zoom_pan_rot(frame, scale=z, shift_y_px=ty)


def _fx_tilt_micro(frame: np.ndarray, t: float, abs_f: int, ev: EffectEvent, ctx: FrameRenderContext) -> np.ndarray:
    inten = _get_intensity(ev)
    if inten <= 1e-6:
        return frame
    max_deg = min(2.0, float(ev.parameters.get("max_degrees", 1.4)))
    angle = max_deg * inten * math.sin(math.pi * t)
    return _warp_zoom_pan_rot(frame, scale=1.0, angle_deg=angle)


def _fx_settle_bounce(frame: np.ndarray, t: float, abs_f: int, ev: EffectEvent, ctx: FrameRenderContext) -> np.ndarray:
    inten = _get_intensity(ev)
    if inten <= 1e-6:
        return frame
    target_z = float(ev.parameters.get("target_zoom", 1.06))
    if t < 0.72:
        p = spring_damped(t / 0.72)
    else:
        p = ease_out_cubic((1.0 - t) / 0.28)
    z = 1.0 + (target_z - 1.0) * inten * p
    return _warp_zoom_pan_rot(frame, scale=z, anchor_x=ev.region.get("cx", 0.5), anchor_y=ev.region.get("cy", 0.5))


def _fx_reframed_focus_hold(frame: np.ndarray, t: float, abs_f: int, ev: EffectEvent, ctx: FrameRenderContext) -> np.ndarray:
    inten = _get_intensity(ev)
    if inten <= 1e-6:
        return frame
    hold_z = float(ev.parameters.get("hold_zoom", 1.06))
    ramp = float(ev.parameters.get("ramp_ratio", 0.25))
    env = smooth_envelope(t, attack=ramp, release=ramp)
    z = 1.0 + (hold_z - 1.0) * inten * env
    return _warp_zoom_pan_rot(frame, scale=z, anchor_x=ev.region.get("cx", 0.5), anchor_y=ev.region.get("cy", 0.5))


def _fx_depth_shadow_push(frame: np.ndarray, t: float, abs_f: int, ev: EffectEvent, ctx: FrameRenderContext) -> np.ndarray:
    inten = _get_intensity(ev)
    if inten <= 1e-6:
        return frame
    fg_scale = float(ev.parameters.get("foreground_scale", 0.92))
    env = smooth_envelope(t, attack=0.22, release=0.22)
    cur_scale = 1.0 - (1.0 - fg_scale) * inten * env + 0.015 * inten * eval_easing(ev.easing, t) * env
    cur_scale = min(1.0, max(0.80, cur_scale))
    if cur_scale >= 0.998:
        return frame
    backdrop = _soft_same_media_backdrop(frame, dim=0.52)
    radius = int(round(min(frame.shape[:2]) * 0.03 * env * inten))
    return _composite_card_on_backdrop(
        frame,
        backdrop,
        scale=cur_scale,
        corner_radius=radius,
        shadow_offset_px=max(4, int(round(min(frame.shape[:2]) * 0.015 * env))),
        shadow_alpha=0.45 * env * inten,
    )


# ======================== MASK, REVEAL & LAYOUT (L01–L10) ========================


def _fx_rounded_mask_reveal(frame: np.ndarray, t: float, abs_f: int, ev: EffectEvent, ctx: FrameRenderContext) -> np.ndarray:
    inten = _get_intensity(ev)
    if inten <= 1e-6:
        return frame
    cr = float(ev.parameters.get("corner_radius_ratio", 0.06))
    mr = float(ev.parameters.get("margin_ratio", 0.06))
    env = smooth_envelope(t, attack=0.25, release=0.25)
    h, w = frame.shape[:2]
    margin_x = int(round(w * mr * inten * env))
    margin_y = int(round(h * mr * inten * env))
    if margin_x < 2 and margin_y < 2:
        return frame
    radius = int(round(min(w, h) * cr * inten * env))
    backdrop = _soft_same_media_backdrop(frame, dim=0.60)
    mask = _rounded_rect_mask(h, w, margin_x, margin_y, w - margin_x, h - margin_y, radius)[:, :, None]
    out = frame.astype(np.float32) * mask + backdrop.astype(np.float32) * (1.0 - mask)
    return np.clip(out, 0, 255).astype(np.uint8)


def _fx_circular_focus_mask(frame: np.ndarray, t: float, abs_f: int, ev: EffectEvent, ctx: FrameRenderContext) -> np.ndarray:
    inten = _get_intensity(ev)
    if inten <= 1e-6:
        return frame
    r_ratio = float(ev.parameters.get("radius_ratio", 0.38))
    dim_out = float(ev.parameters.get("dim_outside", 0.35))
    env = smooth_envelope(t, attack=0.25, release=0.25)
    h, w = frame.shape[:2]
    cx = int(round(ev.region.get("cx", 0.5) * w))
    cy = int(round(ev.region.get("cy", 0.5) * h))
    yy, xx = np.ogrid[:h, :w]
    dist = np.sqrt(((xx - cx) ** 2 + (yy - cy) ** 2).astype(np.float32))
    rad = max(20.0, min(w, h) * r_ratio)
    feather = rad * 0.45
    mask = np.clip((rad + feather - dist) / max(1.0, feather), 0.0, 1.0)
    factor = 1.0 - (1.0 - mask) * dim_out * inten * env
    return np.clip(frame.astype(np.float32) * factor[:, :, None], 0, 255).astype(np.uint8)


def _fx_diagonal_mask_sweep(frame: np.ndarray, t: float, abs_f: int, ev: EffectEvent, ctx: FrameRenderContext) -> np.ndarray:
    inten = _get_intensity(ev)
    if inten <= 1e-6:
        return frame
    feather = float(ev.parameters.get("feather_ratio", 0.10))
    prog = eval_easing(ev.easing, t)
    env = smooth_envelope(t, attack=0.12, release=0.18)
    h, w = frame.shape[:2]
    yy = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    xx = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :]
    diag = (xx + yy) * 0.5  # 0..1
    edge = prog * 1.15 - 0.075
    mask = np.clip((edge - diag) / max(0.02, feather) + 0.5, 0.0, 1.0)
    # Softly dimmed/slightly desaturated base vs crisp slightly brightened foreground
    soft = _soft_same_media_backdrop(frame, dim=0.78)
    eff_mask = (1.0 - inten * env) + (inten * env) * mask
    out = frame.astype(np.float32) * eff_mask[:, :, None] + soft.astype(np.float32) * (1.0 - eff_mask[:, :, None])
    return np.clip(out, 0, 255).astype(np.uint8)


def _fx_soft_edge_vignette_reveal(frame: np.ndarray, t: float, abs_f: int, ev: EffectEvent, ctx: FrameRenderContext) -> np.ndarray:
    inten = _get_intensity(ev)
    if inten <= 1e-6:
        return frame
    strength = float(ev.parameters.get("strength", 0.28))
    # Opens cleanly from vignette -> clear -> slight return
    prog = 1.0 - ease_out_cubic(min(1.0, t / 0.65)) if t < 0.65 else 0.0
    w_eff = (0.3 + 0.7 * prog) * smooth_envelope(t, 0.1, 0.2) * inten * strength
    h, w = frame.shape[:2]
    yy = np.linspace(-1.0, 1.0, h, dtype=np.float32)[:, None]
    xx = np.linspace(-1.0, 1.0, w, dtype=np.float32)[None, :]
    r2 = np.clip(xx * xx + yy * yy, 0.0, 2.0) / 2.0
    factor = 1.0 - w_eff * (r2 ** 1.3)
    return np.clip(frame.astype(np.float32) * factor[:, :, None], 0, 255).astype(np.uint8)


def _fx_split_composition(frame: np.ndarray, t: float, abs_f: int, ev: EffectEvent, ctx: FrameRenderContext) -> np.ndarray:
    inten = _get_intensity(ev)
    if inten <= 1e-6:
        return frame
    gap = int(ev.parameters.get("gap_px", 6))
    env = smooth_envelope(t, attack=0.22, release=0.22) * inten
    h, w = frame.shape[:2]
    out = frame.copy()
    shift = int(round(min(w, h) * 0.025 * env))
    if h >= w:
        # Vertical 9:16 -> top/bottom split with subtle opposing scale/pan
        mid = h // 2
        top = _warp_zoom_pan_rot(frame, scale=1.0 + 0.04 * env, shift_x_px=float(shift))[:mid]
        bot = _warp_zoom_pan_rot(frame, scale=1.0 + 0.06 * env, shift_x_px=float(-shift))[mid:]
        out[:mid] = top
        out[mid:] = bot
        half_g = max(1, int(round(gap * 0.5 * env)))
        if half_g >= 1:
            out[max(0, mid - half_g) : min(h, mid + half_g), :] = np.clip(
                out[max(0, mid - half_g) : min(h, mid + half_g), :].astype(np.float32) * 0.35 + 165.0 * env,
                0,
                255,
            ).astype(np.uint8)
    else:
        mid = w // 2
        left = _warp_zoom_pan_rot(frame, scale=1.0 + 0.04 * env, shift_y_px=float(shift))[:, :mid]
        right = _warp_zoom_pan_rot(frame, scale=1.0 + 0.06 * env, shift_y_px=float(-shift))[:, mid:]
        out[:, :mid] = left
        out[:, mid:] = right
        half_g = max(1, int(round(gap * 0.5 * env)))
        if half_g >= 1:
            out[:, max(0, mid - half_g) : min(w, mid + half_g)] = np.clip(
                out[:, max(0, mid - half_g) : min(w, mid + half_g)].astype(np.float32) * 0.35 + 165.0 * env,
                0,
                255,
            ).astype(np.uint8)
    return out


def _fx_full_to_inset(frame: np.ndarray, t: float, abs_f: int, ev: EffectEvent, ctx: FrameRenderContext) -> np.ndarray:
    inten = _get_intensity(ev)
    if inten <= 1e-6:
        return frame
    inset_scale = float(ev.parameters.get("inset_scale", 0.84))
    env = smooth_envelope(t, attack=0.28, release=0.25)
    cur_scale = 1.0 - (1.0 - inset_scale) * inten * env
    if cur_scale >= 0.998:
        return frame
    backdrop = _soft_same_media_backdrop(frame, dim=0.50)
    radius = int(round(min(frame.shape[:2]) * 0.04 * env * inten))
    return _composite_card_on_backdrop(frame, backdrop, scale=cur_scale, corner_radius=radius, shadow_alpha=0.40 * env * inten)


def _fx_inset_to_full(frame: np.ndarray, t: float, abs_f: int, ev: EffectEvent, ctx: FrameRenderContext) -> np.ndarray:
    inten = _get_intensity(ev)
    if inten <= 1e-6:
        return frame
    start_scale = float(ev.parameters.get("start_scale", 0.84))
    prog = eval_easing(ev.easing, t)
    cur_scale = start_scale + (1.0 - start_scale) * prog
    cur_scale = 1.0 - (1.0 - cur_scale) * inten
    if cur_scale >= 0.998:
        return frame
    backdrop = _soft_same_media_backdrop(frame, dim=0.50)
    rem = (1.0 - prog) * inten
    radius = int(round(min(frame.shape[:2]) * 0.04 * rem))
    return _composite_card_on_backdrop(frame, backdrop, scale=cur_scale, corner_radius=radius, shadow_alpha=0.40 * rem)


def _fx_foreground_frame_accent(frame: np.ndarray, t: float, abs_f: int, ev: EffectEvent, ctx: FrameRenderContext) -> np.ndarray:
    inten = _get_intensity(ev)
    if inten <= 1e-6:
        return frame
    mr = float(ev.parameters.get("margin_ratio", 0.045))
    stroke = int(ev.parameters.get("stroke_px", 3))
    env = smooth_envelope(t, attack=0.20, release=0.22) * inten
    h, w = frame.shape[:2]
    ov = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(ov)
    mx = int(round(w * mr))
    my = int(round(h * mr))
    arm = int(round(min(w, h) * 0.08 * ease_out_cubic(min(1.0, t / 0.35))))
    alpha = int(round(205 * env))
    col = (245, 245, 250, alpha)
    if arm >= 2 and alpha > 2:
        # 4 clean corner brackets
        for (cx, cy, sx, sy) in [
            (mx, my, 1, 1),
            (w - mx, my, -1, 1),
            (mx, h - my, 1, -1),
            (w - mx, h - my, -1, -1),
        ]:
            draw.line([(cx, cy), (cx + sx * arm, cy)], fill=col, width=stroke)
            draw.line([(cx, cy), (cx, cy + sy * arm)], fill=col, width=stroke)
    return _alpha_composite_rgba(frame, ov)


def _fx_offset_layer_echo(frame: np.ndarray, t: float, abs_f: int, ev: EffectEvent, ctx: FrameRenderContext) -> np.ndarray:
    inten = _get_intensity(ev)
    if inten <= 1e-6:
        return frame
    lag = int(ev.parameters.get("lag_frames", 3))
    echo_alpha = float(ev.parameters.get("echo_alpha", 0.18))
    env = smooth_envelope(t, attack=0.20, release=0.25) * inten * echo_alpha
    prev_f = ctx.recent_frames.get(abs_f - lag)
    if prev_f is None:
        # Fallback to slightly scaled copy when lag frame not yet buffered
        prev_f = _warp_zoom_pan_rot(frame, scale=1.03)
    else:
        prev_f = _warp_zoom_pan_rot(prev_f, scale=1.025)
    # Peripheral weighting so central face/subject stays crisp and never doubled
    h, w = frame.shape[:2]
    yy = np.linspace(-1.0, 1.0, h, dtype=np.float32)[:, None]
    xx = np.linspace(-1.0, 1.0, w, dtype=np.float32)[None, :]
    radial = np.clip((xx * xx + yy * yy) * 0.65, 0.15, 1.0)
    w_map = (radial * env)[:, :, None]
    out = frame.astype(np.float32) * (1.0 - w_map) + prev_f.astype(np.float32) * w_map
    return np.clip(out, 0, 255).astype(np.uint8)


def _fx_geometric_window(frame: np.ndarray, t: float, abs_f: int, ev: EffectEvent, ctx: FrameRenderContext) -> np.ndarray:
    inten = _get_intensity(ev)
    if inten <= 1e-6:
        return frame
    chamfer = float(ev.parameters.get("chamfer_ratio", 0.14))
    env = smooth_envelope(t, attack=0.24, release=0.24) * inten
    h, w = frame.shape[:2]
    pad_x = int(round(w * 0.055 * env))
    pad_y = int(round(h * 0.055 * env))
    ch = int(round(min(w, h) * chamfer * env))
    if pad_x < 2 and ch < 3:
        return frame
    x0, y0, x1, y1 = pad_x, pad_y, w - pad_x, h - pad_y
    poly = [
        (x0 + ch, y0),
        (x1 - ch, y0),
        (x1, y0 + ch),
        (x1, y1 - ch),
        (x1 - ch, y1),
        (x0 + ch, y1),
        (x0, y1 - ch),
        (x0, y0 + ch),
    ]
    img = Image.new("L", (w, h), 0)
    ImageDraw.Draw(img).polygon(poly, fill=255)
    mask = cv2.GaussianBlur(np.asarray(img, dtype=np.float32) / 255.0, (5, 5), 1.0)[:, :, None]
    backdrop = _soft_same_media_backdrop(frame, dim=0.54)
    out = frame.astype(np.float32) * mask + backdrop.astype(np.float32) * (1.0 - mask)
    return np.clip(out, 0, 255).astype(np.uint8)


# ======================== TRANSITIONS (T01–T08, DURATION-PRESERVING) ========================


def _get_cut_anchor(frame: np.ndarray, abs_f: int, ev: EffectEvent, ctx: FrameRenderContext) -> np.ndarray:
    """Get pre-cut reference frame captured cleanly at event start (or blurred fallback)."""
    ref = ctx.cut_anchor_frames.get(ev.id)
    if ref is not None and ref.shape == frame.shape:
        return ref
    return _soft_same_media_backdrop(frame, dim=0.92)


def _fx_dissolve_short(frame: np.ndarray, t: float, abs_f: int, ev: EffectEvent, ctx: FrameRenderContext) -> np.ndarray:
    inten = _get_intensity(ev)
    if inten <= 1e-6:
        return frame
    ref = _get_cut_anchor(frame, abs_f, ev, ctx)
    # Blend smoothly across cut boundary without changing duration
    mid = (ev.start_frame + ev.end_frame_exclusive) // 2
    if abs_f < mid:
        alpha = 0.25 * inten * ease_in_out_cubic(t * 2.0)
        return cv2.addWeighted(frame, 1.0 - alpha, cv2.GaussianBlur(frame, (0, 0), 1.8), alpha, 0)
    post_t = clamp01((abs_f - mid) / max(1, ev.end_frame_exclusive - mid))
    alpha_ref = (1.0 - ease_out_cubic(post_t)) * 0.78 * inten
    return cv2.addWeighted(frame, 1.0 - alpha_ref, ref, alpha_ref, 0)


def _fx_directional_wipe(frame: np.ndarray, t: float, abs_f: int, ev: EffectEvent, ctx: FrameRenderContext) -> np.ndarray:
    inten = _get_intensity(ev)
    if inten <= 1e-6:
        return frame
    ref = _get_cut_anchor(frame, abs_f, ev, ctx)
    direction = str(ev.parameters.get("direction", "left"))
    prog = eval_easing(ev.easing, t)
    h, w = frame.shape[:2]
    if direction in ("left", "right"):
        coord = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :]
        if direction == "right":
            coord = 1.0 - coord
        mask = np.clip((prog - coord) / 0.03 + 0.5, 0.0, 1.0)
        mask = np.broadcast_to(mask, (h, w))
    else:
        coord = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
        if direction == "down":
            coord = 1.0 - coord
        mask = np.clip((prog - coord) / 0.03 + 0.5, 0.0, 1.0)
        mask = np.broadcast_to(mask, (h, w))
    m3 = (1.0 - inten + inten * mask)[:, :, None]
    out = frame.astype(np.float32) * m3 + ref.astype(np.float32) * (1.0 - m3)
    return np.clip(out, 0, 255).astype(np.uint8)


def _fx_soft_mask_wipe(frame: np.ndarray, t: float, abs_f: int, ev: EffectEvent, ctx: FrameRenderContext) -> np.ndarray:
    inten = _get_intensity(ev)
    if inten <= 1e-6:
        return frame
    ref = _get_cut_anchor(frame, abs_f, ev, ctx)
    feather = float(ev.parameters.get("feather_ratio", 0.14))
    prog = eval_easing(ev.easing, t)
    h, w = frame.shape[:2]
    coord = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :]
    mask = np.clip((prog * (1.0 + feather) - coord) / max(0.03, feather), 0.0, 1.0)
    m3 = np.broadcast_to((1.0 - inten + inten * mask), (h, w))[:, :, None]
    out = frame.astype(np.float32) * m3 + ref.astype(np.float32) * (1.0 - m3)
    return np.clip(out, 0, 255).astype(np.uint8)


def _fx_blur_through_cut(frame: np.ndarray, t: float, abs_f: int, ev: EffectEvent, ctx: FrameRenderContext) -> np.ndarray:
    inten = _get_intensity(ev)
    if inten <= 1e-6:
        return frame
    max_sig = float(ev.parameters.get("max_sigma", 8.0))
    b = bell_curve(t) * inten
    sigma = max_sig * b
    if sigma < 0.3:
        return frame
    blurred = cv2.GaussianBlur(frame, (0, 0), sigmaX=sigma, sigmaY=sigma * 0.7)
    ref = _get_cut_anchor(frame, abs_f, ev, ctx)
    mid = (ev.start_frame + ev.end_frame_exclusive) // 2
    if abs_f >= mid:
        post_t = clamp01((abs_f - mid) / max(1, ev.end_frame_exclusive - mid))
        ref_w = (1.0 - ease_out_cubic(post_t)) * 0.35 * inten
        ref_b = cv2.GaussianBlur(ref, (0, 0), sigmaX=sigma, sigmaY=sigma * 0.7)
        return cv2.addWeighted(blurred, 1.0 - ref_w, ref_b, ref_w, 0)
    return blurred


def _fx_zoom_through_cut(frame: np.ndarray, t: float, abs_f: int, ev: EffectEvent, ctx: FrameRenderContext) -> np.ndarray:
    inten = _get_intensity(ev)
    if inten <= 1e-6:
        return frame
    peak_z = float(ev.parameters.get("peak_zoom", 1.08))
    b = bell_curve(t) * inten
    z = 1.0 + (peak_z - 1.0) * b
    zoomed = _warp_zoom_pan_rot(frame, scale=z)
    if b > 0.2:
        zoomed = cv2.GaussianBlur(zoomed, (0, 0), sigmaX=2.2 * b, sigmaY=2.2 * b)
    return zoomed


def _fx_dark_dip(frame: np.ndarray, t: float, abs_f: int, ev: EffectEvent, ctx: FrameRenderContext) -> np.ndarray:
    inten = _get_intensity(ev)
    if inten <= 1e-6:
        return frame
    min_b = float(ev.parameters.get("min_brightness", 0.45))
    b = bell_curve(t) * inten
    gain = 1.0 - (1.0 - min_b) * b
    return np.clip(frame.astype(np.float32) * gain, 0, 255).astype(np.uint8)


def _fx_match_scale_bridge(frame: np.ndarray, t: float, abs_f: int, ev: EffectEvent, ctx: FrameRenderContext) -> np.ndarray:
    inten = _get_intensity(ev)
    if inten <= 1e-6:
        return frame
    bz = float(ev.parameters.get("bridge_zoom", 1.06))
    b = bell_curve(t) * inten
    z = 1.0 + (bz - 1.0) * b
    ax, ay = ctx.subject_center(abs_f, (ev.region.get("cx", 0.5), ev.region.get("cy", 0.5)))
    warped = _warp_zoom_pan_rot(frame, scale=z, anchor_x=ax, anchor_y=ay)
    ref = _get_cut_anchor(frame, abs_f, ev, ctx)
    mid = (ev.start_frame + ev.end_frame_exclusive) // 2
    if abs_f >= mid:
        post_t = clamp01((abs_f - mid) / max(1, ev.end_frame_exclusive - mid))
        rw = (1.0 - ease_out_cubic(post_t)) * 0.42 * inten
        return cv2.addWeighted(warped, 1.0 - rw, _warp_zoom_pan_rot(ref, scale=z, anchor_x=ax, anchor_y=ay), rw, 0)
    return warped


def _fx_edge_slide(frame: np.ndarray, t: float, abs_f: int, ev: EffectEvent, ctx: FrameRenderContext) -> np.ndarray:
    inten = _get_intensity(ev)
    if inten <= 1e-6:
        return frame
    max_shift = float(ev.parameters.get("max_shift_ratio", 0.10))
    b = bell_curve(t) * inten
    w = frame.shape[1]
    overscan = 1.0 + max_shift * 2.2 * b
    tx = (1.0 - 2.0 * t) * max_shift * w * b
    slid = _warp_zoom_pan_rot(frame, scale=overscan, shift_x_px=tx)
    ref = _get_cut_anchor(frame, abs_f, ev, ctx)
    mid = (ev.start_frame + ev.end_frame_exclusive) // 2
    if abs_f >= mid:
        post_t = clamp01((abs_f - mid) / max(1, ev.end_frame_exclusive - mid))
        rw = (1.0 - ease_out_cubic(post_t)) * 0.45 * inten
        return cv2.addWeighted(slid, 1.0 - rw, ref, rw, 0)
    return slid


# ======================== COLOR & OPTICAL (V01–V06) ========================


def _fx_warm_subtle(frame: np.ndarray, t: float, abs_f: int, ev: EffectEvent, ctx: FrameRenderContext) -> np.ndarray:
    inten = _get_intensity(ev)
    if inten <= 1e-6:
        return frame
    warmth = float(ev.parameters.get("warmth", 0.04)) * inten * smooth_envelope(t, 0.18, 0.18)
    if warmth <= 1e-5:
        return frame
    f = frame.astype(np.float32)
    # Mild warm lift in R, tiny green balance, subtle cool pull in B; skin tones protected by mild cap
    f[:, :, 0] = np.clip(f[:, :, 0] * (1.0 + warmth * 1.15) + warmth * 28.0, 0, 255)
    f[:, :, 1] = np.clip(f[:, :, 1] * (1.0 + warmth * 0.25), 0, 255)
    f[:, :, 2] = np.clip(f[:, :, 2] * (1.0 - warmth * 0.90), 0, 255)
    return f.astype(np.uint8)


def _fx_neutral_contrast(frame: np.ndarray, t: float, abs_f: int, ev: EffectEvent, ctx: FrameRenderContext) -> np.ndarray:
    inten = _get_intensity(ev)
    if inten <= 1e-6:
        return frame
    c_target = float(ev.parameters.get("contrast", 1.06))
    env = smooth_envelope(t, 0.18, 0.18) * inten
    c = 1.0 + (c_target - 1.0) * env
    if abs(c - 1.0) < 1e-4:
        return frame
    # Smooth S-curve LUT around midpoint 0.5 preserving [0, 255] without hard clipping
    x = np.linspace(0.0, 1.0, 256, dtype=np.float32)
    s_curve = x + (c - 1.0) * 0.55 * np.sin(2.0 * np.pi * (x - 0.5)) * (-1.0) * (1.0 - np.abs(2.0 * x - 1.0) * 0.35)
    lut = np.clip(np.round(s_curve * 255.0), 0, 255).astype(np.uint8)
    return cv2.LUT(frame, lut)


def _fx_soft_glow(frame: np.ndarray, t: float, abs_f: int, ev: EffectEvent, ctx: FrameRenderContext) -> np.ndarray:
    inten = _get_intensity(ev)
    if inten <= 1e-6:
        return frame
    thresh = float(ev.parameters.get("threshold", 0.76)) * 255.0
    gain = float(ev.parameters.get("bloom_gain", 0.16)) * inten * smooth_envelope(t, 0.20, 0.20)
    if gain <= 1e-4:
        return frame
    f = frame.astype(np.float32)
    luma = 0.299 * f[:, :, 0] + 0.587 * f[:, :, 1] + 0.114 * f[:, :, 2]
    # Softly isolate bright highlights above threshold so midtones/skin/text stay untouched
    hi_mask = np.clip((luma - thresh * 0.85) / max(1.0, 255.0 - thresh * 0.85), 0.0, 1.0)[:, :, None]
    highlights = f * hi_mask
    sigma = max(6.0, frame.shape[1] * 0.025)
    bloom = cv2.GaussianBlur(highlights, (0, 0), sigmaX=sigma, sigmaY=sigma)
    # Optical screen-blend halo spreading softly around highlights without clipping whites
    out = f + bloom * (gain * 2.2) * (0.35 + 0.65 * (1.0 - f / 255.0))
    return np.clip(out, 0, 255).astype(np.uint8)


def _fx_edge_vignette(frame: np.ndarray, t: float, abs_f: int, ev: EffectEvent, ctx: FrameRenderContext) -> np.ndarray:
    inten = _get_intensity(ev)
    if inten <= 1e-6:
        return frame
    strength = float(ev.parameters.get("strength", 0.20)) * inten * smooth_envelope(t, 0.18, 0.18)
    if strength <= 1e-4:
        return frame
    h, w = frame.shape[:2]
    yy = np.linspace(-1.0, 1.0, h, dtype=np.float32)[:, None]
    xx = np.linspace(-1.0, 1.0, w, dtype=np.float32)[None, :]
    r2 = np.clip((xx * xx + yy * yy) * 0.5, 0.0, 1.0)
    gain = 1.0 - strength * (r2 ** 1.5)
    return np.clip(frame.astype(np.float32) * gain[:, :, None], 0, 255).astype(np.uint8)


def _fx_light_sweep(frame: np.ndarray, t: float, abs_f: int, ev: EffectEvent, ctx: FrameRenderContext) -> np.ndarray:
    inten = _get_intensity(ev)
    if inten <= 1e-6:
        return frame
    opacity = float(ev.parameters.get("opacity", 0.11)) * inten * smooth_envelope(t, 0.15, 0.15)
    bw = float(ev.parameters.get("band_width", 0.20))
    if opacity <= 1e-4:
        return frame
    prog = eval_easing(ev.easing, t)
    h, w = frame.shape[:2]
    yy = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    xx = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :]
    diag = (xx * 0.72 + yy * 0.28)
    center = -0.15 + prog * 1.30
    band = np.exp(-0.5 * ((diag - center) / max(0.04, bw * 0.5)) ** 2)
    sheen = (band * opacity * 255.0)[:, :, None]
    # Warm white soft sheen
    tint = np.array([1.0, 0.97, 0.91], dtype=np.float32)[None, None, :]
    out = frame.astype(np.float32) + sheen * tint * (1.0 - frame.astype(np.float32) / 280.0)
    return np.clip(out, 0, 255).astype(np.uint8)


def _fx_film_grain_subtle(frame: np.ndarray, t: float, abs_f: int, ev: EffectEvent, ctx: FrameRenderContext) -> np.ndarray:
    inten = _get_intensity(ev)
    if inten <= 1e-6:
        return frame
    amp = float(ev.parameters.get("amplitude", 4.5)) * inten * smooth_envelope(t, 0.15, 0.15)
    if amp <= 0.05:
        return frame
    h, w = frame.shape[:2]
    rng = np.random.default_rng((ctx.seed * 1000003) ^ (abs_f * 9176 + 17))
    # Generate slightly coarse luminance grain (half-res upsampled -> filmic, fast)
    gh, gw = max(16, h // 2), max(16, w // 2)
    noise = rng.normal(0.0, amp, size=(gh, gw)).astype(np.float32)
    noise_full = cv2.resize(noise, (w, h), interpolation=cv2.INTER_LINEAR)[:, :, None]
    return np.clip(frame.astype(np.float32) + noise_full, 0, 255).astype(np.uint8)


# ======================== MOTION GRAPHICS & AUDIO-REACTIVE (G01–G06) ========================


def _fx_edge_lines(frame: np.ndarray, t: float, abs_f: int, ev: EffectEvent, ctx: FrameRenderContext) -> np.ndarray:
    inten = _get_intensity(ev)
    if inten <= 1e-6:
        return frame
    mr = float(ev.parameters.get("margin_ratio", 0.045))
    op = float(ev.parameters.get("opacity", 0.65)) * inten * smooth_envelope(t, 0.20, 0.22)
    if op <= 0.01:
        return frame
    h, w = frame.shape[:2]
    ov = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(ov)
    mx = int(round(w * mr))
    my = int(round(h * mr))
    prog = ease_out_cubic(min(1.0, t / 0.40))
    line_len = int(round(w * 0.22 * prog))
    stroke = max(2, int(round(min(w, h) * 0.004)))
    col = (255, 255, 255, int(round(255 * op)))
    accent = (255, 196, 112, int(round(220 * op)))
    # Top-left & bottom-right crisp geometric lines
    draw.line([(mx, my), (mx + line_len, my)], fill=col, width=stroke)
    draw.line([(mx, my + stroke * 3), (mx + int(line_len * 0.55), my + stroke * 3)], fill=accent, width=max(1, stroke - 1))
    draw.line([(w - mx - line_len, h - my), (w - mx, h - my)], fill=col, width=stroke)
    return _alpha_composite_rgba(frame, ov)


def _fx_ring_accent(frame: np.ndarray, t: float, abs_f: int, ev: EffectEvent, ctx: FrameRenderContext) -> np.ndarray:
    inten = _get_intensity(ev)
    if inten <= 1e-6:
        return frame
    op = float(ev.parameters.get("opacity", 0.55)) * inten * smooth_envelope(t, 0.15, 0.30)
    max_r = float(ev.parameters.get("max_radius_ratio", 0.11))
    if op <= 0.01:
        return frame
    h, w = frame.shape[:2]
    ov = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(ov)
    cx = int(round(clamp01(ev.region.get("cx", 0.84)) * w))
    cy = int(round(clamp01(ev.region.get("cy", 0.86)) * h))
    # Clamp strictly into peripheral band if region wasn't offset
    if 0.30 < cx / w < 0.70 and 0.25 < cy / h < 0.75:
        cx, cy = int(w * 0.85), int(h * 0.86)
    base_r = min(w, h) * max_r
    p1 = ease_out_cubic(t)
    r1 = max(4, int(round(base_r * (0.25 + 0.75 * p1))))
    r2 = max(3, int(round(base_r * (0.15 + 0.55 * ease_out_cubic(max(0.0, t - 0.12) / 0.88)))))
    stroke = max(2, int(round(min(w, h) * 0.004)))
    draw.ellipse([cx - r1, cy - r1, cx + r1, cy + r1], outline=(255, 255, 255, int(round(255 * op))), width=stroke)
    draw.ellipse([cx - r2, cy - r2, cx + r2, cy + r2], outline=(160, 220, 255, int(round(190 * op))), width=max(1, stroke - 1))
    return _alpha_composite_rgba(frame, ov)


def _fx_directional_shapes(frame: np.ndarray, t: float, abs_f: int, ev: EffectEvent, ctx: FrameRenderContext) -> np.ndarray:
    inten = _get_intensity(ev)
    if inten <= 1e-6:
        return frame
    op = float(ev.parameters.get("opacity", 0.50)) * inten * smooth_envelope(t, 0.18, 0.22)
    count = int(ev.parameters.get("count", 3))
    if op <= 0.01:
        return frame
    h, w = frame.shape[:2]
    ov = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(ov)
    base_x = int(round(w * 0.08 + ease_in_out_cubic(t) * w * 0.08))
    base_y = int(round(h * 0.90))
    size = max(8, int(round(min(w, h) * 0.018)))
    stroke = max(2, int(round(min(w, h) * 0.0038)))
    for i in range(count):
        ox = base_x + i * int(size * 1.45)
        a_i = int(round(255 * op * (0.55 + 0.45 * ((i + 1) / count))))
        pts = [(ox - size // 2, base_y - size), (ox + size // 2, base_y), (ox - size // 2, base_y + size)]
        draw.line(pts, fill=(255, 255, 255, a_i), width=stroke, joint="curve")
    return _alpha_composite_rgba(frame, ov)


def _fx_ambient_particles(frame: np.ndarray, t: float, abs_f: int, ev: EffectEvent, ctx: FrameRenderContext) -> np.ndarray:
    inten = _get_intensity(ev)
    if inten <= 1e-6:
        return frame
    count = min(18, max(4, int(ev.parameters.get("particle_count", 12))))
    op = float(ev.parameters.get("opacity", 0.35)) * inten * smooth_envelope(t, 0.20, 0.25)
    if op <= 0.01:
        return frame
    h, w = frame.shape[:2]
    ov = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(ov)
    rng = np.random.default_rng((ctx.seed * 65537) + sum(ord(c) for c in ev.id))
    for i in range(count):
        # Strictly place particles along peripheral frame margins (outside center 68% box)
        side = i % 4
        u = float(rng.uniform(0.05, 0.95))
        v = float(rng.uniform(0.03, 0.13))
        if side == 0:
            px, py = u, v
        elif side == 1:
            px, py = u, 1.0 - v
        elif side == 2:
            px, py = v, u
        else:
            px, py = 1.0 - v, u
        drift_y = -0.04 * t * float(rng.uniform(0.6, 1.3))
        drift_x = 0.015 * math.sin(t * 2.5 * math.pi + i)
        cx = int(round(clamp01(px + drift_x) * w))
        cy = int(round(clamp01(py + drift_y) * h))
        rad = max(2, int(round(min(w, h) * float(rng.uniform(0.003, 0.0075)))))
        alpha = int(round(255 * op * (0.5 + 0.5 * math.sin(math.pi * t + i * 0.7) ** 2)))
        draw.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], fill=(255, 248, 232, alpha))
    return _alpha_composite_rgba(frame, ov)


def _fx_waveform_accent(frame: np.ndarray, t: float, abs_f: int, ev: EffectEvent, ctx: FrameRenderContext) -> np.ndarray:
    inten = _get_intensity(ev)
    if inten <= 1e-6:
        return frame
    bars = min(24, max(8, int(ev.parameters.get("bar_count", 16))))
    op = float(ev.parameters.get("opacity", 0.68)) * inten * smooth_envelope(t, 0.15, 0.20)
    if op <= 0.01:
        return frame
    h, w = frame.shape[:2]
    ov = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(ov)

    # Compact sleek peripheral audio bars anchored cleanly at bottom-center safe margin
    bar_area_w = int(round(w * 0.42))
    x_start = (w - bar_area_w) // 2
    y_base = int(round(h * 0.92))
    max_bar_h = max(12, int(round(h * 0.042)))
    step = bar_area_w / bars
    bar_w = max(2, int(round(step * 0.52)))

    center_lvl = ctx.audio_level(abs_f)
    for i in range(bars):
        # Sample local temporal audio window around current frame
        offset = int(round((i - bars / 2.0) * 1.2))
        lvl = ctx.audio_level(max(0, abs_f + offset))
        if lvl <= 0.01 and center_lvl > 0.01:
            lvl = center_lvl * (0.4 + 0.6 * math.sin((i + 1) * math.pi / (bars + 1)))
        # Taper edges gracefully
        taper = math.sin((i + 0.5) * math.pi / bars)
        bh = max(3, int(round(max_bar_h * (0.12 + 0.88 * lvl * taper))))
        bx = int(round(x_start + i * step))
        col = (245, 248, 255, int(round(255 * op)))
        draw.rounded_rectangle([bx, y_base - bh, bx + bar_w, y_base], radius=max(1, bar_w // 2), fill=col)

    return _alpha_composite_rgba(frame, ov)


def _fx_pulse_border(frame: np.ndarray, t: float, abs_f: int, ev: EffectEvent, ctx: FrameRenderContext) -> np.ndarray:
    inten = _get_intensity(ev)
    if inten <= 1e-6:
        return frame
    op = float(ev.parameters.get("opacity", 0.45)) * inten * smooth_envelope(t, 0.16, 0.20)
    thickness = int(ev.parameters.get("thickness_px", 4))
    if op <= 0.01:
        return frame
    # Smoothed local audio energy (average over +-2 frames -> never strobes!)
    lvls = [ctx.audio_level(max(0, abs_f + d)) for d in (-2, -1, 0, 1, 2)]
    energy = sum(lvls) / len(lvls)
    mod_op = op * (0.45 + 0.55 * clamp01(energy))
    h, w = frame.shape[:2]
    ov = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(ov)
    pad = max(6, int(round(min(w, h) * 0.018)))
    rad = max(8, int(round(min(w, h) * 0.025)))
    col = (255, 240, 210, int(round(255 * mod_op)))
    draw.rounded_rectangle([pad, pad, w - pad, h - pad], radius=rad, outline=col, width=thickness)
    return _alpha_composite_rgba(frame, ov)


# ======================== DISPATCH TABLE (ALL 40 EFFECTS) ========================

EFFECT_FUNCTIONS: dict[str, Callable[[np.ndarray, float, int, EffectEvent, FrameRenderContext], np.ndarray]] = {
    # Camera & Composition (10)
    "subtle_push_in": _fx_subtle_push_in,
    "eased_pull_out": _fx_eased_pull_out,
    "subject_punch_in": _fx_subject_punch_in,
    "subject_follow_pan": _fx_subject_follow_pan,
    "lateral_drift": _fx_lateral_drift,
    "vertical_drift": _fx_vertical_drift,
    "tilt_micro": _fx_tilt_micro,
    "settle_bounce": _fx_settle_bounce,
    "reframed_focus_hold": _fx_reframed_focus_hold,
    "depth_shadow_push": _fx_depth_shadow_push,
    # Mask, Reveal & Layout (10)
    "rounded_mask_reveal": _fx_rounded_mask_reveal,
    "circular_focus_mask": _fx_circular_focus_mask,
    "diagonal_mask_sweep": _fx_diagonal_mask_sweep,
    "soft_edge_vignette_reveal": _fx_soft_edge_vignette_reveal,
    "split_composition": _fx_split_composition,
    "full_to_inset": _fx_full_to_inset,
    "inset_to_full": _fx_inset_to_full,
    "foreground_frame_accent": _fx_foreground_frame_accent,
    "offset_layer_echo": _fx_offset_layer_echo,
    "geometric_window": _fx_geometric_window,
    # Transitions (8)
    "dissolve_short": _fx_dissolve_short,
    "directional_wipe": _fx_directional_wipe,
    "soft_mask_wipe": _fx_soft_mask_wipe,
    "blur_through_cut": _fx_blur_through_cut,
    "zoom_through_cut": _fx_zoom_through_cut,
    "dark_dip": _fx_dark_dip,
    "match_scale_bridge": _fx_match_scale_bridge,
    "edge_slide": _fx_edge_slide,
    # Color & Optical (6)
    "warm_subtle": _fx_warm_subtle,
    "neutral_contrast": _fx_neutral_contrast,
    "soft_glow": _fx_soft_glow,
    "edge_vignette": _fx_edge_vignette,
    "light_sweep": _fx_light_sweep,
    "film_grain_subtle": _fx_film_grain_subtle,
    # Motion Graphics & Audio-Reactive (6)
    "edge_lines": _fx_edge_lines,
    "ring_accent": _fx_ring_accent,
    "directional_shapes": _fx_directional_shapes,
    "ambient_particles": _fx_ambient_particles,
    "waveform_accent": _fx_waveform_accent,
    "pulse_border": _fx_pulse_border,
}


def apply_effect_frame(
    frame_rgb: np.ndarray,
    abs_frame: int,
    event: EffectEvent,
    ctx: FrameRenderContext,
) -> np.ndarray:
    """Apply a single EffectEvent deterministically at integer frame `abs_frame`."""
    fn = EFFECT_FUNCTIONS.get(event.effect_id)
    if fn is None:
        raise KeyError(f"No frame implementation registered for effect '{event.effect_id}'")
    span = max(1, event.end_frame_exclusive - event.start_frame - 1)
    local_t = clamp01((abs_frame - event.start_frame) / span)
    return fn(frame_rgb, local_t, abs_frame, event, ctx)
