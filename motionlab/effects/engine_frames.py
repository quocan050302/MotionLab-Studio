"""Deterministic frame-exact effect evaluator — FULL 40 DISTINCT ML-01 EFFECTS.

Implements all 40 distinct algorithms:
  - Camera (10):      C01..C10
  - Layout/Mask (10): L01..L10
  - Transitions (8):  T01..T08 (strictly duration-preserving via neighbor ring buffer)
  - Color/Opt (6):    V01..V06
  - Graphics (6):     G01..G06 (G05/G06 audio-reactive from measured RMS envelope)

Invariants:
  - intensity == 0.0 produces exact no-op identity (`img`).
  - Zero black corners/borders on spatial transforms via exact overscan math.
  - Strictly deterministic from (frame_idx, event, ring_buffer, audio_env, seed).
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from motionlab.effects.easing import (
    bell_envelope,
    clamp01,
    ease_in_out_cubic,
    ease_out_cubic,
    smoothstep,
    spring_damped,
)
from motionlab.schemas import EffectEvent


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    if len(h) != 6:
        return (244, 246, 250)
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def _affine_zoom_pan(
    img: np.ndarray,
    scale: float,
    anchor_x: float = 0.5,
    anchor_y: float = 0.5,
    pan_x_px: float = 0.0,
    pan_y_px: float = 0.0,
    angle_deg: float = 0.0,
) -> np.ndarray:
    """Subpixel zoom/pan/rotation with exact bounding-box overscan so zero black corners ever appear."""
    h, w = img.shape[:2]
    rad = math.radians(angle_deg)
    cos_a = abs(math.cos(rad))
    sin_a = abs(math.sin(rad))
    min_rot_scale = max(
        cos_a + (h / max(1, w)) * sin_a,
        cos_a + (w / max(1, h)) * sin_a,
    )
    eff_scale = max(1.0, scale, (min_rot_scale + 0.004) if abs(angle_deg) > 1e-4 else 1.0)
    if abs(eff_scale - 1.0) < 1e-5 and abs(pan_x_px) < 1e-3 and abs(pan_y_px) < 1e-3 and abs(angle_deg) < 1e-4:
        return img

    cx = anchor_x * w
    cy = anchor_y * h
    max_shift_x = max(0.0, (w * (eff_scale - min_rot_scale)) * 0.48)
    max_shift_y = max(0.0, (h * (eff_scale - min_rot_scale)) * 0.48)
    dx = max(-max_shift_x, min(max_shift_x, pan_x_px))
    dy = max(-max_shift_y, min(max_shift_y, pan_y_px))

    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, eff_scale)
    M[0, 2] += dx
    M[1, 2] += dy

    if abs(angle_deg) < 1e-4:
        min_tx = (w - 1) * (1.0 - eff_scale)
        min_ty = (h - 1) * (1.0 - eff_scale)
        M[0, 2] = max(min_tx, min(0.0, float(M[0, 2])))
        M[1, 2] = max(min_ty, min(0.0, float(M[1, 2])))

    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)


def _rounded_rect_mask(h: int, w: int, x0: int, y0: int, x1: int, y1: int, radius: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    r = max(1, min(radius, max(1, (x1 - x0) // 2), max(1, (y1 - y0) // 2)))
    cv2.rectangle(mask, (x0 + r, y0), (x1 - r, y1), 255, -1)
    cv2.rectangle(mask, (x0, y0 + r), (x1, y1 - r), 255, -1)
    cv2.circle(mask, (x0 + r, y0 + r), r, 255, -1)
    cv2.circle(mask, (x1 - r, y0 + r), r, 255, -1)
    cv2.circle(mask, (x0 + r, y1 - r), r, 255, -1)
    cv2.circle(mask, (x1 - r, y1 - r), r, 255, -1)
    return cv2.GaussianBlur(mask, (5, 5), 1.0).astype(np.float32) / 255.0


def _ambient_plate(img: np.ndarray, darken: float = 0.56) -> np.ndarray:
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(32, w // 8), max(32, h // 8)), interpolation=cv2.INTER_AREA)
    blurred = cv2.GaussianBlur(small, (15, 15), 5.0)
    up = cv2.resize(blurred, (w, h), interpolation=cv2.INTER_LINEAR)
    return np.clip(up.astype(np.float32) * darken, 0, 255).astype(np.uint8)


def _apply_inset_card(
    img: np.ndarray,
    scale: float,
    corner_radius_frac: float,
    shadow_alpha: float = 0.32,
    bg_override: np.ndarray | None = None,
) -> np.ndarray:
    if scale >= 0.999:
        return img
    h, w = img.shape[:2]
    bg = bg_override if bg_override is not None else _ambient_plate(img, darken=0.55)
    nw = max(16, int(round(w * scale)))
    nh = max(16, int(round(h * scale)))
    fg_resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

    x0 = (w - nw) // 2
    y0 = (h - nh) // 2
    x1, y1 = x0 + nw, y0 + nh
    rad = max(4, int(round(min(nw, nh) * corner_radius_frac)))

    sh_off = max(4, h // 85)
    sh_mask = np.zeros((h, w), dtype=np.float32)
    cv2.rectangle(sh_mask, (x0, min(h - 2, y0 + sh_off)), (x1, min(h, y1 + sh_off)), 1.0, -1)
    sh_mask = cv2.GaussianBlur(sh_mask, (0, 0), max(4.0, min(w, h) * 0.016))
    bg_f = bg.astype(np.float32) * (1.0 - sh_mask[..., None] * shadow_alpha)

    fg_canvas = np.zeros_like(img)
    fg_canvas[y0:y1, x0:x1] = fg_resized
    mask = _rounded_rect_mask(h, w, x0, y0, x1, y1, rad)[..., None]
    out = fg_canvas.astype(np.float32) * mask + bg_f * (1.0 - mask)
    return np.clip(out, 0, 255).astype(np.uint8)


def _neighbor_pair(
    frame_idx: int,
    start_frame: int,
    end_frame_exclusive: int,
    ring_buffer: dict[int, np.ndarray],
    current_img: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    span = max(1, end_frame_exclusive - start_frame)
    cut_f = start_frame + span // 2
    u = clamp01((frame_idx - start_frame) / max(1, span - 1))
    outgoing = ring_buffer.get(min(frame_idx, cut_f - 1), current_img)
    incoming = ring_buffer.get(max(frame_idx, cut_f), current_img)
    return outgoing, incoming, u


def apply_single_effect(
    img: np.ndarray,
    event: EffectEvent,
    frame_idx: int,
    ring_buffer: dict[int, np.ndarray],
    audio_env: np.ndarray | None,
    seed: int,
) -> np.ndarray:
    """Apply one validated EffectEvent (any of C01..C10, L01..L10, T01..T08, V01..V06, G01..G06)."""
    params = event.parameters
    intensity = float(params.get("intensity", 1.0))
    if intensity <= 1e-6:
        return img

    span = max(1, event.end_frame_exclusive - event.start_frame)
    u = clamp01((frame_idx - event.start_frame) / max(1, span - 1))
    eid = event.effect_id
    h, w = img.shape[:2]

    # =========================================================================
    # CAMERA & COMPOSITION (10): C01..C10
    # =========================================================================
    if eid == "C01":  # subtle_push_in
        max_z = 1.0 + (float(params.get("max_zoom", 1.05)) - 1.0) * intensity
        e = ease_in_out_cubic(u) if u <= 0.85 else 1.0 - 0.15 * ease_in_out_cubic((u - 0.85) / 0.15)
        return _affine_zoom_pan(img, 1.0 + (max_z - 1.0) * e, float(params.get("anchor_x", 0.5)), float(params.get("anchor_y", 0.45)))

    if eid == "C02":  # eased_pull_out
        start_z = 1.0 + (float(params.get("start_zoom", 1.06)) - 1.0) * intensity
        scale = start_z - (start_z - 1.0) * ease_out_cubic(u)
        return _affine_zoom_pan(img, scale, float(params.get("anchor_x", 0.5)), float(params.get("anchor_y", 0.45)))

    if eid == "C03":  # subject_punch_in
        tz = 1.0 + (float(params.get("target_zoom", 1.09)) - 1.0) * intensity
        hold = float(params.get("hold_frac", 0.45))
        in_end = (1.0 - hold) * 0.45
        out_start = in_end + hold
        if u < in_end:
            e = spring_damped(u / max(1e-4, in_end), damping=0.78, freq=1.8)
        elif u <= out_start:
            e = 1.0
        else:
            e = 1.0 - ease_in_out_cubic((u - out_start) / max(1e-4, 1.0 - out_start))
        return _affine_zoom_pan(img, 1.0 + (tz - 1.0) * max(0.0, min(1.06, e)), float(params.get("anchor_x", 0.5)), float(params.get("anchor_y", 0.40)))

    if eid == "C04":  # subject_follow_pan
        ov = 1.0 + (float(params.get("overscan_zoom", 1.08)) - 1.0) * intensity
        ax = float(params.get("anchor_x", 0.5))
        ay = float(params.get("anchor_y", 0.42))
        deadband = float(params.get("deadband", 0.03))
        dx_f = 0.5 - ax
        dy_f = 0.48 - ay
        if abs(dx_f) < deadband:
            dx_f = 0.015 if dx_f >= 0 else -0.015  # gentle micro follow
        if abs(dy_f) < deadband:
            dy_f = 0.0
        e = bell_envelope(u, ramp_frac=0.28)
        return _affine_zoom_pan(img, 1.0 + (ov - 1.0) * e, 0.5, 0.5, pan_x_px=dx_f * w * 0.45 * e * intensity, pan_y_px=dy_f * h * 0.35 * e * intensity)

    if eid == "C05":  # lateral_drift
        shift_f = float(params.get("shift_frac", 0.032)) * intensity
        direction = 1.0 if params.get("direction", "right") == "right" else -1.0
        overscan = 1.0 + 2.25 * shift_f
        pan_x = direction * (ease_in_out_cubic(u) - 0.5) * 2.0 * shift_f * w
        return _affine_zoom_pan(img, overscan, 0.5, 0.5, pan_x_px=pan_x)

    if eid == "C06":  # vertical_drift
        shift_f = float(params.get("shift_frac", 0.028)) * intensity
        direction = -1.0 if params.get("direction", "up") == "up" else 1.0
        overscan = 1.0 + 2.25 * shift_f
        pan_y = direction * (ease_in_out_cubic(u) - 0.5) * 2.0 * shift_f * h
        return _affine_zoom_pan(img, overscan, 0.5, 0.5, pan_y_px=pan_y)

    if eid == "C07":  # tilt_micro
        max_deg = float(params.get("max_deg", 1.5)) * intensity
        sign = 1.0 if params.get("direction", "cw") == "cw" else -1.0
        angle = sign * max_deg * bell_envelope(u, ramp_frac=0.30)
        return _affine_zoom_pan(img, 1.0, 0.5, 0.5, angle_deg=angle)

    if eid == "C08":  # settle_bounce
        tz = 1.0 + (float(params.get("target_zoom", 1.06)) - 1.0) * intensity
        damp = float(params.get("damping", 0.72))
        e = spring_damped(u, damping=damp, freq=2.2)
        if u > 0.78:
            e *= 1.0 - ease_in_out_cubic((u - 0.78) / 0.22)
        return _affine_zoom_pan(img, 1.0 + (tz - 1.0) * max(0.0, e), float(params.get("anchor_x", 0.5)), float(params.get("anchor_y", 0.45)))

    if eid == "C09":  # reframed_focus_hold
        tz = 1.0 + (float(params.get("target_zoom", 1.06)) - 1.0) * intensity
        hold = float(params.get("hold_frac", 0.55))
        e = bell_envelope(u, ramp_frac=max(0.10, min(0.42, (1.0 - hold) * 0.5)))
        return _affine_zoom_pan(img, 1.0 + (tz - 1.0) * e, float(params.get("anchor_x", 0.5)), float(params.get("anchor_y", 0.45)))

    if eid == "C10":  # depth_shadow_push (explicit 2D layered scale/shadow illusion)
        e = bell_envelope(u, ramp_frac=0.28)
        bg = _affine_zoom_pan(_ambient_plate(img, darken=0.62), 1.0 + 0.015 * e * intensity, 0.5, 0.5)
        fg_zoom = 1.0 + (float(params.get("push_zoom", 1.05)) - 1.0) * e * intensity
        fg_src = _affine_zoom_pan(img, fg_zoom, 0.5, 0.46)
        base_fg = float(params.get("fg_scale", 0.91))
        cur_fg = 1.0 - (1.0 - base_fg) * e * intensity
        return _apply_inset_card(fg_src, cur_fg, corner_radius_frac=0.04 * e, shadow_alpha=0.38 * e, bg_override=bg)

    # =========================================================================
    # MASK, REVEAL & LAYOUT (10): L01..L10
    # =========================================================================
    if eid == "L01":  # rounded_mask_reveal (1:1 stationary media revealed by expanding rounded aperture mask)
        min_s = 1.0 - (1.0 - float(params.get("min_scale", 0.84))) * intensity
        e = ease_in_out_cubic(u)
        open_frac = (min_s * 0.65) + (1.0 - min_s * 0.65) * e
        bg = _ambient_plate(img, darken=0.45)
        mw = max(16, int(round(w * open_frac)))
        mh = max(16, int(round(h * open_frac)))
        x0, y0 = (w - mw) // 2, (h - mh) // 2
        x1, y1 = x0 + mw, y0 + mh
        rad = max(6, int(round(min(mw, mh) * float(params.get("corner_radius_frac", 0.06)) * (1.4 - e))))
        mask = _rounded_rect_mask(h, w, x0, y0, x1, y1, rad)[..., None]
        out = img.astype(np.float32) * mask + bg.astype(np.float32) * (1.0 - mask)
        return np.clip(out, 0, 255).astype(np.uint8)

    if eid == "L02":  # circular_focus_mask
        e = bell_envelope(u, ramp_frac=0.26)
        ax = float(params.get("anchor_x", 0.5)) * w
        ay = float(params.get("anchor_y", 0.45)) * h
        rad = float(params.get("radius_frac", 0.42)) * min(w, h) * (0.85 + 0.25 * e)
        dim = float(params.get("surround_dim", 0.32)) * intensity * e
        yy, xx = np.ogrid[:h, :w]
        dist = np.sqrt((xx - ax) ** 2 + (yy - ay) ** 2).astype(np.float32)
        soft = np.clip((dist - rad * 0.65) / max(8.0, rad * 0.55), 0.0, 1.0)
        att = (1.0 - dim * smoothstep_arr(soft))[..., None]
        return np.clip(img.astype(np.float32) * att, 0, 255).astype(np.uint8)

    if eid == "L03":  # diagonal_mask_sweep
        angle = math.radians(float(params.get("angle_deg", 28.0)))
        feather = max(0.05, float(params.get("feather_frac", 0.14)))
        e = ease_in_out_cubic(u)
        ys = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
        xs = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :]
        proj = xs * math.cos(angle) + ys * math.sin(angle)
        proj_n = (proj - float(proj.min())) / max(1e-5, float(proj.max() - proj.min()))
        pos = -feather + e * (1.0 + 2.0 * feather)
        alpha = np.clip((pos - proj_n) / feather + 0.5, 0.0, 1.0)
        dimmed = np.clip(img.astype(np.float32) * (1.0 - 0.16 * intensity), 0, 255)
        crisp = np.clip(img.astype(np.float32) * (1.0 + 0.05 * intensity), 0, 255)
        edge_glow = np.exp(-((proj_n - pos) / max(0.02, feather * 0.35)) ** 2) * (18.0 * intensity * bell_envelope(u, 0.2))
        return np.clip(crisp * alpha[..., None] + dimmed * (1.0 - alpha[..., None]) + edge_glow[..., None], 0, 255).astype(np.uint8)

    if eid == "L04":  # soft_edge_vignette_reveal
        e = ease_out_cubic(u)
        strength = float(params.get("start_strength", 0.58)) * intensity * (1.0 - e * 0.85)
        yy = np.linspace(-1.0, 1.0, h, dtype=np.float32)[:, None]
        xx = np.linspace(-1.0, 1.0, w, dtype=np.float32)[None, :]
        r2 = np.clip((xx * xx + yy * yy) / 1.45, 0.0, 1.0)
        att = (1.0 - strength * (r2 ** 1.4))[..., None]
        return np.clip(img.astype(np.float32) * att, 0, 255).astype(np.uint8)

    if eid == "L05":  # split_composition (opt-in aspect-preserving dual pane of same media)
        e = bell_envelope(u, ramp_frac=0.22) * intensity
        detail_z = float(params.get("detail_zoom", 1.35))
        detail = _affine_zoom_pan(img, detail_z, 0.5, 0.45)
        orient = str(params.get("orientation", "vertical"))
        comp = img.copy()

        def _cover_crop(src: np.ndarray, tw: int, th: int) -> np.ndarray:
            sh, sw = src.shape[:2]
            scale = max(tw / max(1, sw), th / max(1, sh))
            rw, rh = max(tw, int(round(sw * scale))), max(th, int(round(sh * scale)))
            resized = cv2.resize(src, (rw, rh), interpolation=cv2.INTER_AREA)
            x0, y0 = (rw - tw) // 2, (rh - th) // 2
            return resized[y0 : y0 + th, x0 : x0 + tw]

        if orient == "horizontal":
            mid = h // 2
            comp[:mid, :] = _cover_crop(img, w, mid)
            comp[mid:, :] = _cover_crop(detail, w, h - mid)
            cv2.line(comp, (0, mid), (w, mid), (240, 244, 250), max(2, h // 320))
        else:
            mid = w // 2
            comp[:, :mid] = _cover_crop(img, mid, h)
            comp[:, mid:] = _cover_crop(detail, w - mid, h)
            cv2.line(comp, (mid, 0), (mid, h), (240, 244, 250), max(2, w // 320))
        return cv2.addWeighted(comp, e, img, 1.0 - e, 0.0)

    if eid == "L06":  # full_to_inset
        target_s = 1.0 - (1.0 - float(params.get("inset_scale", 0.85))) * intensity
        e = bell_envelope(u, ramp_frac=0.30)
        return _apply_inset_card(img, 1.0 - (1.0 - target_s) * e, float(params.get("corner_radius_frac", 0.05)) * e, shadow_alpha=0.32 * e)

    if eid == "L07":  # inset_to_full
        start_s = 1.0 - (1.0 - float(params.get("start_scale", 0.85))) * intensity
        e = ease_out_cubic(u)
        return _apply_inset_card(img, start_s + (1.0 - start_s) * e, float(params.get("corner_radius_frac", 0.05)) * (1.0 - e), shadow_alpha=0.32 * (1.0 - e))

    if eid == "L08":  # foreground_frame_accent
        alpha = float(params.get("opacity", 0.60)) * intensity * bell_envelope(u, 0.22)
        if alpha <= 1e-4:
            return img
        margin = float(params.get("margin_frac", 0.045))
        color = _hex_to_rgb(str(params.get("accent_hex", "#F4F6FA")))
        p = bell_envelope(u, 0.25)
        mx = max(8, int(round(w * margin)))
        my = max(8, int(round(h * margin)))
        arm = max(12, int(round(min(w, h) * 0.075 * p)))
        thick = max(2, int(round(min(w, h) * 0.0042)))
        overlay = img.copy()
        # Four clean architectural corner brackets strictly in outer margin
        corners = [
            ((mx, my), (mx + arm, my), (mx, my + arm)),
            ((w - mx, my), (w - mx - arm, my), (w - mx, my + arm)),
            ((mx, h - my), (mx + arm, h - my), (mx, h - my - arm)),
            ((w - mx, h - my), (w - mx - arm, h - my), (w - mx, h - my - arm)),
        ]
        for o, a, b in corners:
            cv2.line(overlay, o, a, color, thick, cv2.LINE_AA)
            cv2.line(overlay, o, b, color, thick, cv2.LINE_AA)
        return cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0.0)

    if eid == "L09":  # offset_layer_echo (peripheral trailing echo; center/face protected)
        lag = int(params.get("lag_frames", 4))
        echo_src = ring_buffer.get(max(event.start_frame, frame_idx - lag), img)
        shift_px = int(round(w * float(params.get("shift_frac", 0.025)) * intensity * bell_envelope(u, 0.25)))
        e_alpha = float(params.get("echo_alpha", 0.28)) * intensity * bell_envelope(u, 0.25)
        if e_alpha <= 1e-4:
            return img
        M = np.float32([[1.0, 0.0, float(shift_px)], [0.0, 1.0, 0.0]])
        shifted = cv2.warpAffine(echo_src, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        # Protect center 55% ellipse so duplicate faces never dominate center
        yy = np.linspace(-1.0, 1.0, h, dtype=np.float32)[:, None]
        xx = np.linspace(-1.0, 1.0, w, dtype=np.float32)[None, :]
        periph = np.clip((np.sqrt((xx / 0.65) ** 2 + (yy / 0.65) ** 2) - 0.72) / 0.45, 0.0, 1.0)[..., None]
        w_mask = periph * e_alpha
        out = img.astype(np.float32) * (1.0 - w_mask) + shifted.astype(np.float32) * w_mask
        return np.clip(out, 0, 255).astype(np.uint8)

    if eid == "L10":  # geometric_window (animated octagonal chamfered mask over ambient plate)
        min_s = 1.0 - (1.0 - float(params.get("min_scale", 0.78))) * intensity
        e = bell_envelope(u, ramp_frac=0.28)
        cur_s = 1.0 - (1.0 - min_s) * e
        ch_f = float(params.get("chamfer_frac", 0.14)) * e
        bg = _ambient_plate(img, darken=0.52)
        nw = max(16, int(round(w * cur_s)))
        nh = max(16, int(round(h * cur_s)))
        x0, y0 = (w - nw) // 2, (h - nh) // 2
        x1, y1 = x0 + nw, y0 + nh
        c = max(4, int(round(min(nw, nh) * ch_f)))
        poly = np.array([
            [x0 + c, y0], [x1 - c, y0], [x1, y0 + c], [x1, y1 - c],
            [x1 - c, y1], [x0 + c, y1], [x0, y1 - c], [x0, y0 + c],
        ], dtype=np.int32)
        mask_u8 = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask_u8, [poly], 255)
        mask = (cv2.GaussianBlur(mask_u8, (7, 7), 1.4).astype(np.float32) / 255.0)[..., None]
        out = img.astype(np.float32) * mask + bg.astype(np.float32) * (1.0 - mask)
        return np.clip(out, 0, 255).astype(np.uint8)

    # =========================================================================
    # TRANSITIONS (8): T01..T08 — STRICTLY DURATION-PRESERVING
    # =========================================================================
    if eid == "T01":  # dissolve_short
        outgoing, incoming, prog = _neighbor_pair(frame_idx, event.start_frame, event.end_frame_exclusive, ring_buffer, img)
        alpha = smoothstep(prog)
        mixed = outgoing.astype(np.float32) * (1.0 - alpha) + incoming.astype(np.float32) * alpha
        return np.clip(img.astype(np.float32) * (1.0 - intensity) + mixed * intensity, 0, 255).astype(np.uint8)

    if eid == "T02":  # directional_wipe
        outgoing, incoming, prog = _neighbor_pair(frame_idx, event.start_frame, event.end_frame_exclusive, ring_buffer, img)
        direction = str(params.get("direction", "left_to_right"))
        e = ease_in_out_cubic(prog)
        feather = 0.025
        if direction == "top_to_bottom":
            coord = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
        elif direction == "bottom_to_top":
            coord = np.linspace(1.0, 0.0, h, dtype=np.float32)[:, None]
        elif direction == "right_to_left":
            coord = np.linspace(1.0, 0.0, w, dtype=np.float32)[None, :]
        else:
            coord = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :]
        pos = -feather + e * (1.0 + 2.0 * feather)
        mask = np.clip((pos - coord) / feather + 0.5, 0.0, 1.0)[..., None]
        mixed = incoming.astype(np.float32) * mask + outgoing.astype(np.float32) * (1.0 - mask)
        return np.clip(img.astype(np.float32) * (1.0 - intensity) + mixed * intensity, 0, 255).astype(np.uint8)

    if eid == "T03":  # soft_mask_wipe
        outgoing, incoming, prog = _neighbor_pair(frame_idx, event.start_frame, event.end_frame_exclusive, ring_buffer, img)
        direction = str(params.get("direction", "left_to_right"))
        feather = max(0.08, float(params.get("feather_frac", 0.22)))
        e = ease_in_out_cubic(prog)
        if direction == "top_to_bottom":
            coord = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
        elif direction == "right_to_left":
            coord = np.linspace(1.0, 0.0, w, dtype=np.float32)[None, :]
        else:
            coord = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :]
        pos = -feather + e * (1.0 + 2.0 * feather)
        mask = np.clip((pos - coord) / feather + 0.5, 0.0, 1.0)[..., None]
        mixed = incoming.astype(np.float32) * mask + outgoing.astype(np.float32) * (1.0 - mask)
        return np.clip(img.astype(np.float32) * (1.0 - intensity) + mixed * intensity, 0, 255).astype(np.uint8)

    if eid == "T04":  # blur_through_cut
        bell = math.sin(math.pi * u) ** 1.6
        sigma = float(params.get("max_sigma", 14.0)) * intensity * bell
        return cv2.GaussianBlur(img, (0, 0), sigmaX=sigma, sigmaY=sigma) if sigma >= 0.35 else img

    if eid == "T05":  # zoom_through_cut
        outgoing, incoming, prog = _neighbor_pair(frame_idx, event.start_frame, event.end_frame_exclusive, ring_buffer, img)
        peak_z = 1.0 + (float(params.get("peak_zoom", 1.12)) - 1.0) * intensity
        bell = math.sin(math.pi * prog) ** 1.3
        zoomed_out = _affine_zoom_pan(outgoing, 1.0 + (peak_z - 1.0) * bell, 0.5, 0.5)
        zoomed_in = _affine_zoom_pan(incoming, 1.0 + (peak_z - 1.0) * bell, 0.5, 0.5)
        alpha = smoothstep(prog)
        mixed = zoomed_out.astype(np.float32) * (1.0 - alpha) + zoomed_in.astype(np.float32) * alpha
        return np.clip(mixed, 0, 255).astype(np.uint8)

    if eid == "T06":  # dark_dip
        max_dip = float(params.get("max_dip", 0.70)) * intensity
        bell = math.sin(math.pi * u) ** 1.8
        gain = max(0.20, 1.0 - max_dip * bell)
        return np.clip(img.astype(np.float32) * gain, 0, 255).astype(np.uint8)

    if eid == "T07":  # match_scale_bridge
        outgoing, incoming, prog = _neighbor_pair(frame_idx, event.start_frame, event.end_frame_exclusive, ring_buffer, img)
        bz = 1.0 + (float(params.get("bridge_zoom", 1.07)) - 1.0) * intensity
        ax = float(params.get("anchor_x", 0.5))
        ay = float(params.get("anchor_y", 0.45))
        # Outgoing zooms smoothly into bridge_zoom while incoming settles smoothly from bridge_zoom -> 1.0
        s_out = 1.0 + (bz - 1.0) * ease_in_out_cubic(prog)
        s_in = bz - (bz - 1.0) * ease_out_cubic(prog)
        a_img = _affine_zoom_pan(outgoing, s_out, ax, ay)
        b_img = _affine_zoom_pan(incoming, s_in, ax, ay)
        alpha = smoothstep(prog)
        mixed = a_img.astype(np.float32) * (1.0 - alpha) + b_img.astype(np.float32) * alpha
        return np.clip(img.astype(np.float32) * (1.0 - intensity) + mixed * intensity, 0, 255).astype(np.uint8)

    if eid == "T08":  # edge_slide
        outgoing, incoming, prog = _neighbor_pair(frame_idx, event.start_frame, event.end_frame_exclusive, ring_buffer, img)
        direction = str(params.get("direction", "from_right"))
        e = ease_in_out_cubic(prog)
        canvas = outgoing.copy()
        if direction == "from_left":
            cut_x = max(1, min(w, int(round(w * e))))
            canvas[:, :cut_x] = incoming[:, w - cut_x :]
        elif direction == "from_bottom":
            cut_y = max(1, min(h, int(round(h * e))))
            canvas[h - cut_y :, :] = incoming[:cut_y, :]
        else:  # from_right
            cut_x = max(1, min(w, int(round(w * e))))
            canvas[:, w - cut_x :] = incoming[:, :cut_x]
        return cv2.addWeighted(canvas, intensity, img, 1.0 - intensity, 0.0)

    # =========================================================================
    # COLOR & OPTICAL (6): V01..V06
    # =========================================================================
    if eid == "V01":  # warm_subtle
        warmth = float(params.get("warmth", 0.055)) * intensity * bell_envelope(u, 0.18)
        if warmth <= 1e-5:
            return img
        arr = img.astype(np.float32) / 255.0
        mid_w = arr * (1.0 - arr) * 4.0
        arr[..., 0] = np.clip(arr[..., 0] + warmth * (0.45 * arr[..., 0] + 0.55 * mid_w[..., 0]), 0.0, 1.0)
        arr[..., 1] = np.clip(arr[..., 1] + warmth * 0.14 * mid_w[..., 1], 0.0, 1.0)
        arr[..., 2] = np.clip(arr[..., 2] - warmth * (0.45 * arr[..., 2] + 0.40 * mid_w[..., 2]), 0.0, 1.0)
        return np.round(arr * 255.0).astype(np.uint8)

    if eid == "V02":  # neutral_contrast
        amp = float(params.get("contrast_amp", 0.055)) * intensity * bell_envelope(u, 0.15)
        if amp <= 1e-5:
            return img
        x = np.linspace(0.0, 1.0, 256, dtype=np.float32)
        lut = np.round(np.clip(x - amp * np.sin(2.0 * math.pi * x), 0.0, 1.0) * 255.0).astype(np.uint8)
        return cv2.LUT(img, lut)

    if eid == "V03":  # soft_glow (highlight-gated screen bloom)
        strength = float(params.get("bloom_strength", 0.32)) * intensity * bell_envelope(u, 0.20)
        thresh = float(params.get("threshold", 0.68)) * 255.0
        if strength <= 1e-4:
            return img
        arr = img.astype(np.float32)
        luma = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
        hi_mask = np.clip((luma - thresh) / max(10.0, 255.0 - thresh), 0.0, 1.0)[..., None]
        hi = arr * hi_mask
        small = cv2.resize(hi, (max(32, w // 4), max(32, h // 4)), interpolation=cv2.INTER_AREA)
        glow = cv2.resize(cv2.GaussianBlur(small, (0, 0), 5.0), (w, h), interpolation=cv2.INTER_LINEAR)
        # Screen blend prevents harsh blowout on faces/text
        a_n = arr / 255.0
        g_n = np.clip(glow * strength / 255.0, 0.0, 0.55)
        screened = 1.0 - (1.0 - a_n) * (1.0 - g_n)
        return np.clip(np.round(screened * 255.0), 0, 255).astype(np.uint8)

    if eid == "V04":  # edge_vignette
        strength = float(params.get("strength", 0.28)) * intensity * bell_envelope(u, 0.18)
        if strength <= 1e-4:
            return img
        yy = np.linspace(-1.0, 1.0, h, dtype=np.float32)[:, None]
        xx = np.linspace(-1.0, 1.0, w, dtype=np.float32)[None, :]
        r2 = np.clip((xx * xx + yy * yy) / 1.6, 0.0, 1.0)
        att = (1.0 - strength * (r2 ** 1.6))[..., None]
        return np.clip(img.astype(np.float32) * att, 0, 255).astype(np.uint8)

    if eid == "V05":  # light_sweep (non-flashing low-opacity diagonal sheen)
        lift = float(params.get("peak_lift", 26.0)) * intensity * bell_envelope(u, 0.20)
        bw = max(0.08, float(params.get("band_width", 0.22)))
        if lift <= 1e-3:
            return img
        ys = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
        xs = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :]
        diag = 0.65 * xs + 0.35 * ys
        center = -bw + ease_in_out_cubic(u) * (1.0 + 2.0 * bw)
        sheen = np.exp(-((diag - center) / bw) ** 2)[..., None] * lift
        # Warm subtle white-gold sheen
        tint = np.array([1.0, 0.96, 0.88], dtype=np.float32)[None, None, :]
        return np.clip(img.astype(np.float32) + sheen * tint, 0, 255).astype(np.uint8)

    if eid == "V06":  # film_grain_subtle (deterministic seeded per-frame luma grain)
        sigma = float(params.get("sigma", 5.5)) * intensity * bell_envelope(u, 0.15)
        if sigma <= 0.1:
            return img
        rng = np.random.default_rng((seed * 1000003 + frame_idx * 9973) & 0xFFFFFFFF)
        # Generate grain at half-res and upsample mildly for pleasing organic film texture + speed
        gh, gw = max(16, h // 2), max(16, w // 2)
        noise_small = rng.normal(0.0, sigma, size=(gh, gw)).astype(np.float32)
        noise = cv2.resize(noise_small, (w, h), interpolation=cv2.INTER_LINEAR)[..., None]
        return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # =========================================================================
    # MOTION GRAPHICS & AUDIO-REACTIVE (6): G01..G06
    # =========================================================================
    if eid == "G01":  # edge_lines
        alpha = float(params.get("opacity", 0.62)) * intensity * bell_envelope(u, 0.22)
        if alpha <= 1e-4:
            return img
        margin = float(params.get("margin_frac", 0.04))
        color = _hex_to_rgb(str(params.get("accent_hex", "#F4F6FA")))
        p = bell_envelope(u, 0.25)
        overlay = img.copy()
        mx, my = max(8, int(round(w * margin))), max(8, int(round(h * margin)))
        thick = max(2, int(round(min(w, h) * 0.0040)))
        seg_w = int(round((w - 2 * mx) * 0.30 * p))
        tick = max(8, int(round(min(w, h) * 0.028 * p)))
        cv2.line(overlay, (mx, my), (mx + seg_w, my), color, thick, cv2.LINE_AA)
        cv2.line(overlay, (mx, my), (mx, my + tick), color, thick, cv2.LINE_AA)
        cv2.line(overlay, (w - mx - seg_w, h - my), (w - mx, h - my), color, thick, cv2.LINE_AA)
        cv2.line(overlay, (w - mx, h - my - tick), (w - mx, h - my), color, thick, cv2.LINE_AA)
        return cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0.0)

    if eid == "G02":  # ring_accent
        alpha = float(params.get("opacity", 0.62)) * intensity * bell_envelope(u, 0.22)
        if alpha <= 1e-4:
            return img
        cx = int(round(w * float(params.get("cx", 0.84))))
        cy = int(round(h * float(params.get("cy", 0.16))))
        base_r = max(14, int(round(min(w, h) * float(params.get("radius_frac", 0.055)))))
        color = _hex_to_rgb(str(params.get("accent_hex", "#FFD08A")))
        p = ease_out_cubic(u)
        r1 = max(4, int(round(base_r * (0.45 + 0.65 * p))))
        r2 = max(3, int(round(base_r * (0.25 + 0.45 * p))))
        thick = max(2, int(round(min(w, h) * 0.004)))
        overlay = img.copy()
        sweep_deg = int(round(320 * bell_envelope(u, 0.28)))
        cv2.ellipse(overlay, (cx, cy), (r1, r1), -90, 0, max(15, sweep_deg), color, thick, cv2.LINE_AA)
        cv2.circle(overlay, (cx, cy), r2, color, max(1, thick - 1), cv2.LINE_AA)
        return cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0.0)

    if eid == "G03":  # directional_shapes
        alpha = float(params.get("opacity", 0.58)) * intensity * bell_envelope(u, 0.22)
        if alpha <= 1e-4:
            return img
        side = str(params.get("side", "right"))
        color = _hex_to_rgb(str(params.get("accent_hex", "#7CE0FF")))
        overlay = img.copy()
        thick = max(2, int(round(min(w, h) * 0.0042)))
        sz = max(8, int(round(min(w, h) * 0.022)))
        base_x = int(round(w * (0.93 if side == "right" else 0.07)))
        drift_y = int(round((ease_in_out_cubic(u) - 0.5) * h * 0.10))
        cy = h // 2 + drift_y
        sgn = -1 if side == "right" else 1
        for k in (-1, 0, 1):
            yk = cy + k * int(sz * 1.6)
            pts = np.array([[base_x, yk - sz], [base_x + sgn * sz, yk], [base_x, yk + sz]], dtype=np.int32)
            cv2.polylines(overlay, [pts], False, color, thick, cv2.LINE_AA)
        return cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0.0)

    if eid == "G04":  # ambient_particles (strictly bounded count <=20, peripheral zone)
        alpha = float(params.get("opacity", 0.42)) * intensity * bell_envelope(u, 0.22)
        if alpha <= 1e-4:
            return img
        count = min(20, max(4, int(params.get("count", 14))))
        color = _hex_to_rgb(str(params.get("accent_hex", "#FFF4DE")))
        rng = np.random.default_rng((seed * 7919 + 104729) & 0xFFFFFFFF)
        overlay = img.copy()
        for i in range(count):
            # Place motes strictly in outer peripheral ring (never covering central subject)
            ang = float(rng.uniform(0.0, 2.0 * math.pi))
            rad_n = float(rng.uniform(0.36, 0.48))
            vx = float(rng.uniform(-0.04, 0.04))
            vy = float(rng.uniform(-0.08, -0.02))
            px = int(round(w * (0.5 + math.cos(ang) * rad_n + vx * u)))
            py = int(round(h * (0.5 + math.sin(ang) * rad_n + vy * u)))
            pr = max(2, int(round(min(w, h) * float(rng.uniform(0.004, 0.011)))))
            if 0 <= px < w and 0 <= py < h:
                cv2.circle(overlay, (px, py), pr, color, -1, cv2.LINE_AA)
        return cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0.0)

    if eid == "G05":  # waveform_accent
        if audio_env is None or audio_env.size == 0:
            return img
        alpha = float(params.get("opacity", 0.60)) * intensity * bell_envelope(u, 0.20)
        if alpha <= 1e-4:
            return img
        bars = int(params.get("bars", 20))
        color = _hex_to_rgb(str(params.get("accent_hex", "#7CE0FF")))
        pos = str(params.get("position", "bottom"))
        overlay = img.copy()
        total_w = int(round(w * 0.44))
        x_start = (w - total_w) // 2
        base_y = int(round(h * 0.935)) if pos == "bottom" else int(round(h * 0.065))
        max_bar_h = max(10, int(round(h * 0.040)))
        step_x = max(4, total_w // max(1, bars))
        bar_w = max(2, int(round(step_x * 0.54)))
        n_env = audio_env.size
        half_b = bars // 2
        for b in range(bars):
            sample_f = max(0, min(n_env - 1, frame_idx + (b - half_b)))
            level = float(audio_env[sample_f])
            taper = math.sin(math.pi * (b + 0.5) / bars) ** 0.7
            bh = max(3, int(round((0.16 + 0.84 * level) * max_bar_h * taper)))
            bx = x_start + b * step_x
            cv2.rectangle(overlay, (bx, base_y - bh // 2), (bx + bar_w, base_y + bh // 2), color, -1, cv2.LINE_AA)
        return cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0.0)

    if eid == "G06":  # pulse_border (damped audio-reactive soft perimeter glow, non-strobe)
        if audio_env is None or audio_env.size == 0:
            return img
        # 5-frame moving average smooths energy so it never strobes!
        lo = max(0, frame_idx - 2)
        hi = min(audio_env.size, frame_idx + 3)
        smooth_lvl = float(np.mean(audio_env[lo:hi])) if hi > lo else 0.0
        base_op = float(params.get("opacity", 0.45))
        alpha = base_op * intensity * bell_envelope(u, 0.22) * (0.35 + 0.65 * smooth_lvl)
        if alpha <= 1e-4:
            return img
        margin = float(params.get("margin_frac", 0.025))
        color = _hex_to_rgb(str(params.get("accent_hex", "#A8FFCE")))
        mx, my = max(4, int(round(w * margin))), max(4, int(round(h * margin)))
        thick = max(3, int(round(min(w, h) * (0.005 + 0.004 * smooth_lvl))))
        overlay = img.copy()
        cv2.rectangle(overlay, (mx, my), (w - mx, h - my), color, thick, cv2.LINE_AA)
        return cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0.0)

    raise ValueError(f"Unsupported effect_id in frame engine: {eid}")


def smoothstep_arr(x: np.ndarray) -> np.ndarray:
    c = np.clip(x, 0.0, 1.0)
    return c * c * (3.0 - 2.0 * c)


def apply_frame_effects(
    img: np.ndarray,
    active_events: list[EffectEvent],
    frame_idx: int,
    ring_buffer: dict[int, np.ndarray],
    audio_env: np.ndarray | None,
    seed: int,
) -> np.ndarray:
    """Evaluate all active events for `frame_idx` in deterministic z-order."""
    layer_order = {"transform": 0, "mask": 1, "transition": 2, "color": 3, "overlay": 4}
    ordered = sorted(active_events, key=lambda e: (layer_order.get(e.layer, 9), e.priority, e.id))
    out = img
    for ev in ordered:
        out = apply_single_effect(out, ev, frame_idx, ring_buffer, audio_env, seed)
    return out
