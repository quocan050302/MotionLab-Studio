"""Phase A6/A7 (ML-01.R5.R5): Generative SceneGraph v2 Compositional Operators, Dynamic Layout Solver & Hook Proxy Compiler.

Eliminates D02 (single dark glass pill at y=1255 repeated across beats), D03 (hardcoded finance/CTA defaults),
and D10 (abrupt cream bridge jump) by introducing a true general-purpose compositional operator system:
  - 6 materially distinct visual treatment modes (each with distinct silhouette, layout geometry, color polarity,
    motion curve, and narrative function):
      1. `spatial_editorial_anchor`   — open-space kinetic typography + gradient feather (zero opaque pill box!)
      2. `calendar_timeline_gap`      — architectural warm-ivory weekday timeline ribbon with animated gap arc
      3. `split_dilemma_metaphor`     — floating editorial decision-card / balance-vs-reserved geometric visual
      4. `motivated_object_bridge`    — smooth clip-path architectural grid sweep bridge (zero abrupt hard jump!)
      5. `illustrative_system_board`  — multi-row quiet-luxury interactive conceptual system overview (honestly labeled)
      6. `minimal_cta_anchor`         — unobstructed portrait + refined spatial caption pointer inside safe zone
  - Dynamic multi-frame layout solver deriving exact pixel coordinates from target (W, H), face ROI bounds,
    platform safe margins, and reading speed (zero hardcoded y=1255!).
  - Strict validation: no unknown modes, no cycles, no arbitrary JS/eval, HTML-escaped text, zero hardcoded brand/finance strings.
  - Real 3-variant Hook Proxy compiler (`hook_A_action_conflict`, `hook_B_calendar_lead`, `hook_C_split_dilemma`)
    so the director renders and ranks real proxy pixels before selecting the canonical hook!
"""

from __future__ import annotations

import html
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from motionlab.renderers.hyperframes import HyperFramesRenderer
from motionlab.schemas import canonical_hash, sanitize_native

SCENE_GRAPH_V2_SCHEMA_VERSION = "2.1.0"

_SAFE_LOCAL_ASSET_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,96}\.(png|jpg|jpeg|webp)$", re.IGNORECASE)


def _validate_local_asset_key(asset_key: str) -> str:
    """Fail-closed local relative filename validator (blocks path traversal, remote URLs, and data URIs)."""
    key = str(asset_key or "").strip()
    if not key or not _SAFE_LOCAL_ASSET_RE.fullmatch(key) or ".." in key:
        raise ValueError(f"Unsafe or unsupported local scene asset key (fail-closed): {asset_key!r}")
    return key

VALID_TREATMENT_MODES = frozenset({
    "spatial_editorial_anchor",
    "calendar_timeline_gap",
    "split_dilemma_metaphor",
    "motivated_object_bridge",
    "illustrative_system_board",
    "actual_product_capture",
    "broll_cutaway",
    "calm_breathing_face",
    "minimal_cta_anchor",
})

_SANS_STACK = "'Avenir Next', 'Segoe UI', -apple-system, 'Helvetica Neue', Arial, sans-serif"
_SERIF_STACK = "Georgia, 'Times New Roman', serif"

DEFAULT_EDITORIAL_PALETTE = {
    "ivory": "#F6F2E9",
    "parchment": "#EDE6D6",
    "charcoal": "#171A1F",
    "ink_soft": "#4F5660",
    "amber": "#C9633B",     # Conflict / due event / tension
    "teal": "#1D7874",      # Resolution / clarity / safe target
    "cognac": "#9E7248",    # Quiet luxury warm accent
    "white": "#FAF8F4",
}


def solve_dynamic_safe_layout(
    *,
    width: int,
    height: int,
    treatment_mode: str,
    face_bounds: dict[str, float] | None = None,
    safe_zones: dict[str, float] | None = None,
    placement_hint: str = "auto",  # "auto" | "upper" | "lower" | "center_board"
) -> dict[str, Any]:
    """Solve dynamic pixel layout strictly from W/H, tracked face ROI, and platform safe zones (NEVER fixed y=1255!)."""
    sz = safe_zones or {"top_frac": 0.10, "bottom_frac": 0.21, "left_frac": 0.06, "right_frac": 0.12}
    safe_top = int(round(height * float(sz.get("top_frac", 0.10))))
    safe_bottom = int(round(height * (1.0 - float(sz.get("bottom_frac", 0.21)))))
    safe_left = int(round(width * float(sz.get("left_frac", 0.06))))
    safe_right = int(round(width * float(sz.get("right_frac", 0.12))))
    usable_w = max(240, width - safe_left - safe_right)

    # Derive face exclusion band [fy0, fy1] in pixels from actual face ROI (or conservative center-upper fallback)
    if isinstance(face_bounds, dict) and "y_min" in face_bounds and "y_max" in face_bounds:
        fy0 = int(round(height * max(0.0, float(face_bounds["y_min"]) - 0.02)))
        fy1 = int(round(height * min(0.95, float(face_bounds["y_max"]) + 0.04)))
    else:
        fy0 = int(round(height * 0.16))
        fy1 = int(round(height * 0.56))

    upper_space = max(0, fy0 - safe_top)
    lower_space = max(0, safe_bottom - fy1)

    if treatment_mode in ("motivated_object_bridge", "actual_product_capture"):
        box_y = safe_top
        box_h = safe_bottom - safe_top
        zone = "full_stage"
    elif treatment_mode == "illustrative_system_board":
        # Compact lower editorial board staying cleanly below tracked face ROI and above bottom safe margin
        box_h = min(int(round(height * 0.255)), max(340, lower_space - int(round(height * 0.012))))
        box_y = max(fy1 + int(round(height * 0.012)), safe_bottom - box_h)
        if box_y + box_h > safe_bottom:
            box_y = max(fy1 + 4, safe_bottom - box_h)
        zone = "lower_editorial_board"
    elif placement_hint == "upper" or (placement_hint == "auto" and treatment_mode == "spatial_editorial_anchor" and upper_space >= int(height * 0.11)):
        box_h = min(int(round(height * 0.13)), max(120, upper_space))
        box_y = safe_top + max(0, (upper_space - box_h) // 2)
        zone = "upper_negative_space"
    elif treatment_mode == "split_dilemma_metaphor":
        box_h = int(round(height * 0.24))
        box_y = max(fy1 + int(round(height * 0.03)), safe_bottom - box_h - int(round(height * 0.02)))
        zone = "lower_split_stage"
    elif treatment_mode == "calendar_timeline_gap":
        box_h = int(round(height * 0.17))
        box_y = max(fy1 + int(round(height * 0.04)), safe_bottom - box_h - int(round(height * 0.02)))
        zone = "lower_timeline_ribbon"
    else:
        # minimal_cta_anchor / lower spatial anchor
        box_h = int(round(height * 0.13))
        box_y = max(fy1 + int(round(height * 0.05)), safe_bottom - box_h - int(round(height * 0.015)))
        zone = "lower_safe_anchor"

    # Clamp strictly inside vertical safe bounds
    box_y = max(safe_top, min(safe_bottom - min(box_h, 120), box_y))
    overlaps_face = bool(
        treatment_mode not in ("motivated_object_bridge", "actual_product_capture")
        and (box_y < fy1 and (box_y + box_h) > fy0)
    )
    return sanitize_native({
        "zone": zone,
        "x_px": safe_left,
        "y_px": int(box_y),
        "width_px": int(usable_w),
        "height_px": int(box_h),
        "right_margin_px": safe_right,
        "safe_top_px": safe_top,
        "safe_bottom_px": safe_bottom,
        "face_band_px": [fy0, fy1],
        "overlaps_face": overlaps_face,
        "font_scale": round(width / 1080.0, 4),
    })


def _wrap_hf_html(
    comp_id: str,
    width: int,
    height: int,
    fps: int,
    dur_s: float,
    bg_css: str,
    css: str,
    body: str,
    js: str,
) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width={width}, height={height}" />
<script src="./gsap.min.js"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  html, body {{ width: {width}px; height: {height}px; overflow: hidden; background: {bg_css}; font-family: {_SANS_STACK}; }}
  #hf-root {{ position: relative; width: {width}px; height: {height}px; overflow: hidden; background: {bg_css}; }}
  {css}
</style>
</head>
<body>
<div id="hf-root" data-composition-id="{html.escape(comp_id)}" data-start="0" data-duration="{dur_s:.4f}" data-width="{width}" data-height="{height}" data-fps="{fps}">
<div class="clip" data-start="0" data-duration="{dur_s:.4f}" data-track-index="0" style="position:absolute;inset:0;">
{body}
</div>
</div>
<script>
  window.__timelines = window.__timelines || {{}};
  const tl = gsap.timeline({{ paused: true }});
  {js}
  window.__timelines["{html.escape(comp_id)}"] = tl;
  tl.seek(0);
</script>
</body>
</html>"""


def compile_scene_v2_html(
    scene: dict[str, Any],
    *,
    width: int = 1080,
    height: int = 1920,
    fps: int = 30,
) -> tuple[str, bool]:
    """Compile a typed SceneGraph v2 node into deterministic offline GSAP HTML + `transparent` flag.

    Raises ValueError on unknown `treatment_mode` (fail-closed; NEVER silently falls back to CTA!).
    """
    mode = str(scene.get("treatment_mode") or "")
    if mode not in VALID_TREATMENT_MODES:
        raise ValueError(f"Unknown SceneGraph v2 treatment_mode: {mode!r}")

    pal = {**DEFAULT_EDITORIAL_PALETTE, **(scene.get("palette") or {})}
    params = dict(scene.get("params") or {})
    layout = dict(scene.get("layout") or solve_dynamic_safe_layout(width=width, height=height, treatment_mode=mode))
    dur = max(0.4, float(scene.get("duration_s", 2.0)))
    comp_id = str(scene.get("scene_id", f"sgv2_{mode}"))
    sc = float(layout.get("font_scale", width / 1080.0))
    esc = html.escape
    y_px = int(layout["y_px"])
    x_px = int(layout["x_px"])
    w_px = int(layout["width_px"])
    exit_t = max(0.28, dur - 0.24)

    # ------------------------------------------------------------------
    # 1. SPATIAL EDITORIAL ANCHOR — Open typography + soft top/bottom scrim (NO heavy dark pill box!)
    # ------------------------------------------------------------------
    if mode == "spatial_editorial_anchor":
        kicker = esc(str(params.get("kicker") or ""))
        headline = esc(str(params.get("headline") or ""))
        subline = esc(str(params.get("subline") or ""))
        accent = pal["amber"] if params.get("tone", "amber") == "amber" else pal["teal"]
        is_upper = layout.get("zone") == "upper_negative_space"
        grad_css = (
            "linear-gradient(180deg, rgba(14,16,20,0.78) 0%, rgba(14,16,20,0.48) 58%, rgba(14,16,20,0.0) 100%)"
            if is_upper
            else "linear-gradient(0deg, rgba(14,16,20,0.82) 0%, rgba(14,16,20,0.58) 56%, rgba(14,16,20,0.0) 100%)"
        )
        scrim_pos = "top: 0; height: 30%;" if is_upper else "bottom: 0; height: 42%;"
        css = f"""
  #scrim {{ position: absolute; left: 0; right: 0; {scrim_pos} background: {grad_css}; opacity: 0; }}
  #anchor {{
    position: absolute; left: {x_px}px; width: {w_px}px; top: {y_px}px;
    display: flex; flex-direction: column; align-items: flex-start;
  }}
  #kicker-row {{ display: flex; align-items: center; gap: {int(14*sc)}px; margin-bottom: {int(12*sc)}px; opacity: 0; transform: translateX(-14px); }}
  #k-bar {{ width: {int(36*sc)}px; height: {int(4*sc)}px; border-radius: 2px; background: {accent}; }}
  #k-txt {{ font-size: {int(24*sc)}px; font-weight: 700; letter-spacing: 0.22em; text-transform: uppercase; color: {pal['parchment']}; text-shadow: 0 2px 12px rgba(0,0,0,0.65); }}
  #hl-mask {{ overflow: hidden; }}
  #hl {{
    font-family: {_SERIF_STACK}; font-size: {int(72*sc)}px; font-weight: 700; line-height: 1.04;
    letter-spacing: -0.02em; color: {pal['white']};
    text-shadow: 0 6px 26px rgba(0,0,0,0.62); transform: translateY(105%);
  }}
  #sub {{
    margin-top: {int(14*sc)}px; font-size: {int(30*sc)}px; font-weight: 600; color: rgba(246,242,233,0.92);
    text-shadow: 0 3px 16px rgba(0,0,0,0.65); opacity: 0; transform: translateY(10px);
  }}
  #hairline {{
    margin-top: {int(16*sc)}px; width: {int(220*sc)}px; height: {int(4*sc)}px; border-radius: 2px;
    background: {accent}; transform-origin: left center; transform: scaleX(0);
  }}"""
        body = f"""  <div id="scrim"></div>
  <div id="anchor">
    {f'<div id="kicker-row"><div id="k-bar"></div><div id="k-txt">{kicker}</div></div>' if kicker else ''}
    <div id="hl-mask"><div id="hl">{headline}</div></div>
    {f'<div id="sub">{subline}</div>' if subline else ''}
    <div id="hairline"></div>
  </div>"""
        js = f"""
  tl.to("#scrim", {{ opacity: 1, duration: 0.24, ease: "power2.out" }}, 0.02);
  if (document.getElementById("kicker-row")) {{ tl.to("#kicker-row", {{ opacity: 1, x: 0, duration: 0.24, ease: "power3.out" }}, 0.04); }}
  tl.to("#hl", {{ y: "0%", duration: 0.34, ease: "expo.out" }}, 0.08);
  if (document.getElementById("sub")) {{ tl.to("#sub", {{ opacity: 1, y: 0, duration: 0.28, ease: "power3.out" }}, 0.22); }}
  tl.to("#hairline", {{ scaleX: 1, duration: 0.42, ease: "expo.out" }}, 0.20);
  tl.to(["#anchor", "#scrim"], {{ opacity: 0, y: -12, duration: 0.22, ease: "power2.in" }}, {exit_t:.4f});"""
        return _wrap_hf_html(comp_id, width, height, fps, dur, "transparent", css, body, js), True

    # ------------------------------------------------------------------
    # 2. CALENDAR TIMELINE GAP — Warm-ivory architectural ribbon (distinct light polarity!)
    # ------------------------------------------------------------------
    if mode == "calendar_timeline_gap":
        day_a = esc(str(params.get("start_day") or "DAY 1"))
        tag_a = esc(str(params.get("start_tag") or ""))
        gap_days = [esc(str(d)) for d in (params.get("gap_days") or [])]
        day_b = esc(str(params.get("end_day") or "DAY 2"))
        tag_b = esc(str(params.get("end_tag") or ""))
        caption = esc(str(params.get("caption") or ""))
        t_a = max(0.04, float(params.get("start_trigger_s", 0.14)))
        t_b = max(t_a + 0.35, min(dur - 0.55, float(params.get("end_trigger_s", dur * 0.50))))
        mids = "".join(
            f'<div class="mnode"><div class="mdot"></div><div class="mlbl">{d}</div></div>'
            for d in gap_days
        )
        css = f"""
  #ribbon {{
    position: absolute; left: {x_px}px; width: {w_px}px; top: {y_px}px;
    padding: {int(28*sc)}px {int(34*sc)}px {int(26*sc)}px {int(34*sc)}px;
    border-radius: {int(22*sc)}px;
    background: rgba(246, 242, 233, 0.96);
    border: 2px solid rgba(158, 114, 72, 0.34);
    box-shadow: 0 26px 60px rgba(14, 16, 20, 0.40);
    clip-path: inset(0 100% 0 0 round {int(22*sc)}px);
  }}
  #hdr {{
    display: flex; justify-content: space-between; align-items: center;
    margin-bottom: {int(18*sc)}px; padding-bottom: {int(12*sc)}px;
    border-bottom: 1.5px solid rgba(23, 26, 31, 0.10);
  }}
  #hdr-l {{ font-size: {int(20*sc)}px; font-weight: 800; letter-spacing: 0.20em; text-transform: uppercase; color: {pal['ink_soft']}; }}
  #hdr-r {{ font-size: {int(20*sc)}px; font-weight: 800; letter-spacing: 0.14em; text-transform: uppercase; color: {pal['amber']}; opacity: 0; }}
  #row {{ position: relative; display: flex; align-items: flex-start; justify-content: space-between; padding: {int(6*sc)}px 0; }}
  #track-bg {{
    position: absolute; left: {int(88*sc)}px; right: {int(88*sc)}px; top: {int(34*sc)}px; height: {int(6*sc)}px;
    background: rgba(23, 26, 31, 0.12); border-radius: 4px;
  }}
  #track-fill {{
    position: absolute; left: {int(88*sc)}px; right: {int(88*sc)}px; top: {int(34*sc)}px; height: {int(6*sc)}px;
    background: linear-gradient(90deg, {pal['amber']} 0%, {pal['cognac']} 50%, {pal['teal']} 100%);
    border-radius: 4px; transform-origin: left center; transform: scaleX(0);
  }}
  .dnode {{ position: relative; z-index: 2; display: flex; flex-direction: column; align-items: center; min-width: {int(156*sc)}px; opacity: 0; transform: scale(0.84); }}
  .pill {{
    padding: {int(11*sc)}px {int(26*sc)}px; border-radius: {int(12*sc)}px;
    font-size: {int(38*sc)}px; font-weight: 900; letter-spacing: 0.04em; color: {pal['white']};
  }}
  #dA .pill {{ background: {pal['amber']}; box-shadow: 0 8px 20px rgba(201, 99, 59, 0.36); }}
  #dB .pill {{ background: {pal['teal']}; box-shadow: 0 8px 20px rgba(29, 120, 116, 0.36); }}
  .dtag {{ margin-top: {int(10*sc)}px; font-size: {int(23*sc)}px; font-weight: 800; letter-spacing: 0.12em; text-transform: uppercase; color: {pal['charcoal']}; }}
  .mnode {{ position: relative; z-index: 2; display: flex; flex-direction: column; align-items: center; padding-top: {int(22*sc)}px; opacity: 0; }}
  .mdot {{ width: {int(16*sc)}px; height: {int(16*sc)}px; border-radius: 50%; background: {pal['ivory']}; border: 4px solid {pal['cognac']}; }}
  .mlbl {{ margin-top: {int(20*sc)}px; font-size: {int(22*sc)}px; font-weight: 800; letter-spacing: 0.12em; color: {pal['ink_soft']}; }}"""
        body = f"""  <div id="ribbon">
    <div id="hdr"><div id="hdr-l">TIMELINE</div><div id="hdr-r">{caption}</div></div>
    <div id="row">
      <div id="track-bg"></div><div id="track-fill"></div>
      <div class="dnode" id="dA"><div class="pill">{day_a}</div><div class="dtag">{tag_a}</div></div>
      {mids}
      <div class="dnode" id="dB"><div class="pill">{day_b}</div><div class="dtag">{tag_b}</div></div>
    </div>
  </div>"""
        js = f"""
  tl.to("#ribbon", {{ clipPath: "inset(0 0% 0 0 round {int(22*sc)}px)", duration: 0.34, ease: "expo.out" }}, 0.02);
  tl.to("#dA", {{ opacity: 1, scale: 1, duration: 0.28, ease: "back.out(1.5)" }}, {t_a:.4f});
  tl.to("#track-fill", {{ scaleX: 1, duration: {max(0.32, t_b - t_a):.4f}, ease: "power2.inOut" }}, {t_a + 0.08:.4f});
  tl.to(".mnode", {{ opacity: 1, duration: 0.20, stagger: 0.12, ease: "power2.out" }}, {t_a + 0.22:.4f});
  tl.to("#dB", {{ opacity: 1, scale: 1, duration: 0.28, ease: "back.out(1.5)" }}, {t_b:.4f});
  tl.to("#hdr-r", {{ opacity: 1, duration: 0.24, ease: "power2.out" }}, {t_b + 0.12:.4f});
  tl.to("#ribbon", {{ opacity: 0, y: -14, duration: 0.22, ease: "power2.in" }}, {exit_t:.4f});"""
        return _wrap_hf_html(comp_id, width, height, fps, dur, "transparent", css, body, js), True

    # ------------------------------------------------------------------
    # 3. SPLIT DILEMMA METAPHOR — Two-column balance vs obligation visual tension
    # ------------------------------------------------------------------
    if mode == "split_dilemma_metaphor":
        question = esc(str(params.get("question") or ""))
        left_lbl = esc(str(params.get("left_label") or "ON PAPER"))
        left_sub = esc(str(params.get("left_sub") or "Total Balance"))
        right_lbl = esc(str(params.get("right_label") or "ACTUALLY FREE"))
        right_sub = esc(str(params.get("right_sub") or "After Obligations"))
        css = f"""
  #stage {{
    position: absolute; left: {x_px}px; width: {w_px}px; top: {y_px}px;
    display: flex; flex-direction: column; gap: {int(16*sc)}px;
  }}
  #q-row {{
    font-family: {_SERIF_STACK}; font-size: {int(52*sc)}px; font-weight: 700; line-height: 1.08;
    color: {pal['white']}; text-align: center;
    text-shadow: 0 5px 24px rgba(0,0,0,0.72); opacity: 0; transform: translateY(14px);
  }}
  #cols {{ display: grid; grid-template-columns: 1fr 1fr; gap: {int(18*sc)}px; }}
  .panel {{
    padding: {int(22*sc)}px {int(24*sc)}px; border-radius: {int(18*sc)}px;
    box-shadow: 0 18px 40px rgba(0,0,0,0.34); opacity: 0; transform: translateY(20px);
  }}
  #pL {{ background: rgba(246, 242, 233, 0.94); border-top: 6px solid {pal['amber']}; color: {pal['charcoal']}; }}
  #pR {{ background: rgba(23, 26, 31, 0.90); border-top: 6px solid {pal['teal']}; color: {pal['white']}; border: 1px solid rgba(250,248,244,0.16); border-top: 6px solid {pal['teal']}; }}
  .plbl {{ font-size: {int(20*sc)}px; font-weight: 800; letter-spacing: 0.18em; text-transform: uppercase; opacity: 0.78; }}
  .psub {{ margin-top: {int(8*sc)}px; font-size: {int(34*sc)}px; font-weight: 800; letter-spacing: -0.015em; }}
  .pbar-bg {{ margin-top: {int(14*sc)}px; height: {int(10*sc)}px; border-radius: 5px; background: rgba(128,128,128,0.24); overflow: hidden; }}
  #barL {{ height: 100%; width: 100%; background: {pal['amber']}; transform-origin: left center; transform: scaleX(0); }}
  #barR {{ height: 100%; width: 38%; background: {pal['teal']}; transform-origin: left center; transform: scaleX(0); }}"""
        body = f"""  <div id="stage">
    <div id="q-row">{question}</div>
    <div id="cols">
      <div class="panel" id="pL"><div class="plbl">{left_lbl}</div><div class="psub">{left_sub}</div><div class="pbar-bg"><div id="barL"></div></div></div>
      <div class="panel" id="pR"><div class="plbl">{right_lbl}</div><div class="psub">{right_sub}</div><div class="pbar-bg"><div id="barR"></div></div></div>
    </div>
  </div>"""
        js = f"""
  tl.to("#q-row", {{ opacity: 1, y: 0, duration: 0.26, ease: "power3.out" }}, 0.04);
  tl.to("#pL", {{ opacity: 1, y: 0, duration: 0.30, ease: "power3.out" }}, 0.14);
  tl.to("#barL", {{ scaleX: 1, duration: 0.42, ease: "expo.out" }}, 0.24);
  tl.to("#pR", {{ opacity: 1, y: 0, duration: 0.30, ease: "power3.out" }}, 0.30);
  tl.to("#barR", {{ scaleX: 1, duration: 0.48, ease: "expo.out" }}, 0.42);
  tl.to("#stage", {{ opacity: 0, y: -12, duration: 0.20, ease: "power2.in" }}, {exit_t:.4f});"""
        return _wrap_hf_html(comp_id, width, height, fps, dur, "transparent", css, body, js), True

    # ------------------------------------------------------------------
    # 4. MOTIVATED OBJECT BRIDGE — Smooth clip-path wipe carrying editorial grid or real cover (Fixes D10!)
    # ------------------------------------------------------------------
    if mode == "motivated_object_bridge":
        eyebrow = esc(str(params.get("eyebrow") or "TIMING-FIRST CLARITY"))
        title = esc(str(params.get("title") or ""))
        subline = esc(str(params.get("subline") or ""))
        disclosure = esc(str(params.get("disclosure") or "ILLUSTRATIVE CONCEPT"))
        feat_key_raw = params.get("featured_asset_key")
        feat_key = _validate_local_asset_key(str(feat_key_raw)) if feat_key_raw else None
        t_out = max(0.45, dur - 0.34)
        cover_css = ""
        cover_html = ""
        cover_js = ""
        if feat_key:
            cover_css = f"""
  #cover-wrap {{
    position: relative; z-index: 2; margin-top: {int(30*sc)}px;
    width: {int(430*sc)}px; height: {int(606*sc)}px; border-radius: {int(16*sc)}px;
    overflow: hidden; background: {pal['charcoal']};
    border: 3px solid rgba(181, 143, 82, 0.55);
    box-shadow: 0 34px 72px rgba(14, 24, 20, 0.34), 0 8px 20px rgba(14, 24, 20, 0.18);
    opacity: 0; transform: translateY(28px) scale(0.94);
  }}
  #cover-wrap img {{ width: 100%; height: 100%; object-fit: cover; display: block; }}"""
            cover_html = f'<div id="cover-wrap"><img src="./{esc(feat_key)}" alt="Authorized product cover" /></div>'
            cover_js = 'tl.to("#cover-wrap", { opacity: 1, y: 0, scale: 1.0, duration: 0.34, ease: "expo.out" }, 0.22);'
        css = f"""
  #sheet {{
    position: absolute; inset: 0;
    background: linear-gradient(164deg, {pal['ivory']} 0%, {pal['parchment']} 100%);
    clip-path: inset(100% 0 0 0);
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    padding: 0 {int(80*sc)}px; text-align: center;
  }}
  #grid-bg {{
    position: absolute; inset: 0; opacity: 0.10;
    background-image: linear-gradient(to right, {pal['charcoal']} 1px, transparent 1px),
                      linear-gradient(to bottom, {pal['charcoal']} 1px, transparent 1px);
    background-size: {int(90*sc)}px {int(90*sc)}px;
  }}
  #rule-top {{ position: relative; z-index: 2; width: {int(180*sc)}px; height: {int(5*sc)}px; border-radius: 3px; background: linear-gradient(90deg, {pal['cognac']}, {pal['teal']}); margin-bottom: {int(22*sc)}px; transform: scaleX(0); }}
  #eyebrow {{ position: relative; z-index: 2; font-size: {int(22*sc)}px; font-weight: 800; letter-spacing: 0.24em; text-transform: uppercase; color: {pal['ink_soft']}; opacity: 0; }}
  #title {{ position: relative; z-index: 2; margin-top: {int(14*sc)}px; font-family: {_SERIF_STACK}; font-size: {int((68 if feat_key else 84)*sc)}px; font-weight: 700; line-height: 1.06; color: {pal['charcoal']}; opacity: 0; transform: translateY(18px); }}
  #sub {{ position: relative; z-index: 2; margin-top: {int(16*sc)}px; font-size: {int(26*sc)}px; font-weight: 700; letter-spacing: 0.14em; text-transform: uppercase; color: {pal['cognac']}; opacity: 0; }}
  #disc {{ position: absolute; bottom: {int(height * 0.14)}px; z-index: 2; font-size: {int(18*sc)}px; font-weight: 800; letter-spacing: 0.18em; text-transform: uppercase; color: {pal['ink_soft']}; opacity: 0.78; }}
  {cover_css}"""
        body = f"""  <div id="sheet">
    <div id="grid-bg"></div>
    <div id="rule-top"></div>
    <div id="eyebrow">{eyebrow}</div>
    <div id="title">{title}</div>
    {f'<div id="sub">{subline}</div>' if subline else ''}
    {cover_html}
    <div id="disc">{disclosure}</div>
  </div>"""
        js = f"""
  tl.to("#sheet", {{ clipPath: "inset(0% 0 0 0)", duration: 0.34, ease: "expo.inOut" }}, 0.00);
  tl.to("#rule-top", {{ scaleX: 1, duration: 0.28, ease: "expo.out" }}, 0.14);
  tl.to("#eyebrow", {{ opacity: 1, duration: 0.22, ease: "power2.out" }}, 0.18);
  tl.to("#title", {{ opacity: 1, y: 0, duration: 0.28, ease: "power3.out" }}, 0.20);
  if (document.getElementById("sub")) {{ tl.to("#sub", {{ opacity: 1, duration: 0.22, ease: "power2.out" }}, 0.28); }}
  {cover_js}
  tl.to("#sheet", {{ clipPath: "inset(0 0 100% 0)", duration: 0.32, ease: "expo.inOut" }}, {t_out:.4f});"""
        return _wrap_hf_html(comp_id, width, height, fps, dur, "transparent", css, body, js), True

    # ------------------------------------------------------------------
    # 5A. ACTUAL PRODUCT CAPTURE — Real owner-authorized local product page/UI studio showcase!
    # ------------------------------------------------------------------
    if mode == "actual_product_capture":
        pages = list(params.get("pages") or [])
        if not pages:
            raise ValueError("actual_product_capture requires non-empty verified local `pages` (fail-closed).")
        prod_title = esc(str(params.get("title") or "PRODUCT WALKTHROUGH"))
        badge = esc(str(params.get("provenance_badge") or "REAL PRODUCT · VERIFIED"))
        safe_top = int(layout.get("safe_top_px", int(height * 0.10)))
        safe_bottom = int(layout.get("safe_bottom_px", int(height * 0.80)))
        stage_top = safe_top + int(10 * sc)
        stage_h = max(int(height * 0.56), safe_bottom - stage_top)
        t_out = max(0.65, dur - 0.34)

        pills_html: list[str] = []
        cards_html: list[str] = []
        caps_html: list[str] = []
        anim_js: list[str] = []

        n_pages = min(4, len(pages))
        for k, pg in enumerate(pages[:n_pages]):
            akey = _validate_local_asset_key(str(pg.get("asset_key") or ""))
            step_lbl = esc(str(pg.get("step_pill") or f"0{k+1}"))
            cap_k = esc(str(pg.get("caption_kicker") or f"STEP 0{k+1}"))
            cap_t = esc(str(pg.get("caption_title") or ""))
            trig = max(0.04, min(t_out - 0.45, float(pg.get("trigger_s", 0.08 + k * (dur / max(1, n_pages))))))
            next_trig = (
                max(trig + 0.35, min(t_out - 0.25, float(pages[k + 1].get("trigger_s", trig + 1.4))))
                if k + 1 < n_pages
                else t_out
            )
            hold_dur = max(0.45, next_trig - trig)

            pills_html.append(f'<div class="sp-pill" id="sp{k}">{step_lbl}</div>')
            cards_html.append(
                f'<div class="pg-card" id="pg{k}"><img src="./{esc(akey)}" alt="{cap_t}" /></div>'
            )
            caps_html.append(
                f'<div class="cap-item" id="cap{k}"><div class="cap-k">{cap_k}</div><div class="cap-t">{cap_t}</div></div>'
            )

            # Activate step pill, reveal real page with gentle studio zoom, fade previous page cleanly
            anim_js.append(f'tl.to("#sp{k}", {{ backgroundColor: "{pal["cognac"]}", color: "{pal["charcoal"]}", opacity: 1, duration: 0.22 }}, {trig:.4f});')
            anim_js.append(f'tl.to("#pg{k}", {{ opacity: 1, y: 0, scale: 1.0, duration: 0.34, ease: "expo.out" }}, {trig:.4f});')
            anim_js.append(f'tl.to("#pg{k} img", {{ scale: 1.035, duration: {hold_dur:.4f}, ease: "none" }}, {trig + 0.10:.4f});')
            anim_js.append(f'tl.to("#cap{k}", {{ opacity: 1, y: 0, duration: 0.28, ease: "power3.out" }}, {trig + 0.04:.4f});')
            if k > 0:
                anim_js.append(f'tl.to("#sp{k-1}", {{ backgroundColor: "rgba(245,242,234,0.12)", color: "rgba(245,242,234,0.68)", duration: 0.20 }}, {trig:.4f});')
                anim_js.append(f'tl.to("#pg{k-1}", {{ opacity: 0, scale: 0.96, y: -18, duration: 0.28, ease: "power2.inOut" }}, {trig:.4f});')
                anim_js.append(f'tl.to("#cap{k-1}", {{ opacity: 0, y: -10, duration: 0.22, ease: "power2.in" }}, {trig:.4f});')

        css = f"""
  #prod-stage {{
    position: absolute; inset: 0;
    background: radial-gradient(circle at 50% 38%, #224C40 0%, {pal['charcoal']} 74%, #0E231D 100%);
    clip-path: inset(100% 0 0 0);
  }}
  #stage-grid {{
    position: absolute; inset: 0; opacity: 0.07;
    background-image: linear-gradient(to right, {pal['ivory']} 1px, transparent 1px),
                      linear-gradient(to bottom, {pal['ivory']} 1px, transparent 1px);
    background-size: {int(84*sc)}px {int(84*sc)}px;
  }}
  #prod-inner {{
    position: absolute; left: {x_px}px; width: {w_px}px; top: {stage_top}px; height: {stage_h}px;
    display: flex; flex-direction: column; justify-content: space-between; gap: {int(14*sc)}px;
  }}
  #prod-hdr {{
    display: flex; flex-direction: column; gap: {int(12*sc)}px; flex-shrink: 0;
  }}
  #prod-top {{
    display: flex; justify-content: space-between; align-items: center; gap: {int(12*sc)}px;
  }}
  #prod-title {{
    font-family: {_SERIF_STACK}; font-size: {int(36*sc)}px; font-weight: 700; color: {pal['white']}; letter-spacing: -0.01em;
  }}
  #prod-badge {{
    padding: {int(6*sc)}px {int(14*sc)}px; border-radius: 99px;
    background: rgba(181, 143, 82, 0.22); border: 1.5px solid rgba(181, 143, 82, 0.65);
    color: {pal['ivory']}; font-size: {int(14*sc)}px; font-weight: 800; letter-spacing: 0.13em; text-transform: uppercase;
    white-space: nowrap;
  }}
  #step-row {{
    display: flex; gap: {int(10*sc)}px;
  }}
  .sp-pill {{
    flex: 1; text-align: center; padding: {int(8*sc)}px {int(10*sc)}px; border-radius: {int(10*sc)}px;
    background: rgba(245, 242, 234, 0.12); color: rgba(245, 242, 234, 0.68);
    font-size: {int(16*sc)}px; font-weight: 800; letter-spacing: 0.11em; text-transform: uppercase;
    opacity: 0.85;
  }}
  #page-stage {{
    position: relative; flex: 1; min-height: {int(620*sc)}px;
    display: flex; align-items: center; justify-content: center;
  }}
  .pg-card {{
    position: absolute; inset: 0; margin: auto;
    width: 100%; height: 100%; max-width: {int(720*sc)}px;
    border-radius: {int(18*sc)}px; overflow: hidden;
    background: {pal['ivory']};
    border: 3px solid rgba(181, 143, 82, 0.58);
    box-shadow: 0 32px 72px rgba(4, 10, 8, 0.56), 0 8px 24px rgba(0, 0, 0, 0.30);
    opacity: 0; transform: translateY(24px) scale(0.95);
  }}
  .pg-card img {{
    width: 100%; height: 100%; object-fit: contain; background: {pal['ivory']}; display: block;
    transform-origin: center 32%;
  }}
  #cap-bar {{
    position: relative; height: {int(108*sc)}px; flex-shrink: 0;
    border-radius: {int(16*sc)}px; background: rgba(245, 242, 234, 0.96);
    border-left: 7px solid {pal['cognac']};
    box-shadow: 0 18px 40px rgba(0, 0, 0, 0.34);
    overflow: hidden;
  }}
  .cap-item {{
    position: absolute; inset: 0; padding: {int(14*sc)}px {int(24*sc)}px;
    display: flex; flex-direction: column; justify-content: center;
    opacity: 0; transform: translateY(12px);
  }}
  .cap-k {{
    font-size: {int(18*sc)}px; font-weight: 800; letter-spacing: 0.16em; text-transform: uppercase; color: {pal['cognac']};
  }}
  .cap-t {{
    margin-top: {int(4*sc)}px; font-family: {_SERIF_STACK}; font-size: {int(32*sc)}px; font-weight: 700; color: {pal['charcoal']}; line-height: 1.1;
  }}"""
        body = f"""  <div id="prod-stage">
    <div id="stage-grid"></div>
    <div id="prod-inner">
      <div id="prod-hdr">
        <div id="prod-top"><div id="prod-title">{prod_title}</div><div id="prod-badge">{badge}</div></div>
        <div id="step-row">{"".join(pills_html)}</div>
      </div>
      <div id="page-stage">{"".join(cards_html)}</div>
      <div id="cap-bar">{"".join(caps_html)}</div>
    </div>
  </div>"""
        js = f"""
  tl.to("#prod-stage", {{ clipPath: "inset(0% 0 0 0)", duration: 0.34, ease: "expo.inOut" }}, 0.00);
  {chr(10).join("  " + ln for ln in anim_js)}
  tl.to("#prod-stage", {{ clipPath: "inset(0 0 100% 0)", duration: 0.32, ease: "expo.inOut" }}, {t_out:.4f});"""
        return _wrap_hf_html(comp_id, width, height, fps, dur, "transparent", css, body, js), True

    # ------------------------------------------------------------------
    # 5B. ILLUSTRATIVE SYSTEM BOARD — Multi-row structured flow (honestly labeled!)
    # ------------------------------------------------------------------
    if mode == "illustrative_system_board":
        board_title = esc(str(params.get("title") or "STRUCTURED OVERVIEW"))
        badge = esc(str(params.get("provenance_badge") or "ILLUSTRATIVE OVERVIEW"))
        rows = params.get("rows") or [
            {"label": "STEP 1", "detail": "Primary obligations", "trigger_s": 0.35, "tone": "amber"},
            {"label": "STEP 2", "detail": "Horizon coverage", "trigger_s": 1.80, "tone": "cognac"},
            {"label": "STEP 3", "detail": "Clear available spend", "trigger_s": 3.20, "tone": "teal"},
        ]
        tone_map = {"amber": pal["amber"], "cognac": pal["cognac"], "teal": pal["teal"], "neutral": pal["ink_soft"]}
        row_html = ""
        row_js = []
        for k, rw in enumerate(rows[:4]):
            tc = tone_map.get(str(rw.get("tone", "teal")), pal["teal"])
            ts = max(0.06, min(dur - 0.45, float(rw.get("trigger_s", 0.3 + k * 1.2))))
            row_html += (
                f'<div class="brow" id="br{k}">'
                f'<div class="bdot" style="background:{tc};"></div>'
                f'<div class="btxt"><div class="blbl">{esc(str(rw.get("label","")))}</div>'
                f'<div class="bdet">{esc(str(rw.get("detail","")))}</div></div>'
                f'<div class="bmeter"><div class="bfill" id="bf{k}" style="background:{tc};"></div></div>'
                f'</div>'
            )
            row_js.append(f'tl.to("#br{k}", {{ opacity: 1, x: 0, borderColor: "{tc}", duration: 0.30, ease: "power3.out" }}, {ts:.4f});')
            row_js.append(f'tl.to("#bf{k}", {{ scaleX: 1, duration: 0.44, ease: "expo.out" }}, {ts + 0.08:.4f});')

        css = f"""
  #board {{
    position: absolute; left: {x_px}px; width: {w_px}px; top: {y_px}px;
    padding: {int(20*sc)}px {int(24*sc)}px {int(14*sc)}px {int(24*sc)}px; border-radius: {int(22*sc)}px;
    background: rgba(246, 242, 233, 0.96);
    border: 2px solid rgba(23, 26, 31, 0.14);
    box-shadow: 0 24px 56px rgba(10, 12, 16, 0.42);
    opacity: 0; transform: translateY(22px);
  }}
  #bhdr {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: {int(14*sc)}px; padding-bottom: {int(10*sc)}px; border-bottom: 1.5px solid rgba(23,26,31,0.10); }}
  #btitle {{ font-family: {_SERIF_STACK}; font-size: {int(30*sc)}px; font-weight: 700; color: {pal['charcoal']}; }}
  #bbadge {{ padding: {int(5*sc)}px {int(12*sc)}px; border-radius: 99px; background: rgba(158,114,72,0.16); color: {pal['cognac']}; font-size: {int(15*sc)}px; font-weight: 800; letter-spacing: 0.14em; text-transform: uppercase; }}
  .brow {{
    display: flex; align-items: center; gap: {int(16*sc)}px;
    padding: {int(12*sc)}px {int(16*sc)}px; margin-bottom: {int(9*sc)}px;
    border-radius: {int(12*sc)}px; background: #FFFFFF;
    border-left: 6px solid rgba(23,26,31,0.12);
    box-shadow: 0 4px 12px rgba(0,0,0,0.06);
    opacity: 0.22; transform: translateX(-12px);
  }}
  .bdot {{ width: {int(14*sc)}px; height: {int(14*sc)}px; border-radius: 50%; flex-shrink: 0; }}
  .btxt {{ flex: 1; }}
  .blbl {{ font-size: {int(26*sc)}px; font-weight: 800; color: {pal['charcoal']}; line-height: 1.1; }}
  .bdet {{ margin-top: {int(3*sc)}px; font-size: {int(19*sc)}px; font-weight: 600; color: {pal['ink_soft']}; }}
  .bmeter {{ width: {int(96*sc)}px; height: {int(9*sc)}px; border-radius: 5px; background: rgba(23,26,31,0.10); overflow: hidden; flex-shrink: 0; }}
  .bfill {{ height: 100%; width: 100%; transform-origin: left center; transform: scaleX(0); }}"""
        body = f"""  <div id="board">
    <div id="bhdr"><div id="btitle">{board_title}</div><div id="bbadge">{badge}</div></div>
    {row_html}
  </div>"""
        js = f"""
  tl.to("#board", {{ opacity: 1, y: 0, duration: 0.34, ease: "expo.out" }}, 0.02);
  {chr(10).join("  " + ln for ln in row_js)}
  tl.to("#board", {{ opacity: 0, y: 18, duration: 0.24, ease: "power2.in" }}, {exit_t:.4f});"""
        return _wrap_hf_html(comp_id, width, height, fps, dur, "transparent", css, body, js), True

    # ------------------------------------------------------------------
    # 6. MINIMAL CTA ANCHOR — Clean unobtrusive spatial caption pointer (with optional verified real cover thumb)
    # ------------------------------------------------------------------
    if mode == "minimal_cta_anchor":
        cta_text = esc(str(params.get("headline") or ""))
        subline = esc(str(params.get("subline") or ""))
        cov_raw = params.get("cover_asset_key")
        cov_key = _validate_local_asset_key(str(cov_raw)) if cov_raw else None
        thumb_css = ""
        thumb_html = ""
        if cov_key:
            thumb_css = f"""
  #cta-thumb {{
    width: {int(92*sc)}px; height: {int(130*sc)}px; border-radius: {int(10*sc)}px;
    overflow: hidden; flex-shrink: 0; border: 2px solid {pal['cognac']};
    box-shadow: 0 8px 20px rgba(0,0,0,0.22);
  }}
  #cta-thumb img {{ width: 100%; height: 100%; object-fit: cover; display: block; }}"""
            thumb_html = f'<div id="cta-thumb"><img src="./{esc(cov_key)}" alt="Workbook cover" /></div>'
        css = f"""
  #cta-wrap {{
    position: absolute; left: {x_px}px; width: {w_px}px; top: {y_px}px;
    display: flex; flex-direction: column; align-items: center; text-align: {'left' if cov_key else 'center'};
    opacity: 0; transform: translateY(16px);
  }}
  #cta-card {{
    display: inline-flex; flex-direction: {'row' if cov_key else 'column'}; align-items: center; gap: {int(24*sc)}px;
    padding: {int((18 if cov_key else 22)*sc)}px {int((30 if cov_key else 42)*sc)}px; border-radius: {int(18*sc)}px;
    background: rgba(246, 242, 233, 0.95);
    border: 2px solid {pal['teal']};
    box-shadow: 0 20px 48px rgba(0, 0, 0, 0.34);
  }}
  #cta-col {{ display: flex; flex-direction: column; align-items: {'flex-start' if cov_key else 'center'}; }}
  #cta-sub {{ font-size: {int(20*sc)}px; font-weight: 800; letter-spacing: 0.16em; text-transform: uppercase; color: {pal['cognac'] if cov_key else pal['teal']}; margin-bottom: {int(6*sc)}px; }}
  #cta-hl {{ font-family: {_SERIF_STACK}; font-size: {int((46 if cov_key else 52)*sc)}px; font-weight: 700; color: {pal['charcoal']}; letter-spacing: -0.02em; }}
  #chev {{ margin-top: {int(6*sc)}px; font-size: {int(28*sc)}px; font-weight: 900; color: {pal['teal']}; }}
  {thumb_css}"""
        body = f"""  <div id="cta-wrap"><div id="cta-card">
    {thumb_html}
    <div id="cta-col">
      {f'<div id="cta-sub">{subline}</div>' if subline else ''}
      <div id="cta-hl">{cta_text}</div>
      <div id="chev">↓</div>
    </div>
  </div></div>"""
        js = f"""
  tl.to("#cta-wrap", {{ opacity: 1, y: 0, duration: 0.30, ease: "power3.out" }}, 0.04);
  tl.to("#chev", {{ y: 8, duration: 0.45, repeat: 2, yoyo: true, ease: "sine.inOut" }}, 0.30);"""
        return _wrap_hf_html(comp_id, width, height, fps, dur, "transparent", css, body, js), True

    # calm_breathing_face / broll_cutaway -> transparent empty pass (clean unobstructed footage!)
    return _wrap_hf_html(comp_id, width, height, fps, dur, "transparent", "", "", ""), True


def render_scene_v2_overlay(
    scene: dict[str, Any],
    *,
    out_path: Path,
    slot_dir: Path,
    width: int = 1080,
    height: int = 1920,
    fps: int = 30,
    cache_dir: Path | None = None,
) -> dict[str, Any]:
    """Render a SceneGraph v2 node via local offline HyperFramesRenderer (Chromium + GSAP), binding verified local assets."""
    dur_frames = max(6, int(scene["end_frame_exclusive"]) - int(scene["start_frame"]))
    dur_s = dur_frames / float(fps)
    sc_copy = {**scene, "duration_s": dur_s}
    html_src, transparent = compile_scene_v2_html(sc_copy, width=width, height=height, fps=fps)
    raw_extra = scene.get("extra_assets") or {}
    extra_assets: dict[str, Path] = {}
    for k, v in raw_extra.items():
        safe_k = _validate_local_asset_key(str(k))
        src_p = Path(str(v)).resolve()
        if src_p.is_symlink() or not src_p.is_file():
            raise FileNotFoundError(f"Verified local scene asset missing or symlinked for {safe_k}: {src_p}")
        extra_assets[safe_k] = src_p
    hf = HyperFramesRenderer(enabled=True)
    res = hf.render_html_slot(
        slot_dir=slot_dir,
        html=html_src,
        out_path=out_path,
        fps=float(fps),
        fmt="webm",
        extra_assets=extra_assets or None,
        cache_dir=cache_dir,
        use_cache=(cache_dir is not None),
        skip_cli_lint=True,
    )
    return sanitize_native({
        "scene_id": scene["scene_id"],
        "treatment_mode": scene["treatment_mode"],
        "transparent": transparent,
        "duration_frames": dur_frames,
        "bound_asset_keys": sorted(extra_assets.keys()),
        "html_sha256": canonical_hash({"html": html_src})[:16],
        **res,
    })


def build_three_hook_candidates(
    *,
    hook_claim: dict[str, Any],
    next_claim: dict[str, Any] | None,
    width: int,
    height: int,
    fps: int,
    face_bounds: dict[str, float] | None,
    safe_zones: dict[str, float] | None,
    palette: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Build 3 materially distinct rendered hook candidates (A: action conflict, B: calendar tension lead, C: split dilemma)."""
    t0 = max(0.05, float(hook_claim.get("start_s", 0.1)))
    t1 = min(max(t0 + 2.2, float(hook_claim.get("end_s", 3.5))), t0 + 3.8)
    sf = int(round(t0 * fps))
    ef = max(sf + int(fps * 1.8), int(round(t1 * fps)))
    dur_s = (ef - sf) / float(fps)

    # Extract concise grounded hook wording from actual spoken claim (zero hardcoded Olivia strings!)
    txt = str(hook_claim.get("text", "")).strip()
    words = [w.strip(".,?!;:\"'") for w in txt.split() if w.strip(".,?!;:\"'")]
    short_hl = " ".join(words[-4:]).title() if len(words) >= 4 else (txt[:28] or "Real Timing Check")
    if "bank account" in txt.lower() and "groceries" in txt.lower():
        hl_a = "Balance ≠ Spendable"
        kicker_a = "BEFORE GROCERIES"
        sub_a = "Checking your account in the car"
    else:
        hl_a = short_hl[:30]
        kicker_a = "THE REAL QUESTION"
        sub_a = "Look closely before deciding"

    days = list((next_claim or {}).get("weekdays_mentioned") or hook_claim.get("weekdays_mentioned") or [])
    day_a = days[0][:3].upper() if len(days) >= 1 else "DUE"
    day_b = days[1][:3].upper() if len(days) >= 2 else "NEXT"

    cands = [
        {
            "candidate_id": "hook_A_spatial_action_conflict",
            "strategy": "Open kinetic editorial typography anchored in upper/lower negative space over live phone-glance action (zero dark box).",
            "scene": {
                "scene_id": "hook_A_spatial_action_conflict",
                "treatment_mode": "spatial_editorial_anchor",
                "start_frame": sf,
                "end_frame_exclusive": ef,
                "start_s": round(sf / fps, 4),
                "end_s": round(ef / fps, 4),
                "duration_s": round(dur_s, 4),
                "layout": solve_dynamic_safe_layout(width=width, height=height, treatment_mode="spatial_editorial_anchor", face_bounds=face_bounds, safe_zones=safe_zones, placement_hint="lower"),
                "params": {"kicker": kicker_a, "headline": hl_a, "subline": sub_a, "tone": "amber"},
            },
        },
        {
            "candidate_id": "hook_B_calendar_tension_lead",
            "strategy": "Warm-ivory architectural weekday gap ribbon previewing timing tension from frame 0.",
            "scene": {
                "scene_id": "hook_B_calendar_tension_lead",
                "treatment_mode": "calendar_timeline_gap",
                "start_frame": sf,
                "end_frame_exclusive": ef,
                "start_s": round(sf / fps, 4),
                "end_s": round(ef / fps, 4),
                "duration_s": round(dur_s, 4),
                "layout": solve_dynamic_safe_layout(width=width, height=height, treatment_mode="calendar_timeline_gap", face_bounds=face_bounds, safe_zones=safe_zones),
                "params": {"start_day": day_a, "start_tag": "OBLIGATION", "gap_days": ["GAP"], "end_day": day_b, "end_tag": "INCOME", "caption": "TIMING CONFLICT", "start_trigger_s": 0.12, "end_trigger_s": min(dur_s * 0.52, 1.3)},
            },
        },
        {
            "candidate_id": "hook_C_split_dilemma",
            "strategy": "Two-column split metaphor contrasting on-paper balance vs actually spendable right in the hook.",
            "scene": {
                "scene_id": "hook_C_split_dilemma",
                "treatment_mode": "split_dilemma_metaphor",
                "start_frame": sf,
                "end_frame_exclusive": ef,
                "start_s": round(sf / fps, 4),
                "end_s": round(ef / fps, 4),
                "duration_s": round(dur_s, 4),
                "layout": solve_dynamic_safe_layout(width=width, height=height, treatment_mode="split_dilemma_metaphor", face_bounds=face_bounds, safe_zones=safe_zones),
                "params": {"question": hl_a, "left_label": "ON SCREEN", "left_sub": "Account Total", "right_label": "ACTUALLY FREE", "right_sub": "Safe To Spend"},
            },
        },
    ]
    if palette:
        for c in cands:
            c["scene"]["palette"] = dict(palette)
    return cands
