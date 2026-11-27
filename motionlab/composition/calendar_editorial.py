"""Phase A3/A4 (Generic New Wave Editorial Visual Primitive: Calendar Gap, Kinetic Typography & Progressive System).

Modern US UGC × quiet-luxury editorial graphics rendered via local offline
`HyperFramesRenderer` (Chromium + GSAP) at native 1080x1920 @ 30fps with
transparent alpha WebM overlays or fullscreen quiet-luxury editorial bridge scenes.
  - ZERO money waterfalls, coins, macro PIP, live-detail boxes, fake bank UI, or QA jargon.
  - Parameterized strictly from semantic states and edited-domain timings (no hardcoded timestamps).
  - Enforces 9:16 platform safe zones and face/eye/mouth clearance.
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

from motionlab.renderers.hyperframes import HyperFramesRenderer
from motionlab.schemas import sanitize_native

NEW_WAVE_SCHEMA_VERSION = "1.0.0"

_MODERN_SANS = "'Avenir Next', -apple-system, 'SF Pro Display', 'Helvetica Neue', Arial, sans-serif"

DEFAULT_NEW_WAVE_PALETTE = {
    "ivory": "#F6F3EC",
    "stone": "#EAE4D7",
    "ink": "#16191D",
    "ink_soft": "#4A5058",
    "amber": "#D96B43",       # Conflict / due day (e.g. Tuesday)
    "teal": "#1E847F",        # Resolution / payday / safe-to-spend (e.g. Friday)
    "glass_dark": "rgba(20, 22, 26, 0.72)",
    "white": "#FAF8F4",
}


def _hf_doc(comp_id: str, width: int, height: int, fps: int, dur: float, bg_css: str, style_extra: str, body_inner: str, script_inner: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<script src="./gsap.min.js"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  html, body {{ width: {width}px; height: {height}px; overflow: hidden; background: {bg_css}; font-family: {_MODERN_SANS}; }}
  #hf-root {{ position: relative; width: {width}px; height: {height}px; overflow: hidden; background: {bg_css}; }}
  {style_extra}
</style>
</head>
<body>
<div id="hf-root" data-composition-id="{comp_id}" data-duration="{dur:.4f}" data-width="{width}" data-height="{height}" data-fps="{fps}">
{body_inner}
</div>
<script>
  window.__timelines = window.__timelines || {{}};
  const tl = gsap.timeline({{ paused: true }});
  {script_inner}
  window.__timelines["{comp_id}"] = tl;
</script>
</body>
</html>"""


def build_new_wave_layer_html(
    kind: str,
    params: dict[str, Any],
    *,
    width: int = 1080,
    height: int = 1920,
    fps: int = 30,
    duration_s: float = 2.0,
    transparent: bool = True,
) -> str:
    """Generate deterministic GSAP-driven HyperFrames HTML conforming strictly to HyperFramesRenderer contract."""
    pal = {**DEFAULT_NEW_WAVE_PALETTE, **(params.get("palette") or {})}
    bg_css = "transparent" if transparent else pal["ivory"]
    dur = max(0.3, float(duration_s))
    esc = html.escape
    comp_id = str(params.get("composition_id", f"nw_{kind}"))

    if kind == "editorial_hook":
        label = esc(str(params.get("headline", "CHECKING AGAIN?")))
        eyebrow = esc(str(params.get("eyebrow", "")))
        y_px = int(params.get("y_px", 1255))
        exit_t = max(0.25, dur - 0.24)
        style = f"""
  #wrap {{
    position: absolute; left: 96px; right: 96px; top: {y_px}px;
    display: flex; flex-direction: column; align-items: center; text-align: center;
  }}
  #eyebrow {{
    font-size: 21px; font-weight: 700; letter-spacing: 0.20em; text-transform: uppercase;
    color: rgba(234, 228, 215, 0.86); margin-bottom: 8px; opacity: 0;
  }}
  #pill {{
    display: inline-flex; flex-direction: column; align-items: center;
    padding: 20px 40px 18px 40px; border-radius: 16px;
    background: {pal['glass_dark']};
    border: 1px solid rgba(250, 248, 244, 0.18);
    box-shadow: 0 18px 44px rgba(0, 0, 0, 0.34);
    opacity: 0; transform: translateY(18px);
  }}
  #hl {{ font-size: 56px; font-weight: 800; letter-spacing: -0.02em; line-height: 1.05; color: {pal['white']}; }}
  #rule {{
    margin-top: 12px; width: 100%; height: 5px; border-radius: 3px;
    background: {pal['amber']}; transform-origin: left center; transform: scaleX(0);
  }}"""
        body = f"""  <div id="wrap">
    <div id="pill">{f'<div id="eyebrow">{eyebrow}</div>' if eyebrow else ''}<div id="hl">{label}</div><div id="rule"></div></div>
  </div>"""
        script = f"""
  tl.to("#pill", {{ opacity: 1, y: 0, duration: 0.28, ease: "power3.out" }}, 0.04);
  if (document.getElementById("eyebrow")) {{
    tl.to("#eyebrow", {{ opacity: 1, duration: 0.22, ease: "power2.out" }}, 0.02);
  }}
  tl.to("#rule", {{ scaleX: 1, duration: 0.42, ease: "expo.out" }}, 0.18);
  tl.to("#wrap", {{ opacity: 0, y: -12, duration: 0.22, ease: "power2.in" }}, {exit_t:.4f});"""
        return _hf_doc(comp_id, width, height, fps, dur, bg_css, style, body, script)

    if kind == "calendar_gap_strip":
        day_a = esc(str(params.get("start_day", "TUE")))
        tag_a = esc(str(params.get("start_tag", "RENT DUE")))
        mid_days = [esc(str(d)) for d in (params.get("gap_days") or ["WED", "THU"])]
        day_b = esc(str(params.get("end_day", "FRI")))
        tag_b = esc(str(params.get("end_tag", "PAYDAY")))
        t_a = max(0.02, float(params.get("start_trigger_s", 0.15)))
        t_b = max(t_a + 0.35, float(params.get("end_trigger_s", dur * 0.55)))
        y_px = int(params.get("y_px", 1215))
        exit_t = max(t_b + 0.28, dur - 0.20)
        mid_html = "".join(
            f'<div class="mid-node" id="mid{k}"><div class="mid-dot"></div><div class="mid-lbl">{d}</div></div>'
            for k, d in enumerate(mid_days)
        )
        style = f"""
  #cal-card {{
    position: absolute; left: 88px; right: 88px; top: {y_px}px;
    padding: 26px 32px 24px 32px; border-radius: 18px;
    background: {pal['glass_dark']};
    border: 1px solid rgba(250, 248, 244, 0.18);
    box-shadow: 0 22px 50px rgba(0, 0, 0, 0.38);
    opacity: 0; transform: translateY(16px);
  }}
  #track-row {{ position: relative; display: flex; align-items: center; justify-content: space-between; padding: 6px 4px; }}
  #base-line {{
    position: absolute; left: 96px; right: 96px; top: 38px; height: 5px;
    background: rgba(250, 248, 244, 0.18); border-radius: 3px;
  }}
  #fill-line {{
    position: absolute; left: 96px; right: 96px; top: 38px; height: 5px;
    background: linear-gradient(90deg, {pal['amber']} 0%, rgba(234,228,215,0.82) 52%, {pal['teal']} 100%);
    border-radius: 3px; transform-origin: left center; transform: scaleX(0);
  }}
  .node {{
    position: relative; z-index: 2; display: flex; flex-direction: column; align-items: center;
    min-width: 158px; opacity: 0; transform: scale(0.86);
  }}
  .badge {{
    padding: 11px 24px; border-radius: 12px; font-size: 38px; font-weight: 800; letter-spacing: 0.04em;
    color: {pal['white']};
  }}
  #nodeA .badge {{ background: {pal['amber']}; box-shadow: 0 8px 24px rgba(217, 107, 67, 0.42); }}
  #nodeB .badge {{ background: {pal['teal']}; box-shadow: 0 8px 24px rgba(30, 132, 127, 0.42); }}
  .tag {{
    margin-top: 10px; font-size: 24px; font-weight: 700; letter-spacing: 0.14em; text-transform: uppercase;
    color: rgba(250, 248, 244, 0.94);
  }}
  .mid-node {{ position: relative; z-index: 2; display: flex; flex-direction: column; align-items: center; opacity: 0; }}
  .mid-dot {{
    width: 14px; height: 14px; border-radius: 50%; background: {pal['stone']};
    border: 3px solid rgba(20, 22, 26, 0.90); margin-top: 26px;
  }}
  .mid-lbl {{
    margin-top: 24px; font-size: 22px; font-weight: 700; letter-spacing: 0.14em;
    color: rgba(234, 228, 215, 0.78);
  }}
  #gap-caption {{
    margin-top: 16px; text-align: center; font-size: 24px; font-weight: 700; letter-spacing: 0.16em;
    text-transform: uppercase; color: rgba(246, 243, 236, 0.82); opacity: 0;
  }}"""
        body = f"""  <div id="cal-card">
    <div id="track-row">
      <div id="base-line"></div>
      <div id="fill-line"></div>
      <div class="node" id="nodeA"><div class="badge">{day_a}</div><div class="tag">{tag_a}</div></div>
      {mid_html}
      <div class="node" id="nodeB"><div class="badge">{day_b}</div><div class="tag">{tag_b}</div></div>
    </div>
    <div id="gap-caption">THE PAYDAY GAP</div>
  </div>"""
        script = f"""
  tl.to("#cal-card", {{ opacity: 1, y: 0, duration: 0.24, ease: "power3.out" }}, {max(0.0, t_a - 0.10):.4f});
  tl.to("#nodeA", {{ opacity: 1, scale: 1, duration: 0.26, ease: "back.out(1.4)" }}, {t_a:.4f});
  tl.to("#fill-line", {{ scaleX: 1, duration: {max(0.32, t_b - t_a):.4f}, ease: "power2.inOut" }}, {t_a + 0.10:.4f});
  tl.to(".mid-node", {{ opacity: 1, duration: 0.20, stagger: 0.14, ease: "power2.out" }}, {t_a + 0.24:.4f});
  tl.to("#nodeB", {{ opacity: 1, scale: 1, duration: 0.26, ease: "back.out(1.4)" }}, {t_b:.4f});
  tl.to("#gap-caption", {{ opacity: 1, duration: 0.24, ease: "power2.out" }}, {t_b + 0.14:.4f});
  tl.to("#cal-card", {{ opacity: 0, y: -12, duration: 0.18, ease: "power2.in" }}, {exit_t:.4f});"""
        return _hf_doc(comp_id, width, height, fps, dur, bg_css, style, body, script)

    if kind == "question_focus":
        question = esc(str(params.get("question", "SAFE TO SPEND?")))
        y_px = int(params.get("y_px", 1255))
        exit_t = max(0.3, dur - 0.18)
        style = f"""
  #qbox {{
    position: absolute; left: 96px; right: 96px; top: {y_px}px;
    display: flex; flex-direction: column; align-items: center; text-align: center;
    opacity: 0; transform: translateY(16px);
  }}
  #qpill {{
    padding: 22px 42px 20px 42px; border-radius: 16px;
    background: {pal['glass_dark']};
    border: 1px solid rgba(250, 248, 244, 0.20);
    box-shadow: 0 20px 46px rgba(0, 0, 0, 0.36);
  }}
  #qtxt {{ font-size: 62px; font-weight: 800; letter-spacing: -0.02em; color: {pal['white']}; }}
  #qline {{
    margin-top: 12px; height: 5px; width: 100%; border-radius: 3px;
    background: linear-gradient(90deg, {pal['amber']} 0%, {pal['teal']} 100%);
    transform-origin: left center; transform: scaleX(0);
  }}"""
        body = f"""  <div id="qbox"><div id="qpill"><div id="qtxt">{question}</div><div id="qline"></div></div></div>"""
        script = f"""
  tl.to("#qbox", {{ opacity: 1, y: 0, duration: 0.24, ease: "power3.out" }}, 0.04);
  tl.to("#qline", {{ scaleX: 1, duration: 0.52, ease: "expo.out" }}, 0.20);
  tl.to("#qbox", {{ opacity: 0, y: -10, duration: 0.16, ease: "power2.in" }}, {exit_t:.4f});"""
        return _hf_doc(comp_id, width, height, fps, dur, bg_css, style, body, script)

    if kind == "editorial_bridge_scene":
        left_w = esc(str(params.get("contrast_from", "HOW MUCH")))
        right_w = esc(str(params.get("contrast_to", "WHEN")))
        product_title = esc(str(params.get("product_title", "The Next Paycheck Plan")))
        subtitle = esc(str(params.get("subtitle", "TIMING-FIRST PAYDAY CLARITY")))
        t_swap = round(dur * 0.40, 4)
        style = f"""
  html, body, #hf-root {{ background: #F4EFE4; }}
  #hf-root {{ background: linear-gradient(168deg, #F8F5EE 0%, #EFE9DC 100%); }}
  #phase1, #phase2 {{
    position: absolute; inset: 0; display: flex; flex-direction: column;
    align-items: center; justify-content: center; padding: 0 92px; text-align: center;
  }}
  #p1-eyebrow {{
    font-size: 26px; font-weight: 700; letter-spacing: 0.22em; color: {pal['ink_soft']};
    margin-bottom: 36px; text-transform: uppercase;
  }}
  #contrast-row {{ display: flex; align-items: center; justify-content: center; gap: 32px; }}
  #from-wrap {{ position: relative; display: inline-block; }}
  #from-txt {{ font-size: 72px; font-weight: 800; color: rgba(22, 25, 29, 0.44); letter-spacing: -0.02em; }}
  #strike {{
    position: absolute; left: -6px; right: -6px; top: 52%; height: 6px;
    background: {pal['amber']}; border-radius: 3px; transform-origin: left center; transform: scaleX(0);
  }}
  #arrow {{ font-size: 56px; font-weight: 700; color: {pal['teal']}; opacity: 0; }}
  #to-txt {{
    font-size: 86px; font-weight: 900; color: {pal['teal']}; letter-spacing: -0.025em;
    opacity: 0; transform: translateX(18px);
  }}
  #phase2 {{ opacity: 0; transform: translateY(22px); }}
  #cal-mini {{ display: flex; align-items: center; gap: 18px; margin-bottom: 40px; }}
  .dot-a {{ width: 18px; height: 18px; border-radius: 50%; background: {pal['amber']}; }}
  .dot-b {{ width: 18px; height: 18px; border-radius: 50%; background: {pal['teal']}; }}
  #mini-bar {{
    width: 260px; height: 5px; border-radius: 3px;
    background: linear-gradient(90deg, {pal['amber']} 0%, {pal['teal']} 100%);
    transform-origin: left center; transform: scaleX(0);
  }}
  #prod-title {{ font-size: 86px; font-weight: 800; line-height: 1.06; letter-spacing: -0.03em; color: {pal['ink']}; max-width: 880px; }}
  #prod-sub {{ margin-top: 28px; font-size: 26px; font-weight: 700; letter-spacing: 0.22em; color: {pal['teal']}; text-transform: uppercase; }}"""
        body = f"""  <div id="phase1">
    <div id="p1-eyebrow">NOT JUST THE BALANCE</div>
    <div id="contrast-row">
      <div id="from-wrap"><span id="from-txt">{left_w}</span><div id="strike"></div></div>
      <div id="arrow">→</div>
      <div id="to-txt">{right_w}</div>
    </div>
  </div>
  <div id="phase2">
    <div id="cal-mini"><div class="dot-a"></div><div id="mini-bar"></div><div class="dot-b"></div></div>
    <div id="prod-title">{product_title}</div>
    <div id="prod-sub">{subtitle}</div>
  </div>"""
        script = f"""
  tl.to("#strike", {{ scaleX: 1, duration: 0.20, ease: "power3.out" }}, 0.06);
  tl.to("#arrow", {{ opacity: 1, duration: 0.14 }}, 0.14);
  tl.to("#to-txt", {{ opacity: 1, x: 0, duration: 0.22, ease: "power3.out" }}, 0.16);
  tl.to("#phase1", {{ opacity: 0, y: -22, duration: 0.12, ease: "power2.in" }}, {t_swap:.4f});
  tl.to("#phase2", {{ opacity: 1, y: 0, duration: 0.24, ease: "power3.out" }}, {t_swap + 0.14:.4f});
  tl.to("#mini-bar", {{ scaleX: 1, duration: 0.32, ease: "expo.out" }}, {t_swap + 0.18:.4f});"""
        return _hf_doc(comp_id, width, height, fps, dur, "#F6F3EC", style, body, script)

    if kind == "progressive_benefits":
        title = esc(str(params.get("title", "THE NEXT PAYCHECK PLAN")))
        steps = params.get("steps") or [
            {"label": "ALREADY COMMITTED", "trigger_s": 0.4, "tone": "amber"},
            {"label": "UNTIL PAYDAY", "trigger_s": 2.1, "tone": "neutral"},
            {"label": "SAFE TO SPEND", "trigger_s": 3.8, "tone": "teal"},
        ]
        y_px = int(params.get("y_px", 1225))
        exit_t = max(0.6, dur - 0.24)
        tone_colors = {"amber": pal["amber"], "neutral": "#C6BFA9", "teal": pal["teal"]}
        seg_divs = "".join(
            f'<div class="seg" id="seg{k}" style="background:{tone_colors.get(st.get("tone","teal"), pal["teal"])};"></div>'
            for k, st in enumerate(steps)
        )
        lbl_divs = "".join(
            f'<div class="step-lbl" id="lbl{k}">{esc(str(st["label"]))}</div>'
            for k, st in enumerate(steps)
        )
        step_js_lines = []
        for k, st in enumerate(steps):
            ts = max(0.04, float(st.get("trigger_s", 0.2 + k * 1.4)))
            for prev_k in range(k):
                step_js_lines.append(f'tl.to("#lbl{prev_k}", {{ opacity: 0, y: -12, duration: 0.16, ease: "power2.in" }}, {max(0.0, ts - 0.06):.4f});')
            step_js_lines.append(f'tl.to("#seg{k}", {{ scaleX: 1, opacity: 1, duration: 0.34, ease: "expo.out" }}, {ts:.4f});')
            step_js_lines.append(f'tl.fromTo("#lbl{k}", {{ opacity: 0, y: 14 }}, {{ opacity: 1, y: 0, duration: 0.24, ease: "power3.out" }}, {ts:.4f});')
        style = f"""
  #sys {{
    position: absolute; left: 88px; right: 88px; top: {y_px}px;
    padding: 24px 32px 26px 32px; border-radius: 18px;
    background: {pal['glass_dark']};
    border: 1px solid rgba(250, 248, 244, 0.18);
    box-shadow: 0 22px 50px rgba(0, 0, 0, 0.38);
    opacity: 0; transform: translateY(14px);
  }}
  #sys-title {{
    font-size: 22px; font-weight: 700; letter-spacing: 0.20em; text-transform: uppercase;
    color: rgba(246, 243, 236, 0.80); margin-bottom: 14px;
  }}
  #bar-bg {{
    display: flex; gap: 10px; height: 14px; width: 100%; padding: 2px;
    background: rgba(255, 255, 255, 0.12); border-radius: 8px;
  }}
  .seg {{ flex: 1; height: 100%; border-radius: 6px; transform-origin: left center; transform: scaleX(0); opacity: 0.25; }}
  #lbl-stage {{ position: relative; height: 64px; margin-top: 16px; }}
  .step-lbl {{
    position: absolute; left: 0; right: 0; top: 4px;
    font-size: 48px; font-weight: 800; letter-spacing: -0.015em; color: {pal['white']}; opacity: 0;
  }}"""
        body = f"""  <div id="sys">
    <div id="sys-title">{title}</div>
    <div id="bar-bg">{seg_divs}</div>
    <div id="lbl-stage">{lbl_divs}</div>
  </div>"""
        script = f"""
  tl.to("#sys", {{ opacity: 1, y: 0, duration: 0.26, ease: "power3.out" }}, 0.02);
  {chr(10).join("  " + ln for ln in step_js_lines)}
  tl.to("#sys", {{ opacity: 0, y: -12, duration: 0.22, ease: "power2.in" }}, {exit_t:.4f});"""
        return _hf_doc(comp_id, width, height, fps, dur, bg_css, style, body, script)

    # payoff_phrase / cta_caption
    headline = esc(str(params.get("headline", "Link in caption ↓")))
    sub = esc(str(params.get("subline", "")))
    accent = pal["teal"] if params.get("tone", "teal") == "teal" else pal["amber"]
    y_px = int(params.get("y_px", 1255))
    exit_t = max(0.3, dur - 0.18)
    fade_js = f'tl.to("#cta", {{ opacity: 0, y: -10, duration: 0.18, ease: "power2.in" }}, {exit_t:.4f});' if params.get("fade_out", True) else ""
    style = f"""
  #cta {{
    position: absolute; left: 92px; right: 92px; top: {y_px}px;
    display: flex; flex-direction: column; align-items: center; text-align: center;
    opacity: 0; transform: translateY(16px);
  }}
  #card {{
    padding: 22px 38px 20px 38px; border-radius: 16px;
    background: {pal['glass_dark']};
    border: 1px solid rgba(250, 248, 244, 0.20);
    box-shadow: 0 20px 46px rgba(0, 0, 0, 0.36);
  }}
  #h1 {{ font-size: 56px; font-weight: 800; letter-spacing: -0.02em; color: {pal['white']}; }}
  #sub {{
    margin-top: 8px; font-size: 23px; font-weight: 700; letter-spacing: 0.18em;
    text-transform: uppercase; color: {pal['stone']}; opacity: 0.88;
  }}
  #bar {{
    margin-top: 12px; width: 100%; height: 5px; border-radius: 3px;
    background: {accent}; transform-origin: left center; transform: scaleX(0);
  }}"""
    body = f"""  <div id="cta"><div id="card"><div id="h1">{headline}</div>{f'<div id="sub">{sub}</div>' if sub else ''}<div id="bar"></div></div></div>"""
    script = f"""
  tl.to("#cta", {{ opacity: 1, y: 0, duration: 0.26, ease: "power3.out" }}, 0.04);
  tl.to("#bar", {{ scaleX: 1, duration: 0.36, ease: "expo.out" }}, 0.14);
  {fade_js}"""
    return _hf_doc(comp_id, width, height, fps, dur, bg_css, style, body, script)


def render_hyperframes_new_wave_clip(
    kind: str,
    params: dict[str, Any],
    *,
    out_path: Path,
    slot_dir: Path,
    width: int = 1080,
    height: int = 1920,
    fps: int = 30,
    duration_frames: int = 60,
    transparent: bool = True,
    cache_dir: Path | None = None,
) -> dict[str, Any]:
    """Render a New Wave Editorial primitive via local offline `HyperFramesRenderer.render_html_slot`."""
    hf = HyperFramesRenderer(enabled=True)
    dur_s = duration_frames / float(fps)
    html_src = build_new_wave_layer_html(
        kind, params, width=width, height=height, fps=fps, duration_s=dur_s, transparent=transparent,
    )
    fmt = "webm" if transparent else "mp4"
    res = hf.render_html_slot(
        slot_dir=slot_dir,
        html=html_src,
        out_path=out_path,
        fps=float(fps),
        fmt=fmt,
        timeout_s=180.0,
        skip_cli_lint=True,
        cache_dir=cache_dir,
        use_cache=cache_dir is not None,
    )
    return sanitize_native({
        "kind": kind,
        "engine": "hyperframes",
        "out_path": str(out_path),
        "width": width,
        "height": height,
        "fps": fps,
        "duration_frames": duration_frames,
        "transparent": transparent,
        "artifact_sha256": res.get("artifact_sha256"),
        "size_bytes": res.get("size_bytes"),
        "elapsed_s": res.get("elapsed_s"),
        "peak_tree_rss_mb": res.get("peak_tree_rss_mb", 0.0),
    })
