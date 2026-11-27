"""Phase A2/A3 (ML-01.R5): Typed Adapters Wrapping the 9 Real Imported Apache-2.0 HyperFrames Blocks.

Every generator reads or directly parameterizes its imported upstream HTML/CSS/GSAP structure in
`renderers/hyperframes/imported/heygen-hyperframes/<block>/upstream_offline.html` — preserving real
upstream GSAP motion choreography while guaranteeing:
  - 100% offline local `./gsap.min.js` (zero CDN, zero remote fonts, zero demo audio)
  - Portrait 720x1280 responsive editorial hierarchy (hero numbers 88-132px tabular, labels >=30px)
  - Warm editorial finance/product tokens (#F5F0E6 ivory, #292822 espresso, #365B49 forest, #A4B39A sage, #C6A46A brass, #B85D43 muted clay)
  - Strict integer-cent proportional math (NEVER index-based fake bar widths!)
  - HTML/JS escaping & finite numeric guards
"""

from __future__ import annotations

import html as _html
import json
import math
from pathlib import Path
from typing import Any

from motionlab.config import PROJECT_ROOT
from motionlab.semantic.story_graph import format_cents

HF_IMPORTED_DIR = PROJECT_ROOT / "renderers" / "hyperframes" / "imported" / "heygen-hyperframes"

DEFAULT_EDITORIAL_PALETTE = {
    "ivory": "#F5F0E6",
    "espresso": "#40382E",
    "forest": "#243B32",
    "sage": "#84977A",
    "gold": "#BE9A59",
    "brass": "#BE9A59",
    "terracotta": "#AF745E",
    "clay": "#AF745E",
    "ink_dark": "#1C1A16",
}

_SANS = 'Bahnschrift, -apple-system, BlinkMacSystemFont, "Segoe UI", "Helvetica Neue", Arial, sans-serif'
_SERIF = 'Georgia, Cambria, "Times New Roman", serif'


def _esc(val: Any) -> str:
    return _html.escape(str(val or ""), quote=True)


def _j(val: Any) -> str:
    return json.dumps(val, ensure_ascii=True)


def _pal(params: dict[str, Any]) -> dict[str, str]:
    out = dict(DEFAULT_EDITORIAL_PALETTE)
    p = params.get("palette")
    if isinstance(p, dict):
        for k in out:
            if p.get(k):
                out[k] = str(p[k])
    return out


def load_upstream_smoke_html(block_slug: str, duration_s: float = 3.0) -> str:
    """Load the exact imported Apache-2.0 upstream_offline.html (minimal offline patch only) for A2 upstream smoke proof."""
    src_p = HF_IMPORTED_DIR / block_slug / "upstream_offline.html"
    if not src_p.is_file():
        raise FileNotFoundError(f"Missing imported HyperFrames upstream block: {src_p}")
    raw = src_p.read_text(encoding="utf-8").lstrip()
    # Ensure HTML starts cleanly with <!DOCTYPE html> if a leading comment precedes it (HyperFrames lint contract)
    if raw.startswith("<!--"):
        end_c = raw.find("-->")
        if end_c != -1:
            hdr_c = raw[: end_c + 3]
            rest = raw[end_c + 3 :].lstrip()
            if rest.lower().startswith("<!doctype"):
                nl = rest.find("\n")
                raw = (rest[: nl + 1] + hdr_c + "\n" + rest[nl + 1 :]) if nl != -1 else rest
    # Clamp upstream demo duration to bounded smoke duration_s (e.g. 2.0s = 48 frames) without altering animation mechanics
    import re as _re
    raw = _re.sub(r'data-duration="[^"]+"', f'data-duration="{duration_s:.2f}"', raw)
    # Component snippets (like strikethrough-replace) lack full <!doctype html> shell — wrap cleanly preserving 100% of snippet!
    if "<!doctype" not in raw.lower() and "<html" not in raw.lower():
        return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <script src="./gsap.min.js"></script>
  <style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    html, body {{ width: 720px; height: 1280px; overflow: hidden; background: #F5F0E6; display: flex; align-items: center; justify-content: center; }}
    #root {{ width: 720px; height: 1280px; display: flex; align-items: center; justify-content: center; position: relative; }}
  </style>
</head>
<body>
  <div id="root" data-composition-id="{_esc(block_slug)}" data-start="0" data-duration="{duration_s:.2f}" data-width="720" data-height="1280" data-fps="24">
    {raw}
  </div>
  <script>
    window.__timelines = window.__timelines || {{}};
    var tl = gsap.timeline({{ paused: true }});
    tl.fromTo(".hf-catalog-strikethrough-replace i", {{ "--hf-strike-progress": 0 }}, {{ "--hf-strike-progress": 1, duration: 0.45, ease: "power2.out" }}, 0.35);
    tl.fromTo(".hf-catalog-strikethrough-replace strong", {{ "--hf-replace-opacity": 0, y: 14 }}, {{ "--hf-replace-opacity": 1, y: 0, duration: 0.45, ease: "power2.out" }}, 0.72);
    window.__timelines["{_esc(block_slug)}"] = tl;
  </script>
</body>
</html>
"""
    return raw


def build_adapted_hyperframes_html(
    primitive_id: str,
    width: int,
    height: int,
    fps: float,
    duration_s: float,
    parameters: dict[str, Any],
    transparent: bool = True,
) -> str:
    """Build portrait-adapted, offline, data-accurate HTML/GSAP composition from a verified imported HyperFrames block."""
    pal = _pal(parameters)
    dur_s = max(0.4, float(duration_s))
    fps_i = max(1, int(round(fps)))
    comp_id = f"hf_{primitive_id}"
    bg_css = "transparent" if transparent else str(parameters.get("bg_color") or pal["ivory"])

    if primitive_id == "hf_apple_money_count":
        # Adapted directly from heygen-com/hyperframes `apple-money-count` (Apache-2.0)
        # R5.R3 Multi-Layer Upgrade (D04/D05): Zero Olivia hardcodes + `render_mode` ("combined" | "paper_tokens_only" | "typography_only")
        # allowing genuine 2-HF-layer depth compositing (paper tokens behind subject matte + foreground hero amount above!).
        from_cents = int(parameters.get("from_cents", 0))
        to_cents = int(parameters.get("to_cents", from_cents))
        sym = str(parameters.get("currency_symbol", "$"))
        eyebrow = str(parameters.get("eyebrow", ""))
        caption = str(parameters.get("caption", ""))
        render_mode = str(parameters.get("render_mode", "combined")).lower()
        show_plate = render_mode != "paper_tokens_only"
        default_tokens = 0 if render_mode == "typography_only" else 4
        token_count = 0 if render_mode == "typography_only" else max(0, min(6, int(parameters.get("token_count", default_tokens))))
        y_pos = str(parameters.get("y_anchor", "8.8%"))
        dark_plate = bool(parameters.get("dark_plate", transparent))
        allow_frac = bool(parameters.get("allow_fractional_cents", (abs(from_cents) % 100 != 0) or (abs(to_cents) % 100 != 0)))
        hold_exact = bool(parameters.get("hold_exact_amount", not allow_frac and from_cents == 0))
        discrete_steps = [int(x) for x in (parameters.get("discrete_steps_cents") or [])]
        fg_col = pal["ivory"] if dark_plate else pal["forest"]
        accent_col = str(parameters.get("accent_color") or pal["brass"])
        cap_col = pal["ivory"] if dark_plate else pal["espresso"]
        hero_px = int(parameters.get("hero_px", 122 if dark_plate else 146))
        note_top_css = "44%" if dark_plate else "23%"
        scrim_css = (
            "position:absolute;left:0;right:0;top:0;height:360px;z-index:2;pointer-events:none;"
            "background:linear-gradient(180deg, rgba(24,22,18,0.72) 0%, rgba(24,22,18,0.40) 58%, rgba(24,22,18,0.0) 100%);"
            if (dark_plate and show_plate)
            else "display:none;"
        )
        stage_wrap_display = "flex" if show_plate else "none"
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<!-- adapted-from: heygen-com/hyperframes apple-money-count (Apache-2.0) mode={_esc(render_mode)} -->
<meta charset="UTF-8" />
<script src="./gsap.min.js"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  html, body {{ width: {width}px; height: {height}px; overflow: hidden; background: {bg_css}; }}
  #root {{ position: relative; width: {width}px; height: {height}px; overflow: hidden; background: {bg_css}; font-family: {_SANS}; }}
  .top-scrim {{ {scrim_css} }}
  .stage-wrap {{
    position: absolute; left: 48px; right: 105px; top: {y_pos}; z-index: 5;
    display: {stage_wrap_display}; flex-direction: column; align-items: {"flex-start" if dark_plate else "center"};
    text-align: {"left" if dark_plate else "center"};
  }}
  .plate {{
    display: flex; flex-direction: column; align-items: {"flex-start" if dark_plate else "center"};
    padding: 0; background: transparent; border: none; box-shadow: none;
  }}
  .eyebrow-row {{
    display: flex; align-items: center; gap: 12px; margin-bottom: 6px;
  }}
  .gold-rule {{ width: 36px; height: 3px; background: {accent_col}; border-radius: 2px; }}
  .eyebrow {{
    font-size: 26px; font-weight: 800; letter-spacing: 0.16em; text-transform: uppercase;
    color: {accent_col};
    text-shadow: {"0 2px 10px rgba(0,0,0,0.45)" if dark_plate else "none"};
  }}
  .amount {{
    color: {fg_col}; font-size: {hero_px}px; font-weight: 900; line-height: 0.94;
    letter-spacing: -0.025em; font-variant-numeric: tabular-nums; white-space: nowrap;
    text-shadow: {"0 10px 30px rgba(15,13,10,0.58), 0 2px 6px rgba(15,13,10,0.45)" if dark_plate else "0 4px 16px rgba(36,59,50,0.10)"};
  }}
  .caption {{
    margin-top: 10px; font-size: 30px; font-weight: 600; color: {cap_col};
    opacity: 0.92;
    text-shadow: {"0 2px 10px rgba(0,0,0,0.45)" if dark_plate else "none"};
  }}
  .money-burst {{ position: absolute; inset: 0; pointer-events: none; z-index: 3; perspective: 640px; }}
  .paper-note {{
    position: absolute; left: 46%; top: {note_top_css}; width: 108px; height: 54px; border-radius: 5px;
    background: linear-gradient(135deg, #F4EEE0 0%, #E3DBC6 55%, #D5CEB8 100%);
    border: 1.5px solid #84977A;
    box-shadow: 0 14px 26px rgba(22, 20, 16, 0.30), inset 0 0 0 4px rgba(132, 151, 122, 0.24);
    display: flex; align-items: center; justify-content: center; opacity: 0;
  }}
  .paper-note::before {{
    content: ""; position: absolute; inset: 5px; border: 1px solid rgba(36, 59, 50, 0.38); border-radius: 3px;
  }}
  .paper-note .seal {{
    width: 30px; height: 30px; border-radius: 15px;
    border: 1.5px solid rgba(36, 59, 50, 0.45); background: rgba(132, 151, 122, 0.20);
    display: grid; place-items: center;
    color: {pal["forest"]}; font-family: {_SERIF}; font-size: 18px; font-weight: 700;
  }}
</style>
</head>
<body>
<div id="root" data-composition-id="{comp_id}" data-start="0" data-duration="{dur_s:.4f}" data-width="{width}" data-height="{height}" data-fps="{fps_i}">
  <div class="top-scrim"></div>
  <div id="moneyBurst" class="money-burst"></div>
  <div class="stage-wrap">
    <div id="plate" class="plate">
      <div class="eyebrow-row"><span class="gold-rule"></span><div id="eyebrow" class="eyebrow">{_esc(eyebrow)}</div></div>
      <div id="amount" class="amount">{_esc(format_cents(from_cents, sym))}</div>
      {f'<div id="caption" class="caption">{_esc(caption)}</div>' if caption else ''}
    </div>
  </div>
</div>
<script>
  window.__timelines = window.__timelines || {{}};
  const amountEl = document.querySelector("#amount");
  const burstEl = document.querySelector("#moneyBurst");
  const FROM_CENTS = {from_cents};
  const TO_CENTS = {to_cents};
  const SYM = {_j(sym)};
  const DUR = {dur_s:.4f};
  const ALLOW_FRAC = {"true" if allow_frac else "false"};
  const HOLD_EXACT = {"true" if hold_exact else "false"};
  const DISCRETE_STEPS = {_j(discrete_steps)};
  const IS_CUTAWAY = {"true" if not dark_plate else "false"};

  function fmtCents(c) {{
    const sign = c < 0 ? "-" : "";
    if (!ALLOW_FRAC) {{
      // Integer-dollar lock (R5.R2 D01/D05): discrete whole-dollar increments only, ZERO fake cents!
      const wholeDollars = Math.round(Math.abs(c) / 100);
      return sign + SYM + wholeDollars.toLocaleString("en-US");
    }}
    const absC = Math.abs(Math.round(c));
    const whole = Math.floor(absC / 100);
    const rem = absC % 100;
    return sign + SYM + whole.toLocaleString("en-US") + (rem ? "." + String(rem).padStart(2, "0") : "");
  }}

  const icons = [];
  const specsLive = [
    {{ sx: -170, sy: 150, cx: -240, cy: -20, ex: -195, ey: -185, rot: -14, rx: 14, ry: -18, delay: 0.04 }},
    {{ sx: 155, sy: 165, cx: 235, cy: -10, ex: 185, ey: -165, rot: 12, rx: -12, ry: 16, delay: 0.12 }},
    {{ sx: -95, sy: 190, cx: -165, cy: 20, ex: -130, ey: -120, rot: -8, rx: 10, ry: 14, delay: 0.20 }},
    {{ sx: 115, sy: 185, cx: 180, cy: 35, ex: 150, ey: -105, rot: 10, rx: -14, ry: -12, delay: 0.28 }},
    {{ sx: -45, sy: 210, cx: -195, cy: 60, ex: -215, ey: -95, rot: -16, rx: 16, ry: -10, delay: 0.34 }},
    {{ sx: 65, sy: 210, cx: 200, cy: 65, ex: 205, ey: -75, rot: 15, rx: -10, ry: 18, delay: 0.40 }},
  ];
  const specsCutaway = [
    {{ sx: -210, sy: 10, cx: -265, cy: -45, ex: -245, ey: -95, rot: -14, rx: 12, ry: -16, delay: 0.05 }},
    {{ sx: 215, sy: 15, cx: 265, cy: -40, ex: 245, ey: -88, rot: 13, rx: -12, ry: 16, delay: 0.14 }},
    {{ sx: -190, sy: 42, cx: -245, cy: -10, ex: -225, ey: -48, rot: -9, rx: 10, ry: 12, delay: 0.22 }},
    {{ sx: 195, sy: 44, cx: 245, cy: -8, ex: 225, ey: -44, rot: 10, rx: -10, ry: -12, delay: 0.30 }},
  ];
  const specs = (IS_CUTAWAY ? specsCutaway : specsLive).slice(0, {token_count});

  specs.forEach((s) => {{
    const el = document.createElement("div");
    el.className = "paper-note";
    el.innerHTML = '<span class="seal">' + SYM + '</span>';
    burstEl.appendChild(el);
    icons.push({{ el, s }});
  }});

  const state = {{ c: HOLD_EXACT ? TO_CENTS : (DISCRETE_STEPS.length > 0 ? DISCRETE_STEPS[0] : FROM_CENTS), stepP: 0 }};
  amountEl.textContent = fmtCents(state.c);
  const tl = gsap.timeline({{ paused: true }});
  gsap.set("#plate", {{ opacity: 0, y: 20, scale: 0.94 }});
  icons.forEach(({{ el, s }}) => {{
    gsap.set(el, {{ x: s.sx, y: s.sy, rotation: s.rot * 0.35, rotationX: s.rx, rotationY: s.ry, opacity: 0, scale: 0.72 }});
  }});

  // Anticipation -> crisp reveal -> settle (spring geometry ONLY, never numeric overshoot!)
  tl.to("#plate", {{ opacity: 1, y: 0, scale: 1.02, duration: Math.min(0.24, DUR * 0.14), ease: "power3.out" }}, 0.04);
  tl.to("#plate", {{ scale: 1.0, duration: 0.18, ease: "power2.out" }}, 0.28);
  const countDur = Math.min(0.46, Math.max(0.22, DUR * 0.20));
  if (DISCRETE_STEPS.length >= 2) {{
    // Step strictly through verified ledger milestones ONLY (ZERO unverified intermediates!)
    tl.to(state, {{
      stepP: DISCRETE_STEPS.length - 1, duration: countDur, ease: "power2.out",
      onUpdate: () => {{
        const idx = Math.max(0, Math.min(DISCRETE_STEPS.length - 1, Math.floor(state.stepP + 0.0001)));
        amountEl.textContent = fmtCents(DISCRETE_STEPS[idx]);
      }},
      onComplete: () => {{ amountEl.textContent = fmtCents(DISCRETE_STEPS[DISCRETE_STEPS.length - 1]); }},
    }}, 0.08);
    tl.to("#amount", {{ scale: 1.04, duration: 0.10, ease: "power2.out" }}, 0.08 + countDur);
    tl.to("#amount", {{ scale: 1.0, duration: 0.16, ease: "power2.inOut" }}, 0.18 + countDur);
  }} else if (!HOLD_EXACT && FROM_CENTS !== TO_CENTS) {{
    tl.to(state, {{
      c: TO_CENTS, duration: countDur, ease: "power3.out",
      onUpdate: () => {{
        const snapped = ALLOW_FRAC ? Math.round(state.c) : Math.round(state.c / 100) * 100;
        const lo = Math.min(FROM_CENTS, TO_CENTS);
        const hi = Math.max(FROM_CENTS, TO_CENTS);
        amountEl.textContent = fmtCents(Math.max(lo, Math.min(hi, snapped)));
      }},
      onComplete: () => {{ amountEl.textContent = fmtCents(TO_CENTS); }},
    }}, 0.12);
    tl.to("#amount", {{ scale: 1.035, duration: 0.10, ease: "power2.out" }}, 0.12 + countDur);
    tl.to("#amount", {{ scale: 1.0, duration: 0.16, ease: "power2.inOut" }}, 0.22 + countDur);
  }} else {{
    amountEl.textContent = fmtCents(TO_CENTS);
  }}

  // Curved quadratic-Bezier paper token paths with subtle flutter & soft contact settle
  icons.forEach(({{ el, s }}) => {{
    const pObj = {{ t: 0 }};
    tl.to(el, {{ opacity: 0.90, scale: 0.96, duration: 0.16, ease: "power2.out" }}, 0.08 + s.delay);
    tl.to(pObj, {{
      t: 1, duration: 0.56, ease: "power3.out",
      onUpdate: () => {{
        const u = pObj.t;
        const inv = 1 - u;
        const bx = inv * inv * s.sx + 2 * inv * u * s.cx + u * u * s.ex;
        const by = inv * inv * s.sy + 2 * inv * u * s.cy + u * u * s.ey + Math.sin(u * Math.PI * 2.0) * 2.5;
        const br = s.rot * (0.35 + 0.65 * u);
        gsap.set(el, {{ x: bx, y: by, rotation: br, rotationX: s.rx * (1 - 0.7 * u), rotationY: s.ry * (1 - 0.7 * u) }});
      }},
    }}, 0.08 + s.delay);
    tl.to(el, {{ opacity: 0, y: s.ey - 22, duration: 0.34, ease: "power2.in" }}, Math.max(0.68, DUR * 0.66) + s.delay * 0.35);
  }});
  tl.to("#plate", {{ opacity: 0, y: -10, duration: Math.min(0.24, DUR * 0.12), ease: "power2.in" }}, Math.max(0.2, DUR - 0.26));
  window.__timelines["{comp_id}"] = tl;
</script>
</body>
</html>
"""

    if primitive_id == "hf_mk_progress_stat":
        # Adapted directly from heygen-com/hyperframes `mk-progress-stat` (Apache-2.0)
        # R5.R3 Fix D05: Zero Olivia hardcoded amounts/categories!
        amount_cents = int(parameters.get("amount_cents", 0))
        denom_cents = max(1, int(parameters.get("denom_cents", max(100, abs(amount_cents) or 100))))
        remaining_cents = int(parameters.get("remaining_cents", denom_cents - amount_cents))
        sym = str(parameters.get("currency_symbol", "$"))
        is_deduction = bool(parameters.get("is_deduction", True))
        category = str(parameters.get("category", "CATEGORY"))
        raw_cap = str(parameters.get("caption", f"{format_cents(remaining_cents, sym)} left"))
        caption = raw_cap.replace("(derived)", "").replace("derived", "").strip()
        fill_ratio = max(0.02, min(1.0, abs(amount_cents) / denom_cents))
        bar_col = str(parameters.get("bar_color") or (pal["clay"] if is_deduction else pal["forest"]))
        y_top = max(96, int(parameters.get("y_px", 110)))
        layout_var = str(parameters.get("layout_variant", "top_strip"))
        allow_frac = bool(parameters.get("allow_fractional_cents", abs(amount_cents) % 100 != 0))
        token_count = max(0, min(4, int(parameters.get("token_count", 3 if is_deduction else 0))))
        sign_prefix = "-" if (is_deduction and amount_cents > 0) else ""
        is_rail = (layout_var == "compact_editorial_rail")
        box_bg = "rgba(36, 59, 50, 0.90)" if is_rail else "rgba(245, 240, 230, 0.94)"
        ink_main = pal["ivory"] if is_rail else pal["forest"]
        ink_dim = pal["sage"] if is_rail else pal["espresso"]
        track_bg = "rgba(245, 240, 230, 0.18)" if is_rail else "rgba(64, 56, 46, 0.14)"
        bar_accent = pal["brass"] if is_rail else bar_col
        border_rule = f"border-top: 4px solid {bar_accent}; border-left: none;" if is_rail else f"border-left: 6px solid {bar_col};"
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<!-- adapted-from: heygen-com/hyperframes mk-progress-stat (Apache-2.0) -->
<meta charset="UTF-8" />
<script src="./gsap.min.js"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  html, body {{ width: {width}px; height: {height}px; overflow: hidden; background: {bg_css}; }}
  #mk-ps-root {{
    --mk-font: {_SANS};
    --mk-ink: {ink_main};
    --mk-ink-dim: {ink_dim};
    --mk-accent: {bar_accent};
    position: relative; width: {width}px; height: {height}px; overflow: hidden; background: {bg_css};
  }}
  #mk-ps-group {{
    position: absolute; left: 48px; right: 105px; top: {y_top}px; z-index: 3;
    font-family: var(--mk-font);
    padding: 20px 24px 20px 26px; border-radius: {4 if is_rail else 10}px;
    background: {box_bg};
    {border_rule}
    box-shadow: 0 14px 34px rgba(24, 22, 18, 0.28);
  }}
  .top-row {{ display: flex; align-items: baseline; justify-content: space-between; gap: 14px; }}
  #mk-ps-num {{
    font-weight: 900; font-size: {82 if is_rail else 88}px; line-height: 0.96; letter-spacing: -0.025em;
    color: var(--mk-ink); font-variant-numeric: tabular-nums;
  }}
  #mk-ps-label {{
    font-weight: 800; font-size: 32px; letter-spacing: 0.10em; text-transform: uppercase;
    color: {bar_accent};
  }}
  #mk-ps-track {{
    position: relative; margin-top: 16px; width: 100%; height: 14px; border-radius: 4px;
    background: {track_bg}; overflow: hidden;
  }}
  #mk-ps-fill {{
    position: absolute; inset: 0; border-radius: 4px; background: var(--mk-accent);
    transform-origin: left center;
  }}
  .meta-row {{ display: flex; justify-content: space-between; align-items: center; margin-top: 12px; }}
  #mk-ps-caption {{ font-weight: 700; font-size: 28px; color: var(--mk-ink); font-variant-numeric: tabular-nums; }}
  #mk-ps-share {{ font-weight: 700; font-size: 24px; color: var(--mk-ink-dim); font-variant-numeric: tabular-nums; }}
  #allocTokens {{ position: absolute; inset: 0; pointer-events: none; z-index: 1; perspective: 560px; }}
  .alloc-note {{
    position: absolute; left: 52%; top: {y_top + 216}px; width: 92px; height: 46px; border-radius: 5px;
    background: linear-gradient(135deg, #F4EEE0 0%, #E3DBC6 55%, #D5CEB8 100%);
    border: 1.5px solid #84977A;
    box-shadow: 0 10px 22px rgba(22,20,16,0.28), inset 0 0 0 3px rgba(132,151,122,0.24);
    display: flex; align-items: center; justify-content: center; opacity: 0;
    color: {pal["forest"]}; font-family: {_SERIF}; font-size: 18px; font-weight: 700;
  }}
</style>
</head>
<body>
<div id="mk-ps-root" data-composition-id="{comp_id}" data-start="0" data-duration="{dur_s:.4f}" data-width="{width}" data-height="{height}" data-fps="{fps_i}">
  <div id="allocTokens"></div>
  <div id="mk-ps-group">
    <div class="top-row">
      <div id="mk-ps-num">{_esc(sign_prefix)}{_esc(format_cents(abs(amount_cents), sym))}</div>
      <div id="mk-ps-label">{_esc(category)}</div>
    </div>
    <div id="mk-ps-track"><div id="mk-ps-fill"></div></div>
    <div class="meta-row">
      <div id="mk-ps-caption">{_esc(caption)}</div>
      <div id="mk-ps-share">{int(round(fill_ratio * 100))}% of {_esc(format_cents(denom_cents, sym))}</div>
    </div>
  </div>
</div>
<script>
  window.__timelines = window.__timelines || {{}};
  var DUR = {dur_s:.4f};
  var group = document.getElementById("mk-ps-group");
  var num = document.getElementById("mk-ps-num");
  var fill = document.getElementById("mk-ps-fill");
  var tokenRoot = document.getElementById("allocTokens");
  var TARGET = {abs(amount_cents)};
  var SIGN = {_j(sign_prefix)};
  var SYM = {_j(sym)};
  var RATIO = {fill_ratio:.6f};
  var ALLOW_FRAC = {"true" if allow_frac else "false"};
  function fmt(c) {{
    if (!ALLOW_FRAC) {{
      var wd = Math.round(Math.abs(c) / 100);
      return SIGN + SYM + wd.toLocaleString("en-US");
    }}
    var absC = Math.abs(Math.round(c));
    var w = Math.floor(absC / 100);
    var r = absC % 100;
    return SIGN + SYM + w.toLocaleString("en-US") + (r ? "." + String(r).padStart(2, "0") : "");
  }}
  var tokenSpecs = [
    {{ sx: 140, sy: 130, cx: 40, cy: 15, ex: -140, ey: -62, rot: -12, delay: 0.05 }},
    {{ sx: 165, sy: 150, cx: 70, cy: 30, ex: -115, ey: -54, rot: 9, delay: 0.14 }},
    {{ sx: 115, sy: 165, cx: 25, cy: 40, ex: -155, ey: -48, rot: -8, delay: 0.22 }},
    {{ sx: 180, sy: 140, cx: 85, cy: 20, ex: -125, ey: -58, rot: 11, delay: 0.30 }},
  ].slice(0, {token_count});
  var tokenEls = tokenSpecs.map(function (s) {{
    var el = document.createElement("div");
    el.className = "alloc-note";
    el.textContent = SYM;
    tokenRoot.appendChild(el);
    return {{ el: el, s: s }};
  }});
  var tl = gsap.timeline({{ paused: true }});
  gsap.set(group, {{ opacity: 0, x: -18 }});
  gsap.set(fill, {{ scaleX: 0 }});
  num.textContent = fmt(TARGET);
  tl.to(group, {{ opacity: 1, x: 0, duration: Math.min(0.28, DUR * 0.14), ease: "power3.out" }}, 0.03);
  var animDur = Math.min(0.62, Math.max(0.28, DUR * 0.28));
  tl.to(fill, {{ scaleX: RATIO, duration: animDur, ease: "power3.out" }}, 0.10);
  tokenEls.forEach(function (item) {{
    var s = item.s;
    var p = {{ t: 0 }};
    gsap.set(item.el, {{ x: s.sx, y: s.sy, rotation: s.rot * 0.4, opacity: 0, scale: 0.78 }});
    tl.to(item.el, {{ opacity: 0.88, scale: 0.94, duration: 0.12, ease: "power2.out" }}, 0.06 + s.delay);
    tl.to(p, {{
      t: 1, duration: 0.48, ease: "power3.out",
      onUpdate: function () {{
        var u = p.t, inv = 1 - u;
        var bx = inv * inv * s.sx + 2 * inv * u * s.cx + u * u * s.ex;
        var by = inv * inv * s.sy + 2 * inv * u * s.cy + u * u * s.ey + Math.sin(u * Math.PI * 2) * 2.0;
        gsap.set(item.el, {{ x: bx, y: by, rotation: s.rot * (0.4 + 0.6 * u) }});
      }},
    }}, 0.06 + s.delay);
    tl.to(item.el, {{ opacity: 0, scale: 0.76, duration: 0.18, ease: "power2.in" }}, 0.44 + s.delay);
  }});
  tl.to(group, {{ opacity: 0, y: -10, duration: Math.min(0.24, DUR * 0.12), ease: "power2.in" }}, Math.max(0.2, DUR - 0.26));
  window.__timelines["{comp_id}"] = tl;
</script>
</body>
</html>
"""

    if primitive_id == "hf_mk_specs_list":
        # Adapted directly from heygen-com/hyperframes `mk-specs-list` (Apache-2.0)
        # R5.R3 Fix D05: Zero Olivia/Notion/56-page hardcoded defaults!
        title = str(parameters.get("title", "INCLUDED COMPONENTS"))
        rows = parameters.get("rows") or []
        y_top = int(parameters.get("y_px", 720))
        rows_json = _j(rows[:4])
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<!-- adapted-from: heygen-com/hyperframes mk-specs-list (Apache-2.0) -->
<meta charset="UTF-8" />
<script src="./gsap.min.js"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  html, body {{ width: {width}px; height: {height}px; overflow: hidden; background: {bg_css}; }}
  #mk-sl-root {{
    --mk-font: {_SANS};
    --mk-ink: {pal["forest"]};
    --mk-ink-dim: {pal["espresso"]};
    --mk-accent: {pal["brass"]};
    position: relative; width: {width}px; height: {height}px; overflow: hidden; background: {bg_css};
    font-family: var(--mk-font);
  }}
  #mk-sl-card {{
    position: absolute; left: 54px; right: 54px; top: {y_top}px;
    padding: 18px 8px 12px;
    background: transparent;
    border-top: 2px solid {pal["brass"]};
  }}
  #mk-sl-title {{
    font-size: 24px; font-weight: 800; letter-spacing: 0.16em; text-transform: uppercase;
    color: {pal["terracotta"] if "terracotta" in pal else pal["clay"]}; margin-bottom: 20px;
  }}
  #mk-sl-col {{ display: flex; flex-direction: column; gap: 22px; }}
  .mk-sl-row {{ opacity: 0; }}
  .mk-sl-label {{ font-family: {_SERIF}; font-weight: 700; font-size: 44px; line-height: 1.08; color: var(--mk-ink); }}
  .mk-sl-value {{ margin-top: 6px; font-weight: 600; font-size: 28px; color: var(--mk-ink-dim); opacity: 0.85; }}
  .mk-sl-line {{ position: relative; margin-top: 14px; height: 3px; background: rgba(36,59,50,0.14); border-radius: 2px; overflow: hidden; }}
  .mk-sl-line-fill {{ position: absolute; inset: 0; background: var(--mk-accent); transform-origin: left center; }}
</style>
</head>
<body>
<div id="mk-sl-root" data-composition-id="{comp_id}" data-start="0" data-duration="{dur_s:.4f}" data-width="{width}" data-height="{height}" data-fps="{fps_i}">
  <div id="mk-sl-card">
    <div id="mk-sl-title">{_esc(title)}</div>
    <div id="mk-sl-col"></div>
  </div>
</div>
<script>
  window.__timelines = window.__timelines || {{}};
  var ROWS = {rows_json};
  var DUR = {dur_s:.4f};
  var col = document.getElementById("mk-sl-col");
  ROWS.forEach(function (r, i) {{
    var row = document.createElement("div");
    row.id = "mk-sl-row-" + i;
    row.className = "mk-sl-row";
    var l = document.createElement("div"); l.className = "mk-sl-label"; l.textContent = r.label || "";
    row.appendChild(l);
    if (r.value) {{
      var v = document.createElement("div"); v.className = "mk-sl-value"; v.textContent = r.value;
      row.appendChild(v);
    }}
    var line = document.createElement("div"); line.className = "mk-sl-line";
    var fill = document.createElement("div"); fill.id = "mk-sl-fill-" + i; fill.className = "mk-sl-line-fill";
    line.appendChild(fill);
    row.appendChild(line);
    col.appendChild(row);
  }});
  var tl = gsap.timeline({{ paused: true }});
  gsap.set("#mk-sl-card", {{ opacity: 0, y: 18 }});
  tl.to("#mk-sl-card", {{ opacity: 1, y: 0, duration: 0.30, ease: "power3.out" }}, 0.02);
  ROWS.forEach(function (r, i) {{
    var t0 = 0.14 + i * Math.min(0.42, DUR * 0.20);
    gsap.set("#mk-sl-row-" + i, {{ opacity: 0, x: -28 }});
    gsap.set("#mk-sl-fill-" + i, {{ scaleX: 0 }});
    tl.to("#mk-sl-row-" + i, {{ opacity: 1, x: 0, duration: 0.44, ease: "power3.out" }}, t0);
    tl.to("#mk-sl-fill-" + i, {{ scaleX: 1, duration: 0.52, ease: "power2.inOut" }}, t0 + 0.08);
  }});
  tl.to("#mk-sl-card", {{ opacity: 0, y: -12, duration: 0.28, ease: "power2.in" }}, Math.max(0.25, DUR - 0.32));
  window.__timelines["{comp_id}"] = tl;
</script>
</body>
</html>
"""

    if primitive_id == "hf_mk_callout_highlight":
        # Adapted directly from heygen-com/hyperframes `mk-callout-highlight` (Apache-2.0) — D05 clean generic default
        text = str(parameters.get("text", ""))
        subtitle = str(parameters.get("subtitle", ""))
        font_px = int(parameters.get("font_size", 50))
        y_top = int(parameters.get("y_px", int(height * 0.095)))
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<!-- adapted-from: heygen-com/hyperframes mk-callout-highlight (Apache-2.0) -->
<meta charset="UTF-8" />
<script src="./gsap.min.js"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  html, body {{ width: {width}px; height: {height}px; overflow: hidden; background: {bg_css}; }}
  #mk-ch-root {{ position: relative; width: {width}px; height: {height}px; overflow: hidden; background: {bg_css}; font-family: {_SERIF}; }}
  #mk-ch-plate {{
    position: absolute; left: 48px; right: 48px; top: {y_top}px;
    padding: 28px 34px; border-radius: 24px;
    background: rgba(41, 40, 34, 0.88); border-left: 6px solid {pal["brass"]};
    box-shadow: 0 18px 44px rgba(18, 16, 12, 0.42);
  }}
  #mk-ch-box {{ font-size: {font_px}px; font-weight: 700; line-height: 1.22; letter-spacing: -0.01em; }}
  .mk-ch-word {{ display: inline; }}
  #mk-ch-sub {{ margin-top: 14px; font-family: {_SANS}; font-size: 28px; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; color: {pal["brass"]}; opacity: 0; }}
</style>
</head>
<body>
<div id="mk-ch-root" data-composition-id="{comp_id}" data-start="0" data-duration="{dur_s:.4f}" data-width="{width}" data-height="{height}" data-fps="{fps_i}">
  <div id="mk-ch-plate">
    <div id="mk-ch-box"></div>
    {f'<div id="mk-ch-sub">{_esc(subtitle)}</div>' if subtitle else ''}
  </div>
</div>
<script>
  window.__timelines = window.__timelines || {{}};
  var TEXT = {_j(text)};
  var DUR = {dur_s:.4f};
  var FULL = {_j(pal["ivory"])};
  var FAINT = "rgba(245, 240, 230, 0.30)";
  var box = document.getElementById("mk-ch-box");
  var wordEls = [];
  TEXT.split(/\\s+/).filter(Boolean).forEach(function (w, i) {{
    if (i > 0) box.appendChild(document.createTextNode(" "));
    var s = document.createElement("span"); s.className = "mk-ch-word"; s.textContent = w; s.style.color = FAINT;
    box.appendChild(s); wordEls.push(s);
  }});
  var st = {{ p: 0 }};
  var mix = gsap.utils.interpolate(FAINT, FULL);
  function render() {{
    var k = st.p * wordEls.length;
    for (var i = 0; i < wordEls.length; i++) {{
      var f = Math.max(0, Math.min(1, k - i));
      wordEls[i].style.color = mix(f);
    }}
  }}
  var tl = gsap.timeline({{ paused: true }});
  gsap.set("#mk-ch-plate", {{ opacity: 0, y: 20 }});
  tl.to("#mk-ch-plate", {{ opacity: 1, y: 0, duration: 0.30, ease: "power3.out" }}, 0.02);
  tl.to(st, {{ p: 1, duration: Math.max(0.4, DUR * 0.56), ease: "none" }}, 0.14);
  if (document.getElementById("mk-ch-sub")) {{
    tl.to("#mk-ch-sub", {{ opacity: 1, duration: 0.30, ease: "power2.out" }}, Math.min(DUR * 0.65, 0.14 + DUR * 0.45));
  }}
  tl.to("#mk-ch-plate", {{ opacity: 0, y: -14, duration: 0.28, ease: "power2.in" }}, Math.max(0.25, DUR - 0.30));
  tl.eventCallback("onUpdate", render);
  render();
  window.__timelines["{comp_id}"] = tl;
</script>
</body>
</html>
"""

    if primitive_id == "hf_lt_mask_reveal":
        # Adapted directly from heygen-com/hyperframes `lt-mask-reveal` (Apache-2.0) — D05 zero Olivia defaults!
        headline = str(parameters.get("headline", ""))
        subtitle = str(parameters.get("subtitle", ""))
        y_top = max(96, int(parameters.get("y_px", 110)))
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<!-- adapted-from: heygen-com/hyperframes lt-mask-reveal (Apache-2.0) -->
<meta charset="UTF-8" />
<script src="./gsap.min.js"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  html, body {{ width: {width}px; height: {height}px; overflow: hidden; background: {bg_css}; }}
  #root {{ position: relative; width: {width}px; height: {height}px; overflow: hidden; background: {bg_css}; font-family: {_SANS}; }}
  .scrim {{
    position: absolute; left: 0; right: 0; top: 0; height: 340px; pointer-events: none;
    background: linear-gradient(180deg, rgba(24,26,22,0.72) 0%, rgba(24,26,22,0.34) 62%, transparent 100%);
  }}
  .lt {{
    position: absolute; left: 48px; right: 105px; top: {y_top}px;
    padding: 0; background: transparent;
    display: flex; flex-direction: column; align-items: flex-start; gap: 10px;
  }}
  .lt-namewrap {{ position: relative; overflow: hidden; padding: 4px 0; max-width: 100%; }}
  .lt-name {{
    font-family: {_SERIF}; font-size: 54px; font-weight: 700; color: {pal["ivory"]}; line-height: 1.06;
    text-shadow: 0 6px 22px rgba(14,12,10,0.58);
  }}
  .lt-sweep {{ position: absolute; top: 0; bottom: 0; left: 0; width: 6px; background: {pal["brass"]}; border-radius: 2px; }}
  .lt-role {{
    font-size: 26px; font-weight: 800; letter-spacing: 0.14em; text-transform: uppercase;
    color: {pal["brass"]}; text-shadow: 0 2px 10px rgba(0,0,0,0.48);
  }}
</style>
</head>
<body>
<div id="root" data-composition-id="{comp_id}" data-start="0" data-duration="{dur_s:.4f}" data-width="{width}" data-height="{height}" data-fps="{fps_i}">
  <div class="scrim"></div>
  <div id="lt" class="lt">
    <div class="lt-namewrap" id="lt-wrap">
      <div id="lt-name" class="lt-name">{_esc(headline)}</div>
      <div id="lt-sweep" class="lt-sweep"></div>
    </div>
    <div id="lt-role" class="lt-role">{_esc(subtitle)}</div>
  </div>
</div>
<script>
  window.__timelines = window.__timelines || {{}};
  var DUR = {dur_s:.4f};
  var tl = gsap.timeline({{ paused: true }});
  var nameEl = document.getElementById("lt-name");
  var sweep = document.getElementById("lt-sweep");
  var role = document.getElementById("lt-role");
  var travel = Math.max(220, document.getElementById("lt-wrap").clientWidth || {width - 180});
  gsap.set("#lt", {{ opacity: 0 }});
  gsap.set(nameEl, {{ clipPath: "inset(0 100% 0 0)" }});
  gsap.set(sweep, {{ x: 0, opacity: 0 }});
  gsap.set(role, {{ y: 12, opacity: 0 }});
  tl.to("#lt", {{ opacity: 1, duration: 0.18, ease: "power2.out" }}, 0.02);
  tl.to(sweep, {{ opacity: 1, duration: 0.08 }}, 0.06);
  tl.to(sweep, {{ x: travel, duration: 0.48, ease: "power2.inOut" }}, 0.08);
  tl.to(nameEl, {{ clipPath: "inset(0 0% 0 0)", duration: 0.44, ease: "power2.inOut" }}, 0.12);
  tl.to(sweep, {{ opacity: 0, duration: 0.12 }}, 0.54);
  tl.to(role, {{ y: 0, opacity: 1, duration: 0.38, ease: "power3.out" }}, 0.42);
  tl.to(role, {{ opacity: 0, duration: 0.22, ease: "power2.in" }}, Math.max(0.3, DUR - 0.34));
  tl.to(nameEl, {{ clipPath: "inset(0 0 0 100%)", duration: 0.28, ease: "power2.in" }}, Math.max(0.35, DUR - 0.30));
  tl.to("#lt", {{ opacity: 0, duration: 0.18 }}, Math.max(0.4, DUR - 0.20));
  window.__timelines["{comp_id}"] = tl;
</script>
</body>
</html>
"""

    if primitive_id == "hf_strikethrough_replace":
        # Adapted directly from heygen-com/hyperframes `strikethrough-replace` (Apache-2.0) — D05 zero Olivia defaults!
        from_txt = str(parameters.get("from_text", ""))
        to_txt = str(parameters.get("to_text", ""))
        eyebrow = str(parameters.get("eyebrow", ""))
        y_top = int(parameters.get("y_px", int(height * 0.095)))
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<!-- adapted-from: heygen-com/hyperframes strikethrough-replace (Apache-2.0) -->
<meta charset="UTF-8" />
<script src="./gsap.min.js"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  html, body {{ width: {width}px; height: {height}px; overflow: hidden; background: {bg_css}; }}
  #root {{ position: relative; width: {width}px; height: {height}px; overflow: hidden; background: {bg_css}; font-family: {_SANS}; }}
  .box {{
    position: absolute; left: 48px; right: 48px; top: {y_top}px;
    padding: 28px 32px; border-radius: 24px;
    background: rgba(41, 40, 34, 0.90); border: 2px solid {pal["brass"]};
    display: flex; flex-direction: column; align-items: center; text-align: center; gap: 14px;
    box-shadow: 0 20px 50px rgba(18,16,12,0.44);
  }}
  .eyebrow {{ font-size: 24px; font-weight: 800; letter-spacing: 0.16em; color: {pal["brass"]}; }}
  .hf-catalog-strikethrough-replace {{
    position: relative; display: flex; flex-direction: column; align-items: center; gap: 12px;
    color: {pal["ivory"]}; font-weight: 900; letter-spacing: -0.02em;
  }}
  .old-wrap {{ position: relative; display: inline-block; font-size: 52px; font-variant-numeric: tabular-nums; }}
  .hf-catalog-strikethrough-replace i {{
    position: absolute; left: -6px; right: -6px; top: 50%; height: 8px; border-radius: 999px;
    background: {pal["clay"]}; transform: rotate(-1.4deg) scaleX(var(--hf-strike-progress, 0));
    transform-origin: left center;
  }}
  .hf-catalog-strikethrough-replace strong {{
    font-size: 44px; color: {pal["brass"]}; opacity: var(--hf-replace-opacity, 0);
  }}
</style>
</head>
<body>
<div id="root" data-composition-id="{comp_id}" data-start="0" data-duration="{dur_s:.4f}" data-width="{width}" data-height="{height}" data-fps="{fps_i}">
  <div id="box" class="box">
    <div class="eyebrow">{_esc(eyebrow)}</div>
    <div class="hf-catalog-strikethrough-replace">
      <span class="old-wrap">{_esc(from_txt)}<i id="strikeBar"></i></span>
      <strong id="repTxt">{_esc(to_txt)}</strong>
    </div>
  </div>
</div>
<script>
  window.__timelines = window.__timelines || {{}};
  var DUR = {dur_s:.4f};
  var tl = gsap.timeline({{ paused: true }});
  gsap.set("#box", {{ opacity: 0, y: 20 }});
  tl.to("#box", {{ opacity: 1, y: 0, duration: 0.28, ease: "power3.out" }}, 0.02);
  tl.fromTo("#strikeBar", {{ "--hf-strike-progress": 0 }}, {{ "--hf-strike-progress": 1, duration: 0.36, ease: "power2.out" }}, 0.28);
  tl.fromTo("#repTxt", {{ "--hf-replace-opacity": 0, y: 16 }}, {{ "--hf-replace-opacity": 1, y: 0, duration: 0.36, ease: "power3.out" }}, 0.56);
  tl.to("#box", {{ opacity: 0, y: -14, duration: 0.28, ease: "power2.in" }}, Math.max(0.3, DUR - 0.32));
  window.__timelines["{comp_id}"] = tl;
</script>
</body>
</html>
"""

    if primitive_id == "hf_organic_light_leak":
        # Adapted directly from heygen-com/hyperframes `organic-light-leak-overlay` (Apache-2.0)
        # R5.R2 Fix D08: Strictly peripheral warm rim glow (<= 0.22 opacity) with center face cutout mask so skin contrast stays 100% protected!
        max_op = min(0.22, max(0.14, float(parameters.get("opacity", 0.22))))
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<!-- adapted-from: heygen-com/hyperframes organic-light-leak-overlay (Apache-2.0) -->
<meta charset="UTF-8" />
<script src="./gsap.min.js"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  html, body {{ width: {width}px; height: {height}px; overflow: hidden; background: transparent; }}
  #hf-light-leak {{
    position: absolute; inset: 0; opacity: 0; overflow: hidden; pointer-events: none;
    -webkit-mask-image: radial-gradient(ellipse 44% 52% at 50% 42%, transparent 0%, transparent 54%, black 88%);
    mask-image: radial-gradient(ellipse 44% 52% at 50% 42%, transparent 0%, transparent 54%, black 88%);
  }}
  #hf-light-leak .hf-leak-layer {{ position: absolute; inset: -10%; }}
  #hf-light-leak .hf-leak-amber {{
    background: radial-gradient(ellipse 48% 96% at 6% 52%, rgba(224, 178, 96, 1.0), transparent 60%),
                radial-gradient(ellipse 46% 90% at 94% 68%, rgba(245, 240, 230, 0.96), transparent 60%);
    filter: blur(16px);
  }}
  #hf-light-leak .hf-leak-gold {{
    background: linear-gradient(112deg, rgba(212, 168, 92, 0.94) 0%, transparent 28%, transparent 72%, rgba(164, 186, 148, 0.84) 100%);
    filter: blur(12px);
  }}
</style>
</head>
<body>
<div id="root" data-composition-id="{comp_id}" data-start="0" data-duration="{dur_s:.4f}" data-width="{width}" data-height="{height}" data-fps="{fps_i}">
  <div id="hf-light-leak">
    <div class="hf-leak-layer hf-leak-amber"></div>
    <div class="hf-leak-layer hf-leak-gold"></div>
  </div>
</div>
<script>
  window.__timelines = window.__timelines || {{}};
  var dur = {dur_s:.4f};
  var tl = gsap.timeline({{ paused: true }});
  tl.fromTo("#hf-light-leak", {{ opacity: 0 }}, {{ opacity: {max_op:.3f}, duration: Math.min(0.45, dur * 0.16), ease: "sine.out" }}, 0.02);
  tl.to("#hf-light-leak", {{ opacity: 0, duration: Math.min(0.55, dur * 0.22), ease: "sine.in" }}, Math.max(0.5, dur - 0.60));
  tl.fromTo(".hf-leak-amber", {{ xPercent: -8, scale: 0.97, rotation: -2 }}, {{ xPercent: 8, scale: 1.04, rotation: 2, duration: dur, ease: "sine.inOut" }}, 0);
  tl.fromTo(".hf-leak-gold", {{ xPercent: -10, rotation: -3 }}, {{ xPercent: 10, rotation: 3, duration: dur, ease: "sine.inOut" }}, 0);
  window.__timelines["{comp_id}"] = tl;
</script>
</body>
</html>
"""

    if primitive_id == "hf_data_chart":
        # Adapted directly from heygen-com/hyperframes `data-chart` (Apache-2.0) — strictly proportional, zero Olivia defaults!
        title = str(parameters.get("title", "PROPORTIONAL BREAKDOWN"))
        denom_cents = max(1, int(parameters.get("denom_cents", 100)))
        sym = str(parameters.get("currency_symbol", "$"))
        items = parameters.get("items") or []
        y_top = max(96, int(parameters.get("y_px", 112)))
        bars_html = []
        tweens_js = []
        for idx, it in enumerate(items[:5]):
            c = max(0, int(it.get("cents", 0)))
            pct = max(3.0, min(100.0, (c / denom_cents) * 100.0))
            col = str(it.get("color") or pal["brass"])
            bars_html.append(
                f'<div class="bar-row" id="row_{idx}">'
                f'<div class="bar-head"><span class="bar-lbl">{_esc(it.get("label", ""))}</span>'
                f'<span class="bar-val">{_esc(format_cents(c, sym))} <span class="bar-pct">({int(round(pct))}%)</span></span></div>'
                f'<div class="bar-track"><div class="bar-fill" id="fill_{idx}" style="background:{_esc(col)}; width:{pct:.2f}%"></div></div>'
                f'</div>'
            )
            t0 = round(0.12 + idx * 0.14, 3)
            tweens_js.append(f'gsap.set("#row_{idx}", {{ opacity: 0, y: 14 }}); gsap.set("#fill_{idx}", {{ scaleX: 0 }});')
            tweens_js.append(f'tl.to("#row_{idx}", {{ opacity: 1, y: 0, duration: 0.32, ease: "power3.out" }}, {t0});')
            tweens_js.append(f'tl.to("#fill_{idx}", {{ scaleX: 1, duration: 0.52, ease: "power2.out" }}, {t0 + 0.08:.3f});')
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<!-- adapted-from: heygen-com/hyperframes data-chart (Apache-2.0) -->
<meta charset="UTF-8" />
<script src="./gsap.min.js"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  html, body {{ width: {width}px; height: {height}px; overflow: hidden; background: {bg_css}; }}
  #root {{ position: relative; width: {width}px; height: {height}px; overflow: hidden; background: {bg_css}; font-family: {_SANS}; }}
  .chart-card {{
    position: absolute; left: 48px; right: 56px; top: {y_top}px;
    padding: 8px 4px;
    background: transparent; border: none;
  }}
  .chart-hdr {{
    display: flex; justify-content: space-between; align-items: baseline;
    padding-bottom: 14px; margin-bottom: 24px;
    border-bottom: 2px solid {pal["brass"]};
  }}
  .chart-title {{ font-size: 26px; font-weight: 800; letter-spacing: 0.14em; color: {pal["forest"]}; text-transform: uppercase; }}
  .chart-denom {{ font-size: 26px; font-weight: 700; color: {pal["espresso"]}; font-variant-numeric: tabular-nums; }}
  .bar-row {{ margin-bottom: 24px; }}
  .bar-row:last-child {{ margin-bottom: 0; }}
  .bar-head {{ display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 10px; }}
  .bar-lbl {{ font-family: {_SERIF}; font-size: 38px; font-weight: 700; color: {pal["forest"]}; }}
  .bar-val {{ font-size: 40px; font-weight: 900; color: {pal["espresso"]}; font-variant-numeric: tabular-nums; }}
  .bar-pct {{ font-size: 24px; font-weight: 600; color: {pal["sage"]}; }}
  .bar-track {{ width: 100%; height: 22px; border-radius: 5px; background: rgba(64,56,46,0.12); overflow: hidden; }}
  .bar-fill {{ height: 100%; border-radius: 5px; transform-origin: left center; }}
</style>
</head>
<body>
<div id="root" data-composition-id="{comp_id}" data-start="0" data-duration="{dur_s:.4f}" data-width="{width}" data-height="{height}" data-fps="{fps_i}">
  <div id="chartCard" class="chart-card">
    <div class="chart-hdr">
      <span class="chart-title">{_esc(title)}</span>
      <span class="chart-denom">of {_esc(format_cents(denom_cents, sym))}</span>
    </div>
    {"".join(bars_html)}
  </div>
</div>
<script>
  window.__timelines = window.__timelines || {{}};
  var DUR = {dur_s:.4f};
  var tl = gsap.timeline({{ paused: true }});
  gsap.set("#chartCard", {{ opacity: 0, y: 18 }});
  tl.to("#chartCard", {{ opacity: 1, y: 0, duration: 0.28, ease: "power3.out" }}, 0.02);
  {"".join(tweens_js)}
  tl.to("#chartCard", {{ opacity: 0, y: -14, duration: 0.28, ease: "power2.in" }}, Math.max(0.3, DUR - 0.32));
  window.__timelines["{comp_id}"] = tl;
</script>
</body>
</html>
"""

    # Default: hf_exploded_stack (adapted directly from heygen-com/hyperframes `exploded-stack` Apache-2.0) — D05 zero Olivia defaults!
    title = str(parameters.get("title", "SYSTEM OVERVIEW"))
    badge = str(parameters.get("conceptual_badge", "CONCEPTUAL OVERVIEW"))
    layers = parameters.get("layers") or []
    y_top = int(parameters.get("y_px", int(height * 0.12)))
    plates_html = []
    tweens = []
    for idx, ly in enumerate(layers[:3]):
        col = str(ly.get("color") or pal["brass"])
        y_off = idx * 124
        plates_html.append(
            f'<div class="iso-row" id="iso_{idx}" style="top:{y_off}px">'
            f'<div class="iso-plate" style="background:{_esc(col)};"><span class="iso-idx">0{idx + 1}</span></div>'
            f'<div class="iso-text"><div class="iso-lbl">{_esc(ly.get("label", ""))}</div>'
            f'<div class="iso-sub">{_esc(ly.get("sublabel", ""))}</div></div>'
            f'</div>'
        )
        t0 = round(0.14 + idx * 0.20, 3)
        tweens.append(f'gsap.set("#iso_{idx}", {{ opacity: 0, y: 48 }});')
        tweens.append(f'tl.to("#iso_{idx}", {{ opacity: 1, y: 0, duration: 0.48, ease: "power3.out" }}, {t0});')
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<!-- adapted-from: heygen-com/hyperframes exploded-stack (Apache-2.0) -->
<meta charset="UTF-8" />
<script src="./gsap.min.js"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  html, body {{ width: {width}px; height: {height}px; overflow: hidden; background: {bg_css}; }}
  #root {{ position: relative; width: {width}px; height: {height}px; overflow: hidden; background: {bg_css}; font-family: {_SANS}; }}
  .stack-box {{
    position: absolute; left: 48px; right: 48px; top: {y_top}px; height: 470px;
    padding: 26px 30px; border-radius: 26px;
    background: rgba(41, 40, 34, 0.90); border: 2px solid {pal["brass"]};
    box-shadow: 0 22px 54px rgba(18, 16, 12, 0.45);
  }}
  .hdr {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 18px; }}
  .title {{ font-size: 26px; font-weight: 800; letter-spacing: 0.12em; color: {pal["ivory"]}; }}
  .badge {{ font-size: 18px; font-weight: 700; letter-spacing: 0.1em; color: {pal["espresso"]}; background: {pal["brass"]}; padding: 5px 12px; border-radius: 999px; }}
  .iso-stage {{ position: relative; width: 100%; height: 370px; margin-top: 12px; }}
  .iso-row {{ position: absolute; left: 0; right: 0; display: flex; align-items: center; gap: 26px; }}
  .iso-plate {{
    width: 112px; height: 74px; border-radius: 16px; flex: none;
    transform: perspective(480px) rotateX(48deg) rotateZ(-22deg);
    display: grid; place-items: center;
    box-shadow: 0 18px 28px rgba(0,0,0,0.42), inset 0 0 0 2px rgba(245,240,230,0.45);
  }}
  .iso-idx {{ font-size: 26px; font-weight: 900; color: {pal["ivory"]}; transform: rotateZ(22deg); }}
  .iso-lbl {{ font-size: 34px; font-weight: 800; color: {pal["ivory"]}; line-height: 1.1; }}
  .iso-sub {{ margin-top: 4px; font-size: 24px; font-weight: 500; color: {pal["sage"]}; }}
</style>
</head>
<body>
<div id="root" data-composition-id="{comp_id}" data-start="0" data-duration="{dur_s:.4f}" data-width="{width}" data-height="{height}" data-fps="{fps_i}">
  <div id="stackBox" class="stack-box">
    <div class="hdr"><span class="title">{_esc(title)}</span><span class="badge">{_esc(badge)}</span></div>
    <div class="iso-stage">{"".join(plates_html)}</div>
  </div>
</div>
<script>
  window.__timelines = window.__timelines || {{}};
  var DUR = {dur_s:.4f};
  var tl = gsap.timeline({{ paused: true }});
  gsap.set("#stackBox", {{ opacity: 0, y: 18 }});
  tl.to("#stackBox", {{ opacity: 1, y: 0, duration: 0.28, ease: "power3.out" }}, 0.02);
  {"".join(tweens)}
  tl.to("#stackBox", {{ opacity: 0, y: -14, duration: 0.28, ease: "power2.in" }}, Math.max(0.3, DUR - 0.32));
  window.__timelines["{comp_id}"] = tl;
</script>
</body>
</html>
"""
