"""Phase 5 (ML-01.R4): 12 High-Craft Substantive HyperFrames + GSAP Semantic Graphics Templates (`vfx-*`).

Every template is 100% offline (local GSAP, zero external fonts/CDNs/URLs), driven strictly by tracked anchors,
verified grounded labels (XML-escaped, 0-4 words max), and editorial color hierarchy (warm ivory/sage/espresso/gold).
"""

from __future__ import annotations

import html as _html
import math
from typing import Any

SUPPORTED_VFX_TEMPLATES = {
    "vfx-depth-stage",
    "vfx-prop-accent",
    "vfx-macro-pointer",
    "vfx-action-flow",
    "vfx-benefit-ledger",
    "vfx-shape-morph",
    "vfx-occlusion-wipe",
    "vfx-editorial-collage",
    "vfx-spatial-planes",
    "vfx-trail-accent",
    "vfx-payoff-glint",
    "vfx-closing-resolve",
}

_FONT_STACK = "-apple-system, BlinkMacSystemFont, 'Helvetica Neue', Arial, sans-serif"
_SERIF_STACK = "Georgia, 'Times New Roman', serif"


def _esc(s: str) -> str:
    return _html.escape((s or "").strip(), quote=True)


def build_vfx_graphics_html(
    scene_id: str,
    template_name: str,
    width: int,
    height: int,
    fps: float,
    duration_s: float,
    gsap_rel_path: str,
    anchor_bundle: dict[str, Any],
    parameters: dict[str, Any],
) -> str:
    """Build substantive editorial GSAP+SVG HTML composition for any ML-01.R4 `vfx-*` template."""
    pal = (parameters.get("palette") or {}) if isinstance(parameters.get("palette"), dict) else {}
    c_gold = str(pal.get("primary_hex") or "#C8A26A")
    c_esp = str(pal.get("secondary_hex") or "#3B312A")
    c_ivory = str(pal.get("tertiary_hex") or "#F7F1E6")
    c_sage = str(pal.get("accent_sage") or "#6E8B74")

    disp_label = _esc(str(parameters.get("display_label") or ""))
    sec_labels = [_esc(str(x)) for x in (parameters.get("secondary_labels") or []) if str(x).strip()][:4]
    payoff_local_f = int(parameters.get("payoff_local_frame", max(4, int(duration_s * fps * 0.5))))
    payoff_t = max(0.25, min(duration_s * 0.72, payoff_local_f / max(1.0, fps)))

    stage = anchor_bundle.get("behind_stage") or {"cx": 0.5, "cy": 0.38}
    task_mid = anchor_bundle.get("task_mid") or {"cx": 0.52, "cy": 0.70, "w": 0.34, "h": 0.22}
    slot = anchor_bundle.get("detail_slot") or {"cx": 0.25, "cy": 0.12, "w": 0.41, "h": 0.165, "side": "left"}
    sw = max(3.0, round(min(width, height) * 0.0048, 2))
    op = 0.94

    defs = (
        "<defs>"
        '<filter id="edSh" x="-25%" y="-25%" width="150%" height="150%">'
        '<feDropShadow dx="0" dy="8" stdDeviation="9" flood-color="#19120D" flood-opacity="0.38"/>'
        "</filter>"
        '<filter id="txtSh" x="-20%" y="-20%" width="140%" height="140%">'
        '<feDropShadow dx="0" dy="3" stdDeviation="4" flood-color="#150F0A" flood-opacity="0.52"/>'
        "</filter>"
        f'<linearGradient id="archFill" x1="0%" y1="0%" x2="0%" y2="100%">'
        f'<stop offset="0%" stop-color="{c_ivory}" stop-opacity="0.82"/>'
        f'<stop offset="46%" stop-color="{c_gold}" stop-opacity="0.62"/>'
        f'<stop offset="86%" stop-color="{c_esp}" stop-opacity="0.22"/>'
        f'<stop offset="100%" stop-color="{c_esp}" stop-opacity="0.0"/>'
        "</linearGradient>"
        f'<linearGradient id="rimFade" x1="0%" y1="0%" x2="0%" y2="100%">'
        f'<stop offset="0%" stop-color="{c_ivory}" stop-opacity="0.96"/>'
        f'<stop offset="68%" stop-color="{c_gold}" stop-opacity="0.72"/>'
        f'<stop offset="100%" stop-color="{c_gold}" stop-opacity="0.0"/>'
        "</linearGradient>"
        f'<linearGradient id="ribGrad" x1="0%" y1="0%" x2="100%" y2="0%">'
        f'<stop offset="0%" stop-color="{c_ivory}" stop-opacity="0.20"/>'
        f'<stop offset="45%" stop-color="{c_gold}" stop-opacity="0.90"/>'
        f'<stop offset="80%" stop-color="{c_sage}" stop-opacity="0.82"/>'
        f'<stop offset="100%" stop-color="{c_ivory}" stop-opacity="0.86"/>'
        "</linearGradient>"
        f'<linearGradient id="wipeGrad" x1="0%" y1="0%" x2="100%" y2="100%">'
        f'<stop offset="0%" stop-color="{c_ivory}" stop-opacity="0.0"/>'
        f'<stop offset="35%" stop-color="{c_gold}" stop-opacity="0.72"/>'
        f'<stop offset="52%" stop-color="{c_ivory}" stop-opacity="0.86"/>'
        f'<stop offset="70%" stop-color="{c_sage}" stop-opacity="0.58"/>'
        f'<stop offset="100%" stop-color="{c_esp}" stop-opacity="0.0"/>'
        "</linearGradient>"
        "</defs>"
    )

    if template_name == "vfx-depth-stage":
        # Open-bottomed editorial architectural double-arch + optional grounded bold headline strictly BEHIND speaker!
        cx = int(round(width * float(stage["cx"])))
        cy = int(round(height * float(stage["cy"])))
        aw = int(round(width * 0.42))
        aw_in = int(round(aw * 0.89))
        top_y = max(42, cy - int(height * 0.28))
        bot_y = min(int(height * 0.77), cy + int(height * 0.30))
        sp_y = top_y + aw
        sp_in = top_y + int(aw * 0.12) + aw_in
        d_fill = f"M {cx - aw} {bot_y} L {cx - aw} {sp_y} A {aw} {aw} 0 0 1 {cx + aw} {sp_y} L {cx + aw} {bot_y} Z"
        d_rim = f"M {cx - aw} {bot_y} L {cx - aw} {sp_y} A {aw} {aw} 0 0 1 {cx + aw} {sp_y} L {cx + aw} {bot_y}"
        d_in = f"M {cx - aw_in} {bot_y - 24} L {cx - aw_in} {sp_in} A {aw_in} {aw_in} 0 0 1 {cx + aw_in} {sp_in} L {cx + aw_in} {bot_y - 24}"
        rim_len = int(round(2 * (bot_y - sp_y) + math.pi * aw + 20))
        halo_r = int(round(aw * 0.78))
        halo_c = int(round(2 * math.pi * halo_r))
        # Grounded editorial headline tucked crisply at the upper headroom above speaker's crown
        lbl_y = max(84, top_y - 12)
        lbl_svg = (
            f'<g id="lblG" opacity="0" filter="url(#txtSh)">'
            f'<text x="{width // 2}" y="{lbl_y}" text-anchor="middle" font-family="{_FONT_STACK}" '
            f'font-size="{max(32, int(width * 0.068))}" font-weight="800" letter-spacing="2.5" fill="{c_ivory}">{disp_label.upper()}</text>'
            f'<line x1="{int(width * 0.30)}" y1="{lbl_y + 14}" x2="{int(width * 0.70)}" y2="{lbl_y + 14}" stroke="{c_gold}" stroke-width="2.5" opacity="0.9"/>'
            "</g>"
        ) if disp_label else ""
        svg_body = (
            f'<g id="stgG" filter="url(#edSh)" opacity="0">'
            f'<path d="{d_fill}" fill="url(#archFill)" opacity="0.84"/>'
            f'<circle id="orb1" cx="{cx}" cy="{sp_y - 28}" r="{halo_r}" fill="none" stroke="{c_ivory}" stroke-width="{sw * 0.75:.1f}" '
            f'stroke-dasharray="{int(halo_c * 0.28)} {int(halo_c * 0.72)}" stroke-linecap="round" opacity="0.82"/>'
            f'<path id="aRim" d="{d_rim}" fill="none" stroke="url(#rimFade)" stroke-width="{sw * 1.1:.1f}" stroke-linecap="round" stroke-dasharray="{rim_len}" stroke-dashoffset="{rim_len}"/>'
            f'<path d="{d_in}" fill="none" stroke="url(#rimFade)" stroke-width="{sw * 0.55:.1f}" opacity="0.72"/>'
            "</g>"
            f"{lbl_svg}"
        )
        gsap_code = (
            f'tl.fromTo("#stgG", {{opacity: 0, scale: 0.91, transformOrigin: "{cx}px {cy}px"}}, {{opacity: {op}, scale: 1.0, duration: {duration_s * 0.26:.3f}, ease: "power3.out"}}, 0);'
            f'tl.to("#aRim", {{strokeDashoffset: 0, duration: {duration_s * 0.36:.3f}, ease: "power2.out"}}, 0.04);'
            f'tl.to("#orb1", {{rotation: 36, transformOrigin: "{cx}px {sp_y - 28}px", duration: {duration_s:.3f}, ease: "sine.inOut"}}, 0);'
            + (f'tl.fromTo("#lblG", {{opacity: 0, y: 18}}, {{opacity: 0.96, y: 0, duration: {duration_s * 0.24:.3f}, ease: "power3.out"}}, 0.14);' if disp_label else "")
            + f'tl.to(["#stgG"{", \"#lblG\"" if disp_label else ""}], {{opacity: 0, duration: {duration_s * 0.22:.3f}, ease: "power2.in"}}, {duration_s * 0.78:.3f});'
        )

    elif template_name == "vfx-prop-accent":
        # Tactile editorial focus ring + curved lift cue + grounded crisp badge in upper headroom (NEVER covers prop or face!)
        tx = int(round(width * float(task_mid["cx"])))
        ty = int(round(height * min(0.84, float(task_mid["cy"]) + 0.09)))
        pill_y = int(round(height * 0.085))
        pill_w = min(int(width * 0.66), max(210, len(disp_label) * 20 + 68)) if disp_label else 0
        pill_x = (width - pill_w) // 2
        badge_svg = (
            f'<g id="pBadge" filter="url(#edSh)" opacity="0">'
            f'<rect x="{pill_x}" y="{pill_y}" width="{pill_w}" height="58" rx="29" fill="{c_esp}" fill-opacity="0.88" stroke="{c_gold}" stroke-width="2.2"/>'
            f'<circle cx="{pill_x + 28}" cy="{pill_y + 29}" r="6" fill="{c_gold}"/>'
            f'<text x="{pill_x + pill_w // 2 + 8}" y="{pill_y + 37}" text-anchor="middle" font-family="{_FONT_STACK}" font-size="23" font-weight="700" letter-spacing="1.4" fill="{c_ivory}">{disp_label.upper()}</text>'
            "</g>"
        ) if disp_label else ""
        # Subtle curved specular arc under lifted prop + outer breathing pulse rings
        arc_d = f"M {max(40, tx - 140)} {ty + 42} Q {tx} {ty + 86} {min(width - 40, tx + 140)} {ty + 42}"
        svg_body = (
            f'<g id="propG" filter="url(#edSh)">'
            f'<path id="pArc" d="{arc_d}" fill="none" stroke="{c_gold}" stroke-width="{sw * 1.1:.1f}" stroke-linecap="round" stroke-dasharray="360" stroke-dashoffset="360" opacity="0"/>'
            f'<circle id="pR1" cx="{tx}" cy="{ty}" r="22" fill="none" stroke="{c_ivory}" stroke-width="2.5" opacity="0"/>'
            f'<circle id="pR2" cx="{tx}" cy="{ty}" r="30" fill="none" stroke="{c_gold}" stroke-width="1.8" stroke-dasharray="14 14" opacity="0"/>'
            "</g>"
            f"{badge_svg}"
        )
        gsap_code = (
            f'tl.to("#pArc", {{strokeDashoffset: 0, opacity: 0.88, duration: {duration_s * 0.34:.3f}, ease: "power3.out"}}, 0.08);'
            f'tl.fromTo("#pR1", {{opacity: 0, attr: {{r: 18}}}}, {{opacity: 0.82, attr: {{r: 68}}, duration: {duration_s * 0.42:.3f}, ease: "power2.out"}}, 0.10);'
            f'tl.fromTo("#pR2", {{opacity: 0, attr: {{r: 26}}}}, {{opacity: 0.65, attr: {{r: 92}}, duration: {duration_s * 0.48:.3f}, ease: "power2.out"}}, 0.14);'
            + (f'tl.fromTo("#pBadge", {{opacity: 0, y: -16}}, {{opacity: 1, y: 0, duration: {duration_s * 0.24:.3f}, ease: "power3.out"}}, 0.10);' if disp_label else "")
            + f'tl.to(["#propG"{", \"#pBadge\"" if disp_label else ""}], {{opacity: 0, duration: {duration_s * 0.22:.3f}, ease: "power2.in"}}, {duration_s * 0.78:.3f});'
        )

    elif template_name == "vfx-benefit-ledger":
        # Grounded editorial financial/benefit breakdown stage in upper headroom (ZERO face/hands occlusion!)
        # Uses ACTUAL numeric values / proportions (never fake index-based 0.82 - idx*0.18 widths!) and phone-readable type.
        import re as _re

        items = sec_labels if sec_labels else ([disp_label] if disp_label else ["ORGANIZE", "ALLOCATE", "CLARITY"])
        n_it = min(3, len(items))
        items = items[:n_it]
        pad_x = int(width * 0.048)
        gap = 12
        card_w = (width - 2 * pad_x - gap * (n_it - 1)) // max(1, n_it)
        card_h = 112
        top_y = int(round(height * 0.046))
        hdr_svg = ""
        hdr_has_words = any(ch.isalpha() for ch in disp_label)
        if disp_label and hdr_has_words and disp_label not in items:
            hdr_svg = (
                f'<text id="lHdr" x="{width // 2}" y="{top_y + 22}" text-anchor="middle" font-family="{_FONT_STACK}" '
                f'font-size="24" font-weight="800" letter-spacing="1.8" fill="{c_ivory}" opacity="0" filter="url(#txtSh)">{disp_label.upper()}</text>'
            )
            top_y += 38

        # Parse real numeric amounts for true proportional bar width (fallback equal proportion only when non-numeric)
        parsed_vals: list[float | None] = []
        for txt in items:
            m = _re.search(r"(\d[\d,]*(?:\.\d+)?)", txt)
            parsed_vals.append(float(m.group(1).replace(",", "")) if m else None)
        denom = float(parameters.get("ledger_denominator") or 0.0)
        numeric_vals = [v for v in parsed_vals if v is not None and v > 0.0]
        if denom <= 0.0:
            denom = sum(numeric_vals) if numeric_vals else 1.0

        cards_svg = []
        bar_tweens = []
        card_tweens = []
        accent_cols = [c_gold, c_sage, c_ivory]
        for idx, txt in enumerate(items):
            cx_card = pad_x + idx * (card_w + gap)
            val = parsed_vals[idx]
            frac = min(1.0, max(0.06, val / denom)) if (val is not None and denom > 0.0) else (1.0 / max(1, n_it))
            inner_max_w = max(24, card_w - 28)
            bar_w = max(12, int(round(inner_max_w * frac)))
            col = accent_cols[idx % len(accent_cols)]
            parts = txt.split(" ", 1)
            has_num = any(ch.isdigit() for ch in parts[0])
            big_txt = parts[0] if has_num else txt
            sub_txt = parts[1].upper() if (has_num and len(parts) > 1) else ""
            big_size = 38 if (has_num and len(big_txt) <= 7) else (30 if has_num else 26)
            cards_svg.append(
                f'<g id="bc{idx}" opacity="0" filter="url(#edSh)">'
                f'<rect x="{cx_card}" y="{top_y}" width="{card_w}" height="{card_h}" rx="16" fill="{c_esp}" fill-opacity="0.93" stroke="{col}" stroke-width="2.4"/>'
                f'<rect x="{cx_card + 14}" y="{top_y + card_h - 18}" width="{inner_max_w}" height="7" rx="3.5" fill="{c_ivory}" fill-opacity="0.20"/>'
                f'<rect id="bb{idx}" x="{cx_card + 14}" y="{top_y + card_h - 18}" width="4" height="7" rx="3.5" fill="{col}"/>'
                f'<text x="{cx_card + 14}" y="{top_y + 48}" font-family="{_FONT_STACK}" font-size="{big_size}" font-weight="800" font-variant-numeric="tabular-nums" fill="{c_ivory}">{big_txt}</text>'
                + (f'<text x="{cx_card + 14}" y="{top_y + 78}" font-family="{_FONT_STACK}" font-size="22" font-weight="700" letter-spacing="0.8" fill="{c_gold}">{sub_txt}</text>' if sub_txt else "")
                + "</g>"
            )
            st_t = round(0.08 + idx * min(0.55, (duration_s * 0.45) / max(1, n_it)), 3)
            card_tweens.append(f'tl.fromTo("#bc{idx}", {{opacity: 0, y: -18}}, {{opacity: 1, y: 0, duration: 0.36, ease: "power3.out"}}, {st_t});')
            bar_tweens.append(f'tl.to("#bb{idx}", {{attr: {{width: {bar_w}}}, duration: 0.55, ease: "power2.out"}}, {st_t + 0.12:.3f});')

        # Subtle architectural connector hairline underneath ledger
        svg_body = (
            f'<g id="ledGrp">{hdr_svg}{"".join(cards_svg)}'
            f'<line id="ledLine" x1="{pad_x}" y1="{top_y + card_h + 14}" x2="{width - pad_x}" y2="{top_y + card_h + 14}" '
            f'stroke="{c_gold}" stroke-width="1.6" stroke-dasharray="{width}" stroke-dashoffset="{width}" opacity="0.72"/>'
            "</g>"
        )
        gsap_code = (
            (f'tl.to("#lHdr", {{opacity: 0.92, duration: 0.3}}, 0.04);' if hdr_svg else "")
            + "".join(card_tweens)
            + "".join(bar_tweens)
            + f'tl.to("#ledLine", {{strokeDashoffset: 0, duration: {duration_s * 0.42:.3f}, ease: "power2.out"}}, 0.15);'
            + f'tl.to("#ledGrp", {{opacity: 0, y: -10, duration: {duration_s * 0.20:.3f}, ease: "power2.in"}}, {duration_s * 0.80:.3f});'
        )

    elif template_name == "vfx-action-flow":
        # Purposeful allocation flow trajectories sweeping behind speaker matte + flowing bill/envelope chevron accents!
        y_mid = int(round(height * float(stage["cy"])))
        d1 = f"M {-40} {y_mid + 48} C {int(width * 0.26)} {y_mid - 155}, {int(width * 0.74)} {y_mid + 120}, {width + 40} {y_mid - 70}"
        d2 = f"M {-40} {y_mid + 92} C {int(width * 0.28)} {y_mid - 102}, {int(width * 0.72)} {y_mid + 162}, {width + 40} {y_mid - 22}"
        plen = int(round(width * 1.7))
        rw = max(36, int(round(width * 0.082)))
        svg_body = (
            f'<g id="flwG" filter="url(#edSh)">'
            f'<path id="fUnd" d="{d2}" fill="none" stroke="{c_esp}" stroke-width="{int(rw * 0.65)}" stroke-linecap="round" stroke-dasharray="{plen}" stroke-dashoffset="{plen}" opacity="0"/>'
            f'<path id="fMain" d="{d1}" fill="none" stroke="url(#ribGrad)" stroke-width="{rw}" stroke-linecap="round" stroke-dasharray="{plen}" stroke-dashoffset="{plen}" opacity="0"/>'
            f'<path id="fPulse" d="{d1}" fill="none" stroke="{c_ivory}" stroke-width="{sw * 1.05:.1f}" stroke-linecap="round" stroke-dasharray="22 30" opacity="0"/>'
            f'<circle id="fnL" cx="{int(width * 0.12)}" cy="{y_mid + 8}" r="14" fill="none" stroke="{c_ivory}" stroke-width="{sw:.1f}" opacity="0"/>'
            f'<circle id="fnR" cx="{int(width * 0.88)}" cy="{y_mid - 42}" r="18" fill="none" stroke="{c_gold}" stroke-width="{sw * 1.1:.1f}" opacity="0"/>'
            "</g>"
        )
        gsap_code = (
            f'tl.to("#fUnd", {{strokeDashoffset: 0, opacity: 0.68, duration: {duration_s * 0.36:.3f}, ease: "power2.out"}}, 0);'
            f'tl.to("#fMain", {{strokeDashoffset: 0, opacity: 0.90, duration: {duration_s * 0.40:.3f}, ease: "power2.inOut"}}, 0.04);'
            f'tl.fromTo("#fPulse", {{opacity: 0, strokeDashoffset: 160}}, {{opacity: 0.85, strokeDashoffset: -160, duration: {duration_s * 0.72:.3f}, ease: "none"}}, 0.08);'
            f'tl.fromTo("#fnL", {{opacity: 0, attr: {{r: 6}}}}, {{opacity: 0.9, attr: {{r: 20}}, duration: 0.35, ease: "power2.out"}}, 0.10);'
            f'tl.fromTo("#fnR", {{opacity: 0, attr: {{r: 8}}}}, {{opacity: 0.95, attr: {{r: 24}}, duration: 0.38, ease: "power2.out"}}, 0.25);'
            f'tl.to("#flwG", {{opacity: 0, duration: {duration_s * 0.22:.3f}, ease: "power2.in"}}, {duration_s * 0.78:.3f});'
        )

    elif template_name == "vfx-occlusion-wipe":
        # Tactile diagonal paper-fold depth sweep gliding strictly BEHIND MODNet subject matte!
        band_w = int(width * 0.68)
        svg_body = (
            f'<g id="wpG" opacity="0.92">'
            f'<g id="wpBand" transform="translate({-band_w - 80},0)">'
            f'<polygon points="0,0 {band_w},0 {band_w - 160},{height} {-160},{height}" fill="url(#wipeGrad)"/>'
            f'<line x1="{band_w - 14}" y1="0" x2="{band_w - 174}" y2="{height}" stroke="{c_ivory}" stroke-width="3.5" opacity="0.9"/>'
            f'<line x1="{band_w - 34}" y1="0" x2="{band_w - 194}" y2="{height}" stroke="{c_gold}" stroke-width="2" opacity="0.78"/>'
            "</g>"
            "</g>"
        )
        gsap_code = (
            f'tl.fromTo("#wpBand", {{attr: {{transform: "translate({-band_w - 80},0)"}}}}, '
            f'{{attr: {{transform: "translate({width + 240},0)"}}, duration: {duration_s * 0.76:.3f}, ease: "power2.inOut"}}, 0.04);'
            f'tl.to("#wpG", {{opacity: 0, duration: {duration_s * 0.20:.3f}, ease: "power2.in"}}, {duration_s * 0.80:.3f});'
        )

    elif template_name == "vfx-editorial-collage":
        # Asymmetric tactile paper feature cards tucked opposite the live macro tile in upper headroom (ZERO face cover!)
        items = sec_labels if sec_labels else ([disp_label] if disp_label else [])
        card_side_right = (slot.get("side") == "left")
        card_x = int(width * (0.48 if card_side_right else 0.05))
        card_w = int(width * 0.47)
        top_y = int(round(height * 0.042))
        cards = []
        tweens = []
        show_items = items[:2] if items else [disp_label or "ORGANIZED PLAN"]
        for i, txt in enumerate(show_items):
            cy_i = top_y + i * 78
            col = c_gold if i == 0 else c_sage
            cards.append(
                f'<g id="ec{i}" opacity="0" filter="url(#edSh)">'
                f'<rect x="{card_x}" y="{cy_i}" width="{card_w}" height="66" rx="16" fill="{c_ivory}" fill-opacity="0.95" stroke="{col}" stroke-width="2.4"/>'
                f'<rect x="{card_x + 12}" y="{cy_i + 14}" width="6" height="38" rx="3" fill="{col}"/>'
                f'<text x="{card_x + 28}" y="{cy_i + 41}" font-family="{_FONT_STACK}" font-size="19" font-weight="800" fill="{c_esp}">{txt.upper()[:22]}</text>'
                "</g>"
            )
            tweens.append(f'tl.fromTo("#ec{i}", {{opacity: 0, x: {24 if card_side_right else -24}}}, {{opacity: 1, x: 0, duration: 0.36, ease: "power3.out"}}, {0.10 + i * 0.24:.2f});')
        svg_body = f'<g id="colG">{"".join(cards)}</g>'
        gsap_code = "".join(tweens) + f'tl.to("#colG", {{opacity: 0, duration: {duration_s * 0.22:.3f}, ease: "power2.in"}}, {duration_s * 0.78:.3f});'

    elif template_name == "vfx-payoff-glint":
        # Restrained deterministic golden dust motes + converging editorial payoff badge landing on payoff_t!
        cx, cy_top = width // 2, int(round(height * 0.115))
        motes = []
        mote_tw = []
        # Deterministic golden bokeh dust around upper editorial frame (small tasteful count: 10)
        for k in range(10):
            ang = (k / 10.0) * math.pi * 2.0 + 0.3
            rad_x = int(width * (0.28 + 0.16 * ((k * 37) % 10) / 10.0))
            rad_y = int(height * (0.04 + 0.07 * ((k * 53) % 10) / 10.0))
            mx = cx + int(math.cos(ang) * rad_x)
            my = cy_top + int(math.sin(ang) * rad_y)
            r = 3 + (k % 3) * 1.8
            motes.append(f'<circle id="pm{k}" cx="{mx}" cy="{my}" r="{r:.1f}" fill="{c_ivory if k % 2 == 0 else c_gold}" opacity="0"/>')
            mote_tw.append(
                f'tl.fromTo("#pm{k}", {{opacity: 0, y: 12}}, {{opacity: 0.85, y: -14, duration: {duration_s * 0.55:.3f}, ease: "sine.out"}}, {0.08 + k * 0.04:.2f});'
            )
        pill_w = min(int(width * 0.72), max(220, len(disp_label) * 24 + 74)) if disp_label else 0
        pill_x = (width - pill_w) // 2
        badge = (
            f'<g id="payB" opacity="0" filter="url(#edSh)">'
            f'<rect x="{pill_x}" y="{cy_top - 32}" width="{pill_w}" height="68" rx="34" fill="{c_esp}" fill-opacity="0.92" stroke="{c_gold}" stroke-width="2.6"/>'
            f'<text x="{cx}" y="{cy_top + 11}" text-anchor="middle" font-family="{_FONT_STACK}" font-size="29" font-weight="800" letter-spacing="1.8" fill="{c_ivory}">{disp_label.upper()}</text>'
            "</g>"
        ) if disp_label else ""
        # Crisp warm golden glint star
        star = (
            f'<path id="glintS" d="M {cx} {cy_top - 58} L {cx + 5} {cy_top - 43} L {cx + 20} {cy_top - 38} L {cx + 5} {cy_top - 33} '
            f'L {cx} {cy_top - 18} L {cx - 5} {cy_top - 33} L {cx - 20} {cy_top - 38} L {cx - 5} {cy_top - 43} Z" fill="{c_ivory}" opacity="0"/>'
        )
        svg_body = f'<g id="payG">{"".join(motes)}{star}{badge}</g>'
        gsap_code = (
            "".join(mote_tw)
            + f'tl.fromTo("#glintS", {{opacity: 0, rotation: -30, transformOrigin: "{cx}px {cy_top - 38}px"}}, {{opacity: 0.95, rotation: 45, duration: 0.45, ease: "power2.out"}}, {max(0.05, payoff_t - 0.15):.2f});'
            + (f'tl.fromTo("#payB", {{opacity: 0, y: 14}}, {{opacity: 1, y: 0, duration: 0.35, ease: "power3.out"}}, {max(0.05, payoff_t - 0.20):.2f});' if badge else "")
            + f'tl.to("#payG", {{opacity: 0, duration: {duration_s * 0.22:.3f}, ease: "power2.in"}}, {duration_s * 0.78:.3f});'
        )

    elif template_name == "vfx-macro-pointer":
        # Glowing curved editorial seam + crisp macro focus label cleanly aligned with split wave!
        split_y = int(round(height * float(parameters.get("split_ratio", 0.58))))
        dip = int(round(float(parameters.get("curve_depth_px", 44))))
        d_wave = f"M 0 {split_y + dip} Q {int(width * 0.36)} {split_y - dip} {int(width * 0.72)} {split_y + int(dip * 0.4)} T {width} {split_y - int(dip * 0.35)}"
        plen = int(round(width * 1.3))
        pill_w = min(int(width * 0.62), max(190, len(disp_label) * 20 + 60)) if disp_label else 0
        pill_x = (width - pill_w) // 2
        badge = (
            f'<g id="mB" opacity="0" filter="url(#edSh)">'
            f'<rect x="{pill_x}" y="{int(height * 0.055)}" width="{pill_w}" height="52" rx="26" fill="{c_esp}" fill-opacity="0.88" stroke="{c_gold}" stroke-width="2"/>'
            f'<text x="{width // 2}" y="{int(height * 0.055) + 33}" text-anchor="middle" font-family="{_FONT_STACK}" font-size="21" font-weight="700" letter-spacing="1.4" fill="{c_ivory}">{disp_label.upper()}</text>'
            "</g>"
        ) if disp_label else ""
        svg_body = (
            f'<g id="macG" filter="url(#edSh)">'
            f'<path id="mW1" d="{d_wave}" fill="none" stroke="{c_gold}" stroke-width="{sw * 2.1:.1f}" stroke-linecap="round" stroke-dasharray="{plen}" stroke-dashoffset="{plen}" opacity="0"/>'
            f'<path id="mW2" d="{d_wave}" fill="none" stroke="{c_ivory}" stroke-width="{sw * 0.85:.1f}" stroke-linecap="round" stroke-dasharray="{plen}" stroke-dashoffset="{plen}" opacity="0"/>'
            f"{badge}</g>"
        )
        gsap_code = (
            f'tl.to(["#mW1","#mW2"], {{strokeDashoffset: 0, opacity: 0.94, duration: {duration_s * 0.30:.3f}, ease: "power3.out"}}, 0.04);'
            + (f'tl.fromTo("#mB", {{opacity: 0, y: -12}}, {{opacity: 1, y: 0, duration: 0.32, ease: "power3.out"}}, 0.12);' if badge else "")
            + f'tl.to("#macG", {{opacity: 0, duration: {duration_s * 0.22:.3f}, ease: "power2.in"}}, {duration_s * 0.78:.3f});'
        )

    elif template_name == "vfx-spatial-planes":
        # Architectural 2.5D depth horizon brackets & floating parallax depth circles strictly BEHIND speaker!
        cx = int(round(width * float(stage["cx"])))
        cy = int(round(height * float(stage["cy"])))
        r1, r2 = int(width * 0.44), int(width * 0.34)
        c1, c2 = int(2 * math.pi * r1), int(2 * math.pi * r2)
        svg_body = (
            f'<g id="spG" filter="url(#edSh)" opacity="0">'
            f'<circle id="spR1" cx="{cx}" cy="{cy - 30}" r="{r1}" fill="none" stroke="{c_gold}" stroke-width="{sw * 1.3:.1f}" stroke-dasharray="{int(c1 * 0.28)} {int(c1 * 0.22)}" stroke-linecap="round"/>'
            f'<circle id="spR2" cx="{cx}" cy="{cy - 30}" r="{r2}" fill="none" stroke="{c_ivory}" stroke-width="{sw * 0.75:.1f}" stroke-dasharray="{int(c2 * 0.18)} {int(c2 * 0.32)}" stroke-linecap="round" opacity="0.85"/>'
            f'<line x1="{int(width * 0.06)}" y1="{int(height * 0.22)}" x2="{int(width * 0.22)}" y2="{int(height * 0.22)}" stroke="{c_ivory}" stroke-width="2.5" opacity="0.72"/>'
            f'<line x1="{int(width * 0.78)}" y1="{int(height * 0.22)}" x2="{int(width * 0.94)}" y2="{int(height * 0.22)}" stroke="{c_gold}" stroke-width="2.5" opacity="0.72"/>'
            "</g>"
        )
        gsap_code = (
            f'tl.fromTo("#spG", {{opacity: 0, scale: 0.92, transformOrigin: "{cx}px {cy - 30}px"}}, {{opacity: 0.92, scale: 1.0, duration: {duration_s * 0.28:.3f}, ease: "power3.out"}}, 0);'
            f'tl.to("#spR1", {{rotation: 42, transformOrigin: "{cx}px {cy - 30}px", duration: {duration_s:.3f}, ease: "sine.inOut"}}, 0);'
            f'tl.to("#spR2", {{rotation: -32, transformOrigin: "{cx}px {cy - 30}px", duration: {duration_s:.3f}, ease: "sine.inOut"}}, 0);'
            f'tl.to("#spG", {{opacity: 0, duration: {duration_s * 0.22:.3f}, ease: "power2.in"}}, {duration_s * 0.78:.3f});'
        )

    else:  # vfx-shape-morph, vfx-closing-resolve, vfx-trail-accent
        # Architectural rounded editorial trim + optional refined grounded takeaway pill in upper headroom!
        ix, iy = int(width * 0.055), int(height * 0.055)
        rw, rh = width - 2 * ix, height - 2 * iy
        perim = 2 * (rw + rh)
        pill_w = min(int(width * 0.74), max(210, len(disp_label) * 21 + 66)) if disp_label else 0
        pill_x = (width - pill_w) // 2
        badge = (
            f'<g id="cB" opacity="0" filter="url(#edSh)">'
            f'<rect x="{pill_x}" y="{int(height * 0.082)}" width="{pill_w}" height="56" rx="28" fill="{c_esp}" fill-opacity="0.88" stroke="{c_gold}" stroke-width="2.2"/>'
            f'<text x="{width // 2}" y="{int(height * 0.082) + 35}" text-anchor="middle" font-family="{_SERIF_STACK}" font-size="22" font-weight="700" letter-spacing="1.2" fill="{c_ivory}">{disp_label.upper()}</text>'
            "</g>"
        ) if disp_label else ""
        svg_body = (
            f'<g id="mrpG" filter="url(#edSh)" opacity="0">'
            f'<rect id="mRect" x="{ix}" y="{iy}" width="{rw}" height="{rh}" rx="38" ry="38" fill="none" stroke="{c_gold}" '
            f'stroke-width="{sw * 1.15:.1f}" stroke-dasharray="{int(perim * 0.26)} {int(perim * 0.24)}" stroke-linecap="round"/>'
            "</g>"
            f"{badge}"
        )
        gsap_code = (
            f'tl.fromTo("#mrpG", {{opacity: 0, scale: 1.05, transformOrigin: "50% 50%"}}, {{opacity: 0.90, scale: 1.0, duration: {duration_s * 0.26:.3f}, ease: "power3.out"}}, 0);'
            f'tl.to("#mRect", {{strokeDashoffset: -190, duration: {duration_s:.3f}, ease: "sine.inOut"}}, 0);'
            + (f'tl.fromTo("#cB", {{opacity: 0, y: -12}}, {{opacity: 1, y: 0, duration: 0.34, ease: "power3.out"}}, 0.14);' if badge else "")
            + f'tl.to(["#mrpG"{", \"#cB\"" if badge else ""}], {{opacity: 0, duration: {duration_s * 0.24:.3f}, ease: "power2.in"}}, {duration_s * 0.76:.3f});'
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width={width}, initial-scale=1" />
<title>MotionLab R4 HyperFrames {scene_id} ({template_name})</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  html, body {{ width: {width}px; height: {height}px; overflow: hidden; background: transparent; }}
  #root {{ width: {width}px; height: {height}px; position: relative; background: transparent; }}
</style>
<script src="{gsap_rel_path}"></script>
</head>
<body>
<div id="root" data-composition-id="hf_{scene_id}" data-width="{width}" data-height="{height}" data-fps="{int(round(fps))}" data-start="0" data-duration="{duration_s:.4f}">
  <svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">
    {defs}
    {svg_body}
  </svg>
</div>
<script>
  const tl = gsap.timeline({{ paused: true }});
  {gsap_code}
  window.__timelines = window.__timelines || {{}};
  window.__timelines["hf_{scene_id}"] = tl;
</script>
</body>
</html>
"""
