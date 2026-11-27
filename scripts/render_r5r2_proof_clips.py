"""Phase R2-A3 / R2-A7 (ML-01.R5.R2): Separate ART_DIRECTED_QA Visual Lab Proofs (H1–H5).

Strictly isolated from canonical `AUTO_GENERATIVE`!
Renders 5 focused 3-5s 9:16 proof clips sharing the exact generic Remotion + HyperFrames primitives:
  - H1_hook_integer_money  (f0..60): integer-dollar lock ($1,200, ZERO $1,151.93!) + curved paper tokens
  - H2_bills_single_owner  (f138..207): single canonical ledger owner (ZERO duplicate bottom ribbon!) + live detail + Bills short-cue SFX
  - H3_payoff_paper_fold   (f344..417): luminous line-led sweep (ZERO dark/ivory iris!) + integer $100 payoff + PaperFoldExitBridge to live kitchen!
  - H4_product_fold_bridge (f533..635): bridged 3D FlipPageTransition + open editorial specs + PaperFoldExitBridge to live!
  - H5_closing_face_safe   (f635..720): bold serif takeaway + subtle edge-only rim glow (face washout protected!)

Generates phone-scale side-by-side comparison sheet:
  `runs/r5r2_visual_lab/h1_h5_r5r1_vs_r5r2_phone_compare.jpg`
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw

from motionlab.composition.generative_compiler import compile_scene_hyperframes_html, compile_scene_remotion_props
from motionlab.composition.generative_schema import (
    GENERATIVE_SCHEMA_VERSION,
    LayerSpec,
    SceneSpec,
    StoryboardSpec,
    TransitionSpec,
)
from motionlab.config import PROJECT_ROOT
from motionlab.preflight import probe_media
from motionlab.renderers.base import verify_webm_alpha_pixels
from motionlab.renderers.hyperframes import HyperFramesRenderer
from motionlab.renderers.remotion import RemotionRenderer
from motionlab.schemas import atomic_write_json
from motionlab.semantic.asr import transcribe_local
from motionlab.semantic.story_graph import build_story_graph


def _extract_frame(mp4: Path, frame_idx: int) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(mp4))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_idx))
    ok, fr = cap.read()
    cap.release()
    return cv2.cvtColor(fr, cv2.COLOR_BGR2RGB) if (ok and fr is not None) else None


def main() -> int:
    t0 = time.perf_counter()
    src_mp4 = PROJECT_ROOT / "input" / "Olivia_Paycheck_V5_Kitchen_30s_Combined (1).mp4"
    r1_mp4 = PROJECT_ROOT / "runs" / "olivia_ml01_r5r1" / "20261127T073427Z_c56c9ae0_001" / "Olivia_Paycheck_V5_Kitchen_30s_Combined_ML01R5R1_FINAL_9x16.mp4"
    out_dir = PROJECT_ROOT / "runs" / "r5r2_visual_lab"
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = out_dir / "_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    media = probe_media(src_mp4)
    ffmpeg_bin = shutil.which("ffmpeg") or "ffmpeg"
    from motionlab.config import EditConfig
    cfg = EditConfig(offline=True, composition_mode="generative", allow_remotion=True, allow_hyperframes=True, seed=42)
    tr = transcribe_local(media, cfg, PROJECT_ROOT / ".cache" / "motionlab_r5" / "semantic", ffmpeg_bin)
    sg = build_story_graph(tr, None, media)

    pal = {"ivory": "#F5F0E6", "espresso": "#40382E", "forest": "#243B32", "sage": "#84977A", "brass": "#BE9A59", "clay": "#AF745E"}
    steps = sg.ledger.steps
    s_open = steps[0].step_id
    s_bills = steps[1].step_id
    s_payoff = steps[-1].step_id

    proof_scenes = [
        SceneSpec(
            scene_id="H1_hook_integer_money",
            start_frame=0,
            end_frame_exclusive=60,
            role="hook_contrast",
            stage_mode="live_clean_bridge",
            viewer_takeaway="$1,200 integer bank balance hook",
            persistent_ledger_step_id=s_open,
            layers=[
                LayerSpec("h1_hf", "hf_apple_money_count", "hyperframes", "foreground_graphics", 20, {
                    "from_cents": 120000, "to_cents": 120000, "currency_symbol": "$",
                    "eyebrow": "BANK BALANCE", "caption": "", "y_anchor": "8.6%", "dark_plate": True,
                    "token_count": 4, "hero_px": 128, "allow_fractional_cents": False, "hold_exact_amount": True,
                }),
            ],
        ),
        SceneSpec(
            scene_id="H2_bills_single_owner",
            start_frame=138,
            end_frame_exclusive=207,
            role="allocation_step",
            stage_mode="tracked_macro_split",
            viewer_takeaway="-$800 Bills single canonical ledger",
            persistent_ledger_step_id=s_bills,
            show_persistent_ribbon=False,
            layers=[
                LayerSpec("h2_rem", "rem_ru_track_matte", "remotion", "foreground_graphics", 20, {"x_frac": 0.43, "y_frac": 0.46, "detail_zoom": 1.60, "label": "LIVE DETAIL"}),
                LayerSpec("h2_hf", "hf_mk_progress_stat", "hyperframes", "foreground_graphics", 30, {
                    "amount_cents": 80000, "denom_cents": 120000, "remaining_cents": 40000, "currency_symbol": "$",
                    "is_deduction": True, "category": "BILLS", "caption": "$400 left to spend", "y_px": 110,
                    "layout_variant": "top_strip", "allow_fractional_cents": False, "token_count": 3,
                }),
            ],
        ),
        SceneSpec(
            scene_id="H3_payoff_paper_fold",
            start_frame=344,
            end_frame_exclusive=417,
            role="payoff_reveal",
            stage_mode="full_graphic_cutaway",
            viewer_takeaway="$100 left payoff + smooth PaperFold exit",
            persistent_ledger_step_id=s_payoff,
            transition_in=TransitionSpec(kind="line_led", duration_frames=12),
            parameters={"bg_color": pal["ivory"], "transition_out": "paper_fold_out", "transition_out_frames": 12},
            layers=[
                LayerSpec("h3_rem", "rem_ru_waterfall", "remotion", "background_graphics", 10, {
                    "title": "EXAMPLE ALLOCATION", "subtitle": "$1,200 → $100", "currency_symbol": "$",
                    "y_pct": 38, "chart_height": 440, "dark_surface": False,
                }),
                LayerSpec("h3_hf", "hf_apple_money_count", "hyperframes", "foreground_graphics", 30, {
                    "from_cents": 120000, "to_cents": 10000, "currency_symbol": "$",
                    "eyebrow": "ACTUALLY LEFT TO SPEND", "caption": "from $1,200 example balance",
                    "y_anchor": "8.6%", "dark_plate": False, "token_count": 3, "hero_px": 150,
                    "accent_color": pal["brass"], "allow_fractional_cents": False, "hold_exact_amount": False,
                }),
            ],
        ),
        SceneSpec(
            scene_id="H4_product_fold_bridge",
            start_frame=533,
            end_frame_exclusive=635,
            role="product_benefit",
            stage_mode="full_graphic_cutaway",
            viewer_takeaway="Conceptual product flip + smooth PaperFold exit",
            transition_in=TransitionSpec(kind="line_led", duration_frames=12),
            parameters={"bg_color": pal["ivory"], "transition_out": "paper_fold_out", "transition_out_frames": 12},
            layers=[
                LayerSpec("h4_rem", "rem_rc_flip_page", "remotion", "background_graphics", 10, {
                    "label_before": "The Next Paycheck Plan", "sub_before": "CONCEPTUAL PRODUCT OVERVIEW",
                    "label_after": "56-page workbook", "sub_after": "+ Notion dashboard", "y_frac": 0.075,
                }),
                LayerSpec("h4_hf", "hf_mk_specs_list", "hyperframes", "foreground_graphics", 30, {
                    "title": "THE NEXT PAYCHECK PLAN",
                    "rows": [{"label": "Notion dashboard", "value": "Conceptual overview"}, {"label": "56-page workbook", "value": "Conceptual overview"}],
                    "y_px": 728,
                }),
            ],
        ),
        SceneSpec(
            scene_id="H5_closing_face_safe",
            start_frame=635,
            end_frame_exclusive=720,
            role="closing_takeaway",
            stage_mode="live_clean_bridge",
            viewer_takeaway="Give every dollar a purpose + edge-only rim glow",
            layers=[
                LayerSpec("h5_rem", "rem_ru_masked_slide", "remotion", "foreground_graphics", 20, {
                    "eyebrow": "BEFORE YOUR NEXT PAYDAY",
                    "lines": ["Give every dollar", "a purpose"],
                    "subline": "Link in my profile.",
                    "y_pct": 6.8, "font_size": 50, "serif": True,
                }),
                LayerSpec("h5_hf", "hf_organic_light_leak", "hyperframes", "foreground_graphics", 30, {"opacity": 0.18}),
            ],
        ),
    ]

    sb = StoryboardSpec(
        schema_version=GENERATIVE_SCHEMA_VERSION,
        mode="ART_DIRECTED_QA",
        source_hash=media.input_hash,
        transcript_hash=tr.transcript_hash,
        story_graph_hash=sg.story_graph_hash,
        niche="finance",
        niche_confidence=0.99,
        core_value_proposition="ART_DIRECTED_QA H1-H5 Visual Lab Proofs",
        width=media.display_width,
        height=media.display_height,
        fps_num=media.fps.num,
        fps_den=media.fps.den,
        total_frames=media.duration_frames,
        seed=42,
        palette=pal,
        scenes=proof_scenes,
    )

    hf = HyperFramesRenderer(enabled=True)
    rem = RemotionRenderer(license_allowed=True)
    hf_cache = PROJECT_ROOT / ".cache" / "motionlab_r5" / "hyperframes_overlays"
    fps_str = media.fps.to_ffmpeg()

    public_mounts: list[dict[str, str]] = []
    batch_jobs: list[dict[str, Any]] = []
    for sc in proof_scenes:
        hf_pid, hf_html, hf_role = compile_scene_hyperframes_html(sc, sb, sg)
        hf_rel = None
        if hf_pid and hf_html:
            hf_out = tmp_dir / f"hf_{sc.scene_id}.webm"
            hf.render_html_slot(tmp_dir / f"hf_slot_{sc.scene_id}", hf_html, hf_out, media.fps_float, fmt="webm", cache_dir=hf_cache, use_cache=True, skip_cli_lint=True)
            verify_webm_alpha_pixels(hf_out, media.display_width, media.display_height, sample_frame=12, ffmpeg_bin=ffmpeg_bin)
            hf_rel = f"{sc.scene_id}/hf.webm"
            public_mounts.append({"src": str(hf_out), "rel": hf_rel})

        src_slice = tmp_dir / f"src_{sc.scene_id}.mp4"
        rem._slice_source_mezzanine(src_mp4, sc.start_frame, sc.end_frame_exclusive, fps_str, src_slice)
        src_rel = f"{sc.scene_id}/src.mp4"
        public_mounts.append({"src": str(src_slice), "rel": src_rel})

        props = compile_scene_remotion_props(
            sc, sb, sg,
            source_file=src_rel, fg_alpha_file=None, prop_alpha_file=None,
            hf_graphic_file=hf_rel, hf_layer_role=hf_role,
        )
        batch_jobs.append({
            "kind": "media",
            "id": sc.scene_id,
            "compositionId": "MotionLabGenerativeScene",
            "props": props,
            "outputLocation": str(out_dir / f"{sc.scene_id}.mp4"),
            "codec": "h264",
            "pixelFormat": "yuv420p",
            "crf": 16,
            "concurrency": 1,
        })

    batch_res = rem.render_batch_jobs(
        jobs=batch_jobs,
        public_mounts=public_mounts,
        work_dir=tmp_dir / "rem_batch",
        bundle_cache_root=PROJECT_ROOT / ".cache" / "motionlab_r5" / "remotion_bundles",
        timeout_s=360.0,
    )

    # Build side-by-side phone-scale comparison sheet (R5.R1 vs R5.R2 on exact defect frames!)
    compare_rows = [
        ("H1_hook_integer_money", 12, 12, "f12 Hook (R1 $1,151.93 vs R2 $1,200 exact)"),
        ("H2_bills_single_owner", 28, 166, "f166 Bills (R1 dup counters vs R2 single owner)"),
        ("H3_payoff_paper_fold", 12, 356, "f356 Payoff (R1 $144.06 vs R2 integer $100)"),
        ("H3_payoff_paper_fold", 67, 411, "f411 Payoff Exit (R1 hard cut vs R2 PaperFold)"),
        ("H4_product_fold_bridge", 96, 629, "f629 Product Exit (R1 hard cut vs R2 PaperFold)"),
        ("H5_closing_face_safe", 45, 680, "f680 Closing (R1 glare vs R2 face-safe rim)"),
    ]
    tw, th = 220, 390
    pad = 10
    hdr = 46
    sheet = Image.new("RGB", (len(compare_rows) * (tw + pad) + pad, hdr + 2 * (th + 26 + pad) + pad), (20, 19, 17))
    draw = ImageDraw.Draw(sheet)
    draw.text((10, 14), "ART_DIRECTED_QA Visual Lab — Top: R5.R1 AUTO Baseline | Bottom: R5.R2 Proof Clips (Phone Scale)", fill=(245, 240, 230))
    for ci, (sc_id, loc_f, glob_f, lbl) in enumerate(compare_rows):
        x = pad + ci * (tw + pad)
        im_r1 = _extract_frame(r1_mp4, glob_f) if r1_mp4.is_file() else None
        im_r2 = _extract_frame(out_dir / f"{sc_id}.mp4", loc_f)
        if im_r1 is not None:
            sheet.paste(Image.fromarray(im_r1).resize((tw, th), Image.Resampling.LANCZOS), (x, hdr))
            draw.text((x + 2, hdr + th + 4), f"R5.R1 f{glob_f}", fill=(200, 140, 120))
        if im_r2 is not None:
            y2 = hdr + th + 26 + pad
            sheet.paste(Image.fromarray(im_r2).resize((tw, th), Image.Resampling.LANCZOS), (x, y2))
            draw.text((x + 2, y2 + th + 4), lbl[:32], fill=(170, 215, 170))
    cmp_path = out_dir / "h1_h5_r5r1_vs_r5r2_phone_compare.jpg"
    sheet.save(cmp_path, "JPEG", quality=92)
    shutil.rmtree(tmp_dir, ignore_errors=True)

    summary = {
        "mode": "ART_DIRECTED_QA_ISOLATED_PROOFS",
        "elapsed_s": round(time.perf_counter() - t0, 2),
        "remotion_batch": batch_res,
        "compare_sheet": str(cmp_path),
        "clips": [f"{sc.scene_id}.mp4" for sc in proof_scenes],
    }
    atomic_write_json(out_dir / "proof_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
