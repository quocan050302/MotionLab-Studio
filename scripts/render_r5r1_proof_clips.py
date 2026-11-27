"""Phase R1-A3 (ML-01.R5.R1): Visual Lab H1-H4 Critical Proof Clips & R5 vs R5.R1 Side-by-Side Comparison.

STRICT SEPARATION:
  Art-directed QA proof script ONLY (`runs/r5r1_visual_lab/`). NEVER imported by automatic CLI mode.
  Renders 4 critical 3-5s proofs:
    - H1: Hook Money (f0..60) — Open editorial $1,200 + tactile paper notes + MODNet depth matte
    - H2: Continuous Allocation (f138..269) — Bills -$800 (no '(derived)') + balanced Groceries AspectMorph
    - H3: Payoff + No-Dark Morph (f344..417) — Luminous ivory morph + open $100 deep-forest hero + waterfall
    - H4: Product + Protected CTA (f533..720) — 3D workbook + warm ivory specs + eye-safe open closing CTA
  Then builds a 360px phone-scale R5 AUTO vs R5.R1 side-by-side sheet for visual inspection.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from motionlab.composition.art_direction import R5R1_EDITORIAL_PALETTE
from motionlab.composition.generative_compiler import compile_scene_hyperframes_html, compile_scene_remotion_props
from motionlab.composition.generative_schema import (
    GENERATIVE_SCHEMA_VERSION,
    LayerSpec,
    SceneSpec,
    StoryboardSpec,
    TransitionSpec,
)
from motionlab.config import EditConfig
from motionlab.preflight import probe_media
from motionlab.renderers.base import verify_webm_alpha_pixels
from motionlab.renderers.hyperframes import HyperFramesRenderer
from motionlab.renderers.remotion import RemotionRenderer
from motionlab.resource import ResourceBudget
from motionlab.schemas import atomic_write_json
from motionlab.semantic.asr import transcribe_local
from motionlab.semantic.story_graph import build_story_graph
from motionlab.vision.segmentation import build_foreground_alpha_webm, segment_subject_sequence
from motionlab.vision.tracking import track_action_rois

OLIVIA_INPUT = REPO_ROOT / "input" / "Olivia_Paycheck_V5_Kitchen_30s_Combined (1).mp4"
R5_AUTO_FINAL = REPO_ROOT / "runs" / "olivia_ml01_r5" / "20261127T052515Z_c56c9ae0_001" / "Olivia_Paycheck_V5_Kitchen_30s_Combined_ML01R5_FINAL_9x16.mp4"
OUT_DIR = REPO_ROOT / "runs" / "r5r1_visual_lab"


def _extract_frame_rgb(mp4_path: Path, frame_idx: int) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(mp4_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_idx))
    ok, bgr = cap.read()
    cap.release()
    if not ok or bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def main() -> None:
    t0 = time.perf_counter()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    temp_dir = OUT_DIR / "temp"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    ffmpeg_bin = shutil.which("ffmpeg") or "ffmpeg"
    media = probe_media(OLIVIA_INPUT)
    cfg = EditConfig(
        offline=True,
        composition_mode="generative",
        allow_remotion=True,
        allow_hyperframes=True,
        seed=42,
    )
    budget = ResourceBudget(config=cfg.resources, dest_dir=OUT_DIR, temp_dir=temp_dir)

    sem_cache = REPO_ROOT / ".cache" / "motionlab_r5" / "semantic"
    transcript = transcribe_local(media, cfg, sem_cache, ffmpeg_bin)
    from motionlab.analyzer import analyze_media_async
    import asyncio
    analysis, _ = asyncio.run(analyze_media_async(media, cfg, temp_dir, budget=budget))
    roi_tracks = track_action_rois(media, analysis, cfg, ffmpeg_bin)
    story_graph = build_story_graph(transcript, roi_tracks, media)

    pal = dict(R5R1_EDITORIAL_PALETTE)
    steps = story_graph.ledger.steps
    st_open = steps[0].step_id if steps else None
    st_bills = next((s.step_id for s in steps if s.category.lower() == "bills"), None)
    st_groc = next((s.step_id for s in steps if s.category.lower() == "groceries"), None)
    st_rem = next((s.step_id for s in steps if s.step_type == "remaining_confirmed"), None)

    proof_scenes = [
        # H1: Hook Money (open editorial scrim, 124px tabular $1,200, tactile paper notes, MODNet depth matte)
        SceneSpec(
            scene_id="H1_hook_money",
            start_frame=0, end_frame_exclusive=60,
            role="hook_contrast", stage_mode="live_depth_matte",
            viewer_takeaway="Open editorial $1,200 bank balance + tactile paper notes.",
            layers=[
                LayerSpec(
                    layer_id="h1_hf", primitive_id="hf_apple_money_count", engine="hyperframes",
                    layer_role="background_graphics", z_order=15,
                    params={
                        "from_cents": 0, "to_cents": 120000, "currency_symbol": "$",
                        "eyebrow": "BANK BALANCE", "caption": "",
                        "y_anchor": "8.6%", "dark_plate": True, "token_count": 4, "hero_px": 124,
                    },
                ),
            ],
            persistent_ledger_step_id=st_open,
            requires_matte=True,
            parameters={"bg_dim": 0.80, "bg_blur_px": 1.8, "parallax_travel_px": 22},
        ),
        # H2a: Bills (open ivory editorial strip, no '(derived)', tracked macro detail)
        SceneSpec(
            scene_id="H2_alloc_bills",
            start_frame=138, end_frame_exclusive=207,
            role="allocation_step", stage_mode="tracked_macro_split",
            viewer_takeaway="-$800 Bills with clean '$400 left' and real macro cash hands.",
            layers=[
                LayerSpec(
                    layer_id="h2a_hf", primitive_id="hf_mk_progress_stat", engine="hyperframes",
                    layer_role="foreground_graphics", z_order=30,
                    params={
                        "amount_cents": 80000, "denom_cents": 120000, "remaining_cents": 40000,
                        "currency_symbol": "$", "is_deduction": True,
                        "category": "BILLS", "caption": "$400 left", "y_px": 110, "dark_plate": False,
                    },
                ),
                LayerSpec(
                    layer_id="h2a_rem", primitive_id="rem_ru_track_matte", engine="remotion",
                    layer_role="foreground_graphics", z_order=35,
                    params={"x_frac": 0.44, "y_frac": 0.51, "detail_zoom": 1.60, "label": "LIVE DETAIL"},
                ),
            ],
            persistent_ledger_step_id=st_bills,
            show_persistent_ribbon=True,
        ),
        # H2b: Groceries AspectMorph (balanced architectural split, zero empty gap!)
        SceneSpec(
            scene_id="H2_alloc_groceries",
            start_frame=234, end_frame_exclusive=269,
            role="allocation_step", stage_mode="aspect_morph_editorial",
            viewer_takeaway="Balanced editorial breakdown $800 / $200 / $200 remaining.",
            layers=[
                LayerSpec(
                    layer_id="h2b_hf", primitive_id="hf_data_chart", engine="hyperframes",
                    layer_role="foreground_graphics", z_order=30,
                    params={
                        "title": "PROPORTIONAL BREAKDOWN", "denom_cents": 120000, "currency_symbol": "$", "y_px": 112,
                        "items": [
                            {"label": "Bills", "cents": 80000, "color": pal["clay"]},
                            {"label": "Groceries", "cents": 20000, "color": pal["brass"]},
                            {"label": "Remaining", "cents": 20000, "color": pal["forest"]},
                        ],
                    },
                ),
            ],
            persistent_ledger_step_id=st_groc,
            parameters={"kicker": "", "lines": [], "header_title": "", "shifted_y_frac": 0.48, "shifted_h_frac": 0.43},
        ),
        # H3: Payoff + Luminous Ivory No-Dark Morph
        SceneSpec(
            scene_id="H3_payoff_nodark",
            start_frame=344, end_frame_exclusive=417,
            role="payoff_reveal", stage_mode="full_graphic_cutaway",
            viewer_takeaway="Luminous ivory morph -> gigantic $100 deep-forest hero + clean waterfall.",
            layers=[
                LayerSpec(
                    layer_id="h3_rem", primitive_id="rem_ru_waterfall", engine="remotion",
                    layer_role="background_graphics", z_order=10,
                    params={
                        "title": "EXAMPLE ALLOCATION", "subtitle": "$1,200 → $100",
                        "y_pct": 38, "chart_height": 440, "dark_surface": False,
                        "steps": [
                            {"label": "Balance", "value": 1200, "isTotal": True, "color": pal["espresso"]},
                            {"label": "Bills", "value": -800, "color": pal["clay"]},
                            {"label": "Groceries", "value": -200, "color": pal["brass"]},
                            {"label": "Gas", "value": -100, "color": pal["sage"]},
                            {"label": "Left", "value": 100, "isTotal": True, "color": pal["forest"]},
                        ],
                    },
                ),
                LayerSpec(
                    layer_id="h3_hf", primitive_id="hf_apple_money_count", engine="hyperframes",
                    layer_role="foreground_graphics", z_order=30,
                    params={
                        "from_cents": 120000, "to_cents": 10000, "currency_symbol": "$",
                        "eyebrow": "REMAINING TO SPEND", "caption": "after these expenses ($1,200 example)",
                        "y_anchor": "8.8%", "dark_plate": False, "token_count": 3, "hero_px": 148,
                    },
                ),
            ],
            transition_in=TransitionSpec(kind="frame_morph", duration_frames=12, rationale="luminous_ivory_morph"),
            persistent_ledger_step_id=st_rem,
            intentional_cutaway=True,
            parameters={"bg_color": pal["ivory"]},
        ),
        # H4: Closing Protected Editorial CTA (zero dark box, eye-safe upper scrim)
        SceneSpec(
            scene_id="H4_closing_cta",
            start_frame=635, end_frame_exclusive=720,
            role="closing_takeaway", stage_mode="live_clean_bridge",
            viewer_takeaway="Open editorial closing takeaway clear of eyes/mouth.",
            layers=[
                LayerSpec(
                    layer_id="h4_rem", primitive_id="rem_ru_masked_slide", engine="remotion",
                    layer_role="foreground_graphics", z_order=30,
                    params={
                        "eyebrow": "BEFORE YOUR NEXT PAYDAY",
                        "lines": ["Give every dollar", "a purpose"],
                        "subline": "Link in my profile",
                        "y_pct": 8.6, "font_size": 50, "dark_card": False, "serif": True,
                    },
                ),
                LayerSpec(
                    layer_id="h4_hf", primitive_id="hf_organic_light_leak", engine="hyperframes",
                    layer_role="foreground_graphics", z_order=35,
                    params={"opacity": 0.42},
                ),
            ],
        ),
    ]

    sb = StoryboardSpec(
        schema_version=GENERATIVE_SCHEMA_VERSION,
        mode="ART_DIRECTED_QA_PROOF",
        source_hash=media.input_hash,
        transcript_hash=story_graph.transcript_hash,
        story_graph_hash=story_graph.story_graph_hash,
        niche="finance",
        niche_confidence=1.0,
        core_value_proposition="H1-H4 Visual Lab Proof",
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
    cache_base = REPO_ROOT / ".cache" / "motionlab_r5"
    matte_res = segment_subject_sequence(
        media=media, spans=[(0, 60)], workdir=temp_dir, cache_dir=cache_base / "vision",
        preserve_tabletop_y=roi_tracks.get("tabletop_guard_y"), keyframe_stride=2, budget=budget, ffmpeg_bin=ffmpeg_bin,
    )
    matte_p = Path(matte_res["spans"][0]["matte_mp4"])

    # Render HyperFrames blocks
    hf_out: dict[str, tuple[Path, str]] = {}
    for sc in proof_scenes:
        pid, html, role = compile_scene_hyperframes_html(sc, sb, story_graph)
        if pid and html:
            out_w = temp_dir / f"{sc.scene_id}_{pid}.webm"
            hf.render_html_slot(
                slot_dir=temp_dir / f"hf_{sc.scene_id}", html=html, out_path=out_w, fps=media.fps_float,
                fmt="webm", cache_dir=cache_base / "hyperframes_overlays", use_cache=False,
            )
            a_st = verify_webm_alpha_pixels(out_w, width=media.display_width, height=media.display_height, sample_frame=14, ffmpeg_bin=ffmpeg_bin)
            assert a_st.get("has_visible_alpha"), f"Empty alpha in {sc.scene_id}"
            hf_out[sc.scene_id] = (out_w, role)

    # Render all 5 proof clips in ONE single-browser Remotion batch call!
    fps_str = media.fps.to_ffmpeg()
    mounts = []
    jobs = []
    for sc in proof_scenes:
        sdir = temp_dir / sc.scene_id
        sdir.mkdir(parents=True, exist_ok=True)
        src_rel = None
        if sc.stage_mode != "full_graphic_cutaway" or sc.transition_in.kind != "none":
            ss = sdir / "src.mp4"
            rem._slice_source_mezzanine(OLIVIA_INPUT, sc.start_frame, sc.end_frame_exclusive, fps_str, ss)
            src_rel = f"{sc.scene_id}/src.mp4"
            mounts.append({"src": str(ss), "rel": src_rel})
        fg_rel = None
        if sc.requires_matte:
            fgw = sdir / "fg.webm"
            build_foreground_alpha_webm(OLIVIA_INPUT, matte_p, sc.start_frame, sc.end_frame_exclusive, fps_str, fgw, ffmpeg_bin)
            fg_rel = f"{sc.scene_id}/fg.webm"
            mounts.append({"src": str(fgw), "rel": fg_rel})
        hf_rel, hf_role = None, "foreground_graphics"
        if sc.scene_id in hf_out:
            hf_rel = f"{sc.scene_id}/hf.webm"
            hf_role = hf_out[sc.scene_id][1]
            mounts.append({"src": str(hf_out[sc.scene_id][0]), "rel": hf_rel})
        props = compile_scene_remotion_props(
            sc, sb, story_graph, source_file=src_rel, fg_alpha_file=fg_rel, prop_alpha_file=None,
            hf_graphic_file=hf_rel, hf_layer_role=hf_role,
            task_anchor={"cx": 0.50, "cy": 0.72, "w": 0.44, "h": 0.34},
            face_anchor={"cx": 0.48, "cy": 0.28, "w": 0.30, "h": 0.28},
        )
        jobs.append({
            "kind": "media",
            "id": sc.scene_id,
            "compositionId": "MotionLabGenerativeScene",
            "props": props,
            "outputLocation": str(OUT_DIR / f"{sc.scene_id}.mp4"),
            "codec": "h264",
            "pixelFormat": "yuv420p",
            "crf": 16,
            "concurrency": 1,
        })

    b_res = rem.render_batch_jobs(
        jobs=jobs, public_mounts=mounts, work_dir=temp_dir / "batch",
        bundle_cache_root=cache_base / "remotion_bundles", timeout_s=360.0,
    )

    # Build R5 AUTO (top row) vs R5.R1 Upgraded (bottom row) 360px Phone-Scale Comparison Sheet!
    # Also include H3 transition frame (f348 = local f4) right next to H3 peak so we visually confirm ZERO dark iris!
    panels = [
        ("H1 Hook (f28)", OUT_DIR / "H1_hook_money.mp4", 28, 28),
        ("H2a Bills (f172)", OUT_DIR / "H2_alloc_bills.mp4", 34, 172),
        ("H2b Groceries (f252)", OUT_DIR / "H2_alloc_groceries.mp4", 18, 252),
        ("H3 Morph f348", OUT_DIR / "H3_payoff_nodark.mp4", 4, 348),
        ("H3 Payoff Peak", OUT_DIR / "H3_payoff_nodark.mp4", 36, 380),
        ("H4 Closing CTA", OUT_DIR / "H4_closing_cta.mp4", 42, 677),
    ]
    tw, th = 220, 391
    pad = 10
    hdr = 52
    sheet = Image.new("RGB", (len(panels) * (tw + pad) + pad, hdr + 2 * (th + 28 + pad) + pad), (24, 22, 18))
    draw = ImageDraw.Draw(sheet)
    draw.text((pad, 14), "R5 AUTO Baseline (Top Row) vs R5.R1 Premium Editorial Upgrade (Bottom Row) — Phone-Scale Comparison", fill=(245, 240, 230))
    for i, (lbl, r1_mp4, r1_f, r5_f) in enumerate(panels):
        x = pad + i * (tw + pad)
        im_r5 = _extract_frame_rgb(R5_AUTO_FINAL, r5_f) if R5_AUTO_FINAL.is_file() else None
        im_r1 = _extract_frame_rgb(r1_mp4, r1_f)
        if im_r5 is not None:
            sheet.paste(Image.fromarray(im_r5).resize((tw, th), Image.Resampling.LANCZOS), (x, hdr))
        draw.text((x + 4, hdr + th + 4), f"R5: {lbl}", fill=(175, 116, 94))
        y2 = hdr + th + 28 + pad
        if im_r1 is not None:
            sheet.paste(Image.fromarray(im_r1).resize((tw, th), Image.Resampling.LANCZOS), (x, y2))
        draw.text((x + 4, y2 + th + 4), f"R5.R1: {lbl}", fill=(190, 154, 89))

    comp_jpg = OUT_DIR / "h1_h4_r5_vs_r5r1_phone_compare.jpg"
    sheet.save(comp_jpg, "JPEG", quality=92)
    shutil.rmtree(temp_dir, ignore_errors=True)

    summary = {
        "status": "OK",
        "elapsed_s": round(time.perf_counter() - t0, 2),
        "remotion_batch": b_res,
        "compare_sheet": str(comp_jpg),
        "clips": [str(OUT_DIR / f"{sc.scene_id}.mp4") for sc in proof_scenes],
    }
    atomic_write_json(OUT_DIR / "h1_h4_manifest.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
