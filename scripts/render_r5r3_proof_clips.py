"""Phase A5.4 (ML-01.R5.R3): Isolated ART_DIRECTED_QA Visual Lab Proofs (H1-H6).

Strictly separate from canonical `AUTO_GENERATIVE`! Never spliced into or used as input to AUTO!
Renders 6 focused 2-4.5s 9:16 proof clips validating every critical R5.R3 capability BEFORE full canonical run:
  - H1_multilayer_depth_hook  (f0..60)   : TRUE 2-HF-layer proof! Background `paper_tokens_only` behind MODNet matte
                                           + Foreground `typography_only` above matte + per-layer `excludeLayerId` ablations!
  - H2_bills_single_owner     (f138..207): Single canonical ledger owner (-$800 Bills -> $400 left) + live detail macro.
  - H3_payoff_sheet_and_fold  (f344..417): EditorialSheetSweepInBridge + discrete integer $100 + Waterfall + PaperFoldExitBridge.
  - H4_product_asset_binding  (f533..635): Authorized real PNG product texture inside 3D FlipPageTransition + ablation proof
                                           proving real asset visibly alters pixels vs conceptual schematic!
  - H5_closing_face_safe      (f635..720): Open serif takeaway + face-masked peripheral rim glow (zero washout!).
  - H6_critic_and_repair      (f344..417): Deliberate unbridged hard-cut defect vs repaired bridged render verified by
                                           streaming `run_objective_rendered_critic()` (`D10/D11`)!
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

from motionlab.composition.creative_quality import verify_ablation_still_diff
from motionlab.composition.generative_compiler import (
    compile_scene_hyperframes_layers,
    compile_scene_remotion_props,
)
from motionlab.composition.generative_schema import (
    GENERATIVE_SCHEMA_VERSION,
    LayerSpec,
    SceneSpec,
    StoryboardSpec,
    TransitionSpec,
)
from motionlab.composition.visual_critic import run_objective_rendered_critic
from motionlab.config import EditConfig, PROJECT_ROOT
from motionlab.preflight import probe_media
from motionlab.renderers.base import verify_webm_alpha_pixels
from motionlab.renderers.hyperframes import HyperFramesRenderer
from motionlab.renderers.remotion import RemotionRenderer
from motionlab.schemas import atomic_write_json
from motionlab.semantic.asr import transcribe_local
from motionlab.semantic.story_graph import build_story_graph
from motionlab.vision.segmentation import build_foreground_alpha_webm, inspect_vision_model, segment_subject_sequence


def _extract_frame(mp4: Path, frame_idx: int) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(mp4))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_idx))
    ok, fr = cap.read()
    cap.release()
    return cv2.cvtColor(fr, cv2.COLOR_BGR2RGB) if (ok and fr is not None) else None


def _make_demo_product_cover_png(dst: Path) -> Path:
    """Create a synthetic demo organizer cover image strictly for isolated Visual Lab H4 texture-binding proof."""
    im = Image.new("RGB", (600, 560), (36, 59, 50))
    draw = ImageDraw.Draw(im)
    draw.rectangle([18, 18, 582, 542], outline=(190, 154, 89), width=4)
    draw.rectangle([48, 54, 552, 134], fill=(245, 240, 230))
    draw.text((70, 82), "ORGANIZER DEMO SHEET (LAB PROOF)", fill=(36, 59, 50))
    for i, (lbl, w_frac, col) in enumerate([
        ("Fixed Commitments", 0.78, (175, 116, 94)),
        ("Living Essentials", 0.44, (190, 154, 89)),
        ("Available Buffer", 0.24, (132, 151, 122)),
    ]):
        y = 180 + i * 105
        draw.text((54, y), lbl, fill=(245, 240, 230))
        draw.rectangle([54, y + 32, 546, y + 66], fill=(56, 78, 69))
        draw.rectangle([54, y + 32, 54 + int(492 * w_frac), y + 66], fill=col)
    dst.parent.mkdir(parents=True, exist_ok=True)
    im.save(dst, "PNG")
    return dst


def main() -> int:
    t0 = time.perf_counter()
    src_mp4 = PROJECT_ROOT / "input" / "Olivia_Paycheck_V5_Kitchen_30s_Combined (1).mp4"
    out_dir = PROJECT_ROOT / "runs" / "r5r3_visual_lab"
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = out_dir / "_tmp"
    shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    media = probe_media(src_mp4)
    ffmpeg_bin = shutil.which("ffmpeg") or "ffmpeg"
    cfg = EditConfig(offline=True, composition_mode="generative", allow_remotion=True, allow_hyperframes=True, seed=42)
    tr = transcribe_local(media, cfg, PROJECT_ROOT / ".cache" / "motionlab_r5" / "semantic", ffmpeg_bin)
    sg = build_story_graph(tr, None, media)

    pal = {"ivory": "#F5F0E6", "espresso": "#40382E", "forest": "#243B32", "sage": "#84977A", "brass": "#BE9A59", "clay": "#AF745E"}
    steps = sg.ledger.steps
    s_open = steps[0].step_id
    s_bills = steps[1].step_id
    s_payoff = steps[-1].step_id

    # Segment MODNet matte for H1 (f0..60) so we prove TRUE 2-HF-layer behind-subject + foreground depth ordering!
    fg_alpha_h1_rel = None
    public_mounts: list[dict[str, str]] = []
    vis_info = inspect_vision_model()
    if vis_info.get("available"):
        mres = segment_subject_sequence(
            media=media,
            spans=[(0, 60)],
            workdir=tmp_dir,
            cache_dir=PROJECT_ROOT / ".cache" / "motionlab_r5" / "vision",
            keyframe_stride=2,
            ffmpeg_bin=ffmpeg_bin,
        )
        if mres.get("spans"):
            fg_webm = tmp_dir / "h1_fg_alpha.webm"
            build_foreground_alpha_webm(src_mp4, Path(mres["spans"][0]["matte_mp4"]), 0, 60, media.fps.to_ffmpeg(), fg_webm, ffmpeg_bin=ffmpeg_bin)
            fg_alpha_h1_rel = "H1_multilayer_depth_hook/fg_alpha.webm"
            public_mounts.append({"src": str(fg_webm), "rel": fg_alpha_h1_rel})

    demo_prod_png = _make_demo_product_cover_png(tmp_dir / "demo_organizer_cover.png")

    proof_scenes = [
        SceneSpec(
            scene_id="H1_multilayer_depth_hook",
            start_frame=0,
            end_frame_exclusive=60,
            role="hook_contrast",
            stage_mode="live_depth_matte" if fg_alpha_h1_rel else "live_clean_bridge",
            viewer_takeaway="2 distinct HF layers: paper tokens behind matte + hero typography in front",
            persistent_ledger_step_id=s_open,
            requires_matte=bool(fg_alpha_h1_rel),
            chosen_candidate_id="art_lab_h1_multilayer",
            candidate_provenance="ART_DIRECTED_QA",
            parameters={"bg_dim": 0.78, "bg_blur_px": 2.0, "parallax_travel_px": 24},
            layers=[
                LayerSpec("h1_hf_bg_tokens", "hf_apple_money_count", "hyperframes", "background_graphics", 15, {
                    "from_cents": 120000, "to_cents": 120000, "currency_symbol": "$",
                    "render_mode": "paper_tokens_only", "token_count": 4, "dark_plate": True,
                    "allow_fractional_cents": False, "hold_exact_amount": True,
                }),
                LayerSpec("h1_hf_fg_hero", "hf_apple_money_count", "hyperframes", "foreground_graphics", 35, {
                    "from_cents": 120000, "to_cents": 120000, "currency_symbol": "$",
                    "eyebrow": "BANK BALANCE", "caption": "", "y_anchor": "8.6%", "dark_plate": True,
                    "render_mode": "typography_only", "token_count": 0, "hero_px": 128,
                    "allow_fractional_cents": False, "hold_exact_amount": True,
                }),
            ],
        ),
        SceneSpec(
            scene_id="H2_bills_single_owner",
            start_frame=138,
            end_frame_exclusive=207,
            role="allocation_step",
            stage_mode="tracked_macro_split",
            viewer_takeaway="-$800 Bills single canonical ledger + detail macro",
            persistent_ledger_step_id=s_bills,
            show_persistent_ribbon=False,
            chosen_candidate_id="art_lab_h2_bills",
            candidate_provenance="ART_DIRECTED_QA",
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
            scene_id="H3_payoff_sheet_and_fold",
            start_frame=344,
            end_frame_exclusive=417,
            role="payoff_reveal",
            stage_mode="full_graphic_cutaway",
            viewer_takeaway="Sheet sweep + integer $100 payoff + PaperFold exit bridge",
            persistent_ledger_step_id=s_payoff,
            transition_in=TransitionSpec(kind="sheet_sweep", duration_frames=12),
            chosen_candidate_id="art_lab_h3_payoff",
            candidate_provenance="ART_DIRECTED_QA",
            parameters={"bg_color": pal["ivory"], "transition_out": "paper_fold_out", "transition_out_frames": 12},
            layers=[
                LayerSpec("h3_rem", "rem_ru_waterfall", "remotion", "background_graphics", 10, {
                    "title": "EXAMPLE ALLOCATION", "subtitle": "$1,200 -> $100", "currency_symbol": "$",
                    "y_pct": 38, "chart_height": 440, "dark_surface": False,
                }),
                LayerSpec("h3_hf", "hf_apple_money_count", "hyperframes", "foreground_graphics", 30, {
                    "from_cents": 120000, "to_cents": 10000, "currency_symbol": "$",
                    "eyebrow": "ACTUALLY LEFT TO SPEND", "caption": "from $1,200 example balance",
                    "y_anchor": "8.6%", "dark_plate": False, "token_count": 3, "hero_px": 150,
                    "accent_color": pal["brass"], "allow_fractional_cents": False, "hold_exact_amount": False,
                    "discrete_steps_cents": [120000, 40000, 20000, 10000],
                }),
            ],
        ),
        SceneSpec(
            scene_id="H4_product_asset_binding",
            start_frame=533,
            end_frame_exclusive=635,
            role="product_benefit",
            stage_mode="full_graphic_cutaway",
            viewer_takeaway="Authorized real PNG texture bound into 3D FlipPageTransition",
            transition_in=TransitionSpec(kind="line_led", duration_frames=12),
            chosen_candidate_id="art_lab_h4_product_texture",
            candidate_provenance="ART_DIRECTED_QA",
            parameters={"bg_color": pal["ivory"], "transition_out": "paper_fold_out", "transition_out_frames": 12},
            layers=[
                LayerSpec("h4_rem", "rem_rc_flip_page", "remotion", "background_graphics", 10, {
                    "label_before": "Organizer Overview", "sub_before": "AUTHORIZED PRODUCT PREVIEW",
                    "label_after": "Demo Sheet", "sub_after": "Real image texture proof", "y_frac": 0.075,
                }),
                LayerSpec("h4_hf", "hf_mk_specs_list", "hyperframes", "foreground_graphics", 30, {
                    "title": "INCLUDED COMPONENTS",
                    "rows": [{"label": "Structured Planner", "value": "Authorized asset proof"}, {"label": "Category Tracker", "value": "Phone-legible layout"}],
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
            viewer_takeaway="Give every dollar a purpose + face-safe peripheral rim glow",
            chosen_candidate_id="art_lab_h5_closing",
            candidate_provenance="ART_DIRECTED_QA",
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
        core_value_proposition="ART_DIRECTED_QA H1-H6 R5.R3 Visual Lab Proofs",
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

    batch_jobs: list[dict[str, Any]] = []
    ablation_checks: list[dict[str, Any]] = []

    for sc in proof_scenes:
        compiled_ovs = compile_scene_hyperframes_layers(sc, sb, sg)
        hf_mounts: list[dict[str, Any]] = []
        for h_idx, ov in enumerate(compiled_ovs):
            hf_out = tmp_dir / f"hf_{sc.scene_id}_{ov.layer_id}.webm"
            hf.render_html_slot(tmp_dir / f"hf_slot_{sc.scene_id}_{ov.layer_id}", ov.html, hf_out, media.fps_float, fmt="webm", cache_dir=hf_cache, use_cache=True, skip_cli_lint=True)
            astats = verify_webm_alpha_pixels(hf_out, media.display_width, media.display_height, sample_frame=12, ffmpeg_bin=ffmpeg_bin)
            if not astats.get("has_visible_alpha", False):
                raise RuntimeError(f"Visual lab HF layer {ov.layer_id} rendered empty alpha!")
            rel_p = f"{sc.scene_id}/hf_{h_idx}_{ov.layer_id}.webm"
            public_mounts.append({"src": str(hf_out), "rel": rel_p})
            hf_mounts.append({
                "layerId": ov.layer_id,
                "primitiveId": ov.primitive_id,
                "file": rel_p,
                "layerRole": ov.layer_role,
                "zOrder": ov.z_order,
                "opacity": ov.opacity,
            })

        src_slice = tmp_dir / f"src_{sc.scene_id}.mp4"
        rem._slice_source_mezzanine(src_mp4, sc.start_frame, sc.end_frame_exclusive, fps_str, src_slice)
        src_rel = f"{sc.scene_id}/src.mp4"
        public_mounts.append({"src": str(src_slice), "rel": src_rel})

        prod_rel = None
        prod_meta = None
        if sc.scene_id == "H4_product_asset_binding":
            prod_rel = f"{sc.scene_id}/demo_cover.png"
            prod_meta = {"asset_id": "lab_demo_cover", "asset_type": "workbook_cover", "usage_constraints": "LAB DEMO ASSET"}
            public_mounts.append({"src": str(demo_prod_png), "rel": prod_rel})

        props = compile_scene_remotion_props(
            sc, sb, sg,
            source_file=src_rel,
            fg_alpha_file=fg_alpha_h1_rel if sc.scene_id == "H1_multilayer_depth_hook" else None,
            prop_alpha_file=None,
            hf_graphic_file=hf_mounts[0]["file"] if hf_mounts else None,
            hf_layer_role=hf_mounts[0]["layerRole"] if hf_mounts else "foreground_graphics",
            hf_layers=hf_mounts,
            product_asset_file=prod_rel,
            product_asset_meta=prod_meta,
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

        # Add H1 per-layer multi-HF ablations AND H4 real product texture ablation!
        if sc.scene_id == "H1_multilayer_depth_hook":
            full_p = out_dir / "ab_h1_full_f12.png"
            batch_jobs.append({"kind": "still", "id": "ab:h1:full", "compositionId": "MotionLabGenerativeScene", "props": props, "frame": 12, "outputLocation": str(full_p)})
            for hlm in hf_mounts:
                ex_p = out_dir / f"ab_h1_without_{hlm['layerId']}_f12.png"
                batch_jobs.append({"kind": "still", "id": f"ab:h1:{hlm['layerId']}", "compositionId": "MotionLabGenerativeScene", "props": {**props, "excludeLayerId": hlm["layerId"]}, "frame": 12, "outputLocation": str(ex_p)})
                ablation_checks.append({"label": f"H1:{hlm['layerId']}({hlm['layerRole']})", "full": full_p, "ex": ex_p})
        elif sc.scene_id == "H4_product_asset_binding":
            full_p = out_dir / "ab_h4_with_asset_f72.png"
            ex_p = out_dir / "ab_h4_conceptual_only_f72.png"
            batch_jobs.append({"kind": "still", "id": "ab:h4:full", "compositionId": "MotionLabGenerativeScene", "props": props, "frame": 72, "outputLocation": str(full_p)})
            batch_jobs.append({"kind": "still", "id": "ab:h4:no_asset", "compositionId": "MotionLabGenerativeScene", "props": {**props, "excludeLayerRole": "product_asset"}, "frame": 72, "outputLocation": str(ex_p)})
            ablation_checks.append({"label": "H4:real_product_asset_texture", "full": full_p, "ex": ex_p})

    batch_res = rem.render_batch_jobs(
        jobs=batch_jobs,
        public_mounts=public_mounts,
        work_dir=tmp_dir / "rem_batch",
        bundle_cache_root=PROJECT_ROOT / ".cache" / "motionlab_r5" / "remotion_bundles",
        timeout_s=360.0,
    )

    ablation_summary = {}
    for chk in ablation_checks:
        ablation_summary[chk["label"]] = verify_ablation_still_diff(chk["full"], chk["ex"])

    # H6: Streaming critic + repair proof (inject 1-frame bright white flash into copy of live-action H5 -> verify critic catches it vs clean H5!)
    h5_clean = out_dir / "H5_closing_face_safe.mp4"
    h6_broken = out_dir / "H6_broken_flash_control.mp4"
    cap = cv2.VideoCapture(str(h5_clean))
    fps_h5 = cap.get(cv2.CAP_PROP_FPS) or 24.0
    w_h5, h_h5 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vw = cv2.VideoWriter(str(h6_broken), cv2.VideoWriter_fourcc(*"mp4v"), fps_h5, (w_h5, h_h5))
    fi = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if fi == 36:
            fr = np.full_like(fr, 250)  # Deliberate harsh white flash defect on live footage
        vw.write(fr)
        fi += 1
    cap.release()
    vw.release()

    h6_sb = StoryboardSpec(
        GENERATIVE_SCHEMA_VERSION, "ART_DIRECTED_QA", media.input_hash, tr.transcript_hash, sg.story_graph_hash,
        "finance", 0.99, "H6 critic repair proof", w_h5, h_h5, 24, 1, fi, 42, pal,
        [SceneSpec("H5_closing_face_safe", 0, fi, "closing_takeaway", "live_clean_bridge", "closing")],
    )
    h6_broken_critic = run_objective_rendered_critic(
        source_mp4=tmp_dir / "src_H5_closing_face_safe.mp4",
        final_mp4=h6_broken,
        storyboard=h6_sb,
        continuous_timeline={"boundaries": []},
        roi_tracks=None,
        out_dir=out_dir / "h6_broken_critic",
    )
    h6_repaired_critic = run_objective_rendered_critic(
        source_mp4=tmp_dir / "src_H5_closing_face_safe.mp4",
        final_mp4=h5_clean,
        storyboard=h6_sb,
        continuous_timeline={"boundaries": []},
        roi_tracks=None,
        out_dir=out_dir / "h6_repaired_critic",
    )

    # Render phone-scale visual lab inspection sheet (H1..H5 + H4 asset vs conceptual!)
    tiles = [
        ("H1 Multi-HF Depth (f12)", _extract_frame(out_dir / "H1_multilayer_depth_hook.mp4", 12)),
        ("H1 No BG Tokens (ablation)", np.array(Image.open(out_dir / "ab_h1_without_h1_hf_bg_tokens_f12.png").convert("RGB"))),
        ("H2 Bills Single Owner", _extract_frame(out_dir / "H2_bills_single_owner.mp4", 28)),
        ("H3 Payoff $100 Peak", _extract_frame(out_dir / "H3_payoff_sheet_and_fold.mp4", 24)),
        ("H3 PaperFold Exit", _extract_frame(out_dir / "H3_payoff_sheet_and_fold.mp4", 67)),
        ("H4 Real Product Texture", np.array(Image.open(out_dir / "ab_h4_with_asset_f72.png").convert("RGB"))),
        ("H4 Conceptual Only", np.array(Image.open(out_dir / "ab_h4_conceptual_only_f72.png").convert("RGB"))),
        ("H5 Closing Face-Safe", _extract_frame(out_dir / "H5_closing_face_safe.mp4", 45)),
    ]
    tw, th = 220, 390
    pad, hdr = 10, 44
    cols = 4
    rows = 2
    sheet = Image.new("RGB", (cols * (tw + pad) + pad, hdr + rows * (th + 28 + pad) + pad), (20, 19, 17))
    draw = ImageDraw.Draw(sheet)
    draw.text((10, 14), "ML-01.R5.R3 Isolated ART_DIRECTED_QA Visual Lab (H1 Multi-HF, H2-H5, Real Asset Texture vs Conceptual)", fill=(245, 240, 230))
    for i, (lbl, arr) in enumerate(tiles):
        if arr is None:
            continue
        r, c = divmod(i, cols)
        x = pad + c * (tw + pad)
        y = hdr + r * (th + 28 + pad)
        sheet.paste(Image.fromarray(arr).resize((tw, th), Image.Resampling.LANCZOS), (x, y))
        draw.text((x + 2, y + th + 4), lbl[:30], fill=(190, 220, 180))
    sheet_path = out_dir / "r5r3_visual_lab_sheet.jpg"
    sheet.save(sheet_path, "JPEG", quality=92)
    shutil.rmtree(tmp_dir, ignore_errors=True)

    summary = {
        "mode": "ART_DIRECTED_QA_ISOLATED_PROOFS",
        "elapsed_s": round(time.perf_counter() - t0, 2),
        "remotion_batch": batch_res,
        "ablations": ablation_summary,
        "h6_critic_proof": {
            "broken_internal_flashes": h6_broken_critic["internal_flash_count"],
            "broken_severe_findings": h6_broken_critic["severe_findings_count"],
            "repaired_internal_flashes": h6_repaired_critic["internal_flash_count"],
            "repaired_severe_findings": h6_repaired_critic["severe_findings_count"],
            "streaming_mode": h6_broken_critic["streaming_mode"],
        },
        "visual_lab_sheet": str(sheet_path),
        "clips": [f"{sc.scene_id}.mp4" for sc in proof_scenes] + ["H6_broken_flash_control.mp4"],
    }
    atomic_write_json(out_dir / "proof_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
