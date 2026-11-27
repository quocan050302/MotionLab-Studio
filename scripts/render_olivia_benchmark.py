"""Phase A6 (ML-01.R5): Separate Art-Directed QA Benchmark Renderer (`art_directed_benchmark.mp4`).

STRICT SEPARATION NOTICE:
  This script implements the Section 09 Art-Directed QA Reference Blueprint aligned to Olivia's
  verified offline Faster-Whisper word timestamps.
  Output goes STRICTLY into `runs/olivia_ml01_r5_benchmark/<run_id>/art_directed_benchmark.mp4`.
  It is NEVER renamed to the canonical automatic final and is NEVER imported by automatic CLI mode!
"""

from __future__ import annotations

import asyncio
import datetime
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from motionlab.analyzer import analyze_media_async
from motionlab.composition.generative_schema import (
    AssetRef,
    ClaimRef,
    GENERATIVE_SCHEMA_VERSION,
    LayerSpec,
    SceneSpec,
    SoundIntent,
    StoryboardSpec,
    TransitionSpec,
)
from motionlab.composition.r5_engine import render_compiled_storyboard_to_mp4
from motionlab.config import EditConfig
from motionlab.preflight import probe_media
from motionlab.resource import ResourceBudget, ResourceTracker
from motionlab.schemas import atomic_write_json
from motionlab.semantic.asr import transcribe_local
from motionlab.semantic.story_graph import build_story_graph
from motionlab.vision.tracking import track_action_rois

OLIVIA_INPUT = REPO_ROOT / "input" / "Olivia_Paycheck_V5_Kitchen_30s_Combined (1).mp4"
OUT_ROOT = REPO_ROOT / "runs" / "olivia_ml01_r5_benchmark"

PALETTE = {
    "ivory": "#F5F0E6",
    "espresso": "#292822",
    "forest": "#365B49",
    "sage": "#A4B39A",
    "brass": "#C6A46A",
    "clay": "#B85D43",
}


def build_olivia_art_directed_storyboard(media, story_graph) -> StoryboardSpec:
    """Author the Section 09 10-shot art-directed QA reference storyboard aligned to exact ASR timestamps."""
    step_ids = {st.category.lower(): st.step_id for st in story_graph.ledger.steps}
    step_open = story_graph.ledger.steps[0].step_id if story_graph.ledger.steps else None
    step_bills = step_ids.get("bills")
    step_groc = step_ids.get("groceries")
    step_gas = step_ids.get("gas")
    step_rem = next((st.step_id for st in story_graph.ledger.steps if st.step_type == "remaining_confirmed"), None)

    scenes = [
        # SHOT 01 (f0..f60, 0.00-2.50s): Opening contrast — MODNet depth parallax + behind-subject $1,200 bank balance
        SceneSpec(
            scene_id="shot01_opening_balance",
            start_frame=0, end_frame_exclusive=60,
            role="hook_contrast",
            stage_mode="live_depth_matte",
            viewer_takeaway="Immediate contrast: $1,200 bank balance is the starting point.",
            layers=[
                LayerSpec(
                    layer_id="s01_hf_bal", primitive_id="hf_apple_money_count", engine="hyperframes",
                    layer_role="background_graphics", z_order=15,
                    params={
                        "from_cents": 0, "to_cents": 120000, "currency_symbol": "$",
                        "eyebrow": "BANK BALANCE", "caption": "",
                        "y_anchor": "4.5%", "dark_plate": True, "token_count": 8, "hero_px": 100,
                    },
                ),
            ],
            sound_intents=[
                SoundIntent(cue_id="s01_flick", family="cash_flick", target_frame=22, variant=0, gain_db=-4.5, pan=-0.10, rationale="spoken_$1200_at_0.90s"),
                SoundIntent(cue_id="s01_impact", family="soft_impact", target_frame=34, variant=0, gain_db=-5.0, pan=0.0, rationale="hero_counter_settle"),
            ],
            persistent_ledger_step_id=step_open,
            requires_matte=True,
            parameters={"bg_dim": 0.78, "bg_blur_px": 2.0, "parallax_travel_px": 26},
            expected_visual_change="MODNet foreground silhouette over 2.5D parallax kitchen and behind-subject $1,200 counter.",
        ),
        # SHOT 02 (f60..f134, 2.50-5.58s): Contradiction — "$1,200 TO SPEND" struck through -> "BALANCE ≠ SPENDABLE"
        SceneSpec(
            scene_id="shot02_spendability_contradiction",
            start_frame=60, end_frame_exclusive=134,
            role="negation_tension",
            stage_mode="live_clean_bridge",
            viewer_takeaway="Bank balance does not equal spendable money.",
            layers=[
                LayerSpec(
                    layer_id="s02_rem_strike", primitive_id="rem_ru_strikethrough", engine="remotion",
                    layer_role="foreground_graphics", z_order=30,
                    params={
                        "eyebrow": "SPENDABILITY CHECK",
                        "from_text": "$1,200 TO SPEND",
                        "to_text": "BALANCE ≠ SPENDABLE",
                        "y_pct": 10, "font_size": 48, "dark_card": True,
                    },
                ),
            ],
            sound_intents=[
                SoundIntent(cue_id="s02_rev", family="reverse_whoosh", target_frame=68, variant=0, gain_db=-5.5, pan=0.0, rationale="contradiction_lead_in"),
                SoundIntent(cue_id="s02_hit", family="soft_impact", target_frame=86, variant=1, gain_db=-4.5, pan=0.0, rationale="spoken_$1200_strike_at_3.56s"),
            ],
            expected_visual_change="Remotion StrikethroughReplace crossing out '$1,200 TO SPEND' into 'BALANCE ≠ SPENDABLE'.",
        ),
        # SHOT 03 (f134..f232, 5.58-9.67s): Bills -$800 — Tracked hands macro aperture + 67% proportional bar + $400 balance ribbon
        SceneSpec(
            scene_id="shot03_bills_allocation",
            start_frame=134, end_frame_exclusive=232,
            role="allocation_step",
            stage_mode="tracked_macro_split",
            viewer_takeaway="-$800 goes to bills (67%), leaving $400 derived balance.",
            layers=[
                LayerSpec(
                    layer_id="s03_hf_bills", primitive_id="hf_mk_progress_stat", engine="hyperframes",
                    layer_role="foreground_graphics", z_order=30,
                    params={
                        "amount_cents": 80000, "denom_cents": 120000, "remaining_cents": 40000,
                        "currency_symbol": "$", "is_deduction": True,
                        "category": "BILLS", "caption": "Balance left: $400 (derived)", "y_px": 54, "dark_plate": True,
                    },
                ),
                LayerSpec(
                    layer_id="s03_rem_macro", primitive_id="rem_ru_track_matte", engine="remotion",
                    layer_role="foreground_graphics", z_order=35,
                    params={"x_frac": 0.52, "y_frac": 0.50, "detail_zoom": 1.62, "label": "LIVE CASH ACTION"},
                ),
            ],
            sound_intents=[
                SoundIntent(cue_id="s03_flick", family="cash_flick", target_frame=152, variant=1, gain_db=-4.5, pan=0.12, rationale="spoken_800_bills_at_6.28s"),
                SoundIntent(cue_id="s03_rustle", family="paper_rustle", target_frame=186, variant=0, gain_db=-5.0, pan=-0.10, rationale="cash_handling_action_7.75s"),
                SoundIntent(cue_id="s03_tick", family="tick_accent", target_frame=204, variant=0, gain_db=-5.5, pan=0.0, rationale="derived_$400_settle"),
            ],
            persistent_ledger_step_id=step_bills,
            show_persistent_ribbon=True,
            expected_visual_change="Proportional 67% Bills bar + tracked hands/cash TrackMatte macro + persistent $400 ledger ribbon.",
        ),
        # SHOT 04 (f232..f278, 9.67-11.58s): Groceries -$200 — AspectMorph editorial layout + cumulative proportional data-chart
        SceneSpec(
            scene_id="shot04_groceries_morph",
            start_frame=232, end_frame_exclusive=278,
            role="allocation_step",
            stage_mode="aspect_morph_editorial",
            viewer_takeaway="-$200 for groceries advances cumulative proportional breakdown to $200 remaining.",
            layers=[
                LayerSpec(
                    layer_id="s04_hf_chart", primitive_id="hf_data_chart", engine="hyperframes",
                    layer_role="foreground_graphics", z_order=30,
                    params={
                        "title": "CUMULATIVE ALLOCATION", "denom_cents": 120000, "currency_symbol": "$", "y_px": 74,
                        "items": [
                            {"label": "Bills", "cents": 80000, "color": PALETTE["clay"]},
                            {"label": "Groceries", "cents": 20000, "color": PALETTE["brass"]},
                            {"label": "Remaining (derived)", "cents": 20000, "color": PALETTE["forest"]},
                        ],
                    },
                ),
            ],
            sound_intents=[
                SoundIntent(cue_id="s04_flick", family="cash_flick", target_frame=236, variant=2, gain_db=-5.0, pan=-0.12, rationale="spoken_200_groceries_at_9.74s"),
                SoundIntent(cue_id="s04_tick", family="tick_accent", target_frame=258, variant=1, gain_db=-5.5, pan=0.10, rationale="groceries_bar_settle"),
            ],
            persistent_ledger_step_id=step_groc,
            parameters={"kicker": "", "lines": [], "header_title": "", "shifted_y_frac": 0.56, "shifted_h_frac": 0.36},
            expected_visual_change="AspectMorphCard shifts live footage downward revealing warm-ivory cumulative proportional bar chart.",
        ),
        # SHOT 05 (f278..f342, 11.58-14.25s): Gas -$100 — Clean live bridge + 8% proportional Gas bar + $100 remaining ribbon
        SceneSpec(
            scene_id="shot05_gas_allocation",
            start_frame=278, end_frame_exclusive=342,
            role="allocation_step",
            stage_mode="live_clean_bridge",
            viewer_takeaway="-$100 for gas brings derived remaining balance down to $100.",
            layers=[
                LayerSpec(
                    layer_id="s05_hf_gas", primitive_id="hf_mk_progress_stat", engine="hyperframes",
                    layer_role="foreground_graphics", z_order=30,
                    params={
                        "amount_cents": 10000, "denom_cents": 120000, "remaining_cents": 10000,
                        "currency_symbol": "$", "is_deduction": True,
                        "category": "GAS", "caption": "Balance left: $100 (derived)", "y_px": 90, "dark_plate": True,
                    },
                ),
            ],
            sound_intents=[
                SoundIntent(cue_id="s05_flick", family="cash_flick", target_frame=292, variant=0, gain_db=-5.0, pan=0.10, rationale="spoken_100_gas_at_12.10s"),
                SoundIntent(cue_id="s05_tick", family="tick_accent", target_frame=316, variant=2, gain_db=-5.5, pan=0.0, rationale="gas_settle_at_13.16s"),
            ],
            persistent_ledger_step_id=step_gas,
            show_persistent_ribbon=True,
            expected_visual_change="Distinct -$100 GAS bar (8%) + persistent running balance ribbon stepping $200 -> $100.",
        ),
        # SHOT 06 (f342..f416, 14.25-17.33s): Main Payoff — Full editorial cutaway with complete WaterfallChart + $100 hero counter
        SceneSpec(
            scene_id="shot06_payoff_100_left",
            start_frame=342, end_frame_exclusive=416,
            role="payoff_reveal",
            stage_mode="full_graphic_cutaway",
            viewer_takeaway="Dramatic visual payoff: out of $1,200, only $100 is actually left.",
            layers=[
                LayerSpec(
                    layer_id="s06_rem_wf", primitive_id="rem_ru_waterfall", engine="remotion",
                    layer_role="background_graphics", z_order=10,
                    params={
                        "title": "PAYCHECK WATERFALL", "subtitle": "$1,200 → $100 LEFT",
                        "y_pct": 39, "chart_height": 440, "dark_surface": False,
                        "steps": [
                            {"label": "Balance", "value": 1200, "isTotal": True, "color": PALETTE["espresso"]},
                            {"label": "Bills", "value": -800, "color": PALETTE["clay"]},
                            {"label": "Groceries", "value": -200, "color": PALETTE["brass"]},
                            {"label": "Gas", "value": -100, "color": PALETTE["sage"]},
                            {"label": "Left", "value": 100, "isTotal": True, "color": PALETTE["forest"]},
                        ],
                    },
                ),
                LayerSpec(
                    layer_id="s06_hf_payoff", primitive_id="hf_apple_money_count", engine="hyperframes",
                    layer_role="foreground_graphics", z_order=30,
                    params={
                        "from_cents": 120000, "to_cents": 10000, "currency_symbol": "$",
                        "eyebrow": "ACTUALLY LEFT", "caption": "Out of $1,200 example balance",
                        "y_anchor": "6%", "dark_plate": True, "token_count": 6, "accent_color": PALETTE["forest"],
                    },
                ),
            ],
            transition_in=TransitionSpec(kind="frame_morph", duration_frames=10, rationale="editorial_payoff_cutaway"),
            sound_intents=[
                SoundIntent(cue_id="s06_impact", family="soft_impact", target_frame=372, variant=0, gain_db=-4.0, pan=0.0, rationale="spoken_$100_left_at_15.48s"),
                SoundIntent(cue_id="s06_shimmer", family="shimmer_payoff", target_frame=394, variant=0, gain_db=-6.0, pan=0.0, rationale="payoff_resolve"),
            ],
            persistent_ledger_step_id=step_rem,
            intentional_cutaway=True,
            parameters={"bg_color": PALETTE["ivory"]},
            expected_visual_change="Full-frame warm ivory WaterfallChart + $100 hero counter cutaway.",
        ),
        # SHOT 07 (f416..f480, 17.33-20.00s): Intentional natural breathing bridge (raw source preserved cleanly!)
        # SHOT 08 (f480..f534, 20.00-22.25s): Product Intro — LineLed bridge into MODNet depth stage + lt-mask-reveal
        SceneSpec(
            scene_id="shot08_product_intro",
            start_frame=480, end_frame_exclusive=534,
            role="product_intro",
            stage_mode="live_depth_matte",
            viewer_takeaway="Introduces 'The Next Paycheck Plan' cleanly.",
            layers=[
                LayerSpec(
                    layer_id="s08_rem_motes", primitive_id="rem_viv_particle_field", engine="remotion",
                    layer_role="background_graphics", z_order=10,
                    params={"count": 42, "speed": 0.14, "connect": False, "colors": [PALETTE["brass"], PALETTE["ivory"], PALETTE["sage"]]},
                ),
                LayerSpec(
                    layer_id="s08_hf_title", primitive_id="hf_lt_mask_reveal", engine="hyperframes",
                    layer_role="foreground_graphics", z_order=30,
                    params={"headline": "The Next Paycheck Plan", "subtitle": "ORGANIZE YOUR MONEY", "y_px": 104},
                ),
            ],
            transition_in=TransitionSpec(kind="line_led", duration_frames=10, rationale="solution_reveal_aperture"),
            sound_intents=[
                SoundIntent(cue_id="s08_whoosh", family="soft_whoosh", target_frame=484, variant=0, gain_db=-5.0, pan=0.0, rationale="line_led_transition"),
                SoundIntent(cue_id="s08_page", family="page_turn", target_frame=502, variant=0, gain_db=-4.5, pan=0.10, rationale="spoken_plan_name_20.88s"),
            ],
            requires_matte=True,
            parameters={"bg_dim": 0.78, "bg_blur_px": 2.0, "parallax_travel_px": 24},
            expected_visual_change="LineLed transition into MODNet depth stage + sweeping mask reveal of 'The Next Paycheck Plan'.",
        ),
        # SHOT 09 (f534..f634, 22.25-26.42s): Product Benefit — 3D FlipPage workbook turn + grounded Notion/56-page specs
        SceneSpec(
            scene_id="shot09_product_benefit",
            start_frame=534, end_frame_exclusive=634,
            role="product_benefit",
            stage_mode="full_graphic_cutaway",
            viewer_takeaway="Clarifies Notion dashboard + 56-page workbook honestly (conceptual schematic, ASSET_MISSING).",
            layers=[
                LayerSpec(
                    layer_id="s09_rem_flip", primitive_id="rem_rc_flip_page", engine="remotion",
                    layer_role="background_graphics", z_order=10,
                    params={
                        "label_before": "The Next Paycheck Plan",
                        "sub_before": "CONCEPTUAL PRODUCT OVERVIEW",
                        "label_after": "56-page workbook",
                        "sub_after": "+ Notion dashboard · Organize your money",
                        "y_frac": 0.065, "flip_delay": 24, "flip_duration": 24,
                    },
                ),
                LayerSpec(
                    layer_id="s09_hf_specs", primitive_id="hf_mk_specs_list", engine="hyperframes",
                    layer_role="foreground_graphics", z_order=30,
                    params={
                        "title": "INCLUDED IN THE PLAN",
                        "rows": [
                            {"label": "Notion dashboard", "value": "Organize every dollar"},
                            {"label": "56-page workbook", "value": "Conceptual overview"},
                        ],
                        "y_px": 706,
                    },
                ),
            ],
            asset_refs=[AssetRef(asset_id="prod_concept", asset_kind="conceptual_schematic", provenance="grounded_asr_no_owner_screenshot", status="ASSET_MISSING_CONCEPTUAL")],
            sound_intents=[
                SoundIntent(cue_id="s09_flip", family="page_turn", target_frame=562, variant=1, gain_db=-4.5, pan=-0.10, rationale="56_page_workbook_flip_23.44s"),
                SoundIntent(cue_id="s09_pop", family="pop_reveal", target_frame=608, variant=0, gain_db=-5.0, pan=0.08, rationale="organize_your_money_25.32s"),
            ],
            intentional_cutaway=True,
            parameters={"bg_color": PALETTE["ivory"]},
            expected_visual_change="3D FlipPageTransition workbook turn + HyperFrames specs list (honest conceptual view).",
        ),
        # SHOT 10 (f634..f720, 26.42-30.00s): Closing Takeaway & CTA — Return smoothly to live speaker + kinetic takeaway + light leak
        SceneSpec(
            scene_id="shot10_closing_purpose_cta",
            start_frame=634, end_frame_exclusive=720,
            role="closing_takeaway",
            stage_mode="live_clean_bridge",
            viewer_takeaway="Give every dollar a purpose. Link in my profile.",
            layers=[
                LayerSpec(
                    layer_id="s10_rem_slide", primitive_id="rem_ru_masked_slide", engine="remotion",
                    layer_role="foreground_graphics", z_order=30,
                    params={
                        "eyebrow": "BEFORE YOUR NEXT PAYDAY",
                        "lines": ["Give every dollar", "a purpose"],
                        "subline": "LINK IN MY PROFILE",
                        "y_pct": 10, "font_size": 52, "dark_card": True, "serif": True,
                    },
                ),
                LayerSpec(
                    layer_id="s10_hf_leak", primitive_id="hf_organic_light_leak", engine="hyperframes",
                    layer_role="foreground_graphics", z_order=35,
                    params={"opacity": 0.42},
                ),
            ],
            sound_intents=[
                SoundIntent(cue_id="s10_bass", family="bass_settle", target_frame=662, variant=0, gain_db=-5.0, pan=0.0, rationale="every_dollar_purpose_27.54s"),
                SoundIntent(cue_id="s10_resolve", family="shimmer_payoff", target_frame=686, variant=1, gain_db=-6.0, pan=0.0, rationale="cta_resolve_28.58s"),
            ],
            expected_visual_change="Return cleanly to Olivia eye contact with kinetic closing takeaway and warm editorial light leak.",
        ),
    ]

    sb = StoryboardSpec(
        schema_version=GENERATIVE_SCHEMA_VERSION,
        mode="ART_DIRECTED_BENCHMARK",
        source_hash=media.input_hash,
        transcript_hash=story_graph.transcript_hash,
        story_graph_hash=story_graph.story_graph_hash,
        niche="finance",
        niche_confidence=1.0,
        core_value_proposition="Demonstrate $1,200 balance vs $100 remaining example and introduce The Next Paycheck Plan.",
        width=media.display_width,
        height=media.display_height,
        fps_num=media.fps.num,
        fps_den=media.fps.den,
        total_frames=media.duration_frames,
        seed=42,
        palette=PALETTE,
        scenes=scenes,
        director_model_id="art_directed_qa_blueprint_section_09",
    )
    sb.storyboard_hash = sb.compute_hash()
    return sb


def main() -> None:
    media = probe_media(OLIVIA_INPUT)
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{ts}_benchmark_{media.input_hash[:8]}"
    run_dir = (OUT_ROOT / run_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "evidence").mkdir(exist_ok=True)
    (run_dir / "temp").mkdir(exist_ok=True)

    cfg = EditConfig(
        offline=True,
        composition_mode="generative",
        allow_remotion=True,
        allow_hyperframes=True,
        audio_mode="mix",
        sfx_mode="auto",
        music_mode="off",
        render_preview_9x16=True,
        seed=42,
    )
    budget = ResourceBudget(config=cfg.resources, dest_dir=run_dir, temp_dir=run_dir / "temp")
    budget.preflight_budget(media.duration_s, media.display_width, media.display_height, media.fps_float, "high", overlay_events=9)
    tracker = ResourceTracker()
    ffmpeg_bin = shutil.which("ffmpeg") or "ffmpeg"

    analysis, _ = asyncio.run(analyze_media_async(media, cfg, run_dir / "temp", budget=budget))
    roi_tracks = track_action_rois(media, analysis, cfg, ffmpeg_bin)
    transcript = transcribe_local(media, cfg, cache_dir=REPO_ROOT / ".cache" / "a1_asr_cache", ffmpeg_bin=ffmpeg_bin)
    story_graph = build_story_graph(transcript, roi_tracks, media)
    atomic_write_json(run_dir / "story_graph.json", story_graph.to_dict())
    atomic_write_json(run_dir / "ledger_claims.json", story_graph.ledger.to_dict())

    sb = build_olivia_art_directed_storyboard(media, story_graph)
    res = render_compiled_storyboard_to_mp4(
        media=media,
        storyboard=sb,
        story_graph=story_graph,
        roi_tracks=roi_tracks,
        config=cfg,
        run_id=run_id,
        run_dir=run_dir,
        budget=budget,
        tracker=tracker,
        output_filename="art_directed_benchmark.mp4",
        render_preview=True,
        require_dual_engine_reuse_gate=True,
        transcript_obj=transcript,
    )
    print(json.dumps({k: v for k, v in res.items() if k != "resource_stats"}, indent=2))


if __name__ == "__main__":
    main()
