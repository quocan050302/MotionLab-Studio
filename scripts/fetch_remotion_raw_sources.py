"""Download exact pinned-SHA MIT Remotion component source files & LICENSE files into staging."""

from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STAGE = ROOT / ".cache" / "repo_staging"
OUT_RAW = STAGE / "remotion_raw"
OUT_RAW.mkdir(parents=True, exist_ok=True)

TARGETS = [
    # R02: riaz37/remotion-ui (MIT)
    ("R02", "remotion-ui", "riaz37/remotion-ui", "LICENSE", "LICENSE"),
    ("R02", "remotion-ui", "riaz37/remotion-ui", "apps/web/registry/bases/default/primitives/counter.tsx", "counter.tsx"),
    ("R02", "remotion-ui", "riaz37/remotion-ui", "apps/web/registry/bases/default/primitives/waterfall-chart.tsx", "waterfall-chart.tsx"),
    ("R02", "remotion-ui", "riaz37/remotion-ui", "apps/web/registry/bases/default/primitives/strikethrough-replace.tsx", "strikethrough-replace.tsx"),
    ("R02", "remotion-ui", "riaz37/remotion-ui", "apps/web/registry/bases/default/primitives/track-matte.tsx", "track-matte.tsx"),
    ("R02", "remotion-ui", "riaz37/remotion-ui", "apps/web/registry/bases/default/primitives/parallax-layers.tsx", "parallax-layers.tsx"),
    ("R02", "remotion-ui", "riaz37/remotion-ui", "apps/web/registry/bases/default/primitives/masked-slide-reveal.tsx", "masked-slide-reveal.tsx"),
    ("R02", "remotion-ui", "riaz37/remotion-ui", "apps/web/registry/bases/default/primitives/text-animator.tsx", "text-animator.tsx"),
    ("R02", "remotion-ui", "riaz37/remotion-ui", "apps/web/registry/bases/default/scenes/callout-spotlight/index.tsx", "callout-spotlight.tsx"),
    # R06: vivmagarwal/remotion-effects-library (MIT)
    ("R06", "remotion-effects-library", "vivmagarwal/remotion-effects-library", "LICENSE", "LICENSE"),
    ("R06", "remotion-effects-library", "vivmagarwal/remotion-effects-library", "src/effects/motion/aspect-morph-card/AspectMorphCard.tsx", "AspectMorphCard.tsx"),
    ("R06", "remotion-effects-library", "vivmagarwal/remotion-effects-library", "src/effects/motion/particle-field/ParticleField.tsx", "ParticleField.tsx"),
    ("R06", "remotion-effects-library", "vivmagarwal/remotion-effects-library", "src/effects/type/kinetic-word-reveal/KineticWordReveal.tsx", "KineticWordReveal.tsx"),
    ("R06", "remotion-effects-library", "vivmagarwal/remotion-effects-library", "src/effects/motion/parallax-layers/ParallaxLayers.tsx", "ParallaxLayers.tsx"),
    # R08: RenderComp/free-remotion-templates (MIT)
    ("R08", "free-remotion-templates", "RenderComp/free-remotion-templates", "LICENSE", "LICENSE"),
    ("R08", "free-remotion-templates", "RenderComp/free-remotion-templates", "src/components/flip-page-transition/FlipPageTransition.tsx", "FlipPageTransition.tsx"),
    ("R08", "free-remotion-templates", "RenderComp/free-remotion-templates", "src/components/kpi-counter/KpiCounter.tsx", "KpiCounter.tsx"),
    ("R08", "free-remotion-templates", "RenderComp/free-remotion-templates", "src/components/text-mask-reveal/TextMaskReveal.tsx", "TextMaskReveal.tsx"),
    # R10: GTKottman/remotion-transitions (MIT)
    ("R10", "remotion-transitions", "GTKottman/remotion-transitions", "LICENSE", "LICENSE"),
    ("R10", "remotion-transitions", "GTKottman/remotion-transitions", "transitions/07-morph/shape/frame-morph/index.tsx", "frame-morph.tsx"),
    ("R10", "remotion-transitions", "GTKottman/remotion-transitions", "transitions/08-carry/line-led/line-led/index.tsx", "line-led.tsx"),
    ("R10", "remotion-transitions", "GTKottman/remotion-transitions", "transitions/06-distortion/break/paper-tear/index.tsx", "paper-tear.tsx"),
    # R04: lifeprompt-team/remotion-scenes (MIT)
    ("R04", "remotion-scenes", "lifeprompt-team/remotion-scenes", "LICENSE", "LICENSE"),
    ("R04", "remotion-scenes", "lifeprompt-team/remotion-scenes", "src/scenes/DataAnimations/DataBarChart.tsx", "DataBarChart.tsx"),
    ("R04", "remotion-scenes", "lifeprompt-team/remotion-scenes", "src/scenes/TextAnimations/TextKinetic.tsx", "TextKinetic.tsx"),
]


def main() -> None:
    survey = {r["code"]: r for r in json.loads((STAGE / "github_survey_verified.json").read_text(encoding="utf-8"))}
    manifest = []
    for code, slug, repo, upstream_path, local_name in TARGETS:
        sha = survey[code]["head_sha"]
        url = f"https://raw.githubusercontent.com/{repo}/{sha}/{upstream_path}"
        dst = OUT_RAW / slug / local_name
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "MotionLab-R5-A2/1.0"})
            with urllib.request.urlopen(req, timeout=20) as r:
                raw = r.read()
            dst.write_bytes(raw)
            h = hashlib.sha256(raw).hexdigest()
            manifest.append({
                "code": code,
                "origin_slug": slug,
                "repo": repo,
                "sha": sha,
                "upstream_path": upstream_path,
                "local_raw_path": str(dst.relative_to(ROOT)).replace("\\", "/"),
                "bytes": len(raw),
                "sha256": h,
                "status": "FETCHED",
            })
            print(f"OK   {code} {slug:26s} {local_name:26s} ({len(raw):6d} B, sha={h[:10]})", flush=True)
        except Exception as exc:
            manifest.append({
                "code": code,
                "origin_slug": slug,
                "repo": repo,
                "sha": sha,
                "upstream_path": upstream_path,
                "status": f"ERR:{exc}",
            })
            print(f"FAIL {code} {slug:26s} {upstream_path}: {exc}", flush=True)

    (STAGE / "remotion_raw_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

