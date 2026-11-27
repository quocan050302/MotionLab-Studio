"""Survey P0/P1 upstream Remotion and HyperFrames repositories via GitHub API (metadata + license + SHA)."""

from __future__ import annotations

import json
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / ".cache" / "repo_staging"
OUT_DIR.mkdir(parents=True, exist_ok=True)

REPOS = [
    ("H01", "P0", "hyperframes", "heygen-com/hyperframes"),
    ("H02", "P0", "hyperframes", "imjszhang/awesome-code-video"),
    ("H03", "P0", "hyperframes", "chenyuxiaojin/cyxj-hyperframes"),
    ("H04", "P0", "hyperframes", "alvin-ojoy/hyperframes"),
    ("H05", "P0", "hyperframes", "coleam00/hyperframes-ai-video-generation"),
    ("R01", "P0", "remotion", "Remocn/remocn"),
    ("R02", "P0", "remotion", "riaz37/remotion-ui"),
    ("R03", "P0", "remotion", "GovinduSasanka/opus-motion-library"),
    ("R04", "P0", "remotion", "lifeprompt-team/remotion-scenes"),
    ("R05", "P0", "remotion", "av/remotion-bits"),
    ("R06", "P0", "remotion", "vivmagarwal/remotion-effects-library"),
    ("R07", "P0", "remotion", "reactvideoeditor/remotion-templates"),
    ("R08", "P0", "remotion", "RenderComp/free-remotion-templates"),
    ("R09", "P0", "remotion", "degueba/onda"),
    ("R10", "P0", "remotion", "GTKottman/remotion-transitions"),
    ("R11", "P1", "remotion", "ali-abassi/remotion-templates"),
    ("R12", "P1", "remotion", "Thedurancode/locomotion-templates"),
    ("R15", "P1", "remotion", "remotion-dev/skills"),
    ("R16", "P1", "remotion", "remotion-dev/remotion"),
    ("S02", "P1", "shader", "gl-transitions/gl-transitions"),
]


def gh_json(url: str) -> dict | None:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "MotionLab-R5-Survey/1.0", "Accept": "application/vnd.github+json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return {"_error": f"HTTP {exc.code}: {exc.reason}"}
    except Exception as exc:
        return {"_error": str(exc)}


def git_ls_remote_head(slug: str) -> tuple[str | None, str | None]:
    url = f"https://github.com/{slug}.git"
    try:
        proc = subprocess.run(
            ["git", "ls-remote", url, "HEAD"],
            capture_output=True, text=True, timeout=15,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            parts = proc.stdout.strip().splitlines()[0].split()
            return parts[0], "ok"
        return None, (proc.stderr or "empty").strip()[:120]
    except Exception as exc:
        return None, str(exc)[:120]


def main() -> None:
    results = []
    for code, tier, engine, slug in REPOS:
        head_sha, ls_status = git_ls_remote_head(slug)
        meta = gh_json(f"https://api.github.com/repos/{slug}")
        row = {
            "code": code,
            "tier": tier,
            "engine": engine,
            "repo": slug,
            "url": f"https://github.com/{slug}",
            "head_sha": head_sha,
            "ls_remote_status": ls_status,
        }
        if meta and "_error" not in meta:
            lic = meta.get("license") or {}
            row.update({
                "accessible": True,
                "default_branch": meta.get("default_branch"),
                "spdx_license": lic.get("spdx_id"),
                "license_name": lic.get("name"),
                "stars": meta.get("stargazers_count"),
                "description": meta.get("description"),
                "pushed_at": meta.get("pushed_at"),
                "archived": meta.get("archived", False),
            })
        else:
            row.update({
                "accessible": head_sha is not None,
                "api_note": (meta or {}).get("_error", "unknown"),
            })
        results.append(row)
        print(f"{code:4s} {slug:44s} sha={str(head_sha or '-')[:10]} lic={row.get('spdx_license', row.get('api_note', '-'))}", flush=True)

    out_p = OUT_DIR / "github_survey_raw.json"
    out_p.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved {len(results)} survey records -> {out_p}")


if __name__ == "__main__":
    main()

