# /// script
# requires-python = ">=3.12"
# dependencies = ["senko", "numpy", "scipy"]
# ///
"""Evaluate speaker diarization and cross-recording voice matching on recordings that have a VTT.

Usage:  uv run speaker_eval.py DIR [--out REPORT_DIR] [--min-speech 20]

DIR is scanned recursively for media files with a .vtt transcript next to them
(same name, `<name>.transcript.vtt`, or the only .vtt in that folder).

Everything runs locally. The report contains NO names, NO file names, NO transcript text
and NO voice embeddings: people become P01.., recordings R01.., and only durations,
counts and similarity statistics are written. Read report.md before sharing it.
Needs ffmpeg on PATH.
"""

import argparse
import json
import platform
import re
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from importlib.metadata import version
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

MEDIA = {".mp4", ".m4a", ".mkv", ".webm", ".mov", ".mp3", ".wav", ".aac", ".ogg"}
FPS = 10  # analysis resolution: 0.1 s
TS = re.compile(r"((?:\d+:)?\d{2}:\d{2}[.,]\d{3})\s*-->\s*((?:\d+:)?\d{2}:\d{2}[.,]\d{3})")


def to_seconds(ts: str) -> float:
    parts = ts.replace(",", ".").split(":")
    return sum(float(p) * 60**i for i, p in enumerate(reversed(parts)))


def parse_vtt(path: Path) -> list[tuple[float, float, str]]:
    """Return (start, end, speaker) cues. Supports Zoom (`Name: text`) and Teams (`<v Name>text`)."""
    cues = []
    text = path.read_text(encoding="utf-8-sig", errors="replace").replace("\r\n", "\n")
    for block in re.split(r"\n\s*\n", text):
        lines = [ln for ln in block.strip().split("\n") if ln.strip()]
        for i, ln in enumerate(lines):
            m = TS.search(ln)
            if m:
                payload = " ".join(lines[i + 1 :]).strip()
                break
        else:
            continue
        name = None
        v = re.match(r"<v(?:\.[^\s>]*)?\s+([^>]+)>", payload)
        if v:
            name = v[1]
        else:
            z = re.match(r"([^:<>]{1,60}):\s", payload)
            if z:
                name = z[1]
        if name:
            cues.append((to_seconds(m[1]), to_seconds(m[2]), " ".join(name.casefold().split())))
    return cues


def find_pairs(root: Path) -> list[tuple[Path, Path]]:
    pairs = []
    for media in sorted(p for p in root.rglob("*") if p.suffix.lower() in MEDIA):
        candidates = [media.with_suffix(".vtt"), media.with_name(media.stem + ".transcript.vtt")]
        vtt = next((c for c in candidates if c.exists()), None)
        if vtt is None:
            folder_vtts = list(media.parent.glob("*.vtt"))
            folder_media = [p for p in media.parent.iterdir() if p.suffix.lower() in MEDIA]
            if len(folder_vtts) == 1 and len(folder_media) == 1:
                vtt = folder_vtts[0]
        if vtt:
            pairs.append((media, vtt))
    return pairs


def to_frames(segments, n: int, index: dict) -> np.ndarray:
    arr = np.full(n, -1, dtype=np.int32)
    for start, end, label in segments:
        arr[max(0, int(start * FPS)) : min(n, int(end * FPS))] = index[label]
    return arr


def agreement(ref: np.ndarray, hyp: np.ndarray, n_ref: int, n_hyp: int, mask=None):
    both = (ref >= 0) & (hyp >= 0)
    if mask is not None:
        both &= mask
    table = np.zeros((n_ref, n_hyp), dtype=np.int64)
    np.add.at(table, (ref[both], hyp[both]), 1)
    total = table.sum()
    if total == 0:
        return table, None, None, None
    rows, cols = linear_sum_assignment(-table)
    return (
        table,
        table[rows, cols].sum() / total,  # one-to-one mapping agreement
        table.max(axis=0).sum() / total,  # purity: clusters are single-person
        table.max(axis=1).sum() / total,  # coverage: persons are single-cluster
    )


def pct(x) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


def quantiles(values) -> str:
    if len(values) == 0:
        return "n=0"
    q = np.quantile(values, [0, 0.1, 0.5, 0.9, 1])
    return f"n={len(values)}  min {q[0]:.2f}  p10 {q[1]:.2f}  median {q[2]:.2f}  p90 {q[3]:.2f}  max {q[4]:.2f}"


def equal_error_rate(same, diff):
    best = None
    for thr in np.unique(np.concatenate([same, diff])):
        frr, far = np.mean(same < thr), np.mean(diff >= thr)
        if best is None or abs(frr - far) < best[0]:
            best = (abs(frr - far), (frr + far) / 2, float(thr))
    return best[1], best[2]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir", type=Path)
    ap.add_argument("--out", type=Path, default=Path("speaker_eval_report"))
    ap.add_argument("--min-speech", type=float, default=20.0, help="seconds a person/cluster needs to be used")
    args = ap.parse_args()

    pairs = find_pairs(args.dir)
    if not pairs:
        sys.exit(f"No media+vtt pairs found under {args.dir}")
    print(f"Found {len(pairs)} recording(s) with a VTT", file=sys.stderr)

    import senko

    diarizer = senko.Diarizer(device="auto", warmup=True, quiet=True)

    person_ids: dict[str, int] = {}  # normalized name -> global index; never written to disk
    person_total = defaultdict(float)
    recordings, voices = [], []  # voices: one centroid per (recording, person)

    for r, (media, vtt) in enumerate(pairs):
        print(f"[R{r + 1:02d}] processing…", file=sys.stderr)
        cues = parse_vtt(vtt)
        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp) / "audio.wav"
            subprocess.run(
                ["ffmpeg", "-v", "error", "-y", "-i", str(media), "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(wav)],
                check=True,
            )
            duration = wav.stat().st_size / 32000
            t0 = time.time()
            result = diarizer.diarize(str(wav), generate_colors=False)
            elapsed = time.time() - t0
        n = int(duration * FPS) + 1

        seconds = defaultdict(float)
        for s, e, name in cues:
            seconds[name] += e - s
        for name in sorted(seconds, key=seconds.get, reverse=True):
            person_ids.setdefault(name, len(person_ids))
            person_total[name] += seconds[name]
        local_people = sorted(seconds, key=seconds.get, reverse=True)
        p_index = {name: i for i, name in enumerate(local_people)}
        ref = to_frames(cues, n, p_index)

        segs = [(s["start"], s["end"], s["speaker"]) for s in result["merged_segments"]]
        clusters = sorted({s[2] for s in segs})
        c_index = {c: i for i, c in enumerate(clusters)}
        hyp = to_frames(segs, n, c_index)

        # VTT cue times are coarse; also score with 1 s around every cue boundary ignored.
        collar = np.ones(n, dtype=bool)
        for s, e, _ in cues:
            for b in (s, e):
                collar[max(0, int((b - 1) * FPS)) : int((b + 1) * FPS)] = False

        table, agree, purity, coverage = agreement(ref, hyp, len(local_people), len(clusters))
        _, agree_c, purity_c, coverage_c = agreement(ref, hyp, len(local_people), len(clusters), collar)

        # Label each cluster with the person it mostly overlaps; keep the biggest cluster per person.
        best_for_person = {}
        splits = defaultdict(int)
        for c, ci in c_index.items():
            col = table[:, ci]
            if col.sum() / FPS < args.min_speech:
                continue
            pi = int(col.argmax())
            if col[pi] / col.sum() < 0.6:
                continue
            splits[pi] += 1
            if pi not in best_for_person or col[pi] > best_for_person[pi][1]:
                best_for_person[pi] = (c, col[pi])
        centroids = result.get("speaker_centroids", {})
        for pi, (c, frames) in best_for_person.items():
            if c in centroids:
                vec = np.asarray(centroids[c], dtype=np.float64)
                voices.append({"rec": r, "person": local_people[pi], "vec": vec / np.linalg.norm(vec), "sec": frames / FPS})

        recordings.append(
            {
                "id": f"R{r + 1:02d}",
                "minutes": round(duration / 60, 1),
                "diarize_seconds": round(elapsed, 1),
                "vtt_speakers": len(local_people),
                "vtt_speakers_min_speech": sum(1 for p in local_people if seconds[p] >= args.min_speech),
                "detected_speakers": len(clusters),
                "vtt_speech_share": round(float((ref >= 0).mean()), 3),
                "detected_speech_share": round(float((hyp >= 0).mean()), 3),
                "agreement": agree, "purity": purity, "coverage": coverage,
                "agreement_collar": agree_c, "purity_collar": purity_c, "coverage_collar": coverage_c,
                "persons_split_into_several_clusters": sum(1 for v in splits.values() if v > 1),
                "_people": local_people, "_clusters": clusters, "_table": table,
            }
        )

    # Global anonymous ids, ordered by total speech across all recordings.
    order = sorted(person_ids, key=person_total.get, reverse=True)
    anon = {name: f"P{i + 1:02d}" for i, name in enumerate(order)}

    lines = ["# Speaker evaluation report", ""]
    lines += [
        f"- engine: senko {version('senko')}, python {platform.python_version()}, {platform.system()} {platform.machine()}",
        f"- recordings: {len(recordings)}, total {sum(r['minutes'] for r in recordings) / 60:.1f} h, distinct VTT speakers: {len(order)}",
        f"- min speech to count a person/cluster: {args.min_speech:.0f} s; analysis step {1 / FPS:.1f} s",
        "- contains no names, file names, transcript text or voice embeddings",
        "",
        "## Diarization vs VTT",
        "",
        "agreement = same speaker after best one-to-one mapping; purity = clusters hold one person; coverage = a person stays in one cluster.",
        "`±1s` columns ignore 1 s around every VTT cue boundary (VTT times are coarse).",
        "",
        "| rec | min | diarize s | VTT spk (≥min) | detected | agreement | purity | coverage | agreement ±1s | purity ±1s | coverage ±1s | split persons |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in recordings:
        lines.append(
            f"| {r['id']} | {r['minutes']} | {r['diarize_seconds']} | {r['vtt_speakers']} ({r['vtt_speakers_min_speech']}) | {r['detected_speakers']} | "
            f"{pct(r['agreement'])} | {pct(r['purity'])} | {pct(r['coverage'])} | {pct(r['agreement_collar'])} | {pct(r['purity_collar'])} | "
            f"{pct(r['coverage_collar'])} | {r['persons_split_into_several_clusters']} |"
        )
    lines += ["", "### Overlap tables (minutes; rows = VTT person, columns = detected cluster)", ""]
    for r in recordings:
        people, clusters, table = r.pop("_people"), r.pop("_clusters"), r.pop("_table")
        lines += [f"**{r['id']}**", "", "| | " + " | ".join(f"C{i + 1}" for i in range(len(clusters))) + " |", "|---" * (len(clusters) + 1) + "|"]
        for i, name in enumerate(people):
            if table[i].sum() / FPS >= 5:
                lines.append(f"| {anon[name]} | " + " | ".join(f"{v / FPS / 60:.1f}" if v else "" for v in table[i]) + " |")
        lines.append("")

    # Cross-recording voice matching.
    same, diff, within = [], [], []
    for i in range(len(voices)):
        for j in range(i + 1, len(voices)):
            a, b = voices[i], voices[j]
            sim = float(a["vec"] @ b["vec"])
            if a["rec"] == b["rec"]:
                within.append(sim)
            elif a["person"] == b["person"]:
                same.append(sim)
            else:
                diff.append(sim)
    recs_per_person = defaultdict(set)
    for v in voices:
        recs_per_person[v["person"]].add(v["rec"])
    repeat = sum(1 for s in recs_per_person.values() if len(s) > 1)

    lines += [
        "## Same person across recordings (cosine similarity of voice centroids)",
        "",
        f"- people with a usable voice sample: {len(recs_per_person)}; of them in 2+ recordings: {repeat}",
        f"- same person, different recordings: {quantiles(same)}",
        f"- different people, different recordings: {quantiles(diff)}",
        f"- different people, same recording: {quantiles(within)}",
    ]
    cross = {"same": same, "different": diff, "within_recording_different": within}
    ident = []
    if same and diff:
        eer, thr = equal_error_rate(np.array(same), np.array(diff))
        lines.append(f"- equal error rate: {100 * eer:.1f}% at threshold {thr:.2f}")
        cross |= {"eer": eer, "eer_threshold": thr}
        lines += [
            "",
            "### Identification simulation (library = all other recordings)",
            "",
            "| threshold | known: correct | known: wrong name | known: not recognized | unknown: correctly left anonymous | unknown: given a name |",
            "|---|---|---|---|---|---|",
        ]
        for thr in np.arange(0.50, 0.96, 0.05):
            c = defaultdict(int)
            for q in voices:
                lib = defaultdict(list)
                for v in voices:
                    if v["rec"] != q["rec"]:
                        lib[v["person"]].append(v["vec"])
                if not lib:
                    continue
                scores = {p: float(q["vec"] @ (np.mean(vs, axis=0) / np.linalg.norm(np.mean(vs, axis=0)))) for p, vs in lib.items()}
                best = max(scores, key=scores.get)
                hit = scores[best] >= thr
                if q["person"] in lib:
                    c["correct" if hit and best == q["person"] else "wrong" if hit else "missed"] += 1
                else:
                    c["false_accept" if hit else "reject"] += 1
            ident.append({"threshold": round(float(thr), 2), **c})
            lines.append(f"| {thr:.2f} | {c['correct']} | {c['wrong']} | {c['missed']} | {c['reject']} | {c['false_accept']} |")
    else:
        lines.append("- not enough people shared between recordings to measure matching")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (args.out / "report.json").write_text(
        json.dumps({"recordings": recordings, "cross_recording": cross, "identification": ident}, indent=1, default=float), encoding="utf-8"
    )
    print(args.out / "report.md")


if __name__ == "__main__":
    main()
