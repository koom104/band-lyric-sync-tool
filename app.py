from __future__ import annotations

import hashlib
import html
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import wave
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable

import gradio as gr
import numpy as np


os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).resolve().parent
JOBS = ROOT / "jobs"
JOBS.mkdir(exist_ok=True)


@dataclass
class CaptionLine:
    start: float
    end: float
    text: str
    score: float = 1.0


@dataclass
class WhisperSegment:
    start: float
    end: float
    text: str


@dataclass
class LyricBlock:
    text: str
    sync_text: str


@dataclass
class TimedLyric:
    start: float
    end: float
    text: str


def run(cmd: list[str], cwd: Path | None = None) -> None:
    env = os.environ.copy()
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed:\n{' '.join(cmd)}\n\n{proc.stdout[-4000:]}")


def ffprobe_duration(media: Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(media),
    ]
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr)
    return max(0.1, float(proc.stdout.strip()))


def extract_audio(video: Path, wav: Path) -> None:
    run(["ffmpeg", "-y", "-i", str(video), "-vn", "-ac", "1", "-ar", "16000", str(wav)])


def extract_alignment_audio(media: Path, wav: Path) -> None:
    wav.parent.mkdir(parents=True, exist_ok=True)
    run(["ffmpeg", "-y", "-i", str(media), "-vn", "-ac", "1", "-ar", "22050", "-sample_fmt", "s16", str(wav)])


def download_youtube_reference(url: str, work_dir: Path) -> Path:
    if not url.strip():
        raise gr.Error("Reference YouTube URL is empty.")
    work_dir.mkdir(parents=True, exist_ok=True)
    output_template = str(work_dir / "reference_download.%(ext)s")
    run([
        os.sys.executable,
        "-m",
        "yt_dlp",
        "--no-playlist",
        "-f",
        "bestaudio/best",
        "-o",
        output_template,
        url.strip(),
    ])
    candidates = [p for p in work_dir.glob("reference_download.*") if p.is_file()]
    if not candidates:
        raise gr.Error("Reference audio download finished, but no file was found.")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def prepare_reference_audio(reference_file: object, reference_youtube_url: str, work_dir: Path) -> tuple[Path, str]:
    reference_path = resolve_uploaded_path(reference_file)
    source_note = ""
    if reference_path and reference_path.exists():
        source = reference_path
        source_note = f"reference file: {source.name}"
    elif reference_youtube_url and reference_youtube_url.strip():
        source = download_youtube_reference(reference_youtube_url, work_dir / "youtube_reference")
        source_note = "reference YouTube URL"
    else:
        raise gr.Error("Reference audio DTW 모드에서는 reference audio file 또는 reference YouTube URL이 필요합니다.")
    reference_wav = work_dir / "reference_22050.wav"
    extract_alignment_audio(source, reference_wav)
    return reference_wav, source_note


def separate_vocals(audio: Path, work_dir: Path) -> Path:
    out_dir = work_dir / "demucs"
    run([
        os.sys.executable,
        "-m",
        "demucs.separate",
        "--two-stems",
        "vocals",
        "-n",
        "htdemucs",
        "-o",
        str(out_dir),
        str(audio),
    ])
    candidates = list(out_dir.glob("**/vocals.wav"))
    if not candidates:
        raise RuntimeError("Demucs finished, but vocals.wav was not found.")
    return candidates[0]


def parse_lrc(lyrics: str, duration: float) -> list[CaptionLine] | None:
    items = parse_lrc_items(lyrics, duration)
    if not items:
        return None
    return [CaptionLine(item.start, item.end, item.text) for item in items]


def parse_lrc_items(lyrics: str, duration: float) -> list[TimedLyric]:
    pattern = re.compile(r"\[(\d{1,2}):(\d{2})(?:[.:](\d{1,3}))?\]\s*(.*)")
    events: list[tuple[float, str]] = []
    for raw in lyrics.splitlines():
        match = pattern.match(raw.strip())
        if not match:
            continue
        minutes = int(match.group(1))
        seconds = int(match.group(2))
        fraction_raw = match.group(3) or "0"
        fraction = int(fraction_raw) / (10 ** len(fraction_raw))
        text = match.group(4).strip()
        # Empty rows are meaningful boundaries before instrumental sections.
        events.append((minutes * 60 + seconds + fraction, text))
    items: list[TimedLyric] = []
    for idx, (start, text) in enumerate(events):
        if not text:
            continue
        next_times = [event_time for event_time, _ in events[idx + 1 :] if event_time > start]
        if next_times:
            end = min(duration, max(start + 0.2, next_times[0] - 0.05))
        else:
            end = min(duration, start + 3.5)
        items.append(TimedLyric(start=start, end=end, text=text))
    return items


def count_lrc_lines(lrc_text: str) -> int:
    return len(parse_lrc_items(lrc_text, 24 * 3600))


def fetch_lrclib_synced_lyrics(artist: str, song_title: str, duration: float, expected_lines: int | None = None) -> tuple[str | None, str]:
    if not artist.strip() or not song_title.strip():
        return None, "LRCLIB lookup skipped: artist or song title is empty."
    attempts = [
        {
            "artist_name": artist.strip(),
            "track_name": song_title.strip(),
            "duration": str(int(round(duration))),
        },
        {
            "artist_name": artist.strip(),
            "track_name": song_title.strip(),
        },
    ]
    last_status = "LRCLIB synced lyrics not found."
    best_synced: str | None = None
    best_status = last_status
    best_distance = 10**9
    for params in attempts:
        query = urllib.parse.urlencode(params)
        url = f"https://lrclib.net/api/get?{query}"
        req = urllib.request.Request(url, headers={"User-Agent": "band-lyric-sync-tool/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=12) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                last_status = "LRCLIB synced lyrics not found."
                continue
            return None, f"LRCLIB lookup failed: HTTP {exc.code}"
        except Exception as exc:
            return None, f"LRCLIB lookup failed: {exc}"
        synced = data.get("syncedLyrics")
        if synced and synced.strip():
            track = data.get("trackName") or song_title
            found_artist = data.get("artistName") or artist
            if expected_lines is None:
                return synced, f"LRCLIB synced lyrics found: {found_artist} - {track}"
            distance = abs(count_lrc_lines(synced) - expected_lines)
            if distance < best_distance:
                best_synced = synced
                best_status = f"LRCLIB synced lyrics found: {found_artist} - {track}"
                best_distance = distance
        last_status = "LRCLIB result did not include synced lyrics."

    search_query = urllib.parse.urlencode({"artist_name": artist.strip(), "track_name": song_title.strip()})
    search_url = f"https://lrclib.net/api/search?{search_query}"
    try:
        req = urllib.request.Request(search_url, headers={"User-Agent": "band-lyric-sync-tool/1.0"})
        with urllib.request.urlopen(req, timeout=12) as response:
            records = json.loads(response.read().decode("utf-8"))
    except Exception:
        records = []
    for record in records if isinstance(records, list) else []:
        synced = record.get("syncedLyrics")
        if not synced or not synced.strip():
            continue
        track = record.get("trackName") or song_title
        found_artist = record.get("artistName") or artist
        distance = abs(count_lrc_lines(synced) - expected_lines) if expected_lines is not None else 0
        if distance < best_distance:
            best_synced = synced
            best_status = f"LRCLIB synced lyrics found: {found_artist} - {track}"
            best_distance = distance

    if best_synced:
        if expected_lines is not None:
            best_status += f" / LRC lines {count_lrc_lines(best_synced)}, input blocks {expected_lines}"
        return best_synced, best_status
    return None, last_status


def choose_sync_text(display_text: str, sync_source: str) -> str:
    lines = [line.strip() for line in display_text.split("\n") if line.strip()]
    if not lines:
        return display_text
    if sync_source == "Line 1":
        return lines[0]
    if sync_source == "Line 2":
        return lines[min(1, len(lines) - 1)]
    if sync_source == "Line 3":
        return lines[min(2, len(lines) - 1)]
    if sync_source == "Last line":
        return lines[-1]
    return " ".join(lines)


def split_lyric_blocks(lyrics: str, grouping: str, sync_source: str) -> list[LyricBlock]:
    cleaned = re.sub(r"\[[^\]]+\]", "", lyrics).replace("\r\n", "\n").replace("\r", "\n")
    if grouping == "Every non-empty line":
        texts = [line.strip() for line in cleaned.split("\n") if line.strip()]
        return [LyricBlock(text=text, sync_text=choose_sync_text(text, sync_source)) for text in texts]

    blocks = []
    for block in re.split(r"\n\s*\n+", cleaned):
        lines = [line.strip() for line in block.split("\n") if line.strip()]
        if lines:
            text = "\n".join(lines)
            blocks.append(LyricBlock(text=text, sync_text=choose_sync_text(text, sync_source)))
    return blocks


def split_lyrics(lyrics: str, grouping: str) -> list[str]:
    return [block.text for block in split_lyric_blocks(lyrics, grouping, "All lines")]


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^0-9a-z\u3040-\u30ff\u3400-\u9fff\uac00-\ud7a3\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_for_lyric_match(text: str) -> str:
    text = text.lower()
    # Keep CJK/kana/hangul/latin/numbers and remove punctuation. This works better for Japanese LRC.
    text = re.sub(r"[^\u3040-\u30ff\u3400-\u9fff\uac00-\ud7a30-9a-z]+", "", text)
    return text


def similarity(left: str, right: str) -> float:
    left_norm = normalize_for_lyric_match(left) or normalize_text(left)
    right_norm = normalize_for_lyric_match(right) or normalize_text(right)
    if not left_norm or not right_norm:
        return 0.0
    return SequenceMatcher(None, left_norm, right_norm).ratio()


def transcribe(audio: Path, model_size: str, language: str) -> list[WhisperSegment]:
    from faster_whisper import WhisperModel

    compute_type = "float16"
    device = "cuda"
    try:
        model = WhisperModel(model_size, device=device, compute_type=compute_type)
    except Exception:
        model = WhisperModel(model_size, device="cpu", compute_type="int8")

    segments, _info = model.transcribe(
        str(audio),
        language=language if language != "auto" else None,
        vad_filter=True,
        beam_size=5,
        word_timestamps=False,
        condition_on_previous_text=True,
    )
    result = [
        WhisperSegment(start=float(seg.start), end=float(seg.end), text=seg.text.strip())
        for seg in segments
        if seg.text and seg.text.strip()
    ]
    return result


def distribute_blocks(blocks: list[LyricBlock], duration: float) -> list[CaptionLine]:
    if not blocks:
        return []
    usable_start = 0.5
    usable_end = max(usable_start + 1.0, duration - 0.5)
    step = (usable_end - usable_start) / len(blocks)
    captions = []
    for idx, block in enumerate(blocks):
        start = usable_start + idx * step
        next_start = usable_start + (idx + 1) * step if idx + 1 < len(blocks) else usable_end
        end = min(usable_end, max(start + 0.2, next_start - 0.05))
        captions.append(CaptionLine(start, end, block.text, 0.0))
    return captions


def distribute_lines(lines: list[str], duration: float) -> list[CaptionLine]:
    return distribute_blocks([LyricBlock(text=line, sync_text=line) for line in lines], duration)


def align_blocks_to_segments(blocks: list[LyricBlock], segments: list[WhisperSegment], duration: float) -> list[CaptionLine]:
    if not blocks:
        return []
    if not segments:
        return distribute_blocks(blocks, duration)

    # Dynamic programming over monotonic lyric-line to whisper-segment spans.
    n = len(blocks)
    m = len(segments)
    max_span = 6
    penalty_unmatched = -0.18
    dp = [[float("-inf")] * (m + 1) for _ in range(n + 1)]
    back: list[list[tuple[int, int] | None]] = [[None] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0.0

    for i in range(n):
        for j in range(m + 1):
            if dp[i][j] == float("-inf"):
                continue
            # Allow one lyric line to be inferred between recognized regions.
            if dp[i][j] + penalty_unmatched > dp[i + 1][j]:
                dp[i + 1][j] = dp[i][j] + penalty_unmatched
                back[i + 1][j] = (j, j)
            for k in range(j + 1, min(m, j + max_span) + 1):
                span_text = " ".join(seg.text for seg in segments[j:k])
                span_seconds = max(0.1, segments[k - 1].end - segments[j].start)
                duration_penalty = min(0.18, abs(span_seconds - 3.2) * 0.015)
                score = similarity(blocks[i].sync_text, span_text) - duration_penalty
                if dp[i][j] + score > dp[i + 1][k]:
                    dp[i + 1][k] = dp[i][j] + score
                    back[i + 1][k] = (j, k)

    end_j = max(range(m + 1), key=lambda j: dp[n][j] - 0.03 * (m - j))
    spans: list[tuple[int, int]] = []
    i, j = n, end_j
    while i > 0:
        prev = back[i][j]
        if prev is None:
            spans.append((j, j))
            i -= 1
            continue
        spans.append(prev)
        j = prev[0]
        i -= 1
    spans.reverse()

    captions: list[CaptionLine] = []
    last_end = 0.0
    for idx, (start_idx, end_idx) in enumerate(spans):
        block = blocks[idx]
        if start_idx < end_idx:
            start = max(0.0, segments[start_idx].start)
            end = min(duration, segments[end_idx - 1].end)
            score = similarity(block.sync_text, " ".join(seg.text for seg in segments[start_idx:end_idx]))
        else:
            prev_end = captions[-1].end if captions else 0.5
            next_start = duration - 0.5
            for seg_idx in range(start_idx, len(segments)):
                if segments[seg_idx].start > prev_end:
                    next_start = segments[seg_idx].start
                    break
            gap = max(1.2, next_start - prev_end)
            start = prev_end
            end = min(duration, start + min(2.8, gap))
            score = 0.0
        if start < last_end:
            start = last_end + 0.03
        if end <= start:
            end = min(duration, start + 1.4)
        last_end = end
        captions.append(CaptionLine(start=start, end=end, text=block.text, score=score))
    return captions


def align_lines_to_segments(lines: list[str], segments: list[WhisperSegment], duration: float) -> list[CaptionLine]:
    return align_blocks_to_segments([LyricBlock(text=line, sync_text=line) for line in lines], segments, duration)


def active_time_at(intervals: list[tuple[float, float]], fraction: float) -> float:
    total = sum(end - start for start, end in intervals)
    if total <= 0:
        return intervals[0][0] if intervals else 0.0
    target = min(max(fraction, 0.0), 1.0) * total
    elapsed = 0.0
    for start, end in intervals:
        span = end - start
        if elapsed + span >= target:
            return start + max(0.0, target - elapsed)
        elapsed += span
    return intervals[-1][1]


def detect_vocal_activity(audio: Path, work_dir: Path, media_duration: float) -> list[tuple[float, float]]:
    work_dir.mkdir(parents=True, exist_ok=True)
    analysis_wav = work_dir / "activity_mono.wav"
    run(["ffmpeg", "-y", "-i", str(audio), "-vn", "-ac", "1", "-ar", "16000", "-sample_fmt", "s16", str(analysis_wav)])
    with wave.open(str(analysis_wav), "rb") as wav:
        sample_rate = wav.getframerate()
        samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype=np.int16).astype(np.float32)
    if samples.size == 0:
        return []
    samples /= 32768.0
    win = max(1, int(sample_rate * 0.20))
    hop = max(1, int(sample_rate * 0.10))
    rms = []
    times = []
    for start in range(0, max(1, samples.size - win + 1), hop):
        chunk = samples[start : start + win]
        rms.append(float(np.sqrt(np.mean(chunk * chunk) + 1e-12)))
        times.append(start / sample_rate)
    if not rms:
        return []
    values = np.array(rms)
    if values.size >= 5:
        kernel = np.ones(5, dtype=np.float32) / 5.0
        values = np.convolve(values, kernel, mode="same")
    floor = float(np.percentile(values, 35))
    high = float(np.percentile(values, 92))
    threshold = max(floor * 1.8, floor + (high - floor) * 0.22, 0.006)
    active = values >= threshold

    intervals: list[tuple[float, float]] = []
    current_start: float | None = None
    for idx, is_active in enumerate(active):
        t = times[idx]
        if is_active and current_start is None:
            current_start = t
        elif not is_active and current_start is not None:
            intervals.append((current_start, t + 0.2))
            current_start = None
    if current_start is not None:
        intervals.append((current_start, min(media_duration, times[-1] + 0.2)))

    merged: list[tuple[float, float]] = []
    for start, end in intervals:
        start = max(0.0, start - 0.12)
        end = min(media_duration, end + 0.20)
        if end - start < 0.35:
            continue
        if merged and start - merged[-1][1] <= 0.75:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def align_blocks_by_activity(blocks: list[LyricBlock], audio: Path, work_dir: Path, duration: float) -> list[CaptionLine]:
    if not blocks:
        return []
    intervals = detect_vocal_activity(audio, work_dir, duration)
    if not intervals:
        return distribute_blocks(blocks, duration)
    captions: list[CaptionLine] = []
    last_end = 0.0
    for idx, block in enumerate(blocks):
        start = active_time_at(intervals, idx / len(blocks))
        end = active_time_at(intervals, (idx + 1) / len(blocks))
        if idx + 1 < len(blocks):
            end = min(end, active_time_at(intervals, (idx + 1) / len(blocks)) - 0.04)
        if start < last_end:
            start = last_end + 0.03
        end = max(end, start + 0.8)
        end = min(duration, end)
        if end <= start:
            end = min(duration, start + 0.2)
        last_end = end
        captions.append(CaptionLine(max(0.0, start), end, block.text, 0.0))
    return captions


def apply_global_offset(captions: list[CaptionLine], offset_seconds: float, duration: float) -> list[CaptionLine]:
    if abs(offset_seconds) < 0.001:
        return captions
    shifted = []
    for caption in captions:
        start = min(max(0.0, caption.start + offset_seconds), duration)
        end = min(max(start + 0.2, caption.end + offset_seconds), duration)
        shifted.append(CaptionLine(start, end, caption.text, caption.score))
    return shifted


def warp_captions_to_span(
    captions: list[CaptionLine],
    source_start: float,
    source_end: float,
    target_start: float,
    target_end: float,
    duration: float,
) -> list[CaptionLine]:
    source_span = max(0.1, source_end - source_start)
    target_span = max(0.1, target_end - target_start)
    warped: list[CaptionLine] = []
    for caption in captions:
        start = target_start + (caption.start - source_start) / source_span * target_span
        end = target_start + (caption.end - source_start) / source_span * target_span
        start = min(max(0.0, start), duration)
        end = min(max(start + 0.2, end), duration)
        warped.append(CaptionLine(start, end, caption.text, caption.score))
    return warped


def auto_fit_captions_to_vocal_span(
    captions: list[CaptionLine],
    audio: Path,
    work_dir: Path,
    duration: float,
    fit_mode: str,
) -> tuple[list[CaptionLine], str]:
    if len(captions) < 2:
        return captions, "auto-fit skipped: not enough captions"
    intervals = detect_vocal_activity(audio, work_dir / "auto_fit", duration)
    if not intervals:
        return captions, "auto-fit skipped: no vocal activity detected"
    source_start = captions[0].start
    source_end = captions[-1].end
    target_start = source_start if fit_mode == "Keep LRC start, fit end" else intervals[0][0]
    target_end = intervals[-1][1]
    if source_end - source_start < 1 or target_end - target_start < 1:
        return captions, "auto-fit skipped: invalid timing span"
    fitted = warp_captions_to_span(captions, source_start, source_end, target_start, target_end, duration)
    ratio = (target_end - target_start) / max(0.1, source_end - source_start)
    return fitted, f"auto-fit {fit_mode}: {target_start:.2f}s-{target_end:.2f}s, scale {ratio:.3f}x"


def build_dtw_time_map(reference_audio: Path, performance_audio: Path, work_dir: Path) -> tuple[np.ndarray, np.ndarray, str]:
    import librosa

    work_dir.mkdir(parents=True, exist_ok=True)
    sample_rate = 22050
    hop_length = 2048
    y_ref, _ = librosa.load(str(reference_audio), sr=sample_rate, mono=True)
    y_perf, _ = librosa.load(str(performance_audio), sr=sample_rate, mono=True)
    if y_ref.size < sample_rate or y_perf.size < sample_rate:
        raise gr.Error("Reference audio DTW failed: audio is too short.")

    ref_chroma = librosa.feature.chroma_cens(y=y_ref, sr=sample_rate, hop_length=hop_length)
    perf_chroma = librosa.feature.chroma_cens(y=y_perf, sr=sample_rate, hop_length=hop_length)
    ref_chroma = ref_chroma / np.maximum(1e-6, np.linalg.norm(ref_chroma, axis=0, keepdims=True))
    perf_chroma = perf_chroma / np.maximum(1e-6, np.linalg.norm(perf_chroma, axis=0, keepdims=True))

    # Repeated choruses can make unconstrained DTW jump to a later chorus and
    # then jump back. Allow gradual tempo changes, but require both recordings
    # to keep moving forward and keep the path near the song-length diagonal.
    length_delta = abs(ref_chroma.shape[1] - perf_chroma.shape[1]) / max(ref_chroma.shape[1], perf_chroma.shape[1])
    band_radius = min(0.20, max(0.08, length_delta + 0.04))
    step_sizes = np.asarray([[1, 1], [1, 2], [2, 1]], dtype=np.uint32)
    _cost, path = librosa.sequence.dtw(
        X=ref_chroma,
        Y=perf_chroma,
        metric="cosine",
        subseq=False,
        backtrack=True,
        global_constraints=True,
        band_rad=band_radius,
        step_sizes_sigma=step_sizes,
        weights_add=np.asarray([0.0, 0.08, 0.08]),
    )
    path = np.asarray(path[::-1])
    ref_times = librosa.frames_to_time(path[:, 0], sr=sample_rate, hop_length=hop_length)
    perf_times = librosa.frames_to_time(path[:, 1], sr=sample_rate, hop_length=hop_length)

    order = np.argsort(ref_times)
    ref_times = ref_times[order]
    perf_times = perf_times[order]
    unique_ref = []
    mapped_perf = []
    for ref_time in np.unique(ref_times):
        values = perf_times[ref_times == ref_time]
        unique_ref.append(ref_time)
        mapped_perf.append(float(np.median(values)))
    ref_axis = np.asarray(unique_ref, dtype=np.float64)
    perf_axis = np.maximum.accumulate(np.asarray(mapped_perf, dtype=np.float64))
    offsets = perf_axis - ref_axis
    offset_low, offset_mid, offset_high = np.percentile(offsets, [5, 50, 95])
    status = (
        f"DTW path frames {len(path)}, reference {ref_axis[-1]:.2f}s, "
        f"performance {perf_axis[-1]:.2f}s, offset median {offset_mid:+.2f}s "
        f"(5-95% {offset_low:+.2f}s to {offset_high:+.2f}s)"
    )
    return ref_axis, perf_axis, status


def map_reference_time(seconds: float, ref_axis: np.ndarray, perf_axis: np.ndarray, performance_duration: float) -> float:
    if ref_axis.size == 0:
        return min(max(0.0, seconds), performance_duration)
    mapped = float(np.interp(seconds, ref_axis, perf_axis, left=perf_axis[0], right=perf_axis[-1]))
    return min(max(0.0, mapped), performance_duration)


def warp_captions_with_dtw(
    captions: list[CaptionLine],
    reference_audio: Path,
    performance_audio: Path,
    work_dir: Path,
    performance_duration: float,
) -> tuple[list[CaptionLine], str]:
    ref_axis, perf_axis, status = build_dtw_time_map(reference_audio, performance_audio, work_dir / "dtw")
    raw_starts: list[float] = []
    raw_ends: list[float] = []
    for caption in captions:
        raw_starts.append(map_reference_time(caption.start, ref_axis, perf_axis, performance_duration))
        raw_ends.append(map_reference_time(caption.end, ref_axis, perf_axis, performance_duration))

    starts: list[float] = []
    for raw_start in raw_starts:
        start = raw_start if not starts else max(raw_start, starts[-1] + 0.12)
        starts.append(min(start, performance_duration))

    warped: list[CaptionLine] = []
    for idx, caption in enumerate(captions):
        start = starts[idx]
        end = min(performance_duration, max(start + 0.20, raw_ends[idx]))
        if idx + 1 < len(starts):
            end = min(end, starts[idx + 1] - 0.04)
        if end <= start:
            raise gr.Error(f"DTW produced an invalid subtitle interval at block {idx + 1}.")
        warped.append(CaptionLine(start, end, caption.text, caption.score))
    return warped, status


def validate_caption_sequence(captions: list[CaptionLine], expected_count: int) -> None:
    if len(captions) != expected_count:
        raise gr.Error(
            f"입력 가사 블록은 {expected_count}개지만 생성된 자막은 {len(captions)}개입니다. "
            "입력 블록을 잃지 않도록 생성을 중단했습니다."
        )
    for idx, caption in enumerate(captions):
        if caption.end <= caption.start:
            raise gr.Error(f"{idx + 1}번 자막의 종료 시각이 시작 시각보다 빠릅니다.")
        if idx and caption.start < captions[idx - 1].end - 1e-6:
            raise gr.Error(
                f"{idx}번과 {idx + 1}번 자막 시간이 겹칩니다. 잘못된 결과 생성을 중단했습니다."
            )


def parse_time_value(value: str) -> float:
    value = value.strip()
    if not value:
        raise ValueError("empty time")
    parts = value.split(":")
    if len(parts) == 1:
        return float(parts[0])
    if len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    raise ValueError(f"invalid time: {value}")


def parse_timing_anchors(anchor_text: str, block_count: int, duration: float) -> list[tuple[int, float]]:
    anchors: list[tuple[int, float]] = []
    for raw in anchor_text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = re.match(r"^(\d+)\s*(?:=|:|,)\s*([0-9:.]+)\s*$", line)
        if not match:
            raise gr.Error(f"Timing anchors 형식이 잘못됐습니다: {line}")
        index = int(match.group(1))
        if index < 1 or index > block_count:
            raise gr.Error(f"Timing anchor 번호 {index}가 자막 블록 범위를 벗어났습니다. 현재 블록 수: {block_count}")
        seconds = parse_time_value(match.group(2))
        if seconds < 0 or seconds > duration:
            raise gr.Error(f"Timing anchor 시간이 영상 범위를 벗어났습니다: {line}")
        anchors.append((index - 1, seconds))
    if not anchors:
        raise gr.Error("Manual anchors 모드에서는 Timing anchors에 최소 1개 이상의 기준점을 입력해야 합니다.")
    anchors = sorted(set(anchors))
    for prev, current in zip(anchors, anchors[1:]):
        if current[1] <= prev[1]:
            raise gr.Error("Timing anchors 시간은 블록 번호가 커질수록 증가해야 합니다.")
    return anchors


def align_blocks_by_anchors(blocks: list[LyricBlock], anchor_text: str, duration: float) -> list[CaptionLine]:
    if not blocks:
        return []
    anchors = parse_timing_anchors(anchor_text, len(blocks), duration)
    n = len(blocks)
    starts = [0.0] * n

    first_idx, first_time = anchors[0]
    if first_idx == 0:
        for i in range(0, first_idx + 1):
            starts[i] = first_time
    else:
        step = min(3.0, first_time / max(1, first_idx))
        for i in range(0, first_idx + 1):
            starts[i] = max(0.0, first_time - (first_idx - i) * step)

    for (left_idx, left_time), (right_idx, right_time) in zip(anchors, anchors[1:]):
        span = max(1, right_idx - left_idx)
        for i in range(left_idx, right_idx + 1):
            ratio = (i - left_idx) / span
            starts[i] = left_time + (right_time - left_time) * ratio

    last_idx, last_time = anchors[-1]
    if last_idx < n - 1:
        remaining = n - 1 - last_idx
        step = max(0.5, (max(last_time + 0.5, duration - 0.5) - last_time) / max(1, remaining))
        for i in range(last_idx, n):
            starts[i] = min(duration, last_time + (i - last_idx) * step)

    captions = []
    for idx, block in enumerate(blocks):
        start = min(max(0.0, starts[idx]), duration)
        if idx + 1 < n:
            end = min(duration, max(start + 0.2, starts[idx + 1] - 0.05))
        else:
            end = min(duration, start + 3.0)
        captions.append(CaptionLine(start, end, block.text, 1.0))
    return captions


def _expand_lrc_items_for_blocks(blocks: list[LyricBlock], items: list[TimedLyric]) -> list[TimedLyric]:
    """Create one timing slot per block without combining display text."""
    if len(blocks) <= len(items):
        return items

    counts = [1] * len(items)
    text_lengths = np.asarray(
        [max(1, len(normalize_for_lyric_match(item.text))) for item in items], dtype=float
    )
    durations = np.asarray([max(0.2, item.end - item.start) for item in items], dtype=float)
    median_text = max(1.0, float(np.median(text_lengths)))
    median_duration = max(0.2, float(np.median(durations)))

    for _ in range(len(blocks) - len(items)):
        priorities = [
            (text_lengths[idx] / counts[idx]) / median_text
            + 0.20 * min(2.0, (durations[idx] / counts[idx]) / median_duration)
            for idx in range(len(items))
        ]
        counts[int(np.argmax(priorities))] += 1

    expanded: list[TimedLyric] = []
    block_cursor = 0
    for item, part_count in zip(items, counts):
        assigned = blocks[block_cursor : block_cursor + part_count]
        weights = np.sqrt(
            np.asarray(
                [max(1, len(normalize_for_lyric_match(block.sync_text))) for block in assigned],
                dtype=float,
            )
        )
        weights /= max(1e-9, float(weights.sum()))
        boundaries = [item.start]
        for fraction in np.cumsum(weights)[:-1]:
            boundaries.append(item.start + (item.end - item.start) * float(fraction))
        boundaries.append(item.end)
        for part_idx in range(part_count):
            start = boundaries[part_idx]
            end = max(start + 0.20, boundaries[part_idx + 1])
            expanded.append(TimedLyric(start, min(item.end, end), item.text))
        block_cursor += part_count
    return expanded


def _compress_lrc_items_for_blocks(blocks: list[LyricBlock], items: list[TimedLyric]) -> list[TimedLyric]:
    if len(blocks) >= len(items):
        return items
    compressed: list[TimedLyric] = []
    for block_idx in range(len(blocks)):
        start_idx = round(block_idx * len(items) / len(blocks))
        end_idx = round((block_idx + 1) * len(items) / len(blocks))
        grouped = items[start_idx : max(start_idx + 1, end_idx)]
        compressed.append(
            TimedLyric(grouped[0].start, grouped[-1].end, " ".join(item.text for item in grouped))
        )
    return compressed


def align_blocks_to_lrc(blocks: list[LyricBlock], lrc_text: str, duration: float) -> tuple[list[CaptionLine], float, int]:
    lrc_items = parse_lrc_items(lrc_text, duration)
    if not blocks or not lrc_items:
        return [], 0.0, 0

    timing_items = _expand_lrc_items_for_blocks(blocks, lrc_items)
    timing_items = _compress_lrc_items_for_blocks(blocks, timing_items)
    captions: list[CaptionLine] = []
    scores: list[float] = []
    for block, item in zip(blocks, timing_items):
        score = similarity(block.sync_text, item.text)
        scores.append(score)
        captions.append(CaptionLine(item.start, min(duration, item.end), block.text, score))
    avg_score = sum(scores) / max(1, len(scores))
    return captions, avg_score, len(lrc_items)


def ass_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    cs = int(round(seconds * 100))
    h = cs // 360000
    cs %= 360000
    m = cs // 6000
    cs %= 6000
    s = cs // 100
    cs %= 100
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def srt_time(seconds: float) -> str:
    ms = int(round(max(0.0, seconds) * 1000))
    h = ms // 3600000
    ms %= 3600000
    m = ms // 60000
    ms %= 60000
    s = ms // 1000
    ms %= 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def ass_color(hex_color: str, alpha: str = "00") -> str:
    value = hex_color.strip().lstrip("#")
    if len(value) != 6:
        value = "FFFFFF"
    rr, gg, bb = value[0:2], value[2:4], value[4:6]
    return f"&H{alpha}{bb}{gg}{rr}"


def ass_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}").replace("\n", "\\N")


def write_ass(
    captions: list[CaptionLine],
    out_path: Path,
    title: str,
    font: str,
    font_size: int,
    position: str,
    primary_color: str,
    outline_color: str,
    outline: float,
    shadow: float,
    margin_v: int,
) -> None:
    alignment = {"Bottom center": 2, "Middle center": 5, "Top center": 8}[position]
    header = f"""[Script Info]
Title: {title}
ScriptType: v4.00+
WrapStyle: 0
ScaledBorderAndShadow: yes
PlayResX: 1920
PlayResY: 1080

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font},{font_size},{ass_color(primary_color)},&H000000FF,{ass_color(outline_color)},&H88000000,0,0,0,0,100,100,0,0,1,{outline},{shadow},{alignment},90,90,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines = [
        f"Dialogue: 0,{ass_time(c.start)},{ass_time(c.end)},Default,,0,0,0,,{ass_escape(c.text)}"
        for c in captions
    ]
    out_path.write_text(header + "\n".join(lines) + "\n", encoding="utf-8-sig")


def write_srt(captions: list[CaptionLine], out_path: Path) -> None:
    chunks = []
    for idx, caption in enumerate(captions, start=1):
        chunks.append(f"{idx}\n{srt_time(caption.start)} --> {srt_time(caption.end)}\n{caption.text}\n")
    out_path.write_text("\n".join(chunks), encoding="utf-8")


def filter_escape_path(path: Path) -> str:
    value = path.resolve().as_posix()
    value = value.replace(":", "\\:")
    value = value.replace("'", "\\'")
    return value


def burn_subtitles(video: Path, ass_path: Path, output_path: Path) -> None:
    subtitle_filter = f"subtitles='{filter_escape_path(ass_path)}'"
    run([
        "ffmpeg",
        "-y",
        "-i",
        str(video),
        "-vf",
        subtitle_filter,
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-c:a",
        "copy",
        str(output_path),
    ])


def safe_slug(text: str) -> str:
    slug = re.sub(r"[^0-9a-zA-Z가-힣_-]+", "_", text.strip())
    return slug.strip("_") or "subtitle"


def make_job_dir(video_path: str, artist: str, song_title: str) -> Path:
    seed = f"{video_path}|{artist}|{song_title}|{time.time()}".encode("utf-8")
    job = JOBS / hashlib.sha1(seed).hexdigest()[:12]
    job.mkdir(parents=True, exist_ok=True)
    return job


def resolve_uploaded_path(value: object) -> Path | None:
    if value is None:
        return None
    if isinstance(value, str):
        return Path(value)
    if isinstance(value, dict):
        for key in ("path", "name"):
            path_value = value.get(key)
            if path_value:
                return Path(str(path_value))
    return Path(str(value))


def create_subtitles(
    video_file: object,
    reference_file: object,
    reference_youtube_url: str,
    artist: str,
    song_title: str,
    lyrics: str,
    lyric_grouping: str,
    sync_source: str,
    alignment_mode: str,
    global_offset: float,
    timing_anchors: str,
    use_lrclib: bool,
    auto_fit_vocal_span: bool,
    auto_fit_mode: str,
    font: str,
    font_size: int,
    position: str,
    margin_v: int,
    primary_color: str,
    outline_color: str,
    outline: float,
    shadow: float,
    model_size: str,
    language: str,
    separate_first: bool,
    burn_video: bool,
) -> tuple[str, str | None, str | None, str, str]:
    try:
        return _create_subtitles_impl(
            video_file,
            reference_file,
            reference_youtube_url,
            artist,
            song_title,
            lyrics,
            lyric_grouping,
            sync_source,
            alignment_mode,
            global_offset,
            timing_anchors,
            use_lrclib,
            auto_fit_vocal_span,
            auto_fit_mode,
            font,
            font_size,
            position,
            margin_v,
            primary_color,
            outline_color,
            outline,
            shadow,
            model_size,
            language,
            separate_first,
            burn_video,
        )
    except gr.Error as exc:
        return None, None, None, f"오류: {exc}", ""
    except Exception as exc:
        return None, None, None, f"오류: {exc}", ""


def _create_subtitles_impl(
    video_file: object,
    reference_file: object,
    reference_youtube_url: str,
    artist: str,
    song_title: str,
    lyrics: str,
    lyric_grouping: str,
    sync_source: str,
    alignment_mode: str,
    global_offset: float,
    timing_anchors: str,
    use_lrclib: bool,
    auto_fit_vocal_span: bool,
    auto_fit_mode: str,
    font: str,
    font_size: int,
    position: str,
    margin_v: int,
    primary_color: str,
    outline_color: str,
    outline: float,
    shadow: float,
    model_size: str,
    language: str,
    separate_first: bool,
    burn_video: bool,
) -> tuple[str, str | None, str | None, str, str]:
    video = resolve_uploaded_path(video_file)
    if video is None:
        raise gr.Error("영상 파일을 업로드하세요.")
    if not lyrics or not lyrics.strip():
        raise gr.Error("가사를 붙여넣으세요.")

    duration = ffprobe_duration(video)
    job = make_job_dir(str(video), artist, song_title)
    base = safe_slug(f"{artist}_{song_title}") if artist or song_title else "band_lyrics"

    ass_path = job / f"{base}.ass"
    srt_path = job / f"{base}.srt"
    mp4_path = job / f"{base}_subtitled.mp4"
    meta_path = job / "alignment.json"

    lrc_lines = parse_lrc(lyrics, duration)
    if lrc_lines is not None:
        captions = lrc_lines
        status = f"LRC 타임스탬프 {len(captions)}줄을 사용했습니다."
    else:
        lyric_blocks = split_lyric_blocks(lyrics, lyric_grouping, sync_source)
        audio = job / "audio.wav"
        extract_audio(video, audio)
        align_audio = audio
        if separate_first and alignment_mode not in ("Reference audio DTW", "LRCLIB synced lyrics", "Manual anchors", "Even spacing"):
            align_audio = separate_vocals(audio, job)
        segments: list[WhisperSegment] = []
        lrc_status = ""
        if alignment_mode == "Reference audio DTW":
            reference_audio, reference_note = prepare_reference_audio(reference_file, reference_youtube_url, job)
            reference_duration = ffprobe_duration(reference_audio)
            lrc_text, lrc_status = fetch_lrclib_synced_lyrics(artist, song_title, reference_duration, len(lyric_blocks))
            if not lrc_text:
                raise gr.Error(f"Reference audio DTW requires synced lyrics. {lrc_status}")
            reference_captions, avg_score, lrc_count = align_blocks_to_lrc(lyric_blocks, lrc_text, reference_duration)
            performance_alignment_audio = job / "performance_22050.wav"
            extract_alignment_audio(video, performance_alignment_audio)
            captions, dtw_status = warp_captions_with_dtw(
                reference_captions,
                reference_audio,
                performance_alignment_audio,
                job,
                duration,
            )
            mode_note = "Reference audio DTW"
            lrc_status = f"{lrc_status} / {reference_note} / {dtw_status}"
        elif alignment_mode == "LRCLIB synced lyrics" or use_lrclib:
            lrc_text, lrc_status = fetch_lrclib_synced_lyrics(artist, song_title, duration, len(lyric_blocks))
            if lrc_text:
                captions, avg_score, lrc_count = align_blocks_to_lrc(lyric_blocks, lrc_text, duration)
                mode_note = "LRCLIB synced lyrics"
                if auto_fit_vocal_span:
                    fit_audio = align_audio
                    if separate_first:
                        fit_audio = separate_vocals(audio, job)
                    captions, fit_status = auto_fit_captions_to_vocal_span(captions, fit_audio, job, duration, auto_fit_mode)
                    lrc_status = f"{lrc_status} / {fit_status}"
            elif alignment_mode == "LRCLIB synced lyrics":
                if separate_first:
                    align_audio = separate_vocals(audio, job)
                captions = align_blocks_by_activity(lyric_blocks, align_audio, job, duration)
                mode_note = "Vocal activity sequential fallback"
                avg_score = sum(c.score for c in captions) / max(1, len(captions))
                lrc_count = 0
                lrc_status = f"{lrc_status} / LRCLIB 실패로 보컬 activity 기반 자동 싱크로 대체했습니다."
            else:
                captions = []
                avg_score = 0.0
                lrc_count = 0
        if (alignment_mode == "Whisper fuzzy match") and not (use_lrclib and lrc_status.startswith("LRCLIB synced lyrics found")):
            segments = transcribe(align_audio, model_size, language)
            captions = align_blocks_to_segments(lyric_blocks, segments, duration)
            mode_note = "Whisper fuzzy match"
            lrc_count = 0
        elif alignment_mode == "Even spacing":
            captions = distribute_blocks(lyric_blocks, duration)
            mode_note = "Even spacing"
            avg_score = sum(c.score for c in captions) / max(1, len(captions))
            lrc_count = 0
        elif alignment_mode == "Manual anchors":
            captions = align_blocks_by_anchors(lyric_blocks, timing_anchors, duration)
            mode_note = "Manual anchors"
            avg_score = sum(c.score for c in captions) / max(1, len(captions))
            lrc_count = 0
        elif alignment_mode == "Vocal activity sequential" and not (use_lrclib and lrc_status.startswith("LRCLIB synced lyrics found")):
            if separate_first:
                align_audio = separate_vocals(audio, job)
            captions = align_blocks_by_activity(lyric_blocks, align_audio, job, duration)
            mode_note = "Vocal activity sequential"
            avg_score = sum(c.score for c in captions) / max(1, len(captions))
            lrc_count = 0
        captions = apply_global_offset(captions, float(global_offset), duration)
        if alignment_mode == "Reference audio DTW":
            validate_caption_sequence(captions, len(lyric_blocks))
        warning = ""
        if alignment_mode == "Whisper fuzzy match" and (len(segments) < 3 or avg_score < 0.12):
            warning = " / 경고: Whisper 매칭 품질이 낮습니다. 일본어 곡이면 Language=ja, Sync text line=Line 1을 쓰거나 Vocal activity sequential을 권장합니다."
        status = f"{mode_note}: 입력 {len(lyric_blocks)}블록을 자막 {len(captions)}개로 모두 보존했습니다. LRC {lrc_count}줄, 평균 매칭 점수 {avg_score:.2f}. {lrc_status}{warning}"
        meta_path.write_text(
            json.dumps(
                {
                    "artist": artist,
                    "song_title": song_title,
                    "duration": duration,
                    "alignment_mode": alignment_mode,
                    "sync_source": sync_source,
                    "global_offset": global_offset,
                    "input_block_count": len(lyric_blocks),
                    "caption_count": len(captions),
                    "lrc_status": lrc_status,
                    "segments": [seg.__dict__ for seg in segments],
                    "captions": [caption.__dict__ for caption in captions],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    write_ass(
        captions,
        ass_path,
        title=f"{artist} - {song_title}".strip(" -"),
        font=font.strip() or "Malgun Gothic",
        font_size=int(font_size),
        position=position,
        primary_color=primary_color,
        outline_color=outline_color,
        outline=float(outline),
        shadow=float(shadow),
        margin_v=int(margin_v),
    )
    write_srt(captions, srt_path)

    video_output = None
    if burn_video:
        burn_subtitles(video, ass_path, mp4_path)
        video_output = str(mp4_path)

    preview = "\n".join(
        f"{srt_time(c.start)} --> {srt_time(c.end)}  {c.text}"
        for c in captions[:12]
    )
    if len(captions) > 12:
        preview += f"\n... {len(captions) - 12} more lines"

    return str(ass_path), str(srt_path), video_output, status, preview


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Band Lyric Sync Tool") as demo:
        gr.Markdown("# Band Lyric Sync Tool")
        gr.Markdown("공연 영상과 확인된 가사를 넣으면 줄 단위 자막 싱크를 만들고, ASS/SRT와 자막 입힌 MP4를 생성합니다.")

        with gr.Row():
            with gr.Column(scale=2):
                video = gr.File(label="Input video file", file_types=["video"], type="filepath")
                reference_file = gr.File(label="Reference audio/video file", file_types=["audio", "video"], type="filepath")
                reference_youtube_url = gr.Textbox(label="Reference YouTube URL", placeholder="Official audio URL, used only in Reference audio DTW mode")
                artist = gr.Textbox(label="Artist", placeholder="예: 아티스트명")
                song_title = gr.Textbox(label="Song title", placeholder="예: 곡 제목")
                lyrics = gr.Textbox(label="Lyrics or LRC", lines=16, placeholder="빈 줄로 자막 블록을 나누세요. 블록 안 줄바꿈은 한 자막 안에 유지됩니다.")
                lyric_grouping = gr.Dropdown(
                    label="Lyric grouping",
                    choices=["Blank-line blocks", "Every non-empty line"],
                    value="Blank-line blocks",
                )
                sync_source = gr.Dropdown(
                    label="Sync text line",
                    choices=["Line 1", "Line 2", "Line 3", "Last line", "All lines"],
                    value="Line 1",
                )
            with gr.Column(scale=1):
                font = gr.Textbox(label="Font", value="Malgun Gothic")
                font_size = gr.Slider(label="Font size", minimum=24, maximum=96, value=48, step=1)
                position = gr.Dropdown(label="Position", choices=["Bottom center", "Middle center", "Top center"], value="Bottom center")
                margin_v = gr.Slider(label="Vertical margin", minimum=20, maximum=220, value=70, step=5)
                primary_color = gr.Textbox(label="Text color hex", value="#FFFFFF")
                outline_color = gr.Textbox(label="Outline color hex", value="#000000")
                outline = gr.Slider(label="Outline", minimum=0, maximum=8, value=2.5, step=0.5)
                shadow = gr.Slider(label="Shadow", minimum=0, maximum=6, value=1.0, step=0.5)
                alignment_mode = gr.Dropdown(
                    label="Alignment mode",
                    choices=["Reference audio DTW", "LRCLIB synced lyrics", "Vocal activity sequential", "Whisper fuzzy match", "Manual anchors", "Even spacing"],
                    value="Reference audio DTW",
                )
                global_offset = gr.Slider(label="Global offset seconds", minimum=-10.0, maximum=10.0, value=0.0, step=0.1)
                timing_anchors = gr.Textbox(label="Timing anchors", lines=3, placeholder="Manual anchors only, e.g. 1=23.4")
                use_lrclib = gr.Checkbox(label="Try LRCLIB synced lyrics first", value=True)
                auto_fit_vocal_span = gr.Checkbox(label="Auto fit timing to vocal span", value=True)
                auto_fit_mode = gr.Dropdown(
                    label="Auto fit mode",
                    choices=["Keep LRC start, fit end", "Fit start and end"],
                    value="Keep LRC start, fit end",
                )
                model_size = gr.Dropdown(label="Whisper model", choices=["small", "medium", "large-v3"], value="medium")
                language = gr.Dropdown(label="Language", choices=["ko", "en", "ja", "auto"], value="ja")
                separate_first = gr.Checkbox(label="Separate vocals first", value=True)
                burn_video = gr.Checkbox(label="Create subtitled MP4", value=True)

        create = gr.Button("Create subtitles", variant="primary")
        status = gr.Textbox(label="Status", interactive=False)
        preview = gr.Textbox(label="Timing preview", lines=14, interactive=False)
        with gr.Row():
            ass_file = gr.File(label="ASS")
            srt_file = gr.File(label="SRT")
            mp4_file = gr.File(label="Subtitled MP4")

        create.click(
            create_subtitles,
            inputs=[
                video,
                reference_file,
                reference_youtube_url,
                artist,
                song_title,
                lyrics,
                lyric_grouping,
                sync_source,
                alignment_mode,
                global_offset,
                timing_anchors,
                use_lrclib,
                auto_fit_vocal_span,
                auto_fit_mode,
                font,
                font_size,
                position,
                margin_v,
                primary_color,
                outline_color,
                outline,
                shadow,
                model_size,
                language,
                separate_first,
                burn_video,
            ],
            outputs=[ass_file, srt_file, mp4_file, status, preview],
        )
    return demo


if __name__ == "__main__":
    build_app().launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)
