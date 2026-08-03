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
from functools import lru_cache
from pathlib import Path
from typing import Iterable

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr
import numpy as np


os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).resolve().parent
_configured_data_root = os.environ.get("BAND_LYRIC_SYNC_DATA_DIR", "").strip()
if _configured_data_root:
    DATA_ROOT = Path(_configured_data_root).expanduser().resolve()
elif (ROOT / ".venv").exists():
    DATA_ROOT = ROOT
else:
    DATA_ROOT = Path(os.environ.get("LOCALAPPDATA", ROOT)) / "BandLyricSync"
JOBS = DATA_ROOT / "jobs"
JOBS.mkdir(parents=True, exist_ok=True)


@dataclass
class CaptionLine:
    start: float
    end: float
    text: str
    score: float = 1.0
    sync_text: str = ""


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


@dataclass
class ForcedTimingCandidate:
    start: float
    end: float
    confidence: float
    collapsed_ratio: float
    source: str


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


def run_output(cmd: list[str], cwd: Path | None = None) -> str:
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
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed:\n{' '.join(cmd)}\n\n{proc.stderr[-4000:]}")
    return proc.stdout


@lru_cache(maxsize=1)
def torch_cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


@lru_cache(maxsize=None)
def ffmpeg_has_encoder(encoder: str) -> bool:
    try:
        output = run_output(["ffmpeg", "-hide_banner", "-encoders"])
    except Exception:
        return False
    return encoder in output


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
        "--write-info-json",
        "-o",
        output_template,
        url.strip(),
    ])
    candidates = [
        p
        for p in work_dir.glob("reference_download.*")
        if p.is_file() and not p.name.endswith(".info.json")
    ]
    if not candidates:
        raise gr.Error("Reference audio download finished, but no file was found.")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _reference_metadata_from_payload(
    payload: dict,
    fallback_artist: str,
    fallback_title: str,
) -> tuple[str, str]:
    artist = str(payload.get("artist") or payload.get("creator") or "").strip()
    track = str(payload.get("track") or payload.get("alt_title") or "").strip()
    video_title = str(payload.get("title") or "").strip()
    channel = str(payload.get("channel") or payload.get("uploader") or "").strip()

    cleaned_title = re.sub(
        r"[\(\[\{][^)\]\}]*\b(?:official|audio|video|lyrics?|mv)\b[^)\]\}]*[\)\]\}]",
        "",
        video_title,
        flags=re.IGNORECASE,
    ).strip()
    if not track and " - " in cleaned_title:
        title_artist, title_track = cleaned_title.split(" - ", 1)
        artist = artist or title_artist.strip()
        track = title_track.strip()
    if not track:
        track = cleaned_title
    track = re.split(r"\s*[|｜]\s*", track, maxsplit=1)[0].strip()
    if not artist and channel:
        artist = re.sub(r"\s*-\s*Topic\s*$", "", channel, flags=re.IGNORECASE).strip()
    return artist or fallback_artist, track or fallback_title


def read_reference_download_metadata(
    work_dir: Path,
    fallback_artist: str,
    fallback_title: str,
) -> tuple[str, str]:
    info_files = list(work_dir.glob("reference_download*.info.json"))
    if not info_files:
        return fallback_artist, fallback_title
    info_path = max(info_files, key=lambda path: path.stat().st_mtime)
    try:
        payload = json.loads(info_path.read_text(encoding="utf-8"))
    except Exception:
        return fallback_artist, fallback_title
    return _reference_metadata_from_payload(payload, fallback_artist, fallback_title)


def probe_youtube_reference_details(
    url: str,
    fallback_artist: str,
    fallback_title: str,
) -> tuple[str, str, float | None]:
    if not url.strip():
        return fallback_artist, fallback_title, None
    try:
        raw = run_output([
            os.sys.executable,
            "-m",
            "yt_dlp",
            "--dump-single-json",
            "--skip-download",
            "--no-playlist",
            "--no-warnings",
            url.strip(),
        ])
        payload = json.loads(raw)
    except Exception:
        return fallback_artist, fallback_title, None
    artist, title = _reference_metadata_from_payload(
        payload,
        fallback_artist,
        fallback_title,
    )
    duration = float(payload.get("duration") or 0) or None
    return artist, title, duration


def _youtube_search_reference(
    artist: str,
    song_title: str,
    target_duration: float,
    work_dir: Path,
) -> tuple[Path | None, str]:
    query = f"{artist.strip()} {song_title.strip()} official audio".strip()
    if not query:
        return None, "YouTube automatic search skipped: artist/title missing"
    try:
        raw = run_output([
            os.sys.executable,
            "-m",
            "yt_dlp",
            "--dump-single-json",
            "--flat-playlist",
            "--no-warnings",
            f"ytsearch8:{query}",
        ])
        payload = json.loads(raw)
    except Exception as exc:
        return None, f"YouTube automatic search failed: {exc}"

    title_key = normalize_for_lyric_match(song_title)
    artist_key = normalize_for_lyric_match(artist)
    ranked: list[tuple[float, dict]] = []
    for entry in payload.get("entries", []):
        duration = float(entry.get("duration") or 0)
        video_id = entry.get("id")
        if not video_id or duration <= 0:
            continue
        candidate_title = str(entry.get("title") or "")
        channel = str(entry.get("channel") or entry.get("uploader") or "")
        candidate_key = normalize_for_lyric_match(f"{candidate_title} {channel}")
        duration_error = abs(duration - target_duration) / max(1.0, target_duration)
        score = duration_error
        if title_key and title_key not in candidate_key:
            score += 0.35
        if artist_key and artist_key not in candidate_key:
            score += 0.20
        lowered = f"{candidate_title} {channel}".lower()
        if "official audio" in lowered or "topic" in lowered:
            score -= 0.05
        ranked.append((score, entry))
    if not ranked:
        return None, "YouTube automatic search returned no usable audio"

    _score, selected = min(ranked, key=lambda pair: pair[0])
    selected_duration = float(selected.get("duration") or 0)
    duration_error = abs(selected_duration - target_duration) / max(1.0, target_duration)
    if duration_error > 0.18:
        return None, (
            f"YouTube automatic search found no duration-compatible audio "
            f"(closest {selected_duration:.1f}s, performance {target_duration:.1f}s)"
        )
    selected_url = f"https://www.youtube.com/watch?v={selected['id']}"
    source = download_youtube_reference(selected_url, work_dir)
    title = selected.get("title") or song_title
    channel = selected.get("channel") or selected.get("uploader") or "unknown channel"
    return source, f"auto-selected YouTube: {channel} - {title} ({selected_duration:.1f}s)"


def prepare_reference_audio(
    reference_file: object,
    reference_youtube_url: str,
    artist: str,
    song_title: str,
    target_duration: float,
    work_dir: Path,
) -> tuple[Path, str, str, str]:
    reference_path = resolve_uploaded_path(reference_file)
    source_note = ""
    lookup_artist = artist
    lookup_title = song_title
    if reference_path and reference_path.exists():
        source = reference_path
        source_note = f"reference file: {source.name}"
    elif reference_youtube_url and reference_youtube_url.strip():
        youtube_dir = work_dir / "youtube_reference"
        source = download_youtube_reference(reference_youtube_url, youtube_dir)
        lookup_artist, lookup_title = read_reference_download_metadata(
            youtube_dir,
            artist,
            song_title,
        )
        source_note = "reference YouTube URL"
        source_duration = ffprobe_duration(source)
        duration_error = abs(source_duration - target_duration) / max(1.0, target_duration)
        if duration_error > 0.15:
            replacement, search_note = _youtube_search_reference(
                artist,
                song_title,
                target_duration,
                work_dir / "youtube_reference_auto",
            )
            if replacement is not None:
                source = replacement
                lookup_artist, lookup_title = read_reference_download_metadata(
                    work_dir / "youtube_reference_auto",
                    artist,
                    song_title,
                )
                source_note = (
                    f"provided YouTube duration mismatch {source_duration:.1f}s vs {target_duration:.1f}s; "
                    f"{search_note}"
                )
            else:
                raise gr.Error(
                    f"레퍼런스가 {source_duration:.1f}초, 공연이 {target_duration:.1f}초로 길이가 너무 다릅니다. "
                    f"{search_note}"
                )
    else:
        raise gr.Error("Reference audio DTW 모드에서는 reference audio file 또는 reference YouTube URL이 필요합니다.")
    reference_wav = work_dir / "reference_22050.wav"
    extract_alignment_audio(source, reference_wav)
    return reference_wav, source_note, lookup_artist, lookup_title


def separate_vocals(audio: Path, work_dir: Path) -> Path:
    out_dir = work_dir / "demucs"
    device = "cuda" if torch_cuda_available() else "cpu"
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
        "-d",
        device,
        str(audio),
    ])
    candidates = list(out_dir.glob(f"**/{audio.stem}/vocals.wav"))
    if not candidates:
        candidates = list(out_dir.glob("**/vocals.wav"))
    if not candidates:
        raise RuntimeError("Demucs finished, but vocals.wav was not found.")
    return max(candidates, key=lambda path: path.stat().st_mtime)


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


def _simplify_track_metadata(value: str) -> str:
    value = value.casefold()
    value = re.sub(r"[\(\[\{].*?[\)\]\}]", " ", value)
    value = re.sub(r"\b(?:feat(?:uring)?|ft|official|audio|video|lyrics?|mv)\b.*", " ", value)
    return re.sub(r"[^0-9a-z\u3040-\u30ff\u3400-\u9fff\uac00-\ud7a3]+", "", value)


def _metadata_similarity(left: str, right: str) -> float:
    left_key = _simplify_track_metadata(left)
    right_key = _simplify_track_metadata(right)
    if not left_key or not right_key:
        return 0.0
    if left_key in right_key or right_key in left_key:
        return min(len(left_key), len(right_key)) / max(len(left_key), len(right_key))
    return SequenceMatcher(None, left_key, right_key).ratio()


def _lrclib_request(params: dict[str, str]) -> list[dict]:
    query = urllib.parse.urlencode(params)
    url = f"https://lrclib.net/api/search?{query}"
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "band-lyric-sync-tool/1.1 (https://github.com/koom104/band-lyric-sync-tool)"
        },
    )
    with urllib.request.urlopen(req, timeout=12) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return payload if isinstance(payload, list) else []


def _title_search_variants(song_title: str) -> list[str]:
    variants = [song_title.strip()]
    without_suffix = re.split(r"\s*[|｜]\s*", song_title, maxsplit=1)[0].strip()
    variants.append(without_suffix)
    bracket_values = re.findall(
        r"[\(\[\{【「『]([^)\]\}】」』]+)[\)\]\}】」』]",
        without_suffix,
    )
    variants.extend(value.strip() for value in bracket_values)
    variants.extend(
        value.strip()
        for value in re.findall(r"[A-Za-z0-9][A-Za-z0-9 '&.,!?\-]+", without_suffix)
    )
    result: list[str] = []
    seen: set[str] = set()
    for value in variants:
        value = re.sub(
            r"\b(?:official|audio|video|lyrics?|translation|mv)\b.*",
            "",
            value,
            flags=re.IGNORECASE,
        ).strip(" -_.,! ")
        key = value.casefold()
        if len(value) >= 2 and key not in seen:
            seen.add(key)
            result.append(value)
    return result


def search_lrclib_candidates(artist: str, song_title: str) -> tuple[list[dict], str]:
    artist = artist.strip()
    song_title = song_title.strip()
    if not song_title:
        return [], "LRCLIB lookup skipped: song title is empty."

    title_variants = _title_search_variants(song_title)
    attempts = [
        {"track_name": song_title, "artist_name": artist},
        {"q": f"{artist} {song_title}".strip()},
    ]
    attempts.extend({"track_name": variant} for variant in title_variants)

    records_by_key: dict[str, dict] = {}
    errors: list[str] = []
    unique_attempts: list[dict[str, str]] = []
    seen_attempts: set[str] = set()
    for params in attempts:
        key = urllib.parse.urlencode(params)
        if key not in seen_attempts:
            seen_attempts.add(key)
            unique_attempts.append(params)

    for attempt_idx, params in enumerate(unique_attempts[:5]):
        if not all(params.values()):
            continue
        try:
            records = _lrclib_request(params)
        except urllib.error.HTTPError as exc:
            errors.append(f"HTTP {exc.code}")
            if exc.code == 429:
                break
            continue
        except Exception as exc:
            errors.append(str(exc))
            continue
        for record in records:
            key = str(
                record.get("id")
                or (
                    record.get("artistName"),
                    record.get("trackName"),
                    record.get("albumName"),
                    record.get("duration"),
                )
            )
            records_by_key[key] = record
        if attempt_idx + 1 < min(5, len(unique_attempts)):
            time.sleep(0.25)

    if records_by_key:
        return list(records_by_key.values()), ""
    if errors:
        return [], f"LRCLIB lookup failed: {errors[-1]}"
    return [], "LRCLIB returned no candidates."


def rank_lrclib_candidates(
    records: list[dict],
    artist: str,
    song_title: str,
    duration: float | None,
    expected_lines: int | None,
) -> list[tuple[float, dict]]:
    ranked: list[tuple[float, dict]] = []
    for record in records:
        synced = str(record.get("syncedLyrics") or "").strip()
        if not synced:
            continue
        track = str(record.get("trackName") or "")
        found_artist = str(record.get("artistName") or "")
        title_score = _metadata_similarity(song_title, track)
        artist_score = _metadata_similarity(artist, found_artist) if artist.strip() else 0.75

        duration_score = 0.5
        record_duration = float(record.get("duration") or 0)
        if duration is not None and duration > 0 and record_duration > 0:
            duration_tolerance = max(12.0, duration * 0.08)
            duration_score = max(0.0, 1.0 - abs(record_duration - duration) / duration_tolerance)
        cross_script_duration_match = (
            duration is not None
            and title_score >= 0.58
            and duration_score >= 0.70
        )
        if title_score < 0.42 or (
            artist_score < 0.20
            and title_score < 0.82
            and not cross_script_duration_match
        ):
            continue

        line_score = 0.5
        if expected_lines:
            lrc_lines = count_lrc_lines(synced)
            line_score = max(
                0.0,
                1.0 - abs(lrc_lines - expected_lines) / max(lrc_lines, expected_lines),
            )
        score = (
            0.56 * title_score
            + 0.24 * artist_score
            + 0.15 * duration_score
            + 0.05 * line_score
        )
        ranked.append((score, record))
    return sorted(ranked, key=lambda item: item[0], reverse=True)


def fetch_lrclib_synced_lyrics(
    artist: str,
    song_title: str,
    duration: float,
    expected_lines: int | None = None,
) -> tuple[str | None, str]:
    if not artist.strip() or not song_title.strip():
        return None, "LRCLIB lookup skipped: artist or song title is empty."
    records, search_status = search_lrclib_candidates(artist, song_title)
    ranked = rank_lrclib_candidates(
        records,
        artist,
        song_title,
        duration,
        expected_lines,
    )
    if not ranked:
        return None, search_status or "LRCLIB synced lyrics not found."

    confidence, record = ranked[0]
    synced = str(record["syncedLyrics"]).strip()
    track = str(record.get("trackName") or song_title)
    found_artist = str(record.get("artistName") or artist)
    status_prefix = "LRCLIB synced lyrics found"
    if (
        _metadata_similarity(artist, found_artist) < 0.98
        or _metadata_similarity(song_title, track) < 0.98
    ):
        status_prefix = (
            f'LRCLIB auto-matched "{artist} - {song_title}" '
            f'to "{found_artist} - {track}"'
        )
    else:
        status_prefix += f": {found_artist} - {track}"
    status = f"{status_prefix} / confidence {confidence:.2f}"
    if expected_lines is not None:
        status += (
            f" / LRC lines {count_lrc_lines(synced)}, "
            f"input blocks {expected_lines}"
        )
    return synced, status


def autocomplete_lrclib_fields(
    artist: str,
    song_title: str,
    reference_youtube_url: str,
) -> tuple[str, str, str]:
    search_pairs = [(artist, song_title, "입력값")]
    detected_artist, detected_title, detected_duration = probe_youtube_reference_details(
        reference_youtube_url,
        artist,
        song_title,
    )
    if (
        _metadata_similarity(artist, detected_artist) < 0.98
        or _metadata_similarity(song_title, detected_title) < 0.98
    ):
        search_pairs.insert(0, (detected_artist, detected_title, "YouTube 메타데이터"))

    ranked_matches: list[tuple[float, dict, str]] = []
    statuses: list[str] = []
    for lookup_artist, lookup_title, source in search_pairs:
        records, search_status = search_lrclib_candidates(lookup_artist, lookup_title)
        if search_status:
            statuses.append(search_status)
        ranked = rank_lrclib_candidates(
            records,
            lookup_artist,
            lookup_title,
            detected_duration if source == "YouTube 메타데이터" else None,
            None,
        )
        if ranked:
            confidence, record = ranked[0]
            ranked_matches.append((confidence, record, source))
            if confidence >= 0.88:
                break

    if not ranked_matches:
        return artist, song_title, statuses[-1] if statuses else "LRCLIB 후보를 찾지 못했습니다."
    confidence, record, source = max(ranked_matches, key=lambda item: item[0])
    found_artist = str(record.get("artistName") or artist)
    track = str(record.get("trackName") or song_title)
    return (
        found_artist,
        track,
        f'LRCLIB 입력 자동완성({source}): "{found_artist} - {track}" / confidence {confidence:.2f}',
    )


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


def remove_duplicate_full_lyrics(lyrics: str) -> tuple[str, bool]:
    """Remove an accidentally pasted second full-song copy, preserving the first."""
    allowed = re.compile(r"[0-9a-z\u3040-\u30ff\u3400-\u9fff\uac00-\ud7a3]")
    normalized_chars: list[str] = []
    source_positions: list[int] = []
    for source_idx, char in enumerate(lyrics.lower()):
        if allowed.fullmatch(char):
            normalized_chars.append(char)
            source_positions.append(source_idx)
    normalized = "".join(normalized_chars)
    if len(normalized) < 240:
        return lyrics, False

    prefix_length = min(300, max(100, len(normalized) // 5))
    prefix = normalized[:prefix_length]
    search_start = int(len(normalized) * 0.40)
    search_end = int(len(normalized) * 0.60)
    candidates: set[int] = set()

    candidate = normalized.find(prefix, search_start, search_end + prefix_length)
    while candidate != -1 and candidate <= search_end:
        candidates.add(candidate)
        candidate = normalized.find(prefix, candidate + 1, search_end + prefix_length)

    # A second paste often differs by a few characters, so the complete prefix
    # may not occur verbatim. Locate it from several shorter internal anchors.
    anchor_length = min(48, max(24, prefix_length // 6))
    anchor_step = max(12, anchor_length // 2)
    for anchor_offset in range(0, prefix_length - anchor_length + 1, anchor_step):
        anchor = prefix[anchor_offset : anchor_offset + anchor_length]
        occurrence = normalized.find(
            anchor,
            search_start + anchor_offset,
            search_end + anchor_offset + anchor_length,
        )
        while occurrence != -1:
            candidate_start = occurrence - anchor_offset
            if search_start <= candidate_start <= search_end:
                candidates.add(candidate_start)
            occurrence = normalized.find(
                anchor,
                occurrence + 1,
                search_end + anchor_offset + anchor_length,
            )

    # Last-resort fuzzy prefix scan covers edits distributed across every anchor.
    scan_step = max(4, prefix_length // 50)
    for candidate_start in range(search_start, search_end + 1, scan_step):
        window = normalized[candidate_start : candidate_start + prefix_length]
        if len(window) >= prefix_length * 0.9:
            prefix_score = SequenceMatcher(None, prefix, window).ratio()
            if prefix_score >= 0.88:
                candidates.add(candidate_start)

    best: tuple[float, int] | None = None
    for candidate in candidates:
        first_copy = normalized[:candidate]
        second_copy = normalized[candidate:]
        length_ratio = min(len(first_copy), len(second_copy)) / max(len(first_copy), len(second_copy))
        duplicate_score = SequenceMatcher(None, first_copy, second_copy).ratio()
        if length_ratio >= 0.92 and duplicate_score >= 0.97:
            quality = duplicate_score + 0.1 * length_ratio
            if best is None or quality > best[0]:
                best = (quality, candidate)

    if best is not None:
        cutoff = source_positions[best[1]]
        while cutoff > 0 and lyrics[cutoff - 1].isspace():
            cutoff -= 1
        if cutoff > 0 and lyrics[cutoff - 1] in "\"'([{\u300c\u300e\u201c\u2018":
            cutoff -= 1
        return lyrics[:cutoff].rstrip(), True
    return lyrics, False


def transcribe(
    audio: Path,
    model_size: str,
    language: str,
    initial_prompt: str = "",
) -> list[WhisperSegment]:
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
        initial_prompt=initial_prompt[:1600] or None,
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


def select_best_block_lines_for_transcript(
    blocks: list[LyricBlock],
    segments: list[WhisperSegment],
) -> list[LyricBlock]:
    if not segments:
        return blocks
    transcript_spans: list[str] = []
    for start_idx in range(len(segments)):
        for span_length in range(1, 4):
            end_idx = start_idx + span_length
            if end_idx <= len(segments):
                transcript_spans.append(
                    " ".join(segment.text for segment in segments[start_idx:end_idx])
                )

    selected: list[LyricBlock] = []
    for block in blocks:
        candidates = [line.strip() for line in block.text.splitlines() if line.strip()]
        if block.sync_text not in candidates:
            candidates.append(block.sync_text)
        best_text = block.sync_text
        best_score = max(
            (similarity(best_text, span) for span in transcript_spans),
            default=0.0,
        )
        for candidate in candidates:
            candidate_score = max(
                (similarity(candidate, span) for span in transcript_spans),
                default=0.0,
            )
            if candidate_score > best_score + 0.03:
                best_text = candidate
                best_score = candidate_score
        selected.append(LyricBlock(text=block.text, sync_text=best_text))
    return selected


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


def align_blocks_to_reference_without_lrc(
    blocks: list[LyricBlock],
    reference_audio: Path,
    work_dir: Path,
    duration: float,
    model_size: str,
    language: str,
    separate_first: bool,
) -> tuple[list[CaptionLine], list[WhisperSegment], float, str]:
    alignment_audio = reference_audio
    notes: list[str] = []
    if separate_first:
        try:
            alignment_audio = separate_vocals(
                reference_audio,
                work_dir / "reference_forced_alignment",
            )
            notes.append("Demucs reference vocals")
        except Exception as exc:
            notes.append(f"Demucs fallback to full mix ({type(exc).__name__})")

    prompt = " ".join(block.text.replace("\n", " ") for block in blocks)
    segments: list[WhisperSegment] = []
    try:
        segments = transcribe(
            alignment_audio,
            model_size,
            language,
            initial_prompt=prompt,
        )
    except Exception as exc:
        notes.append(f"Whisper failed ({type(exc).__name__})")

    if segments:
        adaptive_blocks = select_best_block_lines_for_transcript(blocks, segments)
        captions = align_blocks_to_segments(adaptive_blocks, segments, duration)
        average_score = sum(caption.score for caption in captions) / max(1, len(captions))
        coverage = sum(caption.score >= 0.15 for caption in captions) / max(1, len(captions))
        if average_score >= 0.18 or coverage >= 0.45:
            notes.append(
                f"Whisper reference alignment {len(segments)} segments, "
                f"score {average_score:.2f}, coverage {coverage:.0%}"
            )
            return captions, segments, average_score, " / ".join(notes)
        notes.append(
            f"Whisper confidence low (score {average_score:.2f}, coverage {coverage:.0%})"
        )

    captions = align_blocks_by_activity(
        blocks,
        alignment_audio,
        work_dir / "reference_activity",
        duration,
    )
    notes.append("vocal-activity sequential fallback")
    return captions, segments, 0.0, " / ".join(notes)


def apply_global_offset(captions: list[CaptionLine], offset_seconds: float, duration: float) -> list[CaptionLine]:
    if abs(offset_seconds) < 0.001:
        return captions
    shifted = []
    for caption in captions:
        start = min(max(0.0, caption.start + offset_seconds), duration)
        end = min(max(start + 0.2, caption.end + offset_seconds), duration)
        shifted.append(CaptionLine(start, end, caption.text, caption.score, caption.sync_text))
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
        warped.append(CaptionLine(start, end, caption.text, caption.score, caption.sync_text))
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


def _path_to_time_axes(
    ref_frames: np.ndarray,
    perf_frames: np.ndarray,
    feature_rate: float,
) -> tuple[np.ndarray, np.ndarray]:
    ref_times = np.asarray(ref_frames, dtype=np.float64) / feature_rate
    perf_times = np.asarray(perf_frames, dtype=np.float64) / feature_rate
    order = np.argsort(ref_times)
    ref_times = ref_times[order]
    perf_times = perf_times[order]
    unique_ref: list[float] = []
    mapped_perf: list[float] = []
    for ref_time in np.unique(ref_times):
        values = perf_times[ref_times == ref_time]
        unique_ref.append(float(ref_time))
        mapped_perf.append(float(np.median(values)))
    return (
        np.asarray(unique_ref, dtype=np.float64),
        np.maximum.accumulate(np.asarray(mapped_perf, dtype=np.float64)),
    )


def _legacy_dtw_time_map(
    y_ref: np.ndarray,
    y_perf: np.ndarray,
    sample_rate: int,
) -> tuple[np.ndarray, np.ndarray]:
    import librosa

    hop_length = 2048
    ref_chroma = librosa.feature.chroma_cens(y=y_ref, sr=sample_rate, hop_length=hop_length)
    perf_chroma = librosa.feature.chroma_cens(y=y_perf, sr=sample_rate, hop_length=hop_length)
    ref_chroma /= np.maximum(1e-6, np.linalg.norm(ref_chroma, axis=0, keepdims=True))
    perf_chroma /= np.maximum(1e-6, np.linalg.norm(perf_chroma, axis=0, keepdims=True))
    length_delta = abs(ref_chroma.shape[1] - perf_chroma.shape[1]) / max(
        ref_chroma.shape[1], perf_chroma.shape[1]
    )
    band_radius = min(0.20, max(0.08, length_delta + 0.04))
    _cost, path = librosa.sequence.dtw(
        X=ref_chroma,
        Y=perf_chroma,
        metric="cosine",
        subseq=False,
        backtrack=True,
        global_constraints=True,
        band_rad=band_radius,
        step_sizes_sigma=np.asarray([[1, 1], [1, 2], [2, 1]], dtype=np.uint32),
        weights_add=np.asarray([0.0, 0.08, 0.08]),
    )
    path = np.asarray(path[::-1])
    feature_rate = sample_rate / hop_length
    return _path_to_time_axes(path[:, 0], path[:, 1], feature_rate)


def _music_sync_features(
    audio: np.ndarray,
    sample_rate: int,
    hop_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    import librosa
    from scipy.ndimage import maximum_filter1d
    from synctoolbox.feature.chroma import quantize_chroma

    chroma = librosa.feature.chroma_stft(
        y=audio,
        sr=sample_rate,
        hop_length=hop_length,
        n_fft=4096,
        n_chroma=12,
        norm=2,
    )
    chroma /= np.maximum(1e-8, np.linalg.norm(chroma, axis=0, keepdims=True))
    quantized = quantize_chroma(chroma)

    onset = np.maximum(0.0, np.diff(chroma, axis=1, prepend=chroma[:, :1]))
    local_max = maximum_filter1d(onset, size=40, axis=1, mode="nearest")
    onset = np.log1p(100.0 * onset) / (np.log1p(100.0 * local_max) + 1e-8)
    decayed = np.zeros_like(onset)
    for decay_idx in range(10):
        decayed[:, decay_idx:] += (
            1.0 / np.sqrt(decay_idx + 1)
        ) * onset[:, : onset.shape[1] - decay_idx]
    decayed /= np.maximum(1e-8, np.linalg.norm(decayed, axis=0, keepdims=True))
    return quantized, decayed


def _multiscale_dtw_time_map(
    ref_chroma: np.ndarray,
    ref_onset: np.ndarray,
    perf_chroma: np.ndarray,
    perf_onset: np.ndarray,
    feature_rate: int,
) -> tuple[np.ndarray, np.ndarray, int, np.ndarray, np.ndarray]:
    from synctoolbox.dtw.mrmsdtw import sync_via_mrmsdtw
    from synctoolbox.dtw.utils import (
        compute_optimal_chroma_shift,
        make_path_strictly_monotonic,
        shift_chroma_vectors,
    )
    from synctoolbox.feature.chroma import quantized_chroma_to_CENS

    ref_cens = quantized_chroma_to_CENS(ref_chroma, 201, 50, feature_rate)[0]
    perf_cens = quantized_chroma_to_CENS(perf_chroma, 201, 50, feature_rate)[0]
    chroma_shift = compute_optimal_chroma_shift(ref_cens, perf_cens)
    shifted_perf_chroma = shift_chroma_vectors(perf_chroma, chroma_shift)
    shifted_perf_onset = shift_chroma_vectors(perf_onset, chroma_shift)
    path = sync_via_mrmsdtw(
        f_chroma1=ref_chroma,
        f_chroma2=shifted_perf_chroma,
        f_onset1=ref_onset,
        f_onset2=shifted_perf_onset,
        input_feature_rate=feature_rate,
        step_weights=np.asarray([1.5, 1.5, 2.0]),
        threshold_rec=10**6,
        alpha=0.65,
        verbose=False,
    )
    path = make_path_strictly_monotonic(path)
    ref_axis, perf_axis = _path_to_time_axes(path[0], path[1], feature_rate)
    return ref_axis, perf_axis, int(chroma_shift), shifted_perf_chroma, shifted_perf_onset


def _alignment_feature_cost(
    ref_chroma: np.ndarray,
    ref_onset: np.ndarray,
    perf_chroma: np.ndarray,
    perf_onset: np.ndarray,
    ref_axis: np.ndarray,
    perf_axis: np.ndarray,
    feature_rate: int,
) -> float:
    ref_times = np.arange(ref_chroma.shape[1], dtype=np.float64) / feature_rate
    mapped_times = np.interp(ref_times, ref_axis, perf_axis)
    perf_indices = np.clip(
        np.rint(mapped_times * feature_rate).astype(int),
        0,
        perf_chroma.shape[1] - 1,
    )

    def cosine_cost(left: np.ndarray, right: np.ndarray) -> float:
        left_norm = np.linalg.norm(left, axis=0)
        right_norm = np.linalg.norm(right, axis=0)
        active = (left_norm > 1e-6) & (right_norm > 1e-6)
        if int(active.sum()) < 20:
            return 1.0
        similarities = np.sum(left[:, active] * right[:, active], axis=0)
        similarities /= left_norm[active] * right_norm[active]
        return float(np.mean(1.0 - np.clip(similarities, -1.0, 1.0)))

    chroma_cost = cosine_cost(ref_chroma, perf_chroma[:, perf_indices])
    onset_cost = cosine_cost(ref_onset, perf_onset[:, perf_indices])
    return 0.75 * chroma_cost + 0.25 * onset_cost


def _time_map_stats(
    ref_axis: np.ndarray,
    perf_axis: np.ndarray,
) -> tuple[float, float, float, float]:
    offsets = perf_axis - ref_axis
    low, median, high = np.percentile(offsets, [5, 50, 95])
    return float(low), float(median), float(high), float(high - low)


def _choose_alignment_method(
    multiscale_quality: float,
    legacy_quality: float,
    multiscale_spread: float,
    legacy_spread: float,
    max_stable_spread: float,
) -> str:
    multiscale_stable = multiscale_spread <= max_stable_spread
    legacy_stable = legacy_spread <= max_stable_spread
    if multiscale_stable and not legacy_stable:
        return "multiscale"
    if legacy_stable and not multiscale_stable:
        return "legacy"
    if not multiscale_stable and not legacy_stable:
        return "invalid"
    return "multiscale" if multiscale_quality <= legacy_quality + 0.025 else "legacy"


def build_dtw_time_map(reference_audio: Path, performance_audio: Path, work_dir: Path) -> tuple[np.ndarray, np.ndarray, str]:
    import librosa

    work_dir.mkdir(parents=True, exist_ok=True)
    sample_rate = 22050
    feature_rate = 50
    hop_length = sample_rate // feature_rate
    y_ref, _ = librosa.load(str(reference_audio), sr=sample_rate, mono=True)
    y_perf, _ = librosa.load(str(performance_audio), sr=sample_rate, mono=True)
    if y_ref.size < sample_rate or y_perf.size < sample_rate:
        raise gr.Error("Reference audio DTW failed: audio is too short.")
    reference_duration = y_ref.size / sample_rate
    performance_duration = y_perf.size / sample_rate
    duration_ratio = max(reference_duration, performance_duration) / min(
        reference_duration, performance_duration
    )
    if duration_ratio > 1.18:
        raise gr.Error(
            f"레퍼런스({reference_duration:.1f}초)와 공연({performance_duration:.1f}초)의 길이 차이가 너무 커서 "
            "안전하게 정렬할 수 없습니다. 잘못된 시작점 생성을 중단했습니다."
        )

    ref_chroma, ref_onset = _music_sync_features(y_ref, sample_rate, hop_length)
    perf_chroma, perf_onset = _music_sync_features(y_perf, sample_rate, hop_length)
    multiscale_result = None
    multiscale_error = ""
    try:
        multiscale_result = _multiscale_dtw_time_map(
            ref_chroma,
            ref_onset,
            perf_chroma,
            perf_onset,
            feature_rate,
        )
    except Exception as exc:
        multiscale_error = f"{type(exc).__name__}: {exc}"

    legacy_result = None
    legacy_error = ""
    try:
        legacy_result = _legacy_dtw_time_map(y_ref, y_perf, sample_rate)
    except Exception as exc:
        legacy_error = f"{type(exc).__name__}: {exc}"

    if multiscale_result is None and legacy_result is None:
        raise gr.Error(
            "새 정렬과 기존 정렬이 모두 실패했습니다. "
            f"multiscale={multiscale_error}; legacy={legacy_error}"
        )

    chroma_shift = 0
    shifted_perf_chroma = perf_chroma
    shifted_perf_onset = perf_onset
    multiscale_quality = float("inf")
    multiscale_spread = float("inf")
    multi_low = multi_median = multi_high = 0.0
    if multiscale_result is not None:
        (
            multiscale_ref,
            multiscale_perf,
            chroma_shift,
            shifted_perf_chroma,
            shifted_perf_onset,
        ) = multiscale_result
        multiscale_quality = _alignment_feature_cost(
            ref_chroma,
            ref_onset,
            shifted_perf_chroma,
            shifted_perf_onset,
            multiscale_ref,
            multiscale_perf,
            feature_rate,
        )
        multi_low, multi_median, multi_high, multiscale_spread = _time_map_stats(
            multiscale_ref, multiscale_perf
        )

    legacy_quality = float("inf")
    legacy_spread = float("inf")
    legacy_low = legacy_median = legacy_high = 0.0
    if legacy_result is not None:
        legacy_ref, legacy_perf = legacy_result
        legacy_quality = _alignment_feature_cost(
            ref_chroma,
            ref_onset,
            shifted_perf_chroma,
            shifted_perf_onset,
            legacy_ref,
            legacy_perf,
            feature_rate,
        )
        legacy_low, legacy_median, legacy_high, legacy_spread = _time_map_stats(
            legacy_ref, legacy_perf
        )

    max_stable_spread = max(
        12.0, min(reference_duration, performance_duration) * 0.10
    )
    if multiscale_result is None:
        selected = "legacy" if legacy_spread <= max_stable_spread else "invalid"
    elif legacy_result is None:
        selected = "multiscale" if multiscale_spread <= max_stable_spread else "invalid"
    else:
        selected = _choose_alignment_method(
            multiscale_quality,
            legacy_quality,
            multiscale_spread,
            legacy_spread,
            max_stable_spread,
        )
    if selected == "invalid":
        raise gr.Error(
            "새 정렬과 기존 정렬 모두 시간축 변동이 너무 큽니다. "
            "반복 구간을 잘못 연결할 가능성이 있어 결과 생성을 중단했습니다."
        )

    disagreement_95 = 0.0
    if multiscale_result is not None and legacy_result is not None:
        comparison_times = np.linspace(
            0.0,
            min(multiscale_ref[-1], legacy_ref[-1]),
            num=500,
        )
        disagreement = np.abs(
            np.interp(comparison_times, multiscale_ref, multiscale_perf)
            - np.interp(comparison_times, legacy_ref, legacy_perf)
        )
        disagreement_95 = float(np.percentile(disagreement, 95))
    if selected == "multiscale":
        ref_axis, perf_axis = multiscale_ref, multiscale_perf
        offset_low, offset_mid, offset_high = multi_low, multi_median, multi_high
    else:
        ref_axis, perf_axis = legacy_ref, legacy_perf
        offset_low, offset_mid, offset_high = legacy_low, legacy_median, legacy_high

    diagnostics = {
        "selected": selected,
        "chroma_shift": chroma_shift,
        "multiscale_quality": (
            multiscale_quality if np.isfinite(multiscale_quality) else None
        ),
        "legacy_quality": legacy_quality if np.isfinite(legacy_quality) else None,
        "multiscale_offset_spread": (
            multiscale_spread if np.isfinite(multiscale_spread) else None
        ),
        "legacy_offset_spread": (
            legacy_spread if np.isfinite(legacy_spread) else None
        ),
        "candidate_disagreement_95_seconds": disagreement_95,
        "multiscale_error": multiscale_error,
        "legacy_error": legacy_error,
    }
    (work_dir / "alignment_candidates.json").write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    status = (
        f"alignment {selected}, chroma shift {chroma_shift:+d}, "
        f"quality multi {multiscale_quality:.3f} / legacy {legacy_quality:.3f}, "
        f"candidate diff95 {disagreement_95:.2f}s, "
        f"offset median {offset_mid:+.2f}s "
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
        warped.append(CaptionLine(start, end, caption.text, caption.score, caption.sync_text))
    return warped, status


def _forced_alignment_chunks(item_count: int) -> list[tuple[int, int]]:
    chunks: set[tuple[int, int]] = set()
    for size, stride in ((6, 6), (8, 4)):
        for start in range(0, item_count, stride):
            end = min(item_count, start + size)
            if end - start >= 3:
                chunks.add((start, end))
        if item_count >= 3:
            chunks.add((max(0, item_count - size), item_count))
    return sorted(chunks)


def _select_forced_timing_consensus(
    caption: CaptionLine,
    candidates: list[ForcedTimingCandidate],
) -> tuple[float, float] | None:
    valid = [
        candidate
        for candidate in candidates
        if abs(candidate.start - caption.start) <= 3.0
        and candidate.end > candidate.start + 0.08
    ]
    if len(valid) < 2:
        return None

    pairs: list[tuple[float, ForcedTimingCandidate, ForcedTimingCandidate]] = []
    for left_index, left in enumerate(valid):
        for right in valid[left_index + 1 :]:
            if left.source == right.source:
                continue
            start_difference = abs(left.start - right.start)
            end_difference = abs(left.end - right.end)
            pairs.append((start_difference + 0.15 * end_difference, left, right))
    if not pairs:
        return None

    _, left, right = min(pairs, key=lambda pair: pair[0])
    if abs(left.start - right.start) > 0.60:
        return None
    if (
        left.collapsed_ratio >= 0.75
        and right.collapsed_ratio >= 0.75
        and max(left.confidence, right.confidence) < 0.04
    ):
        return None

    start = float(np.median([left.start, right.start]))
    support = [
        candidate
        for candidate in valid
        if abs(candidate.start - start) <= 0.45
        and candidate.confidence >= 0.08
        and candidate.collapsed_ratio < 0.70
    ]
    source_count = len({candidate.source for candidate in support})
    strong_pair = (
        abs(left.start - right.start) <= 0.15
        and min(left.confidence, right.confidence) >= 0.25
        and max(left.collapsed_ratio, right.collapsed_ratio) < 0.50
    )
    max_adjustment = 2.8 if source_count >= 3 or strong_pair else 1.0
    if abs(start - caption.start) > max_adjustment:
        return None

    end = caption.end
    if (
        abs(left.end - right.end) <= 1.0
        and min(left.end, right.end) > start + 0.35
    ):
        proposed_end = float(np.median([left.end, right.end]))
        if abs(proposed_end - caption.end) <= 3.0:
            end = proposed_end
    return start, end


def _select_first_line_acoustic_start(
    caption: CaptionLine,
    next_start: float,
    candidates: list[ForcedTimingCandidate],
) -> float | None:
    if caption.start > 1.5:
        return None
    pairs: list[tuple[float, ForcedTimingCandidate, ForcedTimingCandidate]] = []
    for left_index, left in enumerate(candidates):
        for right in candidates[left_index + 1 :]:
            if left.source == right.source:
                continue
            if min(left.confidence, right.confidence) < 0.15:
                continue
            if max(left.collapsed_ratio, right.collapsed_ratio) >= 0.60:
                continue
            difference = abs(left.start - right.start)
            if difference <= 0.35:
                pairs.append((difference, left, right))
    if not pairs:
        return None

    _, left, right = min(pairs, key=lambda pair: pair[0])
    start = float(np.median([left.start, right.start]))
    if start <= caption.start + 0.35 or start > caption.start + 4.0:
        return None
    if start >= next_start - 0.35:
        return None
    return start


def _candidate_source_bounds(source: str) -> tuple[int, int] | None:
    match = re.fullmatch(r"(\d+)-(\d+)", source)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def _select_interlude_boundary_timing(
    index: int,
    captions: list[CaptionLine],
    candidates: list[ForcedTimingCandidate],
) -> tuple[float, float] | None:
    caption = captions[index]
    gap_before = index > 0 and caption.start - captions[index - 1].end >= 4.0
    gap_after = index + 1 < len(captions) and captions[index + 1].start - caption.end >= 4.0
    if not gap_before and not gap_after:
        return None

    valid = [
        candidate
        for candidate in candidates
        if candidate.confidence >= 0.015
        and candidate.collapsed_ratio < 0.70
        and candidate.end > candidate.start + 0.35
    ]
    if gap_before:
        line_number = index + 1
        anchored = [
            candidate
            for candidate in valid
            if (bounds := _candidate_source_bounds(candidate.source)) is not None
            and bounds[0] == line_number
        ]
        if len(anchored) >= 2 and max(c.start for c in anchored) - min(c.start for c in anchored) <= 0.35:
            start = float(np.median([candidate.start for candidate in anchored]))
            end = float(np.median([candidate.end for candidate in anchored]))
            if abs(start - caption.start) <= 3.0:
                return start, end

    if gap_after and len(valid) >= 3:
        starts = [candidate.start for candidate in valid]
        if max(starts) - min(starts) <= 2.0:
            start = float(np.median(starts))
            end = float(np.median([candidate.end for candidate in valid]))
            if abs(start - caption.start) <= 2.5:
                return start, end
    return None


def _select_contextual_pair_timing(
    index: int,
    caption: CaptionLine,
    candidates: list[ForcedTimingCandidate],
) -> tuple[float, float] | None:
    line_number = index + 1
    contextual = [
        candidate
        for candidate in candidates
        if (bounds := _candidate_source_bounds(candidate.source)) is not None
        and bounds[0] < line_number
        and candidate.confidence >= 0.07
        and candidate.collapsed_ratio < 0.70
        and candidate.end > candidate.start + 0.35
    ]
    pairs: list[tuple[float, ForcedTimingCandidate, ForcedTimingCandidate]] = []
    for left_index, left in enumerate(contextual):
        for right in contextual[left_index + 1 :]:
            if left.source == right.source:
                continue
            difference = abs(left.start - right.start)
            if difference <= 0.60:
                pairs.append((difference, left, right))
    if not pairs:
        return None

    _, left, right = min(pairs, key=lambda pair: pair[0])
    start = float(np.median([left.start, right.start]))
    if not 1.0 < abs(start - caption.start) <= 2.5:
        return None
    end = float(np.median([left.end, right.end]))
    return start, end


@lru_cache(maxsize=2)
def _load_forced_alignment_model(device: str):
    import stable_whisper

    return stable_whisper.load_model("base", device=device)


def infer_lyric_language(blocks: list[LyricBlock], configured_language: str) -> str | None:
    if configured_language != "auto":
        return configured_language
    text = "\n".join(block.sync_text for block in blocks)
    hangul_count = len(re.findall(r"[\uac00-\ud7a3]", text))
    kana_count = len(re.findall(r"[\u3040-\u30ff]", text))
    latin_count = len(re.findall(r"[A-Za-z]", text))
    if hangul_count >= max(1, kana_count, latin_count):
        return "ko"
    if kana_count >= max(1, hangul_count):
        return "ja"
    if latin_count:
        return "en"
    return None


def refine_captions_with_performance_vocals(
    captions: list[CaptionLine],
    blocks: list[LyricBlock],
    vocal_audio: Path,
    work_dir: Path,
    duration: float,
    language: str,
) -> tuple[list[CaptionLine], str]:
    if len(captions) < 3 or len(captions) != len(blocks):
        return captions, "vocal refinement skipped: caption/block count mismatch"

    try:
        import librosa

        samples, sample_rate = librosa.load(vocal_audio, sr=16000, mono=True)
        device = "cuda" if torch_cuda_available() else "cpu"
        model = _load_forced_alignment_model(device)
        candidates: list[list[ForcedTimingCandidate]] = [[] for _ in captions]
        diagnostics: list[dict[str, object]] = []

        align_language = infer_lyric_language(blocks, language)
        for start_index, end_index in _forced_alignment_chunks(len(captions)):
            window_start = max(0.0, captions[start_index].start - 1.5)
            window_end = min(duration, captions[end_index - 1].end + 1.5)
            clip = samples[
                int(window_start * sample_rate) : int(window_end * sample_rate)
            ]
            text = "\n".join(
                captions[index].sync_text or blocks[index].sync_text
                for index in range(start_index, end_index)
            )
            result = model.align(
                clip,
                text,
                language=align_language,
                original_split=True,
                nonspeech_skip=2.0,
                failure_threshold=0.5,
            )
            segments = result.to_dict().get("segments", []) if result else []
            if len(segments) != end_index - start_index:
                diagnostics.append(
                    {
                        "source": f"{start_index + 1}-{end_index}",
                        "error": f"expected {end_index - start_index} segments, got {len(segments)}",
                    }
                )
                continue

            source = f"{start_index + 1}-{end_index}"
            for offset, segment in enumerate(segments):
                words = segment.get("words") or []
                probabilities = [
                    float(word.get("probability") or 0.0) for word in words
                ]
                collapsed_ratio = sum(
                    float(word.get("end") or 0.0) - float(word.get("start") or 0.0) < 0.04
                    for word in words
                ) / max(1, len(words))
                candidate = ForcedTimingCandidate(
                    start=window_start + float(segment.get("start") or 0.0),
                    end=window_start + float(segment.get("end") or 0.0),
                    confidence=sum(probabilities) / max(1, len(probabilities)),
                    collapsed_ratio=collapsed_ratio,
                    source=source,
                )
                candidates[start_index + offset].append(candidate)

        proposed = [
            _select_forced_timing_consensus(caption, line_candidates)
            for caption, line_candidates in zip(captions, candidates)
        ]
        interlude_boundaries = 0
        for index, timing in enumerate(proposed):
            if timing is not None:
                continue
            boundary_timing = _select_interlude_boundary_timing(
                index, captions, candidates[index]
            )
            if boundary_timing is not None:
                proposed[index] = boundary_timing
                interlude_boundaries += 1
        contextual_corrections = 0
        for index, timing in enumerate(proposed):
            if timing is not None:
                continue
            contextual_timing = _select_contextual_pair_timing(
                index, captions[index], candidates[index]
            )
            if contextual_timing is not None:
                proposed[index] = contextual_timing
                contextual_corrections += 1
        first_line_anchor = None
        if proposed and len(captions) > 1 and proposed[0] is None:
            first_line_anchor = _select_first_line_acoustic_start(
                captions[0], captions[1].start, candidates[0]
            )
            if first_line_anchor is not None:
                proposed[0] = (first_line_anchor, captions[0].end)
        starts = [
            timing[0] if timing is not None else caption.start
            for caption, timing in zip(captions, proposed)
        ]
        for index in range(1, len(starts)):
            if starts[index] <= starts[index - 1] + 0.12:
                starts[index] = captions[index].start
            if starts[index] <= starts[index - 1] + 0.12:
                starts[index] = starts[index - 1] + 0.12

        refined: list[CaptionLine] = []
        accepted = 0
        extended_local = 0
        for index, caption in enumerate(captions):
            start = min(duration, max(0.0, starts[index]))
            next_start = starts[index + 1] if index + 1 < len(starts) else duration
            proposed_end = proposed[index][1] if proposed[index] is not None else caption.end
            end = min(duration, proposed_end, next_start - 0.05)
            if end <= start + 0.20:
                end = min(duration, max(start + 0.20, next_start - 0.05))
            refined.append(
                CaptionLine(start, end, caption.text, caption.score, caption.sync_text)
            )
            if proposed[index] is not None:
                accepted += 1
                if abs(start - caption.start) > 1.0:
                    extended_local += 1
            diagnostics.append(
                {
                    "index": index + 1,
                    "dtw_start": caption.start,
                    "refined_start": start,
                    "accepted": proposed[index] is not None,
                    "candidates": [
                        candidate.__dict__ for candidate in candidates[index]
                    ],
                }
            )

        work_dir.mkdir(parents=True, exist_ok=True)
        (work_dir / "vocal_refinement.json").write_text(
            json.dumps(diagnostics, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return (
            refined,
            f"LRC-text forced-alignment consensus {accepted}/{len(captions)} lines ({device})"
            + f" / extended local correction {extended_local} lines"
            + f" / interlude boundary correction {interlude_boundaries} lines"
            + f" / contextual correction {contextual_corrections} lines"
            + (
                f" / first-line acoustic anchor {first_line_anchor:.2f}s"
                if first_line_anchor is not None
                else ""
            ),
        )
    except Exception as exc:
        return captions, f"vocal refinement fallback to DTW: {type(exc).__name__}: {exc}"


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


def _match_blocks_and_lrc_groups(
    blocks: list[LyricBlock],
    items: list[TimedLyric],
) -> tuple[list[TimedLyric] | None, float]:
    """Match skipped LRC rows and split/combined lyric rows using text similarity."""
    n = len(blocks)
    m = len(items)
    negative = -1e9
    dp = [[negative] * (m + 1) for _ in range(n + 1)]
    back: list[list[tuple[int, int, str, int, int, float] | None]] = [
        [None] * (m + 1) for _ in range(n + 1)
    ]
    dp[0][0] = 0.0

    for block_idx in range(n + 1):
        for item_idx in range(m + 1):
            current = dp[block_idx][item_idx]
            if current <= negative / 2:
                continue
            if item_idx < m and current - 0.12 > dp[block_idx][item_idx + 1]:
                dp[block_idx][item_idx + 1] = current - 0.12
                back[block_idx][item_idx + 1] = (block_idx, item_idx, "skip", 0, 1, 0.0)

            for block_count in range(1, min(3, n - block_idx) + 1):
                for item_count in range(1, min(3, m - item_idx) + 1):
                    if block_count > 1 and item_count > 1:
                        continue
                    block_text = " ".join(
                        block.sync_text for block in blocks[block_idx : block_idx + block_count]
                    )
                    item_text = " ".join(item.text for item in items[item_idx : item_idx + item_count])
                    score = similarity(block_text, item_text)
                    candidate = current + score - 0.04 * (block_count + item_count - 2)
                    next_block = block_idx + block_count
                    next_item = item_idx + item_count
                    if candidate > dp[next_block][next_item]:
                        dp[next_block][next_item] = candidate
                        back[next_block][next_item] = (
                            block_idx,
                            item_idx,
                            "align",
                            block_count,
                            item_count,
                            score,
                        )

    block_idx, item_idx = n, m
    groups: list[tuple[int, int, int, int, float]] = []
    while block_idx or item_idx:
        previous = back[block_idx][item_idx]
        if previous is None:
            return None, 0.0
        prev_block, prev_item, kind, _block_count, _item_count, score = previous
        if kind == "align":
            groups.append((prev_block, block_idx, prev_item, item_idx, score))
        block_idx, item_idx = prev_block, prev_item
    groups.reverse()

    average_score = sum(group[4] for group in groups) / max(1, len(groups))
    if average_score < 0.22:
        return None, average_score

    timing_items: list[TimedLyric] = []
    for block_start, block_end, item_start, item_end, _score in groups:
        grouped_items = items[item_start:item_end]
        grouped_blocks = blocks[block_start:block_end]
        start = grouped_items[0].start
        end = grouped_items[-1].end
        if len(grouped_blocks) == 1:
            timing_items.append(
                TimedLyric(start, end, " ".join(item.text for item in grouped_items))
            )
            continue

        weights = np.sqrt(
            np.asarray(
                [max(1, len(normalize_for_lyric_match(block.sync_text))) for block in grouped_blocks],
                dtype=float,
            )
        )
        weights /= max(1e-9, float(weights.sum()))
        boundaries = [start]
        for fraction in np.cumsum(weights)[:-1]:
            boundaries.append(start + (end - start) * float(fraction))
        boundaries.append(end)
        item_text = " ".join(item.text for item in grouped_items)
        for part_idx in range(len(grouped_blocks)):
            timing_items.append(TimedLyric(boundaries[part_idx], boundaries[part_idx + 1], item_text))
    return timing_items, average_score


def align_blocks_to_lrc(blocks: list[LyricBlock], lrc_text: str, duration: float) -> tuple[list[CaptionLine], float, int]:
    lrc_items = parse_lrc_items(lrc_text, duration)
    if not blocks or not lrc_items:
        return [], 0.0, 0

    timing_items, structural_score = _match_blocks_and_lrc_groups(blocks, lrc_items)
    if timing_items is None:
        timing_items = _expand_lrc_items_for_blocks(blocks, lrc_items)
        timing_items = _compress_lrc_items_for_blocks(blocks, timing_items)
    captions: list[CaptionLine] = []
    scores: list[float] = []
    for block, item in zip(blocks, timing_items):
        score = similarity(block.sync_text, item.text)
        scores.append(score)
        captions.append(
            CaptionLine(
                item.start,
                min(duration, item.end),
                block.text,
                score,
                item.text,
            )
        )
    avg_score = structural_score if structural_score >= 0.22 else sum(scores) / max(1, len(scores))
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


def burn_subtitles(video: Path, ass_path: Path, output_path: Path) -> str:
    subtitle_filter = f"subtitles='{filter_escape_path(ass_path)}'"
    if ffmpeg_has_encoder("h264_nvenc"):
        try:
            run([
                "ffmpeg",
                "-y",
                "-i",
                str(video),
                "-vf",
                subtitle_filter,
                "-c:v",
                "h264_nvenc",
                "-preset",
                "p5",
                "-tune",
                "hq",
                "-rc",
                "vbr",
                "-cq",
                "20",
                "-b:v",
                "0",
                "-c:a",
                "copy",
                str(output_path),
            ])
            return "h264_nvenc GPU"
        except Exception:
            pass
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
    return "libx264 CPU fallback"


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
        deduplicated_lyrics, duplicate_removed = remove_duplicate_full_lyrics(lyrics)
        lyric_blocks = split_lyric_blocks(deduplicated_lyrics, lyric_grouping, sync_source)
        audio = job / "audio.wav"
        extract_audio(video, audio)
        align_audio = audio
        if separate_first and alignment_mode not in ("Reference audio DTW", "LRCLIB synced lyrics", "Manual anchors", "Even spacing"):
            align_audio = separate_vocals(audio, job)
        segments: list[WhisperSegment] = []
        lrc_status = ""
        if alignment_mode == "Reference audio DTW":
            (
                reference_audio,
                reference_note,
                reference_artist,
                reference_title,
            ) = prepare_reference_audio(
                reference_file,
                reference_youtube_url,
                artist,
                song_title,
                duration,
                job,
            )
            reference_duration = ffprobe_duration(reference_audio)
            lrc_text, lrc_status = fetch_lrclib_synced_lyrics(
                reference_artist,
                reference_title,
                reference_duration,
                len(lyric_blocks),
            )
            if not lrc_text and (
                _metadata_similarity(artist, reference_artist) < 0.98
                or _metadata_similarity(song_title, reference_title) < 0.98
            ):
                lrc_text, lrc_status = fetch_lrclib_synced_lyrics(
                    artist,
                    song_title,
                    reference_duration,
                    len(lyric_blocks),
                )
            elif lrc_text and (
                _metadata_similarity(artist, reference_artist) < 0.98
                or _metadata_similarity(song_title, reference_title) < 0.98
            ):
                lrc_status = (
                    f'YouTube metadata "{reference_artist} - {reference_title}" / '
                    f"{lrc_status}"
                )
            if lrc_text:
                reference_captions, avg_score, lrc_count = align_blocks_to_lrc(
                    lyric_blocks,
                    lrc_text,
                    reference_duration,
                )
            else:
                (
                    reference_captions,
                    segments,
                    avg_score,
                    reference_alignment_status,
                ) = align_blocks_to_reference_without_lrc(
                    lyric_blocks,
                    reference_audio,
                    job,
                    reference_duration,
                    model_size,
                    language,
                    separate_first,
                )
                lrc_count = 0
                lrc_status = (
                    f"{lrc_status} / LRCLIB 미검색으로 공식 음원 자동 강제정렬: "
                    f"{reference_alignment_status}"
                )
            performance_alignment_audio = job / "performance_22050.wav"
            extract_alignment_audio(video, performance_alignment_audio)
            captions, dtw_status = warp_captions_with_dtw(
                reference_captions,
                reference_audio,
                performance_alignment_audio,
                job,
                duration,
            )
            vocal_refinement_status = ""
            if separate_first:
                performance_vocals = separate_vocals(
                    audio,
                    job / "performance_vocal_refinement",
                )
                captions, vocal_refinement_status = refine_captions_with_performance_vocals(
                    captions,
                    lyric_blocks,
                    performance_vocals,
                    job / "vocal_refinement",
                    duration,
                    language,
                )
            mode_note = "Reference audio DTW"
            lrc_status = f"{lrc_status} / {reference_note} / {dtw_status}"
            if vocal_refinement_status:
                lrc_status = f"{lrc_status} / {vocal_refinement_status}"
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
        duplicate_note = " / 전체 가사 중복 붙여넣기 1회 자동 제거" if duplicate_removed else ""
        status = f"{mode_note}: 입력 {len(lyric_blocks)}블록을 자막 {len(captions)}개로 모두 보존했습니다. LRC {lrc_count}줄, 평균 매칭 점수 {avg_score:.2f}. {lrc_status}{warning}{duplicate_note}"
        meta_path.write_text(
            json.dumps(
                {
                    "artist": artist,
                    "song_title": song_title,
                    "duration": duration,
                    "alignment_mode": alignment_mode,
                    "sync_source": sync_source,
                    "global_offset": global_offset,
                    "duplicate_full_lyrics_removed": duplicate_removed,
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
        encoder_note = burn_subtitles(video, ass_path, mp4_path)
        status += f" / video encoder: {encoder_note}"
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
                lrclib_autocomplete = gr.Button("Find LRCLIB match")
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
                language = gr.Dropdown(label="Language", choices=["ko", "en", "ja", "auto"], value="auto")
                separate_first = gr.Checkbox(label="Separate vocals first", value=True)
                burn_video = gr.Checkbox(label="Create subtitled MP4", value=True)

        create = gr.Button("Create subtitles", variant="primary")
        status = gr.Textbox(label="Status", interactive=False)
        preview = gr.Textbox(label="Timing preview", lines=14, interactive=False)
        with gr.Row():
            ass_file = gr.File(label="ASS")
            srt_file = gr.File(label="SRT")
            mp4_file = gr.File(label="Subtitled MP4")

        lrclib_autocomplete.click(
            autocomplete_lrclib_fields,
            inputs=[artist, song_title, reference_youtube_url],
            outputs=[artist, song_title, status],
        )
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
    build_app().launch(
        server_name="127.0.0.1",
        server_port=int(os.environ.get("BAND_LYRIC_SYNC_PORT", "7860")),
        inbrowser=os.environ.get("BAND_LYRIC_SYNC_OPEN_BROWSER", "1") != "0",
        show_error=True,
        allowed_paths=[str(DATA_ROOT)],
    )
