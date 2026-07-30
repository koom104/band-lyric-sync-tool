import unittest
from pathlib import Path
from unittest.mock import patch

import app


class AlignmentTests(unittest.TestCase):
    def test_empty_lrc_rows_end_the_previous_caption(self):
        items = app.parse_lrc_items(
            "[00:01.00] first line\n[00:03.00] \n[00:10.00] second line\n[00:16.00] ",
            20.0,
        )

        self.assertEqual(len(items), 2)
        self.assertAlmostEqual(items[0].end, 2.95)
        self.assertAlmostEqual(items[1].end, 15.95)

    def test_more_blocks_than_lrc_rows_are_preserved_and_split(self):
        blocks = [
            app.LyricBlock("first", "first"),
            app.LyricBlock("second", "second"),
            app.LyricBlock("third", "third"),
        ]
        lrc = "[00:01.00] first\n[00:03.00] second third together\n[00:09.00] "

        captions, _score, lrc_count = app.align_blocks_to_lrc(blocks, lrc, 12.0)

        self.assertEqual(lrc_count, 2)
        self.assertEqual([caption.text for caption in captions], ["first", "second", "third"])
        self.assertEqual(len(captions), len(blocks))
        self.assertLess(captions[1].start, captions[2].start)
        self.assertLessEqual(captions[1].end, captions[2].start)

    def test_text_matching_handles_skipped_and_split_lrc_rows(self):
        blocks = [
            app.LyricBlock("first", "first"),
            app.LyricBlock("second", "second"),
            app.LyricBlock("part one", "part one"),
            app.LyricBlock("part two", "part two"),
        ]
        lrc = (
            "[00:01.00] first\n"
            "[00:03.00] omitted parenthetical\n"
            "[00:05.00] second\n"
            "[00:07.00] part one part two\n"
            "[00:11.00] "
        )

        captions, score, _lrc_count = app.align_blocks_to_lrc(blocks, lrc, 12.0)

        self.assertGreater(score, 0.8)
        self.assertEqual(len(captions), 4)
        self.assertAlmostEqual(captions[1].start, 5.0)
        self.assertAlmostEqual(captions[2].start, 7.0)
        self.assertLess(captions[2].start, captions[3].start)

    def test_full_song_duplicate_is_removed_without_touching_chorus_repeats(self):
        song = "\n".join(f"line {index} unique lyric content" for index in range(30))
        cleaned, removed = app.remove_duplicate_full_lyrics(song + song)

        self.assertTrue(removed)
        self.assertEqual(cleaned, song)

        chorus_song = song + "\nline 1 unique lyric content\nline 2 unique lyric content"
        unchanged, removed = app.remove_duplicate_full_lyrics(chorus_song)
        self.assertFalse(removed)
        self.assertEqual(unchanged, chorus_song)

    def test_alignment_candidate_selection_uses_quality_and_stability(self):
        self.assertEqual(
            app._choose_alignment_method(0.20, 0.19, 2.0, 2.0, 12.0),
            "multiscale",
        )
        self.assertEqual(
            app._choose_alignment_method(0.35, 0.20, 2.0, 2.0, 12.0),
            "legacy",
        )
        self.assertEqual(
            app._choose_alignment_method(0.15, 0.20, 20.0, 2.0, 12.0),
            "legacy",
        )
        self.assertEqual(
            app._choose_alignment_method(0.15, 0.20, 20.0, 19.0, 12.0),
            "invalid",
        )

    def test_lrclib_ranking_tolerates_metadata_variations(self):
        records = [
            {
                "id": 1,
                "trackName": "Cattleya",
                "artistName": "Yorushika",
                "duration": 162,
                "syncedLyrics": "[00:10.00] first\n[00:15.00] second",
            },
            {
                "id": 2,
                "trackName": "Cattleya (Live)",
                "artistName": "Different Artist",
                "duration": 240,
                "syncedLyrics": "[00:10.00] unrelated",
            },
        ]

        ranked = app.rank_lrclib_candidates(
            records,
            "Yorushka",
            "Cattleya (Official Audio)",
            163.0,
            2,
        )

        self.assertEqual(ranked[0][1]["id"], 1)
        self.assertGreater(ranked[0][0], 0.8)

    def test_youtube_metadata_uses_track_fields_and_title_fallback(self):
        self.assertEqual(
            app._reference_metadata_from_payload(
                {"artist": "Yorushika", "track": "Cattleya"},
                "요루시카",
                "카틀레야",
            ),
            ("Yorushika", "Cattleya"),
        )
        self.assertEqual(
            app._reference_metadata_from_payload(
                {
                    "title": "Yorushika - Cattleya (Official Audio)",
                    "channel": "Yorushika - Topic",
                },
                "",
                "",
            ),
            ("Yorushika", "Cattleya"),
        )

    def test_lrclib_search_extracts_english_title_alias(self):
        variants = app._title_search_variants("カトレア【Cattleya】 | Lyrics")

        self.assertIn("Cattleya", variants)
        self.assertIn("カトレア【Cattleya】", variants)

    def test_transcript_can_select_original_line_from_multiline_block(self):
        blocks = [
            app.LyricBlock(
                text="당신은 알지 못해\nあなたにはわからない",
                sync_text="당신은 알지 못해",
            )
        ]
        segments = [
            app.WhisperSegment(10.0, 13.0, "あなたにはわからない"),
        ]

        selected = app.select_best_block_lines_for_transcript(blocks, segments)

        self.assertEqual(selected[0].sync_text, "あなたにはわからない")

    def test_reference_alignment_falls_back_to_prompted_whisper(self):
        blocks = [
            app.LyricBlock("最初の歌詞", "最初の歌詞"),
            app.LyricBlock("次の歌詞", "次の歌詞"),
        ]
        recognized = [
            app.WhisperSegment(8.0, 11.0, "最初の歌詞"),
            app.WhisperSegment(12.0, 15.0, "次の歌詞"),
        ]

        with patch.object(app, "transcribe", return_value=recognized) as mocked:
            captions, segments, score, status = app.align_blocks_to_reference_without_lrc(
                blocks,
                Path("reference.wav"),
                Path("work"),
                30.0,
                "medium",
                "ja",
                False,
            )

        self.assertEqual(len(captions), 2)
        self.assertEqual(segments, recognized)
        self.assertGreater(score, 0.9)
        self.assertIn("Whisper reference alignment", status)
        self.assertIn("最初の歌詞", mocked.call_args.kwargs["initial_prompt"])


if __name__ == "__main__":
    unittest.main()
