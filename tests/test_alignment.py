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

    def test_fuzzy_full_song_duplicate_with_joined_boundary_is_removed(self):
        lines = [f"line {index} unique lyric content section" for index in range(30)]
        song = "\n\n".join(lines)
        changed_copy = song.replace("line 2 unique", "line two unique", 1)
        duplicated = song + "\u300c" + changed_copy

        cleaned, removed = app.remove_duplicate_full_lyrics(duplicated)

        self.assertTrue(removed)
        self.assertEqual(cleaned, song)

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

    def test_forced_timing_requires_two_agreeing_candidates(self):
        caption = app.CaptionLine(10.0, 14.0, "line")
        candidates = [
            app.ForcedTimingCandidate(10.7, 14.4, 0.2, 0.0, "1-6"),
            app.ForcedTimingCandidate(10.9, 14.6, 0.1, 0.0, "1-8"),
        ]

        timing = app._select_forced_timing_consensus(caption, candidates)

        self.assertIsNotNone(timing)
        self.assertAlmostEqual(timing[0], 10.8)
        self.assertAlmostEqual(timing[1], 14.5)

    def test_forced_timing_rejects_disagreement_and_collapsed_words(self):
        caption = app.CaptionLine(10.0, 14.0, "line")
        disagreement = [
            app.ForcedTimingCandidate(8.5, 12.0, 0.3, 0.0, "1-6"),
            app.ForcedTimingCandidate(11.0, 15.0, 0.3, 0.0, "1-8"),
        ]
        collapsed = [
            app.ForcedTimingCandidate(10.2, 13.0, 0.01, 1.0, "1-6"),
            app.ForcedTimingCandidate(10.3, 13.1, 0.02, 1.0, "1-8"),
        ]

        self.assertIsNone(app._select_forced_timing_consensus(caption, disagreement))
        self.assertIsNone(app._select_forced_timing_consensus(caption, collapsed))

    def test_forced_timing_rejects_large_weak_pair_correction(self):
        caption = app.CaptionLine(10.0, 14.0, "line")
        candidates = [
            app.ForcedTimingCandidate(11.4, 15.0, 0.12, 0.0, "1-6"),
            app.ForcedTimingCandidate(11.5, 15.1, 0.12, 0.0, "1-8"),
        ]

        self.assertIsNone(app._select_forced_timing_consensus(caption, candidates))

    def test_forced_timing_allows_large_three_window_local_correction(self):
        caption = app.CaptionLine(119.2, 122.8, "line")
        candidates = [
            app.ForcedTimingCandidate(121.66, 124.0, 0.18, 0.36, "17-24"),
            app.ForcedTimingCandidate(121.64, 124.0, 0.16, 0.36, "19-24"),
            app.ForcedTimingCandidate(121.63, 124.0, 0.16, 0.64, "21-28"),
        ]

        timing = app._select_forced_timing_consensus(caption, candidates)

        self.assertIsNotNone(timing)
        self.assertAlmostEqual(timing[0], 121.65, delta=0.02)

    def test_lrc_text_is_preserved_for_vocal_refinement(self):
        blocks = [app.LyricBlock("雨あめ\ntranslation", "雨あめ")]
        captions, _score, _count = app.align_blocks_to_lrc(
            blocks,
            "[00:01.00] 雨の匂い\n[00:05.00] ",
            8.0,
        )

        self.assertEqual(captions[0].text, "雨あめ\ntranslation")
        self.assertEqual(captions[0].sync_text, "雨の匂い")

    def test_first_line_acoustic_anchor_does_not_shift_later_lines(self):
        caption = app.CaptionLine(0.6, 7.6, "first")
        candidates = [
            app.ForcedTimingCandidate(3.82, 8.2, 0.4, 0.2, "short"),
            app.ForcedTimingCandidate(3.86, 8.3, 0.5, 0.2, "wide"),
        ]

        start = app._select_first_line_acoustic_start(caption, 7.7, candidates)

        self.assertAlmostEqual(start, 3.84)

    def test_first_line_acoustic_anchor_rejects_weak_or_late_consensus(self):
        caption = app.CaptionLine(0.6, 7.6, "first")
        weak = [
            app.ForcedTimingCandidate(3.8, 8.2, 0.1, 0.2, "short"),
            app.ForcedTimingCandidate(3.8, 8.2, 0.1, 0.2, "wide"),
        ]

        self.assertIsNone(app._select_first_line_acoustic_start(caption, 7.7, weak))
        self.assertIsNone(app._select_first_line_acoustic_start(caption, 4.0, [
            app.ForcedTimingCandidate(3.8, 4.2, 0.4, 0.2, "short"),
            app.ForcedTimingCandidate(3.8, 4.2, 0.4, 0.2, "wide"),
        ]))

    def test_interlude_boundary_corrects_both_sides_without_global_shift(self):
        captions = [
            app.CaptionLine(52.4, 55.3, "before"),
            app.CaptionLine(55.4, 60.8, "edge"),
            app.CaptionLine(73.2, 78.5, "after"),
            app.CaptionLine(78.6, 83.9, "later"),
        ]
        before_candidates = [
            app.ForcedTimingCandidate(55.35, 60.35, 0.08, 0.1, "1-2"),
            app.ForcedTimingCandidate(56.59, 60.35, 0.08, 0.1, "2-3"),
            app.ForcedTimingCandidate(57.18, 60.54, 0.06, 0.2, "2-4"),
        ]
        after_candidates = [
            app.ForcedTimingCandidate(60.8, 79.2, 0.2, 0.3, "2-3"),
            app.ForcedTimingCandidate(71.67, 79.2, 0.2, 0.1, "3-4"),
            app.ForcedTimingCandidate(71.69, 79.2, 0.2, 0.1, "3-5"),
        ]

        before = app._select_interlude_boundary_timing(1, captions, before_candidates)
        after = app._select_interlude_boundary_timing(2, captions, after_candidates)

        self.assertIsNotNone(before)
        self.assertAlmostEqual(before[0], 56.59)
        self.assertIsNotNone(after)
        self.assertAlmostEqual(after[0], 71.68)
        self.assertEqual(captions[3].start, 78.6)

    def test_contextual_pair_ignores_current_line_outlier(self):
        caption = app.CaptionLine(108.94, 111.68, "line")
        candidates = [
            app.ForcedTimingCandidate(110.09, 112.81, 0.086, 0.2, "13-20"),
            app.ForcedTimingCandidate(110.62, 112.76, 0.092, 0.1, "17-24"),
            app.ForcedTimingCandidate(107.44, 112.81, 0.074, 0.1, "19-24"),
        ]

        timing = app._select_contextual_pair_timing(18, caption, candidates)

        self.assertIsNotNone(timing)
        self.assertAlmostEqual(timing[0], 110.355)

    def test_contextual_pair_requires_prior_context_and_moderate_confidence(self):
        caption = app.CaptionLine(108.94, 111.68, "line")
        candidates = [
            app.ForcedTimingCandidate(110.1, 112.8, 0.05, 0.1, "13-20"),
            app.ForcedTimingCandidate(110.2, 112.8, 0.05, 0.1, "17-24"),
            app.ForcedTimingCandidate(110.2, 112.8, 0.5, 0.1, "19-24"),
        ]

        self.assertIsNone(app._select_contextual_pair_timing(18, caption, candidates))

    def test_auto_language_uses_the_sync_text_script(self):
        korean = [app.LyricBlock("한국어\ntranslation", "한국어 가사입니다")]
        japanese = [app.LyricBlock("日本語", "これは日本語です")]

        self.assertEqual(app.infer_lyric_language(korean, "auto"), "ko")
        self.assertEqual(app.infer_lyric_language(japanese, "auto"), "ja")
        self.assertEqual(app.infer_lyric_language(korean, "en"), "en")

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
