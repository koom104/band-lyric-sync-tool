import unittest

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


if __name__ == "__main__":
    unittest.main()
