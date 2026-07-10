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


if __name__ == "__main__":
    unittest.main()
