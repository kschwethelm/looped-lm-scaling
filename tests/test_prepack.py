"""
Tests for packing functions from prepack.py.

python -m pytest tests/test_prepack.py -v
"""

from scripts.prepack import _pack_row, _pack_rows


class TestPackRow:
    """Test the BOS-aligned best-fit packing algorithm (single row)."""

    def test_basic_packing(self):
        """Multiple small docs get packed into a single row."""
        doc_buffer = [[1, 2, 3], [4, 5], [6, 7, 8, 9, 10]]
        row = _pack_row(doc_buffer, row_capacity=10)
        assert len(row) == 10

    def test_exact_fit(self):
        """Doc that exactly fills a row should produce that row with no cropping."""
        doc = list(range(10))
        doc_buffer = [doc]
        row = _pack_row(doc_buffer, row_capacity=10)
        assert row == list(range(10))
        assert doc_buffer == []  # buffer should be consumed

    def test_doc_larger_than_capacity_gets_cropped(self):
        """A doc larger than row_capacity should be cropped to fill the row."""
        doc_buffer = [list(range(20))]
        row = _pack_row(doc_buffer, row_capacity=10)
        assert len(row) == 10
        assert row == list(range(10))  # first 10 tokens

    def test_single_doc_smaller_than_capacity(self):
        """Single doc smaller than capacity: buffer empties before row fills,
        returns None since the row can't be completed."""
        doc_buffer = [[1, 2, 3]]
        assert _pack_row(doc_buffer, row_capacity=10) is None

    def test_bestfit_picks_largest_fitting(self):
        """Best-fit should prefer the largest doc that fits entirely."""
        # Row capacity = 10, docs: [3 tokens], [7 tokens], [5 tokens]
        # First pick: largest fitting = [7 tokens], remaining = 3
        # Then: [3 tokens] fits exactly
        doc_buffer = [[1, 1, 1], [2, 2, 2, 2, 2, 2, 2], [3, 3, 3, 3, 3]]
        row = _pack_row(doc_buffer, row_capacity=10)
        assert len(row) == 10
        # Should start with the 7-token doc, then the 3-token doc
        assert row[:7] == [2, 2, 2, 2, 2, 2, 2]
        assert row[7:] == [1, 1, 1]

    def test_buffer_consumed_in_place(self):
        """_pack_row modifies doc_buffer in place, removing consumed docs."""
        doc_buffer = [list(range(5)) for _ in range(10)]
        _pack_row(doc_buffer, row_capacity=10)
        # 1 row * 2 docs per row = 2 docs consumed
        assert len(doc_buffer) == 8

    def test_crops_shortest_when_nothing_fits(self):
        """When no doc fits the remaining space, crops the shortest doc."""
        # Row capacity = 10, one doc of 8 tokens, one of 5 tokens
        # First: 8-token doc fits, remaining = 2
        # Nothing fits entirely -> crop shortest (5-token) to fill 2
        doc_buffer = [[1] * 8, [2] * 5]
        row = _pack_row(doc_buffer, row_capacity=10)
        assert row == [1] * 8 + [2] * 2


class TestPackRows:
    """Test the row generator with buffer refilling."""

    def test_produces_correct_row_lengths(self):
        """All rows should have exactly row_capacity tokens."""
        docs = [[i] * 5 for i in range(20)]
        rows = list(_pack_rows(iter(docs), row_capacity=10, buffer_size=5))
        assert all(len(r) == 10 for r in rows)

    def test_empty_input(self):
        """No docs -> no rows."""
        rows = list(_pack_rows(iter([]), row_capacity=10, buffer_size=5))
        assert rows == []

    def test_buffer_stays_topped_up(self):
        """With enough docs and small buffer_size, should produce many rows
        without the buffer draining to empty mid-packing."""
        # 100 docs of length 3, capacity 9 -> each row fits 3 docs -> ~33 rows
        docs = [[1, 2, 3] for _ in range(100)]
        rows = list(_pack_rows(iter(docs), row_capacity=9, buffer_size=10))
        assert len(rows) == 33  # 99 docs used (3 per row), 1 leftover can't fill a row
        assert all(len(r) == 9 for r in rows)

    def test_drains_all_docs(self):
        """All docs that can form complete rows are consumed."""
        # 10 docs of length 10, capacity 10 -> 10 rows, nothing wasted
        docs = [list(range(10)) for _ in range(10)]
        rows = list(_pack_rows(iter(docs), row_capacity=10, buffer_size=3))
        assert len(rows) == 10
