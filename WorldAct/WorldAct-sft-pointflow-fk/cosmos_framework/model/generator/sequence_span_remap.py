"""Remap CPU span metadata when tokens are appended to each packed sample."""

from bisect import bisect_right
from dataclasses import replace
from itertools import accumulate


class SpanRemapper:
    """Mirror the device index map without reading device scalars per span."""

    def __init__(self, old_lengths, new_lengths):
        self.ends = list(accumulate(old_lengths))
        old_starts = [0, *self.ends[:-1]]
        new_starts = [0, *list(accumulate(new_lengths))[:-1]]
        self.shifts = [new - old for old, new in zip(old_starts, new_starts, strict=True)]

    def index(self, index):
        length = self.ends[-1]
        if index < 0:
            index += length
        if not 0 <= index < length:
            raise IndexError("Span index outside packed sequence")
        return index + self.shifts[bisect_right(self.ends, index)]

    def __call__(self, spans):
        result = []
        for span in spans:
            start = self.index(span.sequence_start)
            if span.sequence_len and self.index(span.sequence_start + span.sequence_len - 1) != (
                start + span.sequence_len - 1
            ):
                raise ValueError("A modality span crosses sample boundaries")
            result.append(replace(span, sequence_start=start))
        return result
