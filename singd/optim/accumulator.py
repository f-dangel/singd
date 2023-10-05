"""Implements an accumulator over micro-batches/devices."""

from typing import Any


class BatchAccumulator:
    """Accumulate a quantity over multiple batches.

    Assumes that the quantity is averaged via mean or sum.

    The quantity's class must support multiplication with a scalar and addition.
    """

    def __init__(self, batch_averaged: bool = True):
        """Initialize the accumulator.

        Args:
            batch_averaged: Whether the quantity is averaged over the batch.
                If ``False``, assumes sum. Default: ``True``.
        """
        self.value = None
        self.batch_averaged = batch_averaged
        self.batch_size_total = 0

    def update(self, other: Any, batch_size: int):
        """Update the accumulator with a new value.

        Args:
            other: The new value to add to the accumulator.
            batch_size: The size of the batch that ``other`` is from.
        """
        if self.value is None:
            self.value = other
            self.batch_size_total = batch_size
        else:
            if self.batch_averaged:
                scale = self.batch_size_total / (self.batch_size_total + batch_size)
                self.value = self.value * scale + other * (1 - scale)
            else:
                self.value += other

            self.batch_size_total += batch_size
