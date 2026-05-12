"""Latency + accuracy eval for an exported wake head."""

from .latency import measure as measure_latency
from .accuracy import measure as measure_accuracy

__all__ = ["measure_latency", "measure_accuracy"]
