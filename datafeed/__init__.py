"""Data pipeline: exchange downloads, product registry, roll calendar, and loading.

Bars come from CZCE, SHFE, and DCE; ``sources`` holds the per-exchange adapters
and ``data_update`` does the venue-agnostic cleaning and OI weighting.
"""
