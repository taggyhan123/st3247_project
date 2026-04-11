"""
Load observed data CSV files.

Data directory: ../data/ relative to this file.
"""

import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"


def load_infected_timeseries(data_dir=None):
    """Load infected_timeseries.csv → (R, T+1) array."""
    d = Path(data_dir) if data_dir else DATA_DIR
    df = pd.read_csv(d / "infected_timeseries.csv")
    return df.pivot(index="replicate_id", columns="time", values="infected_fraction").values


def load_rewiring_timeseries(data_dir=None):
    """Load rewiring_timeseries.csv → (R, T+1) array."""
    d = Path(data_dir) if data_dir else DATA_DIR
    df = pd.read_csv(d / "rewiring_timeseries.csv")
    return df.pivot(index="replicate_id", columns="time", values="rewire_count").values


def load_degree_histograms(data_dir=None):
    """Load final_degree_histograms.csv → (R, 31) array."""
    d = Path(data_dir) if data_dir else DATA_DIR
    df = pd.read_csv(d / "final_degree_histograms.csv")
    return df.pivot(index="replicate_id", columns="degree", values="count").values


def load_all(data_dir=None):
    """Load all three datasets."""
    return (
        load_infected_timeseries(data_dir),
        load_rewiring_timeseries(data_dir),
        load_degree_histograms(data_dir),
    )
