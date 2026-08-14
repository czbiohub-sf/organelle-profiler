"""
Groupby aggregation helpers for feature extraction.

This module is separate from feature_extraction_slurm.py to allow proper
pickling when running via submitit (which executes the main script as __main__).
"""

import os
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm


def _agg_chunk(grouped, chunk_cols, agg_funcs):
    """Aggregate a chunk of columns."""
    chunk_result = grouped[chunk_cols].agg(agg_funcs)
    chunk_result.columns = ["_".join(col).strip() for col in chunk_result.columns.values]
    return chunk_result


def vectorized_groupby_agg(df, groupby_col, feature_cols, agg_funcs, chunk_size=200, n_threads=None):
    """
    Parallel chunked groupby aggregation with progress bar.

    Uses thread pool to parallelize across column chunks. Threads work well here
    because pandas releases the GIL during numpy operations.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    groupby_col : str
        Column to group by (e.g., 'sgRNA', 'gene_name')
    feature_cols : list
        List of feature columns to aggregate
    agg_funcs : list
        List of aggregation functions (e.g., ['mean', 'std'])
    chunk_size : int
        Number of columns to process per chunk (default 200)
    n_threads : int
        Number of threads (default: min(8, cpu_count))

    Returns
    -------
    pd.DataFrame
        Aggregated dataframe with groupby_col as first column
    """
    if n_threads is None:
        n_threads = min(8, int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 4)))

    # Pre-compute groupby object (index computation done once)
    grouped = df.groupby(groupby_col)

    # Split into chunks
    chunks = [feature_cols[i:i + chunk_size] for i in range(0, len(feature_cols), chunk_size)]
    n_chunks = len(chunks)

    # Process chunks in parallel with progress bar
    results = []
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        futures = [executor.submit(_agg_chunk, grouped, chunk, agg_funcs) for chunk in chunks]

        for future in tqdm(futures, total=n_chunks,
                          desc=f"  Aggregating {len(feature_cols)} cols ({n_threads} threads)",
                          unit="chunk"):
            results.append(future.result())

    # Combine chunks
    combined = pd.concat(results, axis=1)
    return combined.reset_index()
