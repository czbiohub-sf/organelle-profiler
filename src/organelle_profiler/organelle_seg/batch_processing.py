"""
Batch Processing Utilities Module
==================================

Provides utilities for batch processing organelle segmentation tasks.

This module handles Dask cluster setup and configuration for distributed
processing of multiple positions and channels.

Main functions:
- setup_dask_cluster(): Configure and initialize Dask LocalCluster and Client
"""

import os
from dask.distributed import Client, LocalCluster


def setup_dask_cluster(num_workers: int = 1):
    """
    Setup Dask LocalCluster and Client with standard configuration.

    This function creates a Dask cluster optimized for position-level
    processing where each position requires significant memory (~100-150 GB).

    Args:
        num_workers: Number of Dask workers (default: 1 for position-level processing)
            For position-level processing, use 1 worker to avoid OOM errors.
            Each worker processes positions sequentially.

    Returns:
        tuple: (client, cluster) - Dask Client and LocalCluster instances

    Configuration:
        - Single thread per worker (let NumPy/BLAS manage parallelism)
        - No memory limit (let OS handle memory management)
        - Respects OMP_NUM_THREADS environment variable for NumPy parallelism

    Example:
        >>> client, cluster = setup_dask_cluster(num_workers=1)
        >>> # Submit jobs to client
        >>> client.close()
        >>> cluster.close()
    """
    omp_threads = os.environ.get("OMP_NUM_THREADS", "not set")
    print(f"Using {num_workers} Dask worker (position needs ~100-150 GB RAM)")
    print(f"  NumPy/BLAS operations will use OMP_NUM_THREADS={omp_threads} for parallelism")

    cluster_kwargs = {
        "n_workers": num_workers,
        "threads_per_worker": 1,  # Single thread - let NumPy/CuPy manage their own threading
        "memory_limit": "0",  # Disable memory limit - let OS handle it
    }

    cluster = LocalCluster(**cluster_kwargs)
    client = Client(cluster)
    # print(f"Dask dashboard: {client.dashboard_link}")

    return client, cluster
