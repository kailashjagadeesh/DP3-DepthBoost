#!/usr/bin/env python3
"""
zarr_visualizer.py

Usage:
    python zarr_visualizer.py /path/to/file_or_folder.zarr
"""

import argparse
import numpy as np
import zarr
from zarr.hierarchy import Group
from zarr.core import Array


def inspect_node(node, path="", indent=0, max_samples=3):
    pad = " " * indent

    if isinstance(node, Group):
        print(f"{pad}Group: {path or '/'}")
        if node.attrs:
            print(f"{pad}  attrs: {dict(node.attrs)}")

        # Iterate over children (subgroups and arrays)
        for name, child in node.items():
            child_path = f"{path}/{name}" if path else name
            inspect_node(child, child_path, indent + 2, max_samples)

    elif isinstance(node, Array):
        print(f"{pad}Array: {path}")
        print(f"{pad}  shape:  {node.shape}")
        print(f"{pad}  dtype:  {node.dtype}")
        print(f"{pad}  chunks: {node.chunks}")
        if node.attrs:
            print(f"{pad}  attrs: {dict(node.attrs)}")

        # Show a small sample of data to understand what it looks like
        try:
            # Take a small slice along the first dimension (or whole array if small)
            if node.size == 0:
                print(f"{pad}  sample: <empty array>")
            else:
                # Build an index like [0:K, ...] depending on dimensionality
                idx = []
                for ax_len in node.shape:
                    if ax_len > max_samples:
                        idx.append(slice(0, max_samples))
                    else:
                        idx.append(slice(0, ax_len))
                sample = np.asarray(node[tuple(idx)])
                print(f"{pad}  sample (up to {max_samples} per dim):")
                print(f"{pad}    {sample}")
        except Exception as e:
            print(f"{pad}  (could not read sample: {e})")

    else:
        print(f"{pad}Unknown node type at {path}: {type(node)}")


def main():
    parser = argparse.ArgumentParser(
        description="Inspect structure of a Zarr file (images, depth, pointclouds, etc.)."
    )
    parser.add_argument(
        "zarr_path",
        help="Path to Zarr directory or .zarr/.zip store (e.g. /data/rgb.zarr)",
    )
    args = parser.parse_args()

    # zarr.open() works for groups or arrays, directory or zipped
    root = zarr.open(args.zarr_path, mode="r")

    print(f"Opened Zarr store: {args.zarr_path}")
    print(f"Root type: {type(root)}")

    # If the root is already an Array, just inspect it
    if isinstance(root, Array):
        inspect_node(root, path="", indent=0)
    elif isinstance(root, Group):
        inspect_node(root, path="", indent=0)
    else:
        print(f"Unexpected root type: {type(root)}")


if __name__ == "__main__":
    main()