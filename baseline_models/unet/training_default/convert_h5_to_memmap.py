"""
One-time conversion: h5py precomputed prediction files -> numpy memmap (.npy).
Usage: python convert_h5_to_memmap.py <input.h5> <output.npy>
"""
import h5py
import numpy as np
import sys
import os

def convert(h5_path, npy_path, chunk_size=100_000):
    print(f"Converting {h5_path} -> {npy_path}")
    with h5py.File(h5_path, 'r') as f:
        N, D = f['data'].shape
        dtype = f['data'].dtype
    print(f"  Shape: ({N:,}, {D}), dtype: {dtype}, "
          f"size: {N * D * np.dtype(dtype).itemsize / 1e9:.2f} GB")

    out = np.lib.format.open_memmap(npy_path, mode='w+', dtype=dtype, shape=(N, D))
    with h5py.File(h5_path, 'r') as f:
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            out[start:end] = f['data'][start:end]
            print(f"  {end:,}/{N:,} rows", end='\r', flush=True)
    del out
    print(f"\nDone -> {npy_path}  ({os.path.getsize(npy_path)/1e9:.2f} GB on disk)")

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python convert_h5_to_memmap.py <input.h5> <output.npy>")
        sys.exit(1)
    convert(sys.argv[1], sys.argv[2])
