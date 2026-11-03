"""Fetch a bounded, reproducible subset of the public PDEBench Darcy shard.

Only the permanent public dataset URL is stored. Temporary redirects stay in memory.
"""

import argparse
import hashlib
import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np

from inspect_pdebench import RangeReader

URL = "https://darus.uni-stuttgart.de/api/access/datafile/133219"


def fetch_range(url, start, end, total):
    for attempt in range(3):
        try:
            request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(request, timeout=90) as response:
                expected = f"bytes {start}-{end}/{total}"
                if response.status != 206 or response.headers.get("Content-Range") != expected:
                    raise RuntimeError("Dataset server returned an incompatible byte range")
                content = response.read(end - start + 2)
            if len(content) != end - start + 1:
                raise RuntimeError("Incomplete dataset download")
            return start, content
        except Exception:
            if attempt == 2:
                raise RuntimeError("Dataset range download failed after three attempts") from None
            time.sleep(attempt + 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=1024)
    parser.add_argument("--output", type=Path, default=Path("data/pdebench-darcy-beta1.npz"))
    args = parser.parse_args()
    if not 16 <= args.count <= 2048:
        parser.error("count must be between 16 and 2048 (bounded subset only)")
    if args.output.exists():
        raise RuntimeError("Output already exists; inspect it or choose a new output path")
    started = time.perf_counter()
    reader = RangeReader(URL)
    with h5py.File(reader, "r") as file:
        if float(file.attrs["beta"]) != 1.0:
            raise RuntimeError("Unexpected PDE forcing")
        coordinates = {k: np.asarray(file[k])[::4].copy()
                       for k in ["x-coordinate", "y-coordinate"]}
        descriptors = {}
        for key in ["nu", "tensor"]:
            dataset = file[key]
            expected = (10000, 128, 128) if key == "nu" else (10000, 1, 128, 128)
            if dataset.shape != expected or dataset.dtype != np.dtype("float32"):
                raise RuntimeError("Unexpected HDF5 layout")
            if dataset.chunks is not None or dataset.id.get_offset() is None:
                raise RuntimeError("This downloader requires the verified contiguous shard")
            descriptors[key] = dataset.id.get_offset()
    arrays = {}
    transferred = reader.transferred
    for key, offset in descriptors.items():
        nbytes = args.count * 128 * 128 * 4
        output = bytearray(nbytes)
        block_size = 2 * 1024 * 1024
        jobs = [(offset + i, min(offset+nbytes-1, offset+i+block_size-1))
                for i in range(0, nbytes, block_size)]
        completed = 0
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(fetch_range, reader.url, a, b, reader.size) for a, b in jobs]
            for future in as_completed(futures):
                begin, content = future.result()
                output[begin-offset:begin-offset+len(content)] = content
                transferred += len(content)
                completed += 1
                print(f"{key}: {completed}/{len(jobs)} ranges verified", flush=True)
        arrays[key] = np.frombuffer(output, dtype="<f4").reshape(args.count, 128, 128)[:, ::4, ::4].copy()
        del output
    for array in arrays.values():
        if not np.isfinite(array).all():
            raise RuntimeError("Non-finite source data")
    if not np.isin(arrays["nu"], np.array([0.1, 1.0], dtype=np.float32)).all():
        raise RuntimeError("Unexpected coefficient range")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, x=arrays["nu"], y=arrays["tensor"],
                        x_coordinates=coordinates["x-coordinate"],
                        y_coordinates=coordinates["y-coordinate"])
    metadata = {
        "source": "PDEBench", "doi": "10.18419/darus-2986", "file_id": 133219,
        "filename": "2D_DarcyFlow_beta1.0_Train.hdf5", "url": URL,
        "license": "CC BY 4.0", "source_size_bytes": reader.size,
        "published_source_md5": "81694ed31306ff2e5f6b76349b0b4389",
        "full_source_checksum_verified": False,
        "checksum_note": "Only bounded ranges were downloaded; full-shard MD5 was not checked.",
        "sample_indices": list(range(args.count)), "source_resolution": [128, 128],
        "resolution": [32, 32], "subsampling": "array[:, ::4, ::4], no interpolation",
        "array_layout": "sample, x_index, y_index", "beta": 1.0,
        "bytes_transferred": transferred, "download_seconds": time.perf_counter()-started,
        "array_sha256": {k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in arrays.items()},
        "ranges": {k: [float(v.min()), float(v.max())] for k, v in arrays.items()},
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2)+"\n")
    print(json.dumps({k: v for k, v in metadata.items() if k != "sample_indices"}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        # urllib exceptions can contain signed redirect URLs; do not print them.
        print(f"Dataset preparation failed ({type(error).__name__}); no output is trusted.", file=sys.stderr)
        sys.exit(1)
