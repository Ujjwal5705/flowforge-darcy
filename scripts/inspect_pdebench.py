"""Read public HDF5 metadata through HTTP ranges; never log signed URLs."""

import io
import json
import re
import urllib.request
from collections import OrderedDict

import h5py


class RangeReader(io.RawIOBase):
    def __init__(self, url: str, block_size: int = 64 * 1024):
        super().__init__()
        req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
        with urllib.request.urlopen(req, timeout=60) as response:
            if response.status != 206:
                raise RuntimeError("Dataset server must support byte ranges")
            content_range = response.headers["Content-Range"]
            self.size = int(re.fullmatch(r"bytes 0-0/(\d+)", content_range).group(1))
            self.url = response.url  # Ephemeral public redirect retained only in memory.
        self.position = 0
        self.block_size = block_size
        self.cache = OrderedDict()
        self.transferred = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        if whence == 0:
            self.position = offset
        elif whence == 1:
            self.position += offset
        elif whence == 2:
            self.position = self.size + offset
        if self.position < 0:
            raise ValueError("Negative seek")
        return self.position

    def readinto(self, buffer):
        data = self.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)

    def read(self, size=-1):
        if size < 0:
            size = self.size - self.position
        end = min(self.size, self.position + size)
        chunks = []
        while self.position < end:
            block = self.position // self.block_size
            start = block * self.block_size
            if block not in self.cache:
                stop = min(self.size, start + self.block_size) - 1
                req = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{stop}"})
                with urllib.request.urlopen(req, timeout=90) as response:
                    if response.status != 206:
                        raise RuntimeError("Unexpected full-file response")
                    data = response.read(self.block_size + 1)
                if len(data) != stop - start + 1:
                    raise RuntimeError("Incomplete dataset range")
                self.transferred += len(data)
                self.cache[block] = data
                if len(self.cache) > 8:
                    self.cache.popitem(last=False)
            count = min(end - self.position, self.block_size - (self.position - start))
            chunks.append(self.cache[block][self.position - start : self.position - start + count])
            self.position += count
        return b"".join(chunks)


if __name__ == "__main__":
    reader = RangeReader("https://darus.uni-stuttgart.de/api/access/datafile/133219")
    with h5py.File(reader, "r") as file:
        report = {
            "size": reader.size,
            "datasets": {
                k: {
                    "shape": list(v.shape),
                    "dtype": str(v.dtype),
                    "chunks": v.chunks,
                    "offset": v.id.get_offset(),
                }
                for k, v in file.items()
                if isinstance(v, h5py.Dataset)
            },
            "attributes": {k: str(v) for k, v in file.attrs.items()},
        }
        for key in file:
            dataset = file[key]
            if isinstance(dataset, h5py.Dataset):
                sample = dataset[0] if dataset.ndim > 1 else dataset[:]
                report["datasets"][key]["sample_range"] = [float(sample.min()), float(sample.max())]
        report["bytes_transferred"] = reader.transferred
        print(json.dumps(report, indent=2))
