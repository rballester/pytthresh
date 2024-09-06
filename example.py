import time
import numpy as np
import pytthresh as pyt


with open("data/3D_sphere_64_uchar.raw", "rb") as f:
    x = np.fromfile(f, dtype=np.uint8).reshape([64, 64, 64]).astype(np.float64)

## Compression
start = time.time()
compressed = pyt.core.to_object(x, topology="tucker", target_eps=0.02)
print("Compression time:", time.time() - start)

## Decompression
start = time.time()
reco = compressed.decompress()
print("Decompression time:", time.time() - start)

## Metrics
print("Compression ratio:", x.size / compressed.n_bytes())
print("Final error:", np.linalg.norm(x - reco.data) / np.linalg.norm(x))
