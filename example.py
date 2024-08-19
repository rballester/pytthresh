import pytthresh as pyt
import time
import numba

numba.config.NUMBA_NUM_THREADS = 8

start = time.time()
pyt.compress("data/3D_sphere_64_uchar.raw", [64, 64, 64], "uint8", 0.02, "tucker")
print(time.time() - start)
