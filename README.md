# pytthresh: Python Implementation of the TTHRESH Grid Data Compressor

### Installation

```
pip install .
```

### Example Run

Compression:

```
python pytthresh/cli.py compress --original data/3D_sphere_64_uchar.raw --compressed compressed.pyt --shape 64,64,64 --dtype uint8 --eps 0.02 --topology tucker
```

Decompression:

```
python pytthresh/cli.py decompress --compressed compressed.pyt --reconstructed decompressed.raw
```

All in one, while printing statistics:

```
python pytthresh/cli.py compress --original data/3D_sphere_64_uchar.raw --reconstructed decompressed.raw --shape 64,64,64 --dtype uint8 --eps 0.02 --topology tucker --statistics
```

### TODO

- Negabinary: saves mask hassle, allows progressive reconstruction
- Algorithm to limit absolute error
- Allow specifying a target compression ratio
- Profiling
- Case when input tensor is all zeros produces nan (normsq is 0)
- coveig method for tensor splitting
- Check different flatten orders for different cores
- Renormalize cores as they are compressed
- Optimize dimension order
- Work with float32 instead of 64?
- Split into bricks and process in parallel