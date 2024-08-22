# pytthresh: Python Implementation of the TTHRESH Grid Data Compressor

### Installation

```
pip install -r requirements.txt
pip install .
```

### Example Run

```
python pytthresh/cli.py compress --input data/3D_sphere_64_uchar.raw --output compressed.pyt --shape 64,64,64 --dtype uint8 --eps 0.02 --topology tucker
```

### TODO

- Negabinary: saves mask hassle, allows progressive reconstruction
- Allow specifying a target compression ratio
- Profiling
- Case when input tensor is all zeros produces nan (normsq is 0)
- coveig method for tensor splitting
- Check different flatten orders for different cores
- Renormalize cores as they are compressed
- Optimize dimension order
- Work with float32 instead of 64?
