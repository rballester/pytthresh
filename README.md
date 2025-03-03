# pytthresh: Python Entropy Coding of Tree Tensor Networks

Implementation of the paper [R. Ballester-Ripoll, R. Bujack: *Entropy Coding Compression of Tree Tensor Networks*](https://openreview.net/forum?id=hGsxrFF0tY), presented in [CoLoRAI 2025](https://april-tools.github.io/colorai/) (part of the [AAAI 2025 conference](https://aaai.org/conference/aaai/aaai-25/)).

![Entropy Coding Compression of Tree Tensor Networks](https://github.com/user-attachments/assets/ba0269f6-9642-4947-a15f-7b7f6ad12a22)

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

- Do the decomposition by iterative splitting
- Skip canonization by absorbing singular values into both sides
- Choose split method (svd, eig, svds, rsvd, etc.) based on matrix size
- Balance rank cutoff (higher speed) with entropy coding (higher quality)
- Negabinary: saves mask hassle, allows progressive reconstruction
- Algorithm to limit absolute error
- Reduce serialization overhead (currently a few KB's)
- Allow specifying a target compression ratio
- Case when input tensor is all zeros produces nan (normsq is 0)
- coveig method for tensor splitting
- Check different flatten orders for different cores
- Maybe: encode in ALS fashion to avoid accumulating errors?
- Optimize dimension order
- Work with float32 instead of 64?
- Split into bricks and process in parallel
