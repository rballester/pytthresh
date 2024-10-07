from enum import Enum
import time
import numpy as np
import typer
from typing_extensions import Annotated
from typing import Optional

from pytthresh import core


class Topology(str, Enum):
    tucker = "tucker"
    tt = "tt"
    ett = "ett"
    qtt = "qtt"
    single = "single"


app = typer.Typer()


@app.command()
def compress(
    original: Annotated[
        str,
        typer.Option(help="Path for the original tensor", show_default=False),
    ],
    shape: Annotated[
        str,
        typer.Option(
            help="Shape of the source tensor, as comma-separated integers or an expression like 64^3.",
            show_default=False,
        ),
    ],
    dtype: Annotated[
        str,
        typer.Option(
            help="Data type of the source tensor.",
            show_default=False,
        ),
    ],
    eps: Annotated[
        float,
        typer.Option(
            help="Target relative error, between 0 and 1.",
            show_default=False,
        ),
    ],
    topology: Annotated[
        Topology,
        typer.Option(
            help="Tensor network topology.",
            show_default=False,
        ),
    ],
    statistics: Annotated[
        bool,
        typer.Option(
            help="Print statistics",
        ),
    ] = False,
    debug: Annotated[
        bool,
        typer.Option(
            help="Turn on debug logging",
        ),
    ] = False,
    compressed: Annotated[
        str, typer.Option(help="Path for the compressed file")
    ] = None,
    reconstructed: Annotated[
        str,
        typer.Option(help="Output file to decompress to, if desired"),
    ] = None,
):
    if debug:
        statistics = True

    shape_list = []
    for x in shape.split(","):
        if "^" in x:
            x = x.split("^")
            shape_list.extend([int(x[0])] * int(x[1]))
        else:
            shape_list.append(int(x))
    assert len(shape_list) >= 2
    with open(original, "rb") as f:
        x = np.fromfile(f, dtype=np.dtype(dtype)).reshape(shape_list)
    start = time.time()
    file = core.to_object(
        x, topology=topology, target_eps=eps, statistics=statistics, debug=debug
    )
    compressiontime = time.time() - start
    if statistics or compressed is not None:
        bson_data = file.encode()
        nbytes = len(bson_data)
        if compressed is not None:
            with open(compressed, "wb") as f:
                f.write(bson_data)
            # file.to_disk(compressed)
    if statistics:
        print(
            f"oldbits = {x.nbytes*8}, newbits = {nbytes*8}, compressionratio = {x.nbytes/nbytes}, bpv = {nbytes*8/x.size}, compressiontime = {compressiontime}, compressionMBps = {x.nbytes/compressiontime/1e6}"
        )
    if reconstructed is not None:
        start = time.time()
        if compressed is None:  # Decompress from memory
            reco = file.decompress(debug)
        else:  # Decompress from disk
            reco = core.File.from_disk(compressed).decompress(debug)
        decompressiontime = time.time() - start
        if statistics:
            diffnorm = np.linalg.norm(x.astype(float) - reco.astype(float))
            eps = diffnorm / np.linalg.norm(x)
            rmse = diffnorm / np.sqrt(x.size)
            psnr = 20 * np.log10((float(x.max()) - float(x.min())) / (2 * rmse))
            print(
                f"eps = {eps}, rmse = {rmse}, psnr = {psnr}, decompressiontime = {decompressiontime}, decompressionMBps = {x.nbytes/decompressiontime/1e6}"
            )
        with open(reconstructed, "wb") as f:
            reco.tofile(f)


@app.command()
def decompress(
    compressed: Annotated[
        str,
        typer.Option(
            help="Path for the compressed file",
            show_default=False,
        ),
    ],
    reconstructed: Annotated[
        str,
        typer.Option(
            help="Path for the output decompressed tensor", show_default=False
        ),
    ],
    statistics: Annotated[
        bool,
        typer.Option(
            help="Print statistics",
        ),
    ] = False,
    debug: Annotated[
        bool,
        typer.Option(
            help="Turn on debug logging",
        ),
    ] = False,
):
    if debug:
        statistics = True

    file = core.File.from_disk(compressed)
    start = time.time()
    reco = file.decompress(debug)
    decompressiontime = time.time() - start
    if statistics:
        print(
            f"decompressiontime = {decompressiontime}, decompressionMBps = {reco.nbytes/decompressiontime/1e6}"
        )
    if reconstructed is not None:
        with open(reconstructed, "wb") as f:
            reco.tofile(f)


if __name__ == "__main__":
    app()
