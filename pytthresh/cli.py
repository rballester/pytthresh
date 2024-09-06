from enum import Enum

import numpy as np
import typer
from typing_extensions import Annotated
from typing import Optional

from pytthresh import core


class Topology(str, Enum):
    tucker = "tucker"
    tt = "tt"
    ett = "ett"
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
    file = core.to_object(x, topology=topology, target_eps=eps, debug=debug)
    if compressed is not None:
        file.to_disk(compressed)
    if reconstructed is not None:
        reco = file.decompress(debug)
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
    debug: Annotated[
        bool,
        typer.Option(
            help="Turn on debug logging",
        ),
    ] = False,
):
    file = core.File.from_disk(compressed)
    reco = file.decompress(debug)
    if reconstructed is not None:
        with open(reconstructed, "wb") as f:
            reco.tofile(f)


if __name__ == "__main__":
    app()
