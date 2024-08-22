import typer
from enum import Enum
import numpy as np
from pytthresh import core
from typing_extensions import Annotated

app = typer.Typer()


class Dtype(str, Enum):
    int8 = "int8"
    int32 = "int32"
    uint8 = "uint8"
    uint32 = "uint32"
    float32 = "float32"
    float64 = "float64"


class Topology(str, Enum):
    tucker = "tucker"
    tt = "tt"
    ett = "ett"
    single = "single"


@app.command()
def compress(
    input: Annotated[str, typer.Option(help="Path of the source tensor")],
    output: Annotated[
        str,
        typer.Option(help="Path for the destination compressed file"),
    ],
    shape: Annotated[
        str,
        typer.Option(help="Shape of the source tensor, as comma-separated integers"),
    ],
    dtype: Annotated[Dtype, typer.Option(help="Data type of the source tensor")],
    eps: Annotated[float, typer.Option(help="Target relative error, between 0 and 1")],
    topology: Annotated[Topology, typer.Option(help="Tensor network topology")],
    debug: Annotated[bool, typer.Option("--debug")] = False,
):
    shape = [int(x) for x in shape.split(",")]
    with open(input, "rb") as f:
        x = np.fromfile(f, dtype=getattr(np, dtype)).reshape(shape).astype(np.float64)
    file = core.compress(x, topology=topology, target_eps=eps, debug=debug)
    file.save(output)


@app.command()
def decompress():
    pass


if __name__ == "__main__":
    app()
