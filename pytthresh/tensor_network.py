import numpy as np
import quimb.tensor as qtn
from collections import defaultdict
import cotengra as ctg
import itertools


def skeleton(N: int, topology: str) -> dict:
    assert topology in ("tucker", "tt", "ett")

    nodes = defaultdict(lambda: set())
    if topology == "tucker":
        nodes[len(nodes)] = [f"r{i}" for i in range(N)]
        for i in range(N):
            nodes[len(nodes)] = [f"i{i}", f"r{i}"]
        return nodes
    if topology == "tt":
        letter = f"i"
    else:
        letter = f"s"
    nodes[len(nodes)] = [f"i0", f"r{1}"]
    for i in range(1, N - 1):
        nodes[len(nodes)] = [f"r{i}", f"{letter}{i}", f"r{i+1}"]
    nodes[len(nodes)] = [f"r{N-1}", f"i{N-1}"]
    if topology == "ett":
        for i in range(1, N - 1):
            nodes[len(nodes)] = [f"i{i}", f"s{i}"]
    return nodes


def peel_ranks(inputs, ranks):

    unique, counts = np.unique(list(itertools.chain(*inputs)), return_counts=True)
    output = unique[counts == 1]

    hg = ctg.get_hypergraph(inputs, output, None)
    graph = defaultdict(lambda: dict())
    for index, edge in hg.edges.items():
        if len(edge) == 1:
            graph[edge[0]][index] = None
        else:
            assert len(edge) == 2
            graph[edge[0]][index] = edge[1]
            graph[edge[1]][index] = edge[0]

    def inwards(tensor, incoming_edge):
        for edge in graph[tensor].keys():
            if edge != incoming_edge and graph[tensor][edge] is not None:
                inwards(graph[tensor][edge], edge)
                ranks[edge] = min(tensor_sizes[graph[tensor][edge]], ranks[edge])
        prod = 1
        for edge in graph[tensor].keys():
            if edge != incoming_edge:
                prod *= ranks[edge]
        tensor_sizes[tensor] = prod
        return tensor_sizes[tensor]

    def outwards(tensor, incoming_edge):
        prod = 1
        for edge in graph[tensor].keys():
            prod *= ranks[edge]
        for edge in graph[tensor].keys():
            if edge != incoming_edge and graph[tensor][edge] is not None:
                ranks[edge] = min(prod // ranks[edge], ranks[edge])
                outwards(graph[tensor][edge], edge)

    ranks = ranks.copy()
    tensor_sizes = {}
    inwards(0, incoming_edge=None)
    outwards(0, incoming_edge=None)
    return ranks


def build_tensor_network(x: np.ndarray, topology: str):
    assert topology in ("single", "tucker", "tt", "ett")

    if topology == "single":
        return qtn.TensorNetwork(
            [qtn.Tensor(x, inds=[f"i{i}" for i in range(x.ndim)], tags=["C0"])]
        )

    if topology == "tucker":
        tensors = [qtn.Tensor(x, inds=[f"r{i}" for i in range(x.ndim)], tags=["C0"])]
        tensors.extend(
            [
                qtn.Tensor(np.eye(x.shape[i]), inds=[f"i{i}", f"r{i}"], tags=[f"U{i}"])
                for i in range(x.ndim)
            ]
        )
        return qtn.TensorNetwork(tensors)

    if topology == "tt":
        site_ind_id = "i{}"
    else:
        site_ind_id = "s{}"
    tn = qtn.MatrixProductState.from_dense(
        x, dims=x.shape, site_ind_id=site_ind_id, site_tag_id="C{}", method="identity"
    )
    if topology == "ett":
        tn.reindex({"s0": "i0"}, inplace=True)
        tn.reindex({f"s{x.ndim-1}": f"i{x.ndim-1}"}, inplace=True)
        tensors = []
        for i in range(1, x.ndim - 1):
            tensors.append(
                qtn.Tensor(np.eye(x.shape[i]), inds=(f"i{i}", f"s{i}"), tags=[f"U{i}"])
            )
        tn = tn | tensors
    return tn


def plain_to_qtt(x):
    orig_shape = x.shape
    # max_shape = max(x.shape)
    # L = np.ceil(np.log2(max_shape)).astype(int)
    x = np.pad(
        x,
        [[0, 2**np.ceil(np.log2(x.shape[i])).astype(int) - x.shape[i]] for i in range(x.ndim)],
    )
    # order = np.arange(x.ndim * L)
    # order = np.arange(x.ndim * L).reshape(x.ndim, L).T.flatten()
    # order = np.arange(x.ndim * L).reshape(L, x.ndim).T.flatten()
    order = None
    # print(order)
    # x_reshaped = x.reshape([2] * (x.ndim * L)).transpose(order)
    x_reshaped = x.reshape([2] * np.ceil(np.log2(x.size)).astype(int))
    info = {"order": order, "padded_shape": x.shape, "shape": orig_shape}
    return x_reshaped, info


def qtt_to_plain(x, info):
    if info["order"] is not None:
        x = x.transpose(np.argsort(info["order"]))
    x = x.reshape(info["padded_shape"])
    return x[tuple(slice(0, sh) for sh in info["shape"])]
