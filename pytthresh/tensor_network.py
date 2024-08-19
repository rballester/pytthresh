import numpy as np
import quimb.tensor as qtn


def build_tensor_network(x: np.ndarray, topology: str):
    assert topology in ("tucker", "tt", "ett")

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
    max_shape = max(x.shape)
    L = np.ceil(np.log2(max_shape)).astype(int)
    x = np.pad(
        x,
        [[0, 2**L - x.shape[i]] for i in range(x.ndim)],
    )
    order = np.arange(x.ndim * L).reshape(x.ndim, L).T.flatten()
    # order = np.arange(x.ndim * L).reshape(L, x.ndim).T
    x_reshaped = x.reshape([2] * (x.ndim * L)).transpose(order)
    info = {"order": order, "padded_shape": x.shape, "shape": orig_shape}
    return x_reshaped, info


def qtt_to_plain(x, info):
    x = x.transpose(np.argsort(info["order"]))
    x = x.reshape(info["padded_shape"])
    return x[tuple(slice(0, sh) for sh in info["shape"])]
