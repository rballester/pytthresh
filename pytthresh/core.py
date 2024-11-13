import time
from collections import defaultdict
import bson
import rich
import pytthresh as pyt
import constriction
import matplotlib.pyplot as plt
import numpy as np
import quimb.tensor as qtn
import scipy

from pytthresh import rle, tensor_network


def entropy_encode(x):
    unique, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
    if len(counts) < 2:  # constriction needs at least 2 symbols
        assert len(counts) > 0
        counts = np.concatenate([counts, [0]])
    entropy_model = constriction.stream.model.Categorical(
        counts / np.sum(counts), perfect=False
    )
    encoder = constriction.stream.queue.RangeEncoder()
    encoder.encode(inverse.astype(np.int32), entropy_model)
    return encoder.get_compressed(), len(inverse), unique, counts


def entropy_decode(bits, howmany, unique, counts):
    coder = constriction.stream.queue.RangeDecoder(bits)
    entropy_model = constriction.stream.model.Categorical(
        counts / np.sum(counts), perfect=False
    )
    inverse = coder.decode(entropy_model, howmany)
    return unique[inverse]


class CompressedPlane:

    def __init__(self, bits, howmany, unique, counts):
        self.bits = bits
        self.howmany = howmany
        self.unique = unique
        self.counts = counts

    def n_bits(self):
        return len(self.bits) * 32 + len(self.unique) * 32 + len(self.counts) * 32

    def n_bytes(self):
        return int(np.ceil(self.n_bits() / 8))

    def serialize(self):
        return [
            self.bits.tobytes(),
            self.howmany,
            self.unique.astype(np.uint32).tobytes(),
            self.counts.astype(np.uint32).tobytes(),
        ]

    @staticmethod
    def deserialize(data):
        return CompressedPlane(
            bits=np.frombuffer(data[0], dtype=np.uint32),
            howmany=data[1],
            unique=np.frombuffer(data[2], dtype=np.uint32),
            counts=np.frombuffer(data[3], dtype=np.uint32),
        )


class CompressedTensor:

    def __init__(self, ind_sizes, compressed_planes, signs, scale):
        self.ind_sizes = ind_sizes
        self.compressed_planes = compressed_planes
        self.signs = signs
        self.scale = scale

    def n_bits(self):
        # get_compressed() returns int32's
        return sum([cp.n_bits() for cp in self.compressed_planes]) + len(self.signs)

    def n_bytes(self):
        return int(np.ceil(self.n_bits() / 8))

    def decode(self):
        coreq = np.zeros(np.prod(list(self.ind_sizes.values())), dtype=np.uint64)
        for i, cp in enumerate(self.compressed_planes):
            q = 63 - i
            plane_rle = entropy_decode(cp.bits, cp.howmany, cp.unique, cp.counts)
            plane = rle.decode(plane_rle).astype(np.uint64)
            coreq[: len(plane)] += plane << q
        mask = coreq != 0
        if q >= 1:
            coreq[mask == True] += 1 << (q - 1)
        c = coreq.astype(np.float64) / self.scale
        c[mask == True] *= self.signs * 2 - 1
        c = c.reshape(list(self.ind_sizes.values()))
        # shape = list(self.ind_sizes.values())
        # c_padded = np.zeros(shape)
        # c_padded[tuple(slice(0, r) for r in self.ranks)] = c
        return qtn.Tensor(c, inds=list(self.ind_sizes.keys()))

    def serialize(self):
        return [
            {k: int(v) for k, v in self.ind_sizes.items()},
            [cp.serialize() for cp in self.compressed_planes],
            np.packbits(self.signs).tobytes(),
            len(self.signs),
            self.scale,
        ]

    @staticmethod
    def deserialize(data):
        return CompressedTensor(
            ind_sizes=data[0],
            compressed_planes=[CompressedPlane.deserialize(cp) for cp in data[1]],
            signs=np.unpackbits(
                np.frombuffer(data[2], dtype=np.uint8), count=data[3]
            ).astype(np.int32),
            scale=data[4],
        )


class File:
    def __init__(self, compressed_tensors, shape, dtype, min, max, tid0, qtt_info):
        self.compressed_tensors = compressed_tensors

        # "Metadata" needed to reconstruct the tensor
        self.shape = shape
        self.dtype = dtype

        # Min and max values for clipping. This prevents underflow/overflow and preserves the original data range (which helps when rendering)
        self.min = min
        self.max = max

        self.tid0 = tid0
        self.qtt_info = qtt_info

    def n_bytes(self):
        return sum([ct.n_bytes() for ct in self.compressed_tensors])

    def encode(self) -> bytes:
        d = {
            "compressed_tensors": [ct.serialize() for ct in self.compressed_tensors],
            "shape": self.shape,
            "dtype": self.dtype.str,
            "min": self.min,
            "max": self.max,
            "tid0": self.tid0,
            "qtt_info": self.qtt_info,
        }
        return bson.encode(d)

    def to_disk(self, filename):
        bson_data = self.encode()
        with open(filename, "wb") as f:
            f.write(bson_data)

    @staticmethod
    def from_disk(filename):
        with open(filename, "rb") as f:
            bson_data = f.read()
        d = bson.BSON.decode(bson_data)
        return File(
            compressed_tensors=[
                CompressedTensor.deserialize(ct) for ct in d["compressed_tensors"]
            ],
            shape=d["shape"],
            dtype=np.dtype(d["dtype"]),
            min=d["min"],
            max=d["max"],
            tid0=d["tid0"],
            qtt_info=d["qtt_info"],
        )

    def decompress(self, debug=False) -> np.ndarray:

        tensors = [ct.decode() for ct in self.compressed_tensors]
        if len(tensors) == 1:
            result = tensors[0].data
        else:
            tensor_map = {i: tensors[i] for i in range(len(tensors))}

            tid0 = self.tid0
            # tid0 = 3
            tn = qtn.TensorNetwork()
            for k, v in sorted(tensor_map.items()):
                # if v.ndim == 3:
                # v = qtn.Tensor().transpose(np.argsort([0, 2, 1]))
                tn.add_tensor(v, tid=k)

            seen = np.zeros(len(tn.tensors), dtype=bool)
            seen[tid0] = True

            graph = defaultdict(lambda: set())
            for e in tn.get_tree_span(tids=[tid0]):
                graph[e[0]].add(e[1])
                graph[e[1]].add(e[0])

            def recursion(tid, seen):
                for neighbor in graph[tid]:  # .edges(tid):
                    # neighbor = edge[1]
                    if not seen[neighbor]:
                        seen[neighbor] = True
                        recursion(neighbor, seen)
                        # tn._canonize_between_tids(
                        #     neighbor, tid, absorb="right", cutoff=None
                        # )
                        qtn.tensor_core.tensor_compress_bond(
                            tn.tensor_map[neighbor],
                            tn.tensor_map[tid],
                            method="qr",
                            absorb="right",
                            cutoff=None,
                        )
                        # lix = qtn.tensor_core.tensor_make_single_bond(
                        #     tn.tensor_map[neighbor], tn.tensor_map[tid]
                        # )[0]
                        # q, r = tn.tensor_map[neighbor].split(
                        #     left_inds=lix,
                        #     cutoff=None,
                        #     get="tensors",s
                        #     method="svd",
                        #     absorb="right",
                        # )
                        # q = q.transpose_like(tn.tensor_map[neighbor])
                        # tn.tensor_map[neighbor].modify(data=q.data)
                        data = tensor_map[tid].data
                        tn.tensor_map[tid].modify(data=data)

            recursion(tid0, seen)
            if debug:
                for t in tn.tensors:
                    print(t)
                print(tn.outer_inds())
            result = tn.contract(
                output_inds=[f"i{i}" for i in range(len(tn.outer_inds()))]
            ).data

        # Recover original shape
        result = np.pad(
            result,
            [[0, self.shape[i] - result.shape[i]] for i in range(len(self.shape))],
        )

        # Clip and cast to original dtype
        np.clip(result, self.min, self.max, out=result)

        if self.qtt_info is not None:
            result = pyt.tensor_network.qtt_to_plain(result, self.qtt_info)
        return result.astype(self.dtype)


class TensorEncoder:
    """
    Class to encode a single tensor within a tensor decomposition.
    """

    def __init__(self, x: qtn.Tensor):
        self.ind_sizes = {i: x.ind_size(i) for i in x.inds}
        # if x.ndim == 3:
        #     x = x.data.transpose(0, 2, 1).flatten()
        # else:
        #     x = x.data.flatten()
        x = x.data.flatten()
        self.signs = np.sign(x) > 0
        cabs = np.abs(x)
        msplane = np.floor(np.log2(cabs.max())).astype(int)
        self.scale = np.ldexp(1, 63 - msplane)  # TODO is msplane ilogb(maximum)?
        self.coreq = (cabs * self.scale).astype(np.uint64)
        self.plane_Bs = [
            0
        ]  # plane_Bs[i] is the cumulative number of bits before encoding plane i
        self.plane_epss = [
            1
        ]  # plane_epss[i] is the cumulative relative error before encoding plane i
        self.q = 63
        self.cumulative_plane_sumsq = self.coreq.astype(np.float64) ** 2
        self.normsq = np.sum(self.cumulative_plane_sumsq)
        self.mask = np.zeros(x.size, dtype=np.uint64)  # TODO use a bitarray
        self.counts = []
        self.compressed_planes = []

    def _encode(self, plane):
        plane_rle = rle.encode(plane).astype(np.int32)
        bits, howmany, unique, counts = entropy_encode(plane_rle)
        self.compressed_planes.append(
            CompressedPlane(
                bits=bits,
                howmany=howmany,
                unique=unique,
                counts=counts,
            )
        )

    def advance(self):
        plane = (self.coreq >> self.q) & 1  # == 1
        n = (plane << self.q).astype(np.float64)
        self.cumulative_plane_sumsq += (
            n**2 - 2 * np.sqrt(self.cumulative_plane_sumsq) * n
        )
        self._encode(plane)
        self.mask = np.logical_or(self.mask, plane)
        self.plane_Bs.append(
            # len(self.compressed_planes[-1]) * 4 * 8
            # + np.sum(self.mask)
            int(sum([len(cp.bits) for cp in self.compressed_planes])) * 4 * 8
            + np.sum(self.mask)
        )
        self.plane_epss.append(np.sum(self.cumulative_plane_sumsq) / self.normsq)
        self.q -= 1

    def get_ranks(self, cutoff: float):
        last_plane = int(np.ceil(cutoff)) - 1
        mask = (self.coreq >> (64 - last_plane)) != 0
        bp = max(2, int(len(self.coreq) * (cutoff - np.floor(cutoff))))
        plane = (self.coreq[:bp] >> (63 - last_plane)) & 1
        mask[:bp] = np.logical_or(mask[:bp], plane)
        mask = mask.reshape(list(self.ind_sizes.values()))
        return {
            k: np.where(
                np.sum(mask, axis=tuple(np.delete(np.arange(mask.ndim), i))) > 0
            )[0][-1]
            + 1
            for i, k in enumerate(self.ind_sizes)
        }

    def finish(self, cutoff: float, ranks) -> CompressedTensor:

        ranks = {k: ranks[k] for k in self.ind_sizes}

        # TODO compact a bit
        self.coreq = self.coreq.reshape(list(self.ind_sizes.values()))
        self.coreq = self.coreq[tuple(slice(0, r) for r in ranks.values())]
        self.coreq = self.coreq.flatten()
        self.signs = self.signs.reshape(list(self.ind_sizes.values()))
        self.signs = self.signs[tuple(slice(0, r) for r in ranks.values())]
        self.signs = self.signs.flatten()

        self.compressed_planes = []
        last_plane = int(np.floor(cutoff))
        self.mask = (self.coreq >> (64 - last_plane)) != 0
        for plane in range(63, 63 - last_plane, -1):
            plane = (self.coreq >> plane) & 1
            self._encode(plane)
            # self.mask = np.logical_or(self.mask, plane)
        bp = max(2, int(len(self.coreq) * (cutoff - np.floor(cutoff))))
        plane = (self.coreq[:bp] >> (63 - last_plane)) & 1
        self._encode(plane)
        self.mask[:bp] = np.logical_or(self.mask[:bp], plane)
        signs = self.signs[self.mask == True]

        return CompressedTensor(
            ind_sizes=ranks,
            compressed_planes=self.compressed_planes,
            signs=signs,
            scale=self.scale,
        )

    def get_convex_curve(self):

        def convex_envelope(B, eps):
            points = np.array([B, eps]).T

            # Remove duplicate x entries, keeping the best one in each case
            points = np.array(sorted(points.tolist()))
            _, index = np.unique(points[:, 0], return_index=True)
            points = points[index]

            if len(points) <= 2:
                hull = np.arange(len(points))
            else:
                hull = scipy.spatial.ConvexHull(points).vertices
            bp = np.where(hull == 0)[0][0]
            hull = np.roll(hull, -bp)
            points = points[hull]

            # Remove any trailing points that are above the smallest eps
            indices = np.where(points[:, 1] <= points[:, 1].min())[0]
            assert len(indices) > 0
            last_index = indices[-1]
            points = points[: last_index + 1]
            return points[:, 0], points[:, 1]

        if len(self.plane_Bs) == 2:
            pass
        return convex_envelope(self.plane_Bs, self.plane_epss)

    def _B_to_index(self, B):
        return scipy.interpolate.interp1d(
            self.plane_Bs,
            np.arange(len(self.plane_Bs)),
            bounds_error=False,
            fill_value=(0, len(self.plane_Bs)),
        )(B).item()


# def to_disk(*args, **kwargs):
#     filename = kwargs.pop("filename")
#     file = to_object(*args, **kwargs)
#     file.to_disk(filename)


def to_object(
    x: np.ndarray,
    topology: str,
    target_eps: float = None,
    statistics: bool = False,
    qtt: bool = False,
    debug: bool = False,
) -> File:
    assert topology in ("tucker", "qtt", "tt", "ett", "single")

    if topology == "qtt":
        x, qtt_info = pyt.tensor_network.plain_to_qtt(x)
        topology = "tt"
    else:
        qtt_info = None

    info = {}
    start = time.time()
    dtype = x.dtype
    a_min, a_max = x.min().item(), x.max().item()
    x = x.astype(np.float64)
    info["statistics_time"] = time.time() - start

    start = time.time()
    tn = tensor_network.build_tensor_network(x, topology)
    info["build_time"] = time.time() - start

    # order out spanning tree by depth first search
    def sorter(t, tn, distances, connectivity):
        return distances[t]

    # tid0 = tn.most_central_tid()
    if topology in ("single", "tucker"):
        tid0 = 0
    else:
        tid0 = (x.ndim - 1) // 2
    span = tn.get_tree_span([tid0], sorter=sorter)

    # Make graph a default dict
    graph = defaultdict(lambda: set())
    for e in tn.get_tree_span(tids=[tid0]):
        graph[e[0]].add(e[1])
        graph[e[1]].add(e[0])

    ###########################
    # Recursively traverse tree
    ###########################

    seen = np.zeros(len(tn.tensors), dtype=bool)
    seen[tid0] = True
    info["canonize_time"] = 0
    info["compress_time"] = 0

    # def multiply_diagonal(t, s):
    #     data = t.data
    #     s = np.expand_dims(
    #         s, tuple(np.delete(np.arange(s.ndim), t.inds.index(s.inds[0])))
    #     )
    #     t.modify(data=np.multiply(data, s))

    def recursion(tid, seen):
        # Canonized and compress `tid`
        for neighbor in graph[tid]:
            if not seen[neighbor]:
                start = time.time()
                # TODO use get='tensors' to avoid having to canonize afterwards
                ret = qtn.tensor_compress_bond(
                    tn.tensor_map[tid],
                    tn.tensor_map[neighbor],
                    absorb="left",
                    reduced="left",
                    method="eig",
                    cutoff=target_eps**2 / len(tn.tensors) * 1e-1,
                    # cutoff=1e-10,
                    cutoff_mode="rsum2",
                    gauge_smudge=0,
                )
                # reco = tn.contract(output_inds=tn.outer_inds())
                # np.linalg.norm(reco.data - x) / np.linalg.norm(x)
                # T1 = tn.tensor_map[tid]
                # T2 = tn.tensor_map[neighbor]
                # lix, bix, rix = qtn.tensor_core.tensor_make_single_bond(T1, T2)
                # T1C, *s, M = T1.split(
                #     left_inds=lix,
                #     right_inds=bix,
                #     get="tensors",
                #     absorb=None,
                #     cutoff=0,
                # )
                # T2C = M @ T2
                # T1C.transpose_like_(T1)
                # T2C.transpose_like_(T2)
                # np.broadcast_to(s[0].data, T1C.shape)
                # multiply_diagonal(T1C, s[0])
                # T1C.multiply_index_diagonal(s[0].inds[0], s[0], inplace=False)
                info["compress_time"] += time.time() - start

        for neighbor in graph[tid]:
            if not seen[neighbor]:
                start = time.time()
                seen[neighbor] = True
                src = tn.tensor_map[tid].copy(deep=True)
                # Tl = tn.tensor_map[tid]
                # Tr = tn.tensor_map[neighbor]
                # qtn.tensor_core.tensor_canonize_bond(
                #     Tl,
                #     Tr,
                #     method='qr',
                #     absorb='right',
                # )
                # tn._canonize_between_tids(
                #     tid, neighbor, method="qr", absorb="right", gauge_smudge=0
                # )
                qtn.tensor_core.tensor_compress_bond(
                    tn.tensor_map[tid],
                    tn.tensor_map[neighbor],
                    method="qr",
                    absorb="right",
                    gauge_smudge=0,
                )
                info["canonize_time"] += time.time() - start
                recursion(neighbor, seen)
                tn.tensor_map[tid] = src

    # start = time.time()
    recursion(tid0, seen)
    # info['decomposition_time'] = time.time() - start
    # decomposition_time = time.time() - start
    # return tensor_map

    ##########################
    # Lossy bit-plane encoding
    ##########################

    info["encode_time"] = 0
    start = time.time()
    encoders = []
    for k, v in sorted(tn.tensor_map.items()):
        e = TensorEncoder(v)
        e.advance()
        encoders.append(e)
    info["encode_time"] += time.time() - start

    done = False
    last_cutoffs = None
    info["convex_hull_time"] = 0
    info["optimize_time"] = 0
    while not done:
        start = time.time()
        curves = [e.get_convex_curve() for e in encoders]
        info["convex_hull_time"] = time.time() - start
        Bs = [c[0] for c in curves]
        epss = [c[1] for c in curves]
        try:
            start = time.time()
            optimized_Bs = optimize(Bs, epss, target_eps=target_eps)
            info["optimize_time"] += time.time() - start
        except ValueError:
            # Did not converge: we advance all encoders and try again
            if debug:
                print("Warning: optimization did not converge")
            start = time.time()
            for i in range(len(encoders)):
                encoders[i].advance()
            info["encode_time"] += time.time() - start
            continue
        cutoffs = []
        for i in range(len(encoders)):
            cutoffs.append(encoders[i]._B_to_index(optimized_Bs[i]))
        if last_cutoffs is not None and np.all(last_cutoffs == cutoffs):
            if debug:
                print("Cutoffs converged, we break here")
            break
        last_cutoffs = cutoffs
        done = True
        # print("*", cutoffs, [len(e.plane_Bs) for e in encoders])
        start = time.time()
        for i in range(len(encoders)):
            if (
                cutoffs[i] >= len(encoders[i].plane_Bs)
                and encoders[i].plane_epss[-1] > 1e-20
            ):
                encoders[i].advance()
                done = False
        info["encode_time"] += time.time() - start
    # optimized_Bs = optimize(Bs, epss, target_eps=target_eps)  # Useful breakpoint
    # cutoffs = []
    # for i in range(len(encoders)):
    # cutoffs.append(encoders[i]._B_to_index(optimized_Bs[i]))
    # if debug:
    #     import matplotlib.pyplot as plt
    #     import pandas as pd

    #     ks = sorted(tensor_map.keys())
    #     for i in range(len(curves)):
    #         curve = curves[i]
    #         v = tensor_map[ks[i]]
    #         df = pd.DataFrame({"B": curve[0], "epssq": curve[1]})
    #         df.to_csv("{}.csv".format("_".join(v.inds)), index=False)
    #         plt.plot(np.log10(curve[0]), np.log10(curve[1]), label="_".join(v.inds))
    #     plt.xlabel("log10(B)")
    #     plt.ylabel("log10(epssq)")
    #     plt.legend()
    #     plt.savefig("curve.pdf")

    if debug:
        print(
            "Cutoffs:",
            ", ".join(
                f"{cutoffs[i]}/{len(encoders[i].plane_Bs)}"
                for i in range(len(encoders))
            ),
        )
    # for e in encoders[3:6]:
    # plt.plot(e.plane_Bs, e.plane_epss)
    # cc = e.get_convex_curve()
    # plt.plot(cc[0], cc[1])

    ###########################
    # Save with the final ranks
    ###########################

    result = []
    start = time.time()
    global_ranks = defaultdict(lambda: float("inf"))
    for i in range(len(encoders)):
        ranks = encoders[i].get_ranks(cutoffs[i])
        for k, v in ranks.items():
            global_ranks[k] = min(global_ranks[k], v)

    # Peel ranks
    inputs = [t.inds for t in tn.tensor_map.values()]
    global_ranks = pyt.tensor_network.peel_ranks(inputs, global_ranks)
    for i in range(len(encoders)):
        result.append(encoders[i].finish(cutoffs[i], global_ranks))
    info["finish_time"] = time.time() - start
    if debug:
        rich.print(info)
    file = File(
        result,
        shape=x.shape,
        dtype=dtype,
        min=a_min,
        max=a_max,
        tid0=tid0,
        qtt_info=qtt_info,
    )
    return file


# def from_disk(filename=None, debug=False):

#     file = File.from_disk(filename)
#     return file.decompress(debug)


def optimize(Bs, epss, target_cr=None, target_eps=None, datalen=None):
    # TODO deal with target_cr and datalen
    A = []
    b = []
    N = len(Bs)
    factors = []
    for i in range(N):
        slopes = np.diff(epss[i]) / np.diff(Bs[i])
        intercepts = epss[i][:-1] - slopes * Bs[i][:-1]
        assert all(slopes <= 0)  # First derivative must be <= 0
        assert all(np.diff(slopes) >= 0)  # Convexity: 2nd derivative must be >= 0
        Ablock = np.zeros([len(slopes), 2 * N])
        Ablock[:, i * 2] = slopes
        Ablock[:, i * 2 + 1] = -1
        A.append(Ablock)
        b.append(-intercepts)

    # Sum of errors must not exceed given threshold
    if target_cr is not None:
        A.append(np.array([[1, 0] * N]))
        b.append(np.array([(datalen * 8) / target_cr]))  # TODO fix for non uint8 data
    else:
        assert target_eps is not None
        A.append(np.array([[0, 1] * N]))
        b.append(np.array([(target_eps**2)]))

    A = np.vstack(A)
    b = np.concatenate(b)
    c = np.array([1, 0] * N)
    # factor = np.linalg.norm(A[:, ::2], axis=0)
    factor = A[:, ::2].flatten()
    factor = factor[factor != 0]
    factor = -np.max(factor)
    # factor = -np.max(A[:, ::2])
    # factor = 1
    A[:, ::2] /= factor

    results = scipy.optimize.linprog(
        c=c,
        A_ub=A,
        b_ub=b,
        bounds=[[0, None]] * (2 * N),
        method="highs",
        options=dict(),
    )
    if results.status != 0:
        raise ValueError
    # assert results.status == 0
    return results.x[::2] / factor
