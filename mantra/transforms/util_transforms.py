from itertools import combinations
from typing import Dict, List, Literal, TypeAlias, Union

import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform

Representation: TypeAlias = Literal["graph", "sc"]


class SelectAttributes(BaseTransform):
    """Select attributes to keep.

    This transform simplifies pipelines by selecting which attributes to
    keep in the resulting tensor. Thus, this transform is best used last
    in a pipeline of transforms.
    """

    def __init__(self, keep_keys=None):
        """Create new attribute selector transform.

        Parameters
        ----------
        keep_keys : iterable or `None`
            Specify which keys of a torch_geometric.data.Data object to
            keep. If set to `None`, the transform will keep these keys:

            * `x`
            * `y`
            * `edge_index`
        """
        super().__init__()

        self.keep_keys = keep_keys

        if self.keep_keys is None:
            self.keep_keys = ["x", "y", "edge_index"]

        self.keep_keys = set(self.keep_keys)

    def forward(self, data):
        """Modify `data` object and remove unnecessary attributes.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Input data object. All keys that are not mentioned upon
            creating this transform will be *removed*. Non-existent
            keys will be silently ignored.

        Returns
        -------
        torch_geometric.data.Data
            Adjusted data object with the `triangulation` key removed,
            all other keys maintained, and `edge_index` information of
            the dual graph being present.
        """
        for k, v in data.items():
            if k not in self.keep_keys:
                del data[k]

        return data


class SelectFeatures(BaseTransform):
    """Select features to be assigned to nodes / simplices.

    This transform assigns a computed value in a `Data`
    object (i.e. `node_degrees`) the name of a feature tensor (i.e. `x`).
    This process is needed as a formatting step for libraries that have
    a naming convention for feature tensors, like PyG.
    """

    def __init__(
        self,
        src: str,
        dst: Union[str, List[str], None],
        representation: Representation = "graph",
    ):
        """Creates a feature selector transform.

        Parameters
        ----------
        src : str
            Name of the source `Tensor` to be contained in each `Data`
            obj.
        dst : str or None
            Name of the destination `Tensor` to be contained in the
            `Data` obj. If `None` defaults to canonical encoding per
            `representation`. Note that if `representation=sc` then
            the string should be formatable with the argument `{d}`
            to assign each dimension or should be a list of strings.
        representation : Representation
            Specify which representation type to use. The choices are
            `graph` or `sc`. Graphs use the PyG encoding (i.e. `x` and
            `edge_attr`) and simplicial complexes encode features as
            `x_i` where `i` ranges from `0 -> d` for `d` the dimension
            of the simplicial complex.
        """
        super().__init__()

        assert representation in [
            "graph",
            "sc",
        ], f"Invalid value: {representation}"

        self.src = src
        self.dst = dst
        self.representation = representation

        # Use the canonical representation str
        if self.dst is None:
            if representation == "graph":
                self.dst = "x"
            else:  # The case for `sc`
                self.dst = "x_{d}"

    def _select_dst_list(self, src_tensor: Tensor, data: Data):
        assert isinstance(
            src_tensor, Dict
        ), f"The attribute {self.src} is not of type dict"
        assert self.dst is not None, "`dst` is None"

        assert len(self.dst) == len(
            src_tensor.keys()
        ), f"There is a mismatch between num of `src` keys ( {len(src_tensor.keys())} ) and `dst` targets ( { len(self.dst) } )"

        for i, (k, v) in enumerate(src_tensor.items()):
            dst_i = self.dst[i]
            data[dst_i] = v

    def _select_dst_single(self, src_tensor: Tensor, data: Data):
        if self.representation == "graph":
            assert isinstance(
                src_tensor, Tensor
            ), "Attribute `src` is not a `torch.Tensor`"
            data[self.dst] = src_tensor  # noqa

        else:  # The case for `sc`
            assert isinstance(
                src_tensor, Dict
            ), f"The attribute {self.src} is not of type dict"

            # Iterate over each key in the `src` tensor.
            # NOTE: The keys here should be an integer with the dimension of the
            # simplices or a str that can be cast to int, that's why we
            # explicitly cast it to flag possibe miss-alignment errors
            for k, v in src_tensor.items():
                dst_str = self.dst.format(d=int(k))  # noqa
                data[dst_str] = v

    def forward(self, data: Data):
        """Modify `data` object and assign feature tensors.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Input data object.

        Returns
        -------
        torch_geometric.data.Data
            Adjusted data object with the `triangulation` key removed,
            all other keys maintained, and `edge_index` information of
            the dual graph being present.
        """

        assert (
            self.src in data
        ), f"The `data` obj does not contain `{self.src}`"

        src_tensor = getattr(data, self.src)

        # This is the completely explicit case, we pass the source and target
        # and the caller is in charge of making sure everything fits.
        if isinstance(self.dst, List):
            self._select_dst_list(src_tensor, data)

        # This is the case where defaults are used, so the canonical
        # representations for graph are PyG and indexed feature tensors based for simplicial complexes
        else:
            self._select_dst_single(src_tensor, data)

        return data


class PropagateConvexComb(BaseTransform):
    """Propagates the features of a tensor `source` describing
    0-simplices to higher-simplices based on the barycenter
    of the feature.
    """

    def __init__(self, source: str = "x"):
        """
        Parameters
        ----------
        source : str
            Name of the source `Tensor` to be contained in each `Data`
            obj. Defaults to `x`
        """
        self.source = source

    def forward(self, data):

        assert (
            "triangulation" in data
        ), "Data object is missing `triangulation`"
        assert (
            self.source in data
        ), f"Data object is missing source tensor `{self.source}`"

        assert isinstance(getattr(data, self.source), torch.Tensor), f"Input {self.source} is not a torch.Tensor"

        x = getattr(data, self.source)
        triangulation = getattr(data, "triangulation")


        simplices = set([tuple(s) for s in triangulation])
        max_dim = len(next(iter(simplices)))

        for simplex in triangulation:
            for dim in range(1, max_dim):
                simplices.update(s for s in combinations(simplex, r=dim))

        # To sort lexicographically, we need to turn this back into
        # something mutable.
        simplices = list(simplices)
        simplices.sort()
        simplices.sort(key=len)

        # Dictionary containing the new attribute keys
        values = {"x_0": x}

        for dim in range(1, max_dim):
            simplices_ = [s for s in simplices if len(s) == dim + 1]

            # Every simplex of this rank has `dim + 1` vertices, so the
            # index tensor is rectangular; subtract one to map the
            # 1-indexed vertex labels to tensor rows.
            idx = torch.tensor(simplices_) - 1

            # Calculate all barycenters of the current rank at once.
            # TODO: Change this to another type of combination function
            values[f"x_{dim}"] = x[idx].mean(dim=1)

        # Assignment to data object
        for k, v in values.items():
            data[k] = v

        return data
