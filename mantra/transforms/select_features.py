from typing import TypeAlias, Literal, Union, Dict, List
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from torch import Tensor


Representation: TypeAlias = Literal["graph", "sc"]


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
            assert isinstance(
                src_tensor, Dict
            ), f"The attribute {self.src} is not of type dict"
            assert len(self.dst) == len(
                src_tensor.keys()
            ), f"There is a mismatch between num of `src` keys ( {len(src_tensor.keys())} ) and `dst` targets ( { len(self.dst) } )"

            for i, (k, v) in enumerate(src_tensor.items()):
                dst_i = self.dst[i]
                data[dst_i] = v

        # This is the case where defaults are used, so the canonical
        # representations for graph are PyG and indexed feature tensors based for simplicial complexes
        else:
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
                    dst_str = self.dst.format(int(k))  # noqa
                    data[dst_str] = v
        return data
