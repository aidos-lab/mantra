from torch_geometric.transforms import BaseTransform


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
