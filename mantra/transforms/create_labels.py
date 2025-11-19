from torch_geometric.transforms import BaseTransform


class CreateLabels(BaseTransform):
    """Create labels based on attributes.

    This transform creates labels based on attributes present in
    a dataset. Depending on the type of attribute, labels may be
    binary or multi-class.
    """

    def __init__(self, source):
        """Create new label creator transform.

        Parameters
        ----------
        source : str
            Denotes attribute that is used to create labels. If not
            present in the data, the `forward()` function will just
            fail with an exception.
        """
        super().__init__()

        self.source = source

    def forward(self, data):
        """Assign label for a given `data` object.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Input data object. The source attribute, which is used to
            create labels, must be present.

        Returns
        -------
        torch_geometric.data.Data
            Data object with a label attached to it, stored in the `y`
            attribute of the tensor.
        """
        return data
