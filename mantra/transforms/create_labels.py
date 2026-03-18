from torch_geometric.transforms import BaseTransform
import torch


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
        self.label_to_index = {}

    def forward(self, data):
        """Assign label for a given `data` object.

        Given a source attribute to create a label, assign a numerical
        index to be used for downstream classification tasks. There is
        one interesting thing happening here: The class assigns labels
        based on the data type. If a boolean property is detected, the
        mapping will default to `False = 0` and `True = 1`. Otherwise,
        for string-based attributes, indices will be assigned based on
        the order in which they are encountered.

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
        assert (
            self.source in data
        ), f"Source attribute '{self.source}' is not present in data"

        label = data[self.source]

        if isinstance(label, bool):
            data.y = int(label)
        else:
            if label not in self.label_to_index:
                self.label_to_index[label] = len(self.label_to_index)

        data.y = torch.tensor([self.label_to_index[label]])

        return data
