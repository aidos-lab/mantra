import os
from typing import Tuple

import torch
from sklearn.model_selection import train_test_split
from torch_geometric.data import InMemoryDataset

from mantra.configs import Mode, SplitConfig
from mantra.datasets import PairwiseSimplicialDS
from mantra.manifold_types import Manifold2Type, Manifold3Type


class PropertyPredictionDS(InMemoryDataset):
    """
    Wrapper of ManifoldTriangulations to extend it with train/test/val split functionality.

    train/test/val splits depend on the task type due to stratification, i.e. that
    a proper split shall maintain the same class imbalance in all splits.
    Since for every task type the class imbalance differs, stratification can only be done dependent
    on the task type.
    """

    def __init__(
        self,
        root: str,
        split: Mode,
        split_config: SplitConfig,
        comparison_pair: (
            Tuple[Manifold3Type, Manifold3Type]
            | Tuple[Manifold2Type, Manifold2Type]
        ) = (Manifold2Type.S_2, Manifold2Type.T_2),
        manifold="2",
        version="latest",
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        self.manifold = manifold
        self.split_config = split_config
        self.pairwise_simplicial_ds = PairwiseSimplicialDS(
            root=os.path.join(root, "raw_simplicial"),
            comparison_pair=comparison_pair,
            manifold=manifold,
            version=version,
            transform=transform,
            pre_transform=pre_transform,
            pre_filter=pre_filter,
        )
        self.transform = transform
        super().__init__(root, transform=transform)

        self.load(
            self._get_processed_path(
                self.comparison_pair[0],
                self.comparison_pair[1],
                split,
            )
        )

    @property
    def raw_file_names(self):
        return []

    def download(self) -> None:
        pass

    def _data_filename(
        self,
        m_1: Manifold2Type | Manifold3Type,
        m_2: Manifold2Type | Manifold3Type,
        split: str,
    ):
        """
        Parameters
        -----------
            task_type: TaskType
                Type of the task to perform.
            split: ManifoldType
                Type of the second manifold the triangulations represent.
        Returns
        -------
            The name of the data file in .pt format.

        """
        return f"data_{m_1.name.lower()}_{m_2.name.lower()}_{split}.pt"

    def _get_processed_path(
        self,
        m_1: Manifold2Type | Manifold3Type,
        m_2: Manifold2Type | Manifold3Type,
        split: str,
    ):
        fnames = self.processed_file_names
        idx = 0
        for fname in fnames:
            if fname == self._data_filename(m_1, m_2, split):
                return self.processed_paths[idx]
            idx += 1
        raise ValueError(
            "Can not find processed data: Unknown config with task type and mode."
        )

    @property
    def processed_file_names(self):
        if self.manifold == "2":
            f_names = [
                self._data_filename(m_1, m_2, split)
                for m_1 in Manifold2Type
                for m_2 in Manifold2Type
                for split in ["train", "val", "test"]
            ]
        else:
            raise NotImplementedError("TODO")
        return f_names

    def process(self):
        print("---> Preprocessing dataset...)")

        if self.manifold == "3":
            raise NotImplementedError("TODO")

        indices = range(self.pairwise_simplicial_ds.len())

        if self.transform is not None:
            data_list_processed = [
                self.transform(d) for d in self.pairwise_simplicial_ds
            ]
        else:
            data_list_processed = [d for d in self.pairwise_simplicial_ds]

        stratified = torch.vstack([data.y for data in data_list_processed])
        train_val_indices, test_indices = train_test_split(
            indices,
            test_size=self.split_config.split[2],
            shuffle=True,
            stratify=(
                stratified.numpy()
                if self.split_config.use_stratified
                else None
            ),
            random_state=self.split_config.seed,
        )

        train_indices, val_indices = train_test_split(
            train_val_indices,
            test_size=self.split_config.split[1]
            / (self.split_config.split[0] + self.split_config.split[1]),
            shuffle=True,
            stratify=(
                stratified[train_val_indices]
                if self.split_config.use_stratified
                else None
            ),
            random_state=self.split_config.seed,
        )

        #  Save splits
        data_list = [self.pairwise_simplicial_ds.get(idx) for idx in indices]
        train_data_list = [data_list[idx] for idx in train_indices]
        val_data_list = [data_list[idx] for idx in val_indices]
        test_data_list = [data_list[idx] for idx in test_indices]

        m_1, m_2 = self.comparison_pair[0].value, self.comparison_pair[1].value

        self.save(train_data_list, self._get_processed_path(m_1, m_2, "train"))
        self.save(val_data_list, self._get_processed_path(m_1, m_2, "val"))
        self.save(test_data_list, self._get_processed_path(m_1, m_2, "test"))
