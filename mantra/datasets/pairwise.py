import os
import typing
from typing import Literal, Tuple, TypeAlias
from tqdm import tqdm

import torch

from mantra.datasets import ManifoldTriangulations

from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.transforms import Compose

from mantra.configs import SplitConfig, Mode

from mantra.manifold_types import Manifold2Type, Manifold3Type
from mantra.tasks.task_types import TaskType

from mantra.transforms.structural_transforms import (
    AddSimplicialComplexTransform,
    SetNumNodesTransform,
)


class PairwiseSimplicialDS(InMemoryDataset):
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
        self.task_type = TaskType.HOMEOMORPHIC_DIST
        self.manifold = manifold
        self.comparison_pair = comparison_pair
        self.raw_simplicial_ds = ManifoldTriangulations(
            os.path.join(root, "raw_simplicial"),
            manifold,
            version,
            None,
            pre_transform=pre_transform,
            pre_filter=pre_filter,
        )
        super().__init__(root, transform=transform)

        self.load(
            self._get_processed_path(
                self.comparison_pair[0], self.comparison_pair[1]
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
    ):
        """
        Parameters
        -----------
            m_1: ManifoldType
                Type of the first manifold the triangulations represent.
            m_2: ManifoldType
                Type of the second manifold the triangulations represent.
        Returns
        -------
            The name of the data file in .pt format.

        """
        return f"data_{m_1.name.lower()}_{m_2.name.lower()}.pt"

    def _get_processed_path(
        self,
        m_1: Manifold2Type | Manifold3Type,
        m_2: Manifold2Type | Manifold3Type,
    ):
        fnames = self.processed_file_names
        idx = 0
        for fname in fnames:
            if fname == self._data_filename(m_1, m_2):
                return self.processed_paths[idx]
            idx += 1
        raise ValueError(
            "Can not find processed data: Unknown config with task type and mode."
        )

    @property
    def processed_file_names(self):
        if self.manifold == "2":
            f_names = [
                self._data_filename(m_1, m_2)
                for m_1 in Manifold2Type
                for m_2 in Manifold2Type
            ]
        else:
            raise NotImplementedError("TODO")
        return f_names

    def process(self):
        print("---> Preprocessing dataset...)")

        if self.manifold == "3":
            raise NotImplementedError("TODO")

        # Convert to strings
        m_1, m_2 = self.comparison_pair[0].value, self.comparison_pair[1].value

        data_list_preprocessed = [obj for obj in self.raw_simplicial_ds]

        print("---> Constructing pairwise comparison...")

        data_list = []
        # NOTE: Makeup pairwise comparisons between the triangulations
        for i, _m1 in enumerate(tqdm(data_list_preprocessed)):
            for j, _m2 in enumerate(data_list_preprocessed):
                # NOTE: Do not add a pair of the same triangulation
                if i == j:
                    continue

                # We are not interested in this pairwise comparison
                if _m1.name not in [m_1, m_2] or _m2.name not in [m_1, m_2]:
                    # print("Excluding pair from comparison ...)")
                    continue

                # Construct a 2-tensor for each object
                dict_obj = dict()

                for k, v in _m1.to_dict().items():
                    if isinstance(v, torch.Tensor):
                        dict_obj[k] = torch.cat(
                            [v.unsqueeze(0), getattr(_m2, k).unsqueeze(0)],
                            dim=0,
                        )
                    else:
                        dict_obj[k] = tuple([v, getattr(_m2, k)])

                # Create the data object with two triangulations
                data_obj = Data(**dict_obj)
                data_list.append(data_obj)
        self.save(data_list, self._get_processed_path(*self.comparison_pair))
