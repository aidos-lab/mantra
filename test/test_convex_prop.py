import pytest
import torch
import numpy as np
from torch_geometric.data import Data
from mantra.transforms import PropagateConvexComb
from mantra.representations import DualGraph

class TestComb:
    def setup_method(self):
        self.trans = PropagateConvexComb()

    def make_data(self, x, triangulation, source='x'):
        data = Data()
        setattr(data, source, x)
        data.triangulation = triangulation

        return data

    def test_correct_key_1(self):
        """ Tests if it contains the correct tensors
        
        """
        x = np.zeros((4,2), dtype=np.float32)
        triangulation = [[1,2,3,4]]
        data = self.make_data(x, triangulation)
        data = self.trans(data)

        assert getattr(data, 'x_0', None) is not None
        assert getattr(data, 'x_1', None) is not None
        assert getattr(data, 'x_2', None) is not None
        assert getattr(data, 'x_3', None) is not None

    def test_correct_key_2(self):
        """ Tests if it contains the corect
            shapes 
        """
        x = np.zeros((4,2), dtype=np.float32)
        triangulation = [[1,2,3,4]]
        data = self.make_data(x, triangulation)
        data = self.trans(data)

        assert getattr(data, 'x_0', None).shape[0] == 4
        assert getattr(data, 'x_1', None).shape[0] == 6
        assert getattr(data, 'x_2', None).shape[0] == 4
        assert getattr(data, 'x_3', None).shape[0] == 1

    def test_share_edge(self):
        """ Tests if shared edges get 
            mapped correctly 
        
        """
        x = np.array(
            [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0], [2.0, 2.0]], dtype=np.float32
        )
        out = self.trans(self.make_data(x, [[1, 2, 3], [2, 3, 4]]))
    
        expected_edges = np.array(
            [[1, 0], [0, 1], [1, 1], [2, 1], [1, 2]], dtype=np.float32
        )
        torch.testing.assert_close(out["x_1"], torch.from_numpy(expected_edges))
    
        expected_tris = np.array(
            [[2 / 3, 2 / 3], [4 / 3, 4 / 3]], dtype=np.float32
        )
        torch.testing.assert_close(out["x_2"], torch.from_numpy(expected_tris))
    #TODO: Add a test including DualGraph or OneSkeleton