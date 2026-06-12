"""Tests for SimplexTrie data structure."""

import pytest

from mantra.representations.internal.simplex_trie import (
    SimplexNode,
    SimplexTrie,
    is_ordered_subset,
)


class TestIsOrderedSubset:
    def test_single_element_subset(self):
        assert is_ordered_subset((2,), (1, 2))

    def test_proper_subset(self):
        assert is_ordered_subset((1, 2), (1, 2, 3))

    def test_equal_sequences(self):
        assert is_ordered_subset((1, 2, 3), (1, 2, 3))

    def test_not_subset_longer(self):
        assert not is_ordered_subset((1, 2, 3), (1, 2))

    def test_not_subset_different(self):
        assert not is_ordered_subset((1, 2, 3), (1, 2, 4))

    def test_empty_is_subset(self):
        assert is_ordered_subset((), (1, 2, 3))


class TestSimplexNode:
    def test_root_node(self):
        root = SimplexNode(None)
        assert root.label is None
        assert root.depth == 0
        assert root.elements == ()
        assert root.simplex is None

    def test_root_must_have_none_label(self):
        with pytest.raises(ValueError, match="Root node"):
            SimplexNode(1, parent=None)

    def test_child_node(self):
        root = SimplexNode(None)
        child = SimplexNode(1, parent=root)
        assert child.label == 1
        assert child.depth == 1
        assert child.elements == (1,)
        assert root.children[1] is child

    def test_grandchild_elements(self):
        root = SimplexNode(None)
        child = SimplexNode(1, parent=root)
        grandchild = SimplexNode(2, parent=child)
        assert grandchild.elements == (1, 2)
        assert grandchild.depth == 2

    def test_simplex_property(self):
        root = SimplexNode(None)
        child = SimplexNode(1, parent=root)
        s = child.simplex
        assert s is not None
        assert set(s) == {1}

    def test_len_counts_elements(self):
        root = SimplexNode(None)
        child = SimplexNode(1, parent=root)
        grandchild = SimplexNode(2, parent=child)
        assert len(root) == 0
        assert len(grandchild) == 2

    def test_repr_mentions_simplex_node(self):
        root = SimplexNode(None)
        assert repr(SimplexNode(1, parent=root)).startswith("SimplexNode(")


class TestSimplexTrieInsert:
    def test_insert_single_simplex(self):
        trie = SimplexTrie()
        trie.insert((1, 2, 3))
        # triangle has 7 simplices: 3 vertices + 3 edges + 1 triangle
        assert len(trie) == 7

    def test_insert_creates_subsimplices(self):
        trie = SimplexTrie()
        trie.insert((1, 2, 3))
        assert (1,) in trie
        assert (2,) in trie
        assert (1, 2) in trie
        assert (1, 3) in trie
        assert (2, 3) in trie
        assert (1, 2, 3) in trie

    def test_insert_unsorted(self):
        """Elements are sorted internally."""
        trie = SimplexTrie()
        trie.insert((3, 1, 2))
        assert (1, 2, 3) in trie

    def test_insert_two_simplices_shared_face(self):
        trie = SimplexTrie()
        trie.insert((1, 2, 3))
        trie.insert((1, 2, 4))
        # vertices: 1,2,3,4=4; edges: 12,13,23,14,24=5; triangles: 2
        assert len(trie) == 11

    def test_shape(self):
        trie = SimplexTrie()
        trie.insert((1, 2, 3))
        assert trie.shape == [3, 3, 1]


class TestSimplexTrieContains:
    def test_contains_existing(self):
        trie = SimplexTrie()
        trie.insert((1, 2, 3))
        assert (1, 2, 3) in trie

    def test_not_contains(self):
        trie = SimplexTrie()
        trie.insert((1, 2, 3))
        assert (1, 2, 4) not in trie


class TestSimplexTrieGetItem:
    def test_getitem_existing(self):
        trie = SimplexTrie()
        trie.insert((1, 2, 3))
        node = trie[(1, 2)]
        assert node.elements == (1, 2)

    def test_getitem_missing_raises(self):
        trie = SimplexTrie()
        trie.insert((1, 2, 3))
        with pytest.raises(KeyError):
            trie[(1, 4)]


class TestSimplexTrieIteration:
    def test_iter_yields_all(self):
        trie = SimplexTrie()
        trie.insert((1, 2, 3))
        nodes = list(trie)
        assert len(nodes) == 7


class TestSimplexTrieSkeleton:
    def test_skeleton_0(self):
        trie = SimplexTrie()
        trie.insert((1, 2, 3))
        verts = sorted(node.elements for node in trie.skeleton(0))
        assert verts == [(1,), (2,), (3,)]

    def test_skeleton_1(self):
        trie = SimplexTrie()
        trie.insert((1, 2, 3))
        edges = sorted(node.elements for node in trie.skeleton(1))
        assert edges == [(1, 2), (1, 3), (2, 3)]

    def test_skeleton_2(self):
        trie = SimplexTrie()
        trie.insert((1, 2, 3))
        faces = sorted(node.elements for node in trie.skeleton(2))
        assert faces == [(1, 2, 3)]

    def test_skeleton_negative_raises(self):
        trie = SimplexTrie()
        trie.insert((1, 2, 3))
        with pytest.raises(ValueError, match="positive"):
            list(trie.skeleton(-1))

    def test_skeleton_too_high_raises(self):
        trie = SimplexTrie()
        trie.insert((1, 2, 3))
        with pytest.raises(ValueError, match="exceeds"):
            list(trie.skeleton(3))


class TestSimplexTrieCofaces:
    def test_cofaces_of_vertex(self):
        trie = SimplexTrie()
        trie.insert((1, 2, 3))
        trie.insert((1, 2, 4))
        coface_elems = sorted(node.elements for node in trie.cofaces((1,)))
        assert (1,) in coface_elems
        assert (1, 2) in coface_elems
        assert (1, 2, 3) in coface_elems
        assert (1, 2, 4) in coface_elems

    def test_cofaces_of_edge(self):
        trie = SimplexTrie()
        trie.insert((1, 2, 3))
        trie.insert((1, 2, 4))
        coface_elems = sorted(node.elements for node in trie.cofaces((1, 2)))
        assert (1, 2) in coface_elems
        assert (1, 2, 3) in coface_elems
        assert (1, 2, 4) in coface_elems

    def test_cofaces_of_top_simplex(self):
        trie = SimplexTrie()
        trie.insert((1, 2, 3))
        coface_elems = [node.elements for node in trie.cofaces((1, 2, 3))]
        assert coface_elems == [(1, 2, 3)]

    def test_cofaces_skips_depths_without_the_vertex(self):
        # Two disjoint simplices: querying cofaces of vertex 1 must skip
        # the depth-3 layer (the triangle 3-4-5 does not contain 1).
        trie = SimplexTrie()
        trie.insert((1, 2))
        trie.insert((3, 4, 5))
        coface_elems = sorted(node.elements for node in trie.cofaces((1,)))
        assert coface_elems == [(1,), (1, 2)]

    def test_cofaces_excludes_same_top_vertex_non_superset(self):
        # (1, 3) and (2, 3) share the largest vertex 3 but neither
        # contains the other, so cofaces((1, 3)) must exclude (2, 3).
        trie = SimplexTrie()
        trie.insert((1, 3))
        trie.insert((2, 3))
        coface_elems = sorted(node.elements for node in trie.cofaces((1, 3)))
        assert coface_elems == [(1, 3)]
