"""
Implementation of a simplex trie datastructure for simplicial complexes as presented in [1]_.

This module is intended for internal use by the `SimplicialComplex` class only. Any direct interactions with this
module or its classes may break at any time. In particular, this also means that the `SimplicialComplex` class must not
leak any object references to the trie or its nodes.

Some implementation details:
- Inside this module, simplices are represented as ordered sequences with unique elements. It is expected that all
  inputs from outside are already pre-processed and ordered accordingly. This is not checked and the behavior is
  undefined if this is not the case.

References
----------
.. [1] Jean-Daniel Boissonnat and Clément Maria. The Simplex Tree: An Efficient Data Structure for General Simplicial
       Complexes. Algorithmica, pages 1-22, 2014
"""

from collections.abc import Generator, Hashable, Iterable, Iterator, Sequence
from typing import Any, Generic, TypeVar

from .simplex import Simplex

ElementType = TypeVar("ElementType", bound=Hashable)
T = TypeVar("T")


def is_ordered_subset(one: Sequence[T], other: Sequence[T]) -> bool:
    """Return True if the first iterable is a subset of the second iterable.

    This method is specifically optimized for ordered iterables to use return as early as possible for non-subsets.

    Parameters
    ----------
    one : Sequence
        The first iterable.
    other : Sequence
        The second iterable.

    Returns
    -------
    bool
        True if the first iterable is a subset of the second iterable, False otherwise.

    Examples
    --------
    >>> is_ordered_subset((2,), (1, 2))
    True
    >>> is_ordered_subset((1, 2), (1, 2, 3))
    True
    >>> is_ordered_subset((1, 2, 3), (1, 2, 3))
    True
    >>> is_ordered_subset((1, 2, 3), (1, 2))
    False
    >>> is_ordered_subset((1, 2, 3), (1, 2, 4))
    False
    """
    index = 0
    for item in one:
        while (
            index < len(other)
            and isinstance(item, type(other[index]))
            and other[index] < item
        ):
            index += 1
        if index >= len(other) or other[index] != item:
            return False
    return True


class SimplexNode(Generic[ElementType]):
    """Node in a simplex trie.

    Parameters
    ----------
    label : ElementType or None
        The label of the node. May only be `None` for the root node.
    parent : SimplexNode, optional
        The parent node of this node. If `None`, this node is the root node.
    """

    label: ElementType | None
    elements: tuple[ElementType, ...]
    attributes: dict[Hashable, Any]

    depth: int
    parent: "SimplexNode | None"
    children: dict[ElementType, "SimplexNode[ElementType]"]

    def __init__(
        self,
        label: ElementType | None,
        parent: "SimplexNode[ElementType] | None" = None,
    ) -> None:
        """Node in a simplex trie.

        Parameters
        ----------
        label : ElementType or None
            The label of the node. May only be `None` for the root node.
        parent : SimplexNode, optional
            The parent node of this node. If `None`, this node is the root node.
        """
        self.label = label
        self.attributes = {}

        self.children = {}

        self.parent = parent
        if parent is not None:
            parent.children[label] = self
            self.elements = (*parent.elements, label)
            self.depth = parent.depth + 1
        else:
            if label is not None:
                raise ValueError("Root node must have label `None`.")
            self.elements = ()
            self.depth = 0

    def __len__(self) -> int:
        """Return the number of elements in this trie node.

        Returns
        -------
        int
            Number of elements in this trie node.
        """
        return len(self.elements)

    def __repr__(self) -> str:
        """Return a string representation of this trie node.

        Returns
        -------
        str
            A string representation of this trie node.
        """
        return f"SimplexNode({self.label}, {self.parent!r})"

    @property
    def simplex(self) -> Simplex[ElementType] | None:
        """Return a `Simplex` object representing this node.

        Returns
        -------
        Simplex or None
            A `Simplex` object representing this node, or `None` if this node is the root node.
        """
        if self.label is None:
            return None
        simplex = Simplex(self.elements)
        simplex._attributes = self.attributes
        return simplex

    def iter_all(self) -> Generator["SimplexNode[ElementType]", None, None]:
        """Iterate over all nodes in the subtree rooted at this node.

        Ordering is according to breadth-first search, i.e., simplices are yielded in increasing order of dimension and
        increasing order of labels within each dimension.

        Yields
        ------
        SimplexNode
        """
        queue = [self]
        while queue:
            node = queue.pop(0)
            if node.label is not None:
                # skip root node
                yield node
            queue += [
                node.children[label] for label in sorted(node.children.keys())
            ]


class SimplexTrie(Generic[ElementType]):
    """
    Simplex trie data structure as presented in [1]_.

    This class is intended for internal use by the `SimplicialComplex` class only. Any
    direct interactions with this class may break at any time.

    References
    ----------
    .. [1] Jean-Daniel Boissonnat and Clément Maria. The Simplex Tree: An Efficient
           Data Structure for General Simplicial Complexes. Algorithmica, pages 1-22, 2014
    """

    root: SimplexNode[ElementType]
    label_lists: dict[int, dict[ElementType, list[SimplexNode[ElementType]]]]
    shape: list[int]

    def __init__(self) -> None:
        """Simplex trie data structure as presented in [1]_.

        This class is intended for internal use by the `SimplicialComplex` class only.
        Any direct interactions with this class may break at any time.

        References
        ----------
        .. [1] Jean-Daniel Boissonnat and Clément Maria. The Simplex Tree: An Efficient
               Data Structure for General Simplicial Complexes. Algorithmica, pages 1-22, 2014
        """
        self.root = SimplexNode(None)
        self.label_lists = {}
        self.shape = []

    def __len__(self) -> int:
        """Return the number of simplices in the trie.

        Returns
        -------
        int
            Number of simplices in the trie.

        Examples
        --------
        >>> trie = SimplexTrie()
        >>> trie.insert((1, 2, 3))
        >>> len(trie)
        7
        """
        return sum(self.shape)

    def __contains__(self, item: Iterable[ElementType]) -> bool:
        """Check if a simplex is contained in this trie.

        Parameters
        ----------
        item : Iterable of ElementType
            The simplex to check for. Must be ordered and contain unique elements.

        Returns
        -------
        bool
            Whether the given simplex is contained in this trie.

        Examples
        --------
        >>> trie = SimplexTrie()
        >>> trie.insert((1, 2, 3))
        >>> (1, 2, 3) in trie
        True
        >>> (1, 2, 4) in trie
        False
        """
        return self.find(item) is not None

    def __getitem__(
        self, item: Iterable[ElementType]
    ) -> SimplexNode[ElementType]:
        """Return the simplex node for a given simplex.

        Parameters
        ----------
        item : Iterable of ElementType
            The simplex to return the node for. Must be ordered and contain only unique
            elements.

        Returns
        -------
        SimplexNode
            The trie node that represents the given simplex.

        Raises
        ------
        KeyError
            If the given simplex does not exist in this trie.
        """
        node = self.find(item)
        if node is None:
            raise KeyError(f"Simplex {item} not found in trie.")
        return node

    def __iter__(self) -> Iterator[SimplexNode[ElementType]]:
        """Iterate over all simplices in the trie.

        Simplices are ordered by increasing dimension and increasing order of labels within each dimension.

        Yields
        ------
        tuple of ElementType
        """
        yield from self.root.iter_all()

    def insert(
        self,
        item: Sequence[ElementType],
        subtree: None | SimplexNode[ElementType] = None,
    ) -> None:
        """Insert a simplex into the trie.

        Any lower-dimensional simplices that do not exist in the trie are also inserted
        to fulfill the simplex property. If the simplex already exists, its properties
        are updated.

        Parameters
        ----------
        items : Sequence
            The (partial) simplex to insert under the subtree.
        subtree : SimplexNode
            The subtree to insert the simplex node under. Defaults to root for the base-case
        """

        # This is what the outside caller sees
        if subtree is None:
            item = sorted(frozenset(item))
            self.insert(item, self.root)
        # This is the recursive call
        else:
            for i, label in enumerate(item):
                if label not in subtree.children:
                    self._insert_child(subtree, label)
                self.insert(item[i + 1 :], subtree.children[label])

    def _insert_child(
        self, parent: SimplexNode[ElementType], label: ElementType
    ) -> SimplexNode[ElementType]:
        """Insert a child node with a given label.

        Parameters
        ----------
        parent : SimplexNode
            The parent node.
        label : ElementType
            The label of the child node.

        Returns
        -------
        SimplexNode
            The new child node.
        """
        node = SimplexNode(label, parent=parent)

        if node.depth not in self.label_lists:
            self.label_lists[node.depth] = {}
        if label in self.label_lists[node.depth]:
            self.label_lists[node.depth][label].append(node)
        else:
            self.label_lists[node.depth][label] = [node]

        if node.depth > len(self.shape):
            self.shape += [0]
        self.shape[node.depth - 1] += 1

        return node

    def find(
        self, search: Iterable[ElementType]
    ) -> SimplexNode[ElementType] | None:
        """Find the node in the trie that matches the search.

        Parameters
        ----------
        search : Iterable of ElementType
            The simplex to search for. Must be ordered and contain only unique elements.

        Returns
        -------
        SimplexNode or None
            The node that matches the search, or `None` if no such node exists.
        """
        node = self.root
        for item in search:
            if item not in node.children:
                return None
            node = node.children[item]
        return node

    def cofaces(
        self, simplex: Sequence[ElementType]
    ) -> Generator[SimplexNode[ElementType], None, None]:
        """Return the cofaces of the given simplex.

        No ordering is guaranteed by this method.

        Parameters
        ----------
        simplex : Sequence of ElementType
            The simplex to find the cofaces of. Must be ordered and contain only unique elements.

        Yields
        ------
        SimplexNode
            The cofaces of the given simplex, including the simplex itself.

        Examples
        --------
        >>> trie = SimplexTrie()
        >>> trie.insert((1, 2, 3))
        >>> trie.insert((1, 2, 4))
        >>> sorted(map(lambda node: node.elements, trie.cofaces((1,))))
        [(1,), (1, 2), (1, 2, 3), (1, 2, 4), (1, 3), (1, 4)]
        """
        # Find simplices of the form [*, l_1, *, l_2, ..., *, l_n], i.e. simplices that contain all elements of the
        # given simplex plus some additional elements, but sharing the same largest element. This can be done by the
        # label lists.
        simplex_nodes = self._coface_roots(simplex)

        # Found all simplices of the form [*, l_1, *, l_2, ..., *, l_n] in the simplex trie. All nodes in the subtrees
        # rooted at these nodes are cofaces of the given simplex.
        for simplex_node in simplex_nodes:
            yield from simplex_node.iter_all()

    def _coface_roots(
        self, simplex: Sequence[ElementType]
    ) -> Generator[SimplexNode[ElementType], None, None]:
        """Return the roots of the coface subtrees.

        A coface subtree is a subtree of the trie whose simplices are all cofaces of a
        given simplex.

        Parameters
        ----------
        simplex : Sequence of ElementType
            The simplex to find the cofaces of. Must be ordered and contain only unique
            elements.

        Yields
        ------
        SimplexNode
            The trie nodes that are roots of the coface subtrees.
        """
        for depth in range(len(simplex), len(self.shape) + 1):
            if simplex[-1] not in self.label_lists[depth]:
                continue
            for simplex_node in self.label_lists[depth][simplex[-1]]:
                if is_ordered_subset(simplex, simplex_node.elements):
                    yield simplex_node

    def skeleton(self, rank: int) -> Generator[SimplexNode, None, None]:
        """Return the simplices of the given rank.

        No particular ordering is guaranteed and is dependent on insertion order.

        Parameters
        ----------
        rank : int
            The rank of the simplices to return.

        Yields
        ------
        SimplexNode
            The simplices of the given rank.

        Raises
        ------
        ValueError
            If the given rank is negative or exceeds the maximum rank of the trie.

        Examples
        --------
        >>> trie = SimplexTrie()
        >>> trie.insert((1, 2, 3))
        >>> sorted(map(lambda node: node.elements, trie.skeleton(0)))
        [(1,), (2,), (3,)]
        >>> sorted(map(lambda node: node.elements, trie.skeleton(1)))
        [(1, 2), (1, 3), (2, 3)]
        >>> sorted(map(lambda node: node.elements, trie.skeleton(2)))
        [(1, 2, 3)]
        """
        if rank < 0:
            raise ValueError(f"`rank` must be a positive integer, got {rank}.")
        if rank >= len(self.shape):
            raise ValueError(
                f"`rank` {rank} exceeds maximum rank {len(self.shape)}."
            )

        for nodes in self.label_lists[rank + 1].values():
            yield from nodes
