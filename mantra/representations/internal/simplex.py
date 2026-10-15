"""Simplex Class.
Adapted from https://github.com/pyt-team/TopoNetX/blob/main/toponetx/classes/simplex.py
"""

from collections.abc import Collection, Hashable, Iterable
from functools import total_ordering
from typing import Any, Generic, TypeVar

from typing_extensions import Self

ElementType = TypeVar("ElementType", bound=Hashable)


@total_ordering
class Simplex(Generic[ElementType]):
    """A class representing a simplex in a simplicial complex.

    This class represents a simplex in a simplicial complex, which is a set of nodes with a specific dimension. The
    simplex is immutable, and the nodes in the simplex must be hashable and unique.

    Parameters
    ----------
    elements : Collection[Hashable]
        The nodes in the simplex.

    Examples
    --------
    >>> # Create a 0-dimensional simplex (point)
    >>> s = Simplex((1,))
    >>> # Create a 1-dimensional simplex (line segment)
    >>> s = Simplex((1, 2))
    >>> # Create a 2-dimensional simplex (triangle)
    >>> simplex1 = Simplex((1, 2, 3))
    >>> simplex2 = Simplex(("a", "b", "c"))
    >>> # Create a 3-dimensional simplex (tetrahedron)
    >>> simplex3 = Simplex((1, 2, 4, 5))
    """

    elements: tuple[Hashable, ...]

    def __init__(self, elements: Collection[Hashable]) -> None:
        if len(set(elements)) != len(elements):
            raise ValueError("A simplex cannot contain duplicate nodes.")

        # Sorted makes iteration, `repr` and the ordering lexicographic
        self.elements = tuple(sorted(elements))

    def __hash__(self) -> int:
        """Returns a hash of the simplex.

        Returns
        -------
        int
            Hash of the elements.
        """
        return hash(self.elements)

    def __len__(self) -> int:
        """Returns the number of nodes in the simplex.

        Returns
        -------
        int
            Number of nodes.
        """
        return len(self.elements)

    def __iter__(self) -> Iterable:
        """Returns an iterator over the elements in the simplex.

        Returns
        -------
        Iterable
            Iterator over simplex elements.
        """

        return iter(self.elements)

    def __eq__(self, other: Any) -> bool:
        """Return `True` if the given simplex is equal to this simplex.

        Simplices are considered equal if they have the same elements but may have
        different attributes.

        Parameters
        ----------
        other : Any
            The simplex to compare.

        Returns
        -------
        bool
            Returns `True` if the given simplex is equal to this simplex and `False` otherwise.
        """
        return type(self) is type(other) and self.elements == other.elements

    def __contains__(self, item: ElementType) -> bool:
        """Return True if the given node is in the simplex.

        Parameters
        ----------
        item : Any
            The node to be checked.

        Returns
        -------
        bool
            True if the given node is in the simplex.

        Examples
        --------
        >>> s = Simplex((1, 2, 3))
        >>> 1 in s
        True
        >>> 4 in s
        False
        """
        return item in self.elements

    def __le__(self, other) -> bool:
        """Return True if this simplex comes before the other simplex in the lexicographic order.

        Parameters
        ----------
        other : Simplex
            The other simplex to compare with.

        Returns
        -------
        bool
            True if this simplex comes before the other simplex in the lexicographic order.
        """
        assert isinstance(
            other, Simplex
        ), f"Comparison object {other} is not a `Simplex`"
        return self.elements <= other.elements

    def __repr__(self) -> str:
        """Return string representation of the simplex.

        Returns
        -------
        str
            A string representation of the simplex.
        """
        return f"Simplex({tuple(self.elements)})"

    def __str__(self) -> str:
        """Return human readable (str) simplex.

        Returns
        -------
        str
            A human readable representation of the simplex.
        """
        return f"Nodes: {tuple(self.elements)}"

    def clone(self) -> Self:
        """Return a copy of the simplex.

        The clone method by default returns an independent shallow copy of the simplex
        and attributes. That is, if an attribute is a container, that container is
        shared by the original and the copy. Use Python's `copy.deepcopy` for new
        containers.

        Returns
        -------
        Simplex
            A copy of this simplex.
        """
        return self.__class__(self.elements)
