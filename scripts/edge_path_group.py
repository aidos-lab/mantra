import json
import sys

from collections import defaultdict
from collections import deque

from itertools import combinations
from itertools import permutations


def faces(top_simplices, k):
    face_set = set()

    for s in top_simplices:
        s = sorted(s)

        for f in combinations(s, k + 1):
            face_set.add(f)

    return face_set


def spanning_tree(vertices, edges):
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    root = next(iter(vertices))
    parent = {root: None}
    tree_edges = set()
    q = deque([root])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if v not in parent:
                parent[v] = u
                tree_edges.add(tuple(sorted((u, v))))
                q.append(v)
    return tree_edges, parent, root


def path_to_root(v, parent):
    path = [v]
    while parent[path[-1]] is not None:
        path.append(parent[path[-1]])
    return path


def edge_word(vertices, edges, top_simplices):
    tree_edges, parent, root = spanning_tree(vertices, edges)
    non_tree_edges = [e for e in edges if e not in tree_edges]
    gen_index = {e: i for i, e in enumerate(non_tree_edges)}

    def gen_of_directed_edge(u, v):
        e = tuple(sorted((u, v)))
        if e in tree_edges:
            return None
        sign = 1 if (u, v) == e else -1
        return (gen_index[e], sign)

    relators = []
    triangles = faces(top_simplices, 2)
    for a, b, c in triangles:
        word = []
        for u, v in [(a, b), (b, c), (c, a)]:
            g = gen_of_directed_edge(u, v)
            if g is not None:
                word.append(g)
        relators.append(word)

    return non_tree_edges, relators


def format_word(word):
    return " ".join(f"g{i}" if s == 1 else f"g{i}^-1" for i, s in word) or "1"


def edge_path_presentation(top_simplices):
    vertices = sorted(faces(top_simplices, 0))
    vertices = [v[0] for v in vertices]
    edges = sorted(faces(top_simplices, 1))

    gens, relators = edge_word(vertices, edges, top_simplices)
    return gens, relators


def hom_count(gens, relators, elements, mult):
    n = len(gens)

    ready_at = defaultdict(list)
    relator_gens = []
    for w in relators:
        used = sorted({i for i, _ in w})
        relator_gens.append(used)
        if used:
            ready_at[used[-1]].append(len(relator_gens) - 1)
        else:
            ready_at[-1].append(
                len(relator_gens) - 1
            )  # empty word, check immediately

    def eval_word(word, assignment, identity):
        val = identity
        for i, sign in word:
            e = assignment[i]
            if sign == -1:
                e = inverse[e]
            val = mult(val, e)
        return val

    identity = next(
        e for e in elements if all(mult(e, x) == x for x in elements)
    )
    inverse = {}
    for a in elements:
        for b in elements:
            if mult(a, b) == identity:
                inverse[a] = b
                break

    count = 0
    assignment = [None] * n

    def backtrack(i):
        nonlocal count
        if i == n:
            count += 1
            return
        for g in elements:
            assignment[i] = g
            ok = True
            for r_idx in ready_at.get(i, []):
                if (
                    eval_word(relators[r_idx], assignment, identity)
                    != identity
                ):
                    ok = False
                    break
            if ok:
                backtrack(i + 1)
        assignment[i] = None

    # handle relators with no generators (only matters if n == 0)
    for r_idx in ready_at.get(-1, []):
        if relators[
            r_idx
        ]:  # non-empty but somehow no gens used: skip, shouldn't happen
            continue

    backtrack(0)
    return count


def symmetric_group(n):
    from itertools import permutations

    elements = list(permutations(range(n)))

    def mult(a, b):
        return tuple(a[b[i]] for i in range(n))

    return elements, mult


def cyclic_group(n):
    elements = list(range(n))

    def mult(a, b):
        return (a + b) % n

    return elements, mult


def barycentric_subdivision(top_simplices):
    new = set()

    for s in top_simplices:
        s = tuple(sorted(s))
        for p in permutations(s):
            new.add(
                tuple(
                    sorted(tuple(sorted(p[:k])) for k in range(1, len(p) + 1))
                )
            )

    return sorted(new)


def free_reduce(word):
    stack = []
    for g, s in word:
        if stack and stack[-1] == (g, -s):
            stack.pop()
        else:
            stack.append((g, s))
    return stack


def invert_word(word):
    return [(g, -s) for g, s in reversed(word)]


def substitute(word, gen_i, sub_word, sub_word_inv):
    new = []
    for g, s in word:
        if g == gen_i:
            new.extend(sub_word if s == 1 else sub_word_inv)
        else:
            new.append((g, s))
    return free_reduce(new)


def tietze_simplify(gens, relators):
    relators = [free_reduce(list(w)) for w in relators]
    remaining = set(range(len(gens)))
    eliminated = {}

    progress = True
    while progress:
        progress = False
        for ridx, r in enumerate(relators):
            if not r:
                continue
            counts = defaultdict(int)
            for g, _ in r:
                counts[g] += 1
            candidate = next(
                (g for g, c in counts.items() if c == 1 and g in remaining),
                None,
            )
            if candidate is None:
                continue

            g = candidate
            pos = next(i for i, (gg, _) in enumerate(r) if gg == g)
            A, (_, s), B = r[:pos], r[pos], r[pos + 1 :]
            W = (
                free_reduce(invert_word(A) + invert_word(B))
                if s == 1
                else free_reduce(B + A)
            )
            Winv = invert_word(W)

            del relators[ridx]
            remaining.discard(g)
            eliminated[g] = W

            new_relators = []
            for rr in relators:
                rr2 = substitute(rr, g, W, Winv)
                if rr2:
                    new_relators.append(rr2)
            relators = new_relators
            progress = True
            break

    uniq, seen = [], set()
    for r in relators:
        key, keyinv = tuple(r), tuple(invert_word(r))
        if key in seen or keyinv in seen:
            continue
        seen.add(key)
        uniq.append(r)

    remaining_sorted = sorted(remaining)
    reindex = {g: i for i, g in enumerate(remaining_sorted)}
    new_gens = [gens[g] for g in remaining_sorted]
    new_relators = [[(reindex[g], s) for g, s in r] for r in uniq]

    return new_gens, new_relators, eliminated


if __name__ == "__main__":
    with open(sys.argv[1]) as f:
        data = json.load(f)

        # FIXME: Should make this configurable since it only applies to
        # dimension 2. Maybe get a class count and only pick *some*?
        if data[0]["dimension"] == 2:
            data = list(
                filter(
                    lambda manifold: manifold["name"]
                    in ["Klein bottle", "RP^2", "S^2", "T^2"],
                    data,
                )
            )

    dim = [manifold["dimension"] for manifold in data]
    assert min(dim) == max(dim), "Require same dimension"
    dim = dim[0]

    for manifold in data:
        K = manifold["triangulation"]
        L = barycentric_subdivision(K)

        print("#" * 72)
        print(manifold["name"])
        print("#" * 72)
        print("")

        g1, r1 = edge_path_presentation(K)
        g1, r1, _ = tietze_simplify(g1, r1)

        g2, r2 = edge_path_presentation(L)
        g2, r2, _ = tietze_simplify(g2, r2)

        for name, (gens, relators) in [("K", (g1, r1)), ("sd K", (g2, r2))]:
            print(name)
            for name, (elements, mult) in [
                ("Z/2", cyclic_group(2)),
                ("Z/3", cyclic_group(3)),
                ("S3", symmetric_group(3)),
            ]:
                c = hom_count(gens, relators, elements, mult)
                print(f"  |Hom(pi_1, {name})| = {c}")
