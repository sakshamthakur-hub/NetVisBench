"""Task family 1: routing on a weighted topology diagram.

The image shows routers (letters) joined by links labelled with an
integer cost, the way topologies are drawn in textbooks for link-state
(Dijkstra / OSPF) exercises. Two question variants:

* ``path_cost`` - total cost of the least-cost path from S to D.
* ``next_hop``  - first router after S on the least-cost path to D.

We only keep source/destination pairs whose least-cost path is unique,
so every question has exactly one correct answer.
"""
from __future__ import annotations

import math
import string
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx

from ..common import Item, make_rng, save_fig

SIZES = {"easy": (5, 1), "medium": (7, 3), "hard": (9, 5),
         "expert": (13, 9)}  # (nodes, extra links)


def _random_topology(rng, n_nodes: int, extra: int) -> nx.Graph:
    names = list(string.ascii_uppercase[:n_nodes])
    g = nx.Graph()
    g.add_nodes_from(names)
    # random spanning tree -> always connected
    order = names[:]
    rng.shuffle(order)
    for i in range(1, n_nodes):
        g.add_edge(order[i], rng.choice(order[:i]), weight=rng.randint(1, 9))
    # extra links create alternative paths (this is what makes it non-trivial)
    tries = 0
    while extra > 0 and tries < 200:
        tries += 1
        u, v = rng.sample(names, 2)
        if not g.has_edge(u, v):
            g.add_edge(u, v, weight=rng.randint(1, 9))
            extra -= 1
    return g


def _grid_topology(rng, n_nodes: int, extra: int):
    """Large topologies: routers on a jittered 4x4 grid, links only between
    neighbouring grid cells (as in real regional networks). This keeps big
    diagrams readable where random placement fails."""
    names = list(string.ascii_uppercase[:n_nodes])
    cells = rng.sample([(x, y) for x in range(4) for y in range(4)], n_nodes)
    cell = dict(zip(names, cells))
    pos = {k: ((x + rng.uniform(-0.18, 0.18)) / 3.0, (y + rng.uniform(-0.18, 0.18)) / 3.0)
           for k, (x, y) in cell.items()}

    def near(a, b):
        (x1, y1), (x2, y2) = cell[a], cell[b]
        return max(abs(x1 - x2), abs(y1 - y2)) == 1

    pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1:] if near(a, b)]
    g = nx.Graph()
    g.add_nodes_from(names)
    rng.shuffle(pairs)

    def crosses_existing(a, b):
        return any(_crosses(pos[a], pos[b], pos[c], pos[d])
                   for c, d in g.edges if len({a, b, c, d}) == 4)

    for a, b in pairs:  # random planar spanning tree over neighbouring cells
        if not nx.has_path(g, a, b) and not crosses_existing(a, b):
            g.add_edge(a, b, weight=rng.randint(1, 9))
    if not nx.is_connected(g):
        return None, None

    for a, b in pairs:
        if extra == 0:
            break
        if not g.has_edge(a, b) and not crosses_existing(a, b):
            g.add_edge(a, b, weight=rng.randint(1, 9))
            extra -= 1
    return g, pos


def _unique_shortest(g: nx.Graph, s: str, d: str):
    paths = list(nx.all_shortest_paths(g, s, d, weight="weight"))
    if len(paths) != 1 or len(paths[0]) < 3:  # want >=2 hops and uniqueness
        return None
    return paths[0]


def _seg_point_dist(p, a, b) -> float:
    ax_, ay = a
    bx, by = b
    px, py = p
    dx, dy = bx - ax_, by - ay
    t = max(0.0, min(1.0, ((px - ax_) * dx + (py - ay) * dy) / (dx * dx + dy * dy + 1e-12)))
    return math.hypot(px - (ax_ + t * dx), py - (ay + t * dy))


def _crosses(p1, p2, p3, p4) -> bool:
    def orient(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    return (orient(p1, p2, p3) * orient(p1, p2, p4) < 0
            and orient(p3, p4, p1) * orient(p3, p4, p2) < 0)


def _layout_score(g: nx.Graph, pos) -> float | None:
    """Score a placement in the unit square; None if it is not readable.

    Hard constraints: no link passes close to a router it does not touch,
    and every cost label (drawn at a link's midpoint) is clear of other
    links and other labels. Among readable placements we prefer few link
    crossings and short links.
    """
    nodes, edges = list(g.nodes), list(g.edges)
    s = 1.0 if len(nodes) <= 9 else 0.68  # larger figure -> same clearance on paper
    for n in nodes:
        for u, v in edges:
            if n not in (u, v) and _seg_point_dist(pos[n], pos[u], pos[v]) < 0.085 * s:
                return None
    mids = {e: ((pos[e[0]][0] + pos[e[1]][0]) / 2, (pos[e[0]][1] + pos[e[1]][1]) / 2)
            for e in edges}
    for e, m in mids.items():
        if any(math.dist(m, pos[n]) < 0.09 * s for n in nodes):
            return None
        for f in edges:
            if f != e and _seg_point_dist(m, pos[f[0]], pos[f[1]]) < 0.045 * s:
                return None
    crossings = sum(
        _crosses(pos[a], pos[b], pos[c], pos[d])
        for i, (a, b) in enumerate(edges) for c, d in edges[i + 1:]
        if len({a, b, c, d}) == 4)
    length = sum(math.dist(pos[u], pos[v]) for u, v in edges)
    return -crossings - 0.1 * length


def _scatter(nodes, rng, min_d: float):
    """Random positions in the unit square with a guaranteed minimum spacing."""
    pos: dict[str, tuple[float, float]] = {}
    for n in nodes:
        for _ in range(500):
            p = (rng.random(), rng.random())
            if all(math.dist(p, q) >= min_d for q in pos.values()):
                pos[n] = p
                break
        else:
            return None
    return pos


def _best_layout(g: nx.Graph, rng):
    """Sample many well-spaced placements and keep the most readable one.

    Readability = no link passes close to a router it does not touch, few
    link crossings, and short links (so cost labels stay near their link).
    """
    n = g.number_of_nodes()
    min_d = 0.42 if n <= 5 else (0.34 if n <= 7 else (0.29 if n <= 9 else 0.22))
    best, best_score = None, -1e9
    for _ in range(1500):
        pos = _scatter(list(g.nodes), rng, min_d)
        if pos is None:
            continue
        score = _layout_score(g, pos)
        if score is not None and score > best_score:
            best, best_score = pos, score
    return best


def _draw(g: nx.Graph, rng, pos=None) -> plt.Figure:
    pos = pos or _best_layout(g, rng)
    if pos is None:
        return None
    big = g.number_of_nodes() > 9
    fig, ax = plt.subplots(figsize=(7.0, 5.6) if big else (5.2, 4.2))
    nx.draw_networkx_edges(g, pos, ax=ax, width=1.6, edge_color="#555555")
    nx.draw_networkx_nodes(g, pos, ax=ax, node_size=620, node_color="#cfe3ff",
                           edgecolors="#1f4e9a", linewidths=1.5)
    nx.draw_networkx_labels(g, pos, ax=ax, font_size=12, font_weight="bold")
    nx.draw_networkx_edge_labels(
        g, pos, ax=ax, edge_labels=nx.get_edge_attributes(g, "weight"),
        font_size=11, font_color="#b00020", rotate=False,  # upright: avoids 6/9 confusion
        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"))
    ax.set_axis_off()
    ax.margins(0.12)
    return fig


def _text(g: nx.Graph) -> str:
    links = sorted((min(u, v), max(u, v), w) for u, v, w in g.edges(data="weight"))
    lines = [f"Routers: {', '.join(sorted(g.nodes))}", "Links (router-router: cost):"]
    lines += [f"  {u}-{v}: {w}" for u, v, w in links]
    return "\n".join(lines)


def generate(out_dir: Path, seed: int, per_difficulty: int) -> list[Item]:
    items: list[Item] = []
    for diff, (n, extra) in SIZES.items():
        k = 0
        attempt = 0
        while k < per_difficulty:
            attempt += 1
            rng = make_rng(seed, "topology", diff, attempt)
            pos = None
            if n > 9:
                g, pos = _grid_topology(rng, n, extra)
                if g is None:
                    continue
            else:
                g = _random_topology(rng, n, extra)
            s, d = rng.sample(sorted(g.nodes), 2)
            path = _unique_shortest(g, s, d)
            if path is None or (n > 9 and len(path) < 4):  # expert: at least 3 hops
                continue
            subtype = "path_cost" if k % 2 == 0 else "next_hop"
            iid = f"topology-{diff}-{k:03d}"
            fig = _draw(g, rng, pos)
            if fig is None:  # no readable layout found for this graph: skip it
                continue
            img = save_fig(fig, out_dir, iid)
            if subtype == "path_cost":
                cost = nx.path_weight(g, path, weight="weight")
                q = (f"The diagram shows a network of routers. Numbers on the links are link costs. "
                     f"What is the total cost of the least-cost path from router {s} to router {d}? "
                     f"Answer with a single integer.")
                ans, atype = str(cost), "int"
            else:
                q = (f"The diagram shows a network of routers. Numbers on the links are link costs. "
                     f"Using least-cost (shortest-path) routing, what is the next-hop router from "
                     f"router {s} towards router {d}? Answer with a single router letter.")
                ans, atype = path[1], "label"
            items.append(Item(
                id=iid, family="topology", subtype=subtype, difficulty=diff, question=q,
                answer=ans, answer_type=atype, image=img, text_rendering=_text(g),
                meta={"nodes": g.number_of_nodes(), "links": g.number_of_edges(),
                      "path": path}))
            k += 1
    return items
