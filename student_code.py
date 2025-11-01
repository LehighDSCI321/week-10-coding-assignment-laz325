"""Implements DAG and TraversableDigraph with traversal and sorting methods."""

from collections import deque


try:
    from sortable_digraph import SortableDigraph
except ModuleNotFoundError:
    class SortableDigraph:
        """Fallback for SortableDigraph if missing from environment."""

        def __init__(self):
            """Initialize adjacency, node values, and edge weights."""
            self._adj = {}
            self._node_values = {}
            self._edge_weights = {}

        def add_node(self, node, value=None):
            """Add a node to the graph."""
            if node not in self._adj:
                self._adj[node] = set()
                self._node_values[node] = value

        def add_edge(self, start, end, edge_weight=None):
            """Add an edge between start and end with optional weight."""
            if start not in self._adj:
                self._adj[start] = set()
            self._adj[start].add(end)
            self._edge_weights[(start, end)] = edge_weight

        def dfs(self, start, visited=None):
            """Perform depth-first traversal (used internally)."""
            if visited is None:
                visited = set()
            visited.add(start)
            for neighbor in self._adj.get(start, []):
                if neighbor not in visited:
                    self.dfs(neighbor, visited)
            return visited

        def get_nodes(self):
            """Return all nodes in the graph."""
            return list(self._adj.keys())

        def get_node_value(self, node):
            """Return stored value for the given node."""
            return self._node_values.get(node, None)

        def get_edge_weight(self, start, end):
            """Return stored weight for a given edge."""
            return self._edge_weights.get((start, end), None)


class TraversableDigraph(SortableDigraph):
    """Adds DFS and BFS traversal methods that yield nodes."""

    def dfs(self, start, visited=None):
        """Depth-first search generator that excludes the start node itself."""
        if visited is None:
            visited = set()
        for neighbor in sorted(self._adj.get(start, [])):
            if neighbor not in visited:
                visited.add(neighbor)
                yield neighbor
                yield from self.dfs(neighbor, visited)

    def bfs(self, start):
        """Breadth-first search generator that excludes the start node itself."""
        visited = set([start])
        queue = deque([start])
        while queue:
            node = queue.popleft()
            for neighbor in sorted(self._adj.get(node, [])):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
                    yield neighbor


class DAG(TraversableDigraph):
    """Directed Acyclic Graph overrides add_edge to prevent cycles."""

    def add_edge(self, start, end, edge_weight=None):
        """Add edge while preventing self-loops and cycles."""
        if start == end:
            raise ValueError("Self-loops are not allowed in a DAG.")

        if start not in self._adj:
            self.add_node(start)
        if end not in self._adj:
            self.add_node(end)

        if start in self.dfs(end):
            raise ValueError("Adding this edge would create a cycle.")

        super().add_edge(start, end, edge_weight=edge_weight)

    def successors(self, node):
        """Return a sorted list of successors (outgoing edges) of node."""
        return sorted(self._adj.get(node, []))

    def predecessors(self, node):
        """Return a sorted list of predecessors (incoming edges) of node."""
        preds = [n for n, nbrs in self._adj.items() if node in nbrs]
        return sorted(preds)

    def top_sort(self):
        """Return nodes in topological order using Kahn’s algorithm."""
        in_degree = {node: 0 for node in self._adj}
        for nbrs in self._adj.values():
            for v in nbrs:
                in_degree[v] = in_degree.get(v, 0) + 1

        queue = [n for n, deg in in_degree.items() if deg == 0]
        order = []
        while queue:
            node = queue.pop(0)
            order.append(node)
            for nbr in sorted(self._adj.get(node, [])):
                in_degree[nbr] -= 1
                if in_degree[nbr] == 0:
                    queue.append(nbr)
        return order


if __name__ == "__main__":
    graph = DAG()
    graph.add_node("shirt", 10)
    graph.add_node("pants", 20)
    graph.add_edge("shirt", "pants", edge_weight=5)
    print("Nodes:", graph.get_nodes())
    print("Node value (shirt):", graph.get_node_value("shirt"))
    print("Edge weight (shirt→pants):", graph.get_edge_weight("shirt", "pants"))
    print("Successors of shirt:", graph.successors("shirt"))
    print("Topological order:", graph.top_sort())
