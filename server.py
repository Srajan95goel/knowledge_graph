"""
Knowledge Graph MCP Server — Hash-based, bidirectional graph with flat nodes.

This server loads a graph YAML file (hash-keyed, collision-free) into a
NetworkX directed graph, and exposes MCP tools for codebase structural queries
with bidirectional traversal.

Key features:
  - Nodes are addressed by 12-char sha256 hashes → zero collisions
  - All edges are bidirectional (imports/imported_by, calls/called_by, etc.)
  - Fixture → Class "instantiates" links resolve the fixture↔class gap
  - Methods are first-class nodes in YAML (not promoted at load time)
  - Parent/children hierarchy enables tree traversal
  - Index section enables O(1) name→hash lookups

Tools provided:
  - get_architecture: Package tree overview
  - find_impact: BFS reverse traversal for blast radius
  - find_dependencies: DFS forward traversal for deps
  - find_path: Shortest path between two nodes
  - get_domain_subgraph: Everything in a domain/module
  - get_fixture_chain: Full fixture dependency chain
  - query_node: Full details of a node
  - search_nodes: Search by name/pattern
  - get_change_recipe: Step-by-step guides
  - get_graph_stats: Codebase statistics
  - get_source_snippet: Read actual source code via graph
  - get_clarification_checklist: Decision gate for tasks
  - validate_task_context: Readiness check

Usage:
    python server.py
"""

import os
import re
import sys
import json
import logging
import difflib
from pathlib import Path
from typing import Any, Optional
from collections import deque

try:
    import yaml
    import networkx as nx
    from mcp.server.fastmcp import FastMCP
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install pyyaml networkx mcp[cli]")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Graph Loader V2
# ─────────────────────────────────────────────────────────────

class CodebaseGraph:
    """Loads a hash-keyed graph YAML into NetworkX."""

    def __init__(self, graph_yaml_path: str):
        self.yaml_path = graph_yaml_path
        self.G = nx.DiGraph()
        self.raw_data = {}
        self.nodes_dict = {}    # hash → node dict (from YAML)
        self.index = {}         # name→hash, type→[hashes], file→hash lookups
        self._load()

    def _load(self):
        """Load YAML and build NetworkX graph from flat nodes."""
        logger.info(f"Loading graph from {self.yaml_path}")
        with open(self.yaml_path, "r", encoding="utf-8") as f:
            self.raw_data = yaml.safe_load(f)

        self.nodes_dict = self.raw_data.get("nodes", {})
        self.index = self.raw_data.get("index", {})

        # Phase 1: Add all nodes to NetworkX
        for h, node in self.nodes_dict.items():
            self.G.add_node(h, **{
                "type": node.get("type", "unknown"),
                "name": node.get("name", ""),
                "fqn": node.get("fqn", ""),
                "hash": h,
                "file": node.get("file", node.get("path", node.get("defined_in", ""))),
                "line": node.get("line", 0),
                "purpose": node.get("purpose", ""),
            })

        # Phase 2: Build edges from embedded relationships
        edge_count = 0
        for h, node in self.nodes_dict.items():
            # Parent/children (tree hierarchy)
            for child_hash in node.get("children", []):
                if child_hash in self.nodes_dict:
                    self.G.add_edge(h, child_hash, edge_type="contains")
                    edge_count += 1

            # Imports (file → file)
            for imp_hash in node.get("imports", []):
                if imp_hash in self.nodes_dict:
                    self.G.add_edge(h, imp_hash, edge_type="imports")
                    edge_count += 1

            # Calls (method → method)
            for call_hash in node.get("calls", []):
                if call_hash in self.nodes_dict:
                    self.G.add_edge(h, call_hash, edge_type="calls")
                    edge_count += 1

            # Fixture depends_on
            for dep_hash in node.get("depends_on", []):
                if dep_hash in self.nodes_dict:
                    self.G.add_edge(h, dep_hash, edge_type="fixture_depends")
                    edge_count += 1

            # Inheritance (bases_resolved)
            for base in node.get("bases_resolved", []):
                base_hash = base.get("hash")
                if base_hash and base_hash in self.nodes_dict:
                    self.G.add_edge(h, base_hash, edge_type="inherits")
                    edge_count += 1

            # Fixture instantiates → class
            inst = node.get("instantiates", {})
            if isinstance(inst, dict) and "hash" in inst:
                self.G.add_edge(h, inst["hash"], edge_type="instantiates")
                edge_count += 1

            # Fixtures provided by class
            for fix_hash in node.get("fixtures_provided", []):
                if fix_hash in self.nodes_dict:
                    self.G.add_edge(h, fix_hash, edge_type="provides_fixture")
                    edge_count += 1

            # Subclasses (reverse inheritance)
            for sub_hash in node.get("subclasses", []):
                if sub_hash in self.nodes_dict:
                    self.G.add_edge(h, sub_hash, edge_type="has_subclass")
                    edge_count += 1

            # Exposed_by_fixtures (reverse instantiates: class ← fixture)
            for efx_hash in node.get("exposed_by_fixtures", []):
                if efx_hash in self.nodes_dict:
                    self.G.add_edge(h, efx_hash, edge_type="exposed_by_fixture")
                    edge_count += 1

        logger.info(
            f"Graph built: {self.G.number_of_nodes()} nodes, "
            f"{edge_count} edges loaded"
        )

    # ── Node Resolution ────────────────────────────────────

    def _find_node(self, name: str) -> Optional[str]:
        """Find a node by name/hash, using the index for fast lookup.

        Resolution order (fast → slow, exact → fuzzy):
          1. Direct hash match
          2. Exact fixture name (highest priority for named lookups)
          3. Exact name in by_name index (prefer fixture/class over method)
          4. Exact file path
          5. Case-insensitive exact name
          6. Partial filename (without extension)
          7. Substring match in FQN
          8. Fuzzy match (typo-tolerant via difflib)
        """
        # 1. Direct hash match
        if name in self.nodes_dict:
            return name

        by_name = self.index.get("by_name", {})
        fix_names = self.index.get("fixture_names", {})
        by_file = self.index.get("by_file", {})

        # 2. Exact fixture name (fixtures are the most common lookup target)
        if name in fix_names:
            return fix_names[name]

        # 3. Exact name in by_name index — prefer fixture/class/file over method
        if name in by_name:
            val = by_name[name]
            if isinstance(val, str):
                return val
            # Multiple matches — prefer fixture > class > file > function > method
            type_priority = {"fixture": 0, "class": 1, "file": 2, "function": 3,
                             "package": 4, "method": 5}
            sorted_hashes = sorted(
                val,
                key=lambda h: type_priority.get(
                    self.nodes_dict.get(h, {}).get("type", ""), 99
                )
            )
            return sorted_hashes[0]

        # 4. Exact file path
        if name in by_file:
            return by_file[name]

        name_lower = name.lower()

        # 5. Case-insensitive exact name
        for n, val in by_name.items():
            if n.lower() == name_lower:
                return val if isinstance(val, str) else val[0]

        # 6. Partial filename match (user says 'my_module' without .py)
        for path, h in by_file.items():
            filename = path.rsplit("/", 1)[-1]
            stem = filename.rsplit(".", 1)[0] if "." in filename else filename
            if name_lower == stem.lower() or name_lower == filename.lower():
                return h

        # 7. Substring match in names/FQNs
        for h, node in self.nodes_dict.items():
            fqn = node.get("fqn", "").lower()
            if name_lower in fqn:
                return h

        # 8. Fuzzy match — typo-tolerant (e.g. "modle" → "Module")
        fuzzy_hit = self._fuzzy_find(name)
        if fuzzy_hit:
            return fuzzy_hit

        return None

    def _fuzzy_find(self, name: str, cutoff: float = 0.6) -> Optional[str]:
        """Fuzzy match using difflib.get_close_matches.

        Searches across all node names and fixture names for the
        closest match above the cutoff threshold (0.6 = 60% similar).
        """
        # Build candidate pool: all unique names from the index
        by_name = self.index.get("by_name", {})
        fix_names = self.index.get("fixture_names", {})
        by_file = self.index.get("by_file", {})

        # Pool 1: all node names (class, function, method, etc.)
        all_names = list(by_name.keys())

        # Pool 2: fixture names
        all_names.extend(fix_names.keys())

        # Pool 3: file stems (e.g. "my_module" from "src/.../my_module.py")
        file_stems = {}
        for path, h in by_file.items():
            stem = path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
            file_stems[stem] = h
        all_names.extend(file_stems.keys())

        # Get close matches (case-insensitive by comparing lowered)
        lower_map = {}  # lowered_name → original_name
        for n in all_names:
            lower_map.setdefault(n.lower(), n)

        matches = difflib.get_close_matches(
            name.lower(), lower_map.keys(), n=1, cutoff=cutoff
        )
        if not matches:
            return None

        best_lower = matches[0]
        best_original = lower_map[best_lower]

        # Resolve back to hash
        if best_original in by_name:
            val = by_name[best_original]
            return val if isinstance(val, str) else val[0]
        if best_original in fix_names:
            return fix_names[best_original]
        if best_original in file_stems:
            return file_stems[best_original]

        return None

    def _suggest_nodes(self, name: str, limit: int = 5) -> list:
        """Suggest similar nodes when exact match fails.

        Uses 3 strategies:
          1. Substring containment (fast)
          2. Token overlap (splits on _ and checks parts)
          3. Fuzzy/edit-distance via difflib (typo-tolerant)
        """
        name_lower = name.lower()
        by_name = self.index.get("by_name", {})
        fix_names = self.index.get("fixture_names", {})

        # Strategy 1 & 2: substring + token overlap
        substring_hits = []
        token_hits = []
        tokens = [t for t in name_lower.split("_") if len(t) > 2]

        for n, val in by_name.items():
            n_lower = n.lower()
            h = val if isinstance(val, str) else val[0]
            entry = {"name": n, "hash": h, "type": self.nodes_dict.get(h, {}).get("type", "")}

            if name_lower in n_lower or n_lower in name_lower:
                substring_hits.append(entry)
            elif tokens and any(t in n_lower for t in tokens):
                token_hits.append(entry)

        # Strategy 3: fuzzy matches via difflib
        all_names = list(by_name.keys()) + list(fix_names.keys())
        lower_map = {}
        for n in all_names:
            lower_map.setdefault(n.lower(), n)

        fuzzy_matches = difflib.get_close_matches(
            name_lower, lower_map.keys(), n=limit, cutoff=0.5
        )
        fuzzy_hits = []
        for fm in fuzzy_matches:
            orig = lower_map[fm]
            if orig in by_name:
                val = by_name[orig]
                h = val if isinstance(val, str) else val[0]
            elif orig in fix_names:
                h = fix_names[orig]
            else:
                continue
            fuzzy_hits.append({
                "name": orig, "hash": h,
                "type": self.nodes_dict.get(h, {}).get("type", ""),
                "match": "fuzzy",
            })

        # Merge, deduplicate, prioritize: substring > token > fuzzy
        seen = set()
        results = []
        for entry in substring_hits + token_hits + fuzzy_hits:
            if entry["hash"] not in seen:
                seen.add(entry["hash"])
                results.append(entry)
                if len(results) >= limit:
                    break

        return results

    def _node_summary(self, h: str) -> dict:
        """Get a compact summary of a node for result formatting."""
        node = self.nodes_dict.get(h, {})
        return {
            "hash": h,
            "name": node.get("name", ""),
            "type": node.get("type", "unknown"),
            "fqn": node.get("fqn", ""),
            "file": node.get("file", node.get("path", node.get("defined_in", ""))),
            "purpose": node.get("purpose", ""),
        }

    # ── Graph Algorithms ───────────────────────────────────

    def bfs_impact(self, start_node: str, depth: int = 3,
                   edge_types: Optional[list] = None) -> dict:
        """BFS reverse traversal: who depends on this node?"""
        start_id = self._find_node(start_node)
        if not start_id:
            return {"error": f"Node '{start_node}' not found",
                    "suggestions": self._suggest_nodes(start_node)}

        reverse_g = self.G.reverse()
        visited = {}
        queue = deque([(start_id, 0)])
        visited[start_id] = 0

        while queue:
            node, d = queue.popleft()
            if d >= depth:
                continue
            for neighbor in reverse_g.neighbors(node):
                edge_data = reverse_g.edges[node, neighbor]
                if edge_types and edge_data.get("edge_type") not in edge_types:
                    continue
                if neighbor not in visited:
                    visited[neighbor] = d + 1
                    queue.append((neighbor, d + 1))

        by_depth = {}
        for node_id, d in visited.items():
            if node_id == start_id:
                continue
            d_key = str(d)
            if d_key not in by_depth:
                by_depth[d_key] = []
            by_depth[d_key].append(self._node_summary(node_id))

        return {
            "start": self._node_summary(start_id),
            "depth": depth,
            "total_impacted": len(visited) - 1,
            "by_depth": by_depth,
        }

    def dfs_dependencies(self, start_node: str, depth: int = 5,
                         edge_types: Optional[list] = None) -> dict:
        """DFS forward traversal: what does this node depend on?"""
        start_id = self._find_node(start_node)
        if not start_id:
            return {"error": f"Node '{start_node}' not found",
                    "suggestions": self._suggest_nodes(start_node)}

        visited = set()
        tree = {}

        def _dfs(node, d):
            if d > depth or node in visited:
                return
            visited.add(node)
            children = []
            for neighbor in self.G.neighbors(node):
                edge_data = self.G.edges[node, neighbor]
                if edge_types and edge_data.get("edge_type") not in edge_types:
                    continue
                child_info = self._node_summary(neighbor)
                child_info["edge_type"] = edge_data.get("edge_type", "unknown")
                children.append(child_info)
                _dfs(neighbor, d + 1)
            tree[node] = children

        _dfs(start_id, 0)
        return {
            "start": self._node_summary(start_id),
            "total_dependencies": len(visited) - 1,
            "tree": tree,
        }

    def shortest_path(self, source: str, target: str) -> dict:
        """Find shortest path between two nodes."""
        src_id = self._find_node(source)
        tgt_id = self._find_node(target)

        if not src_id:
            return {"error": f"Source '{source}' not found",
                    "suggestions": self._suggest_nodes(source)}
        if not tgt_id:
            return {"error": f"Target '{target}' not found",
                    "suggestions": self._suggest_nodes(target)}

        # Try directed, reversed, then undirected
        path = None
        for attempt in ["directed", "reversed", "undirected"]:
            try:
                if attempt == "directed":
                    path = nx.shortest_path(self.G, src_id, tgt_id)
                elif attempt == "reversed":
                    path = list(reversed(nx.shortest_path(self.G, tgt_id, src_id)))
                else:
                    path = nx.shortest_path(self.G.to_undirected(), src_id, tgt_id)
                break
            except nx.NetworkXNoPath:
                continue

        if not path:
            return {"error": f"No path between '{source}' and '{target}'"}

        result_path = []
        for i, node_hash in enumerate(path):
            info = self._node_summary(node_hash)
            if i < len(path) - 1:
                edge_data = self.G.edges.get((path[i], path[i + 1]), {})
                if not edge_data:
                    edge_data = self.G.edges.get((path[i + 1], path[i]), {})
                info["edge_to_next"] = edge_data.get("edge_type", "connected")
            result_path.append(info)

        return {
            "source": self._node_summary(src_id),
            "target": self._node_summary(tgt_id),
            "length": len(path),
            "path": result_path,
        }

    def get_domain_subgraph(self, domain_name: str) -> dict:
        """Extract everything related to a domain."""
        domain_name = domain_name.lower().strip()

        related = []
        for h, node in self.nodes_dict.items():
            searchable = " ".join([
                node.get("fqn", ""),
                node.get("name", ""),
                node.get("file", ""),
                node.get("path", ""),
                node.get("defined_in", ""),
            ]).lower()
            if domain_name in searchable:
                related.append(h)

        # Extend 2 hops outward
        extended = set(related)
        for node_h in related:
            for neighbor in self.G.neighbors(node_h):
                extended.add(neighbor)
                for n2 in self.G.neighbors(neighbor):
                    extended.add(n2)

        # Reverse deps
        reverse_g = self.G.reverse()
        dependents = set()
        for node_h in related:
            for neighbor in reverse_g.neighbors(node_h):
                dependents.add(neighbor)

        # Classify
        files = []
        classes = []
        fixtures = []
        methods = []
        for h in related:
            node = self.nodes_dict.get(h, {})
            ntype = node.get("type", "")
            summary = self._node_summary(h)
            if ntype == "file":
                files.append(summary)
            elif ntype == "class":
                summary["bases"] = node.get("bases", [])
                summary["method_count"] = len(node.get("children", []))
                classes.append(summary)
            elif ntype == "fixture":
                summary["scope"] = node.get("scope", "")
                summary["instantiates"] = node.get("instantiates", {})
                fixtures.append(summary)
            elif ntype == "method":
                methods.append(summary)

        deps_list = [self._node_summary(h) for h in (extended - set(related))][:50]
        dep_by_list = [self._node_summary(h) for h in dependents][:50]

        return {
            "domain": domain_name,
            "files": files,
            "classes": classes,
            "fixtures": fixtures,
            "methods": methods[:30],
            "dependencies": deps_list,
            "dependents": dep_by_list,
            "total_domain_nodes": len(related),
        }

    def get_fixture_chain(self, fixture_name: str) -> dict:
        """Trace fixture dependency chain via fixture_depends edges."""
        fixture_id = self._find_node(fixture_name)
        if not fixture_id:
            # Try fixture_names index
            fix_names = self.index.get("fixture_names", {})
            fixture_id = fix_names.get(fixture_name)
            if not fixture_id:
                return {"error": f"Fixture '{fixture_name}' not found",
                        "suggestions": self._suggest_nodes(fixture_name)}

        visited = set()
        chain = []

        def _trace(node_hash, depth):
            if node_hash in visited:
                return
            visited.add(node_hash)
            node = self.nodes_dict.get(node_hash, {})
            entry = {
                "hash": node_hash,
                "fixture": node.get("name", ""),
                "defined_in": node.get("defined_in", node.get("file", "")),
                "scope": node.get("scope", ""),
                "depth": depth,
                "instantiates": node.get("instantiates", {}),
            }
            chain.append(entry)

            for neighbor in self.G.neighbors(node_hash):
                edge_data = self.G.edges[node_hash, neighbor]
                if edge_data.get("edge_type") == "fixture_depends":
                    _trace(neighbor, depth + 1)

        _trace(fixture_id, 0)
        return {
            "fixture": fixture_name,
            "chain_length": len(chain),
            "chain": chain,
        }

    def query_node(self, node_name: str) -> dict:
        """Full details of a node including all embedded edges."""
        node_id = self._find_node(node_name)
        if not node_id:
            return {"error": f"Node '{node_name}' not found",
                    "suggestions": self._suggest_nodes(node_name)}

        node = dict(self.nodes_dict.get(node_id, {}))

        # NetworkX edges (for completeness)
        outgoing = []
        for _, target in self.G.out_edges(node_id):
            edge_data = self.G.edges[node_id, target]
            target_summary = self._node_summary(target)
            target_summary["edge_type"] = edge_data.get("edge_type", "unknown")
            outgoing.append(target_summary)

        incoming = []
        for source, _ in self.G.in_edges(node_id):
            edge_data = self.G.edges[source, node_id]
            source_summary = self._node_summary(source)
            source_summary["edge_type"] = edge_data.get("edge_type", "unknown")
            incoming.append(source_summary)

        return {
            "hash": node_id,
            "attributes": node,
            "outgoing_edges": outgoing,
            "incoming_edges": incoming,
            "out_degree": len(outgoing),
            "in_degree": len(incoming),
        }

    def search_nodes(self, pattern: str, node_type: Optional[str] = None,
                     limit: int = 20) -> dict:
        """Search nodes by name/FQN pattern (regex supported)."""
        try:
            regex = re.compile(pattern, re.IGNORECASE)
        except re.error:
            regex = re.compile(re.escape(pattern), re.IGNORECASE)

        results = []

        # Optimization: if node_type specified, only search those hashes
        if node_type:
            candidate_hashes = self.index.get("by_type", {}).get(node_type, [])
        else:
            candidate_hashes = self.nodes_dict.keys()

        for h in candidate_hashes:
            node = self.nodes_dict.get(h, {})
            searchable = " ".join([
                node.get("name", ""),
                node.get("fqn", ""),
                node.get("file", ""),
                node.get("path", ""),
                node.get("purpose", ""),
            ])
            if regex.search(searchable):
                results.append(self._node_summary(h))
                if len(results) >= limit:
                    break

        return {"pattern": pattern, "count": len(results), "results": results}

    def get_architecture(self) -> dict:
        """Return root package tree as architecture overview."""
        # Build tree from package nodes
        root_packages = []
        by_type = self.index.get("by_type", {})
        package_hashes = by_type.get("package", [])

        for h in package_hashes:
            node = self.nodes_dict.get(h, {})
            if "parent" not in node:  # root packages
                root_packages.append(self._build_package_subtree(h, max_depth=2))

        meta = self.raw_data.get("meta", {})
        return {
            "meta": meta,
            "package_tree": root_packages,
        }

    def _build_package_subtree(self, h: str, max_depth: int, depth: int = 0) -> dict:
        """Build a nested package tree for architecture view."""
        node = self.nodes_dict.get(h, {})
        result = {
            "name": node.get("name", ""),
            "hash": h,
            "type": node.get("type", ""),
        }
        if node.get("purpose"):
            result["purpose"] = node["purpose"]

        if depth < max_depth:
            children = []
            for child_hash in node.get("children", []):
                child_node = self.nodes_dict.get(child_hash, {})
                if child_node.get("type") == "package":
                    children.append(self._build_package_subtree(child_hash, max_depth, depth + 1))
            if children:
                result["subpackages"] = children

            # Count non-package children
            file_count = sum(
                1 for ch in node.get("children", [])
                if self.nodes_dict.get(ch, {}).get("type") == "file"
            )
            if file_count:
                result["file_count"] = file_count

        return result

    def get_stats(self) -> dict:
        """Get overall graph statistics."""
        meta = self.raw_data.get("meta", {})

        # Most connected
        degree_list = sorted(self.G.degree(), key=lambda x: x[1], reverse=True)
        top_connected = [
            {**self._node_summary(n), "connections": d}
            for n, d in degree_list[:15]
        ]

        return {
            "meta": meta,
            "graph_stats": {
                "nodes": self.G.number_of_nodes(),
                "edges": self.G.number_of_edges(),
                "weakly_connected_components": nx.number_weakly_connected_components(self.G),
            },
            "most_connected_nodes": top_connected,
        }


# ─────────────────────────────────────────────────────────────
# Change Recipes (same as V1)
# ─────────────────────────────────────────────────────────────

CHANGE_RECIPES = {
    "add_module": {
        "description": "Add a new module/package to the codebase",
        "steps": [
            "1. Create directory: src/<package_name>/",
            "2. Create src/<package_name>/__init__.py",
            "3. Create main module files",
            "4. Add imports in parent package __init__.py if needed",
            "5. Optionally create tests/<package_name>/ for unit tests",
        ],
        "files_to_modify": [
            "src/<package_name>/__init__.py (create)",
            "src/<package_name>/<module>.py (create)",
        ],
    },
    "add_fixture": {
        "description": "Add a new pytest fixture",
        "steps": [
            "1. Open the relevant conftest.py or fixtures module",
            "2. Add @pytest.fixture decorated function/method",
            "3. Name convention: function = fixture_<name>, exposed name = <name>",
            "4. Dependencies: add other fixtures as function parameters",
            "5. Use yield for setup/teardown pattern",
        ],
        "files_to_modify": [
            "conftest.py or <package>/fixtures.py (edit)",
        ],
    },
    "add_class": {
        "description": "Add a new class to an existing module",
        "steps": [
            "1. Identify the appropriate module/package",
            "2. Create the class with proper inheritance if needed",
            "3. Register in __init__.py exports if public",
            "4. Add unit tests",
        ],
        "files_to_modify": [
            "src/<package>/<module>.py (edit or create)",
        ],
    },
    "add_interface": {
        "description": "Add a new abstract interface (ABC)",
        "steps": [
            "1. Create interface module: src/<package>/interfaces/<name>.py",
            "2. Define ABC class with @abstractmethod methods",
            "3. Create concrete implementation(s)",
            "4. Register in factory if applicable",
        ],
        "files_to_modify": [
            "src/<package>/interfaces/<name>.py (create)",
            "src/<package>/<implementation>.py (create)",
        ],
    },
    "fix_bug": {
        "description": "Fix a bug in existing code",
        "steps": [
            "1. Identify the component from bug description",
            "2. Use get_domain_subgraph('<component>') to see all related files",
            "3. Use find_dependencies('<component>') to understand the dependency chain",
            "4. Locate the buggy code, make the fix",
            "5. Run existing tests",
        ],
        "files_to_modify": [
            "src/<package>/<relevant_file>.py (edit)",
        ],
    },
}

# ─────────────────────────────────────────────────────────────
# Context Checklists (Decision Gate)
# ─────────────────────────────────────────────────────────────

CONTEXT_CHECKLISTS = {
    "debug_fix": {
        "description": "Debug an issue and fix code",
        "required_context": [
            {"id": "problem_statement", "question": "What is the exact problem or error?",
             "why": "Without the exact error, the fix may target the wrong root cause"},
            {"id": "component", "question": "Which component or module is affected?",
             "why": "Need to know which subgraph to search"},
            {"id": "repro_conditions", "question": "How is this issue reproduced?",
             "why": "Many bugs are environment/config-specific"},
            {"id": "expected_vs_actual", "question": "Expected behavior vs actual?",
             "why": "Needed to verify the fix resolves the issue"},
        ],
        "optional_context": [
            {"id": "ticket_id", "question": "Is there a ticket/issue ID?"},
        ],
    },
    "write_test": {
        "description": "Write a new test",
        "required_context": [
            {"id": "target_class_or_function", "question": "What class/function should this test cover?",
             "why": "Need exact target to determine mock strategy and imports"},
            {"id": "test_type", "question": "Unit test (mocked) or integration test?",
             "why": "Determines the test pattern and dependencies"},
            {"id": "behaviors_to_test", "question": "What specific behaviors/scenarios should be tested?",
             "why": "Determines which test methods to write and what to assert"},
        ],
    },
    "code_change": {
        "description": "Make a code change to the codebase",
        "required_context": [
            {"id": "what_to_change", "question": "What exactly needs to change?",
             "why": "Vague requests lead to wrong implementations"},
            {"id": "where_to_change", "question": "Which file(s) or component(s)?",
             "why": "Need to narrow the target"},
            {"id": "backward_compatibility", "question": "Should this be backward-compatible?",
             "why": "Breaking changes may cascade to dependents"},
        ],
    },
}


# ─────────────────────────────────────────────────────────────
# MCP Server
# ─────────────────────────────────────────────────────────────

GRAPH_YAML = os.path.join(os.path.dirname(__file__), "graph.yaml")
mcp = FastMCP("knowledge-graph", instructions="""
Knowledge Graph MCP Server — hash-based, bidirectional, collision-free.
Provides structural queries over any Python codebase. All nodes are addressed
by deterministic 12-char hashes. Start with get_architecture or search_nodes
to orient, then drill down.
""")

# Lazy-loaded graph
graph: Optional[CodebaseGraph] = None


def _ensure_graph() -> CodebaseGraph:
    global graph
    if graph is None:
        graph = CodebaseGraph(GRAPH_YAML)
    return graph


@mcp.tool()
def get_architecture() -> str:
    """Get the high-level package tree architecture of the codebase.

    Returns the top-level packages, their sub-packages, and purpose.
    Always call this FIRST to orient yourself.
    """
    g = _ensure_graph()
    result = g.get_architecture()
    return yaml.dump(result, default_flow_style=False, sort_keys=False)


@mcp.tool()
def find_impact(node_name: str, depth: int = 3) -> str:
    """BFS impact analysis — find everything affected by changing a node.

    Traverses REVERSE dependency edges to find all impacted components.

    Args:
        node_name: Name of class, fixture, file, or method
        depth: How many levels deep (default 3, max 6)
    """
    g = _ensure_graph()
    result = g.bfs_impact(node_name, min(depth, 6))
    return json.dumps(result, indent=2, default=str)


@mcp.tool()
def find_dependencies(node_name: str, depth: int = 5) -> str:
    """DFS dependency analysis — find everything a node depends on.

    Args:
        node_name: Name of class, fixture, or file
        depth: How many levels deep (default 5, max 8)
    """
    g = _ensure_graph()
    result = g.dfs_dependencies(node_name, min(depth, 8))
    return json.dumps(result, indent=2, default=str)


@mcp.tool()
def find_path(source: str, target: str) -> str:
    """Find the shortest connection path between two nodes.

    Args:
        source: Starting node name
        target: Ending node name
    """
    g = _ensure_graph()
    result = g.shortest_path(source, target)
    return json.dumps(result, indent=2, default=str)


@mcp.tool()
def get_domain_subgraph(domain_name: str) -> str:
    """Extract the complete subgraph for a specific domain or module.

    Args:
        domain_name: Domain/module name keyword to search for
    """
    g = _ensure_graph()
    result = g.get_domain_subgraph(domain_name)
    return json.dumps(result, indent=2, default=str)


@mcp.tool()
def get_fixture_chain(fixture_name: str) -> str:
    """Trace the full fixture dependency chain.

    Args:
        fixture_name: Name of the fixture
    """
    g = _ensure_graph()
    result = g.get_fixture_chain(fixture_name)
    return json.dumps(result, indent=2, default=str)


@mcp.tool()
def query_node(node_name: str) -> str:
    """Get full details of a node including all edges, attributes, and relationships.

    Args:
        node_name: Name, hash, or FQN of the node
    """
    g = _ensure_graph()
    result = g.query_node(node_name)
    return json.dumps(result, indent=2, default=str)


@mcp.tool()
def search_nodes(pattern: str, node_type: str = "", limit: int = 20) -> str:
    """Search for nodes by name pattern (regex supported).

    Args:
        pattern: Regex pattern (e.g. 'auth.*handler', 'service', 'factory')
        node_type: Filter: 'class', 'fixture', 'file', 'package', 'method', 'function'
        limit: Max results (default 20)
    """
    g = _ensure_graph()
    result = g.search_nodes(pattern, node_type if node_type else None, limit)
    return json.dumps(result, indent=2, default=str)


@mcp.tool()
def get_change_recipe(change_type: str) -> str:
    """Get a step-by-step recipe for common codebase changes.

    Args:
        change_type: 'add_module', 'add_fixture', 'add_class',
                     'add_interface', 'fix_bug'
    """
    recipe = CHANGE_RECIPES.get(change_type)
    if not recipe:
        return json.dumps({"error": f"Unknown '{change_type}'",
                           "available": list(CHANGE_RECIPES.keys())}, indent=2)
    return json.dumps(recipe, indent=2, default=str)


@mcp.tool()
def get_graph_stats() -> str:
    """Get overall statistics of the codebase graph."""
    g = _ensure_graph()
    return json.dumps(g.get_stats(), indent=2, default=str)


@mcp.tool()
def get_source_snippet(node_name: str, context_lines: int = 20) -> str:
    """Read actual source code for any graph node (class, method, fixture, file).

    Args:
        node_name: Node name, hash, or FQN
        context_lines: Lines of context around target (default 20)
    """
    g = _ensure_graph()

    node_id = g._find_node(node_name)
    if not node_id:
        return json.dumps({
            "error": f"Node '{node_name}' not found",
            "suggestions": g._suggest_nodes(node_name),
        }, indent=2)

    node = g.nodes_dict.get(node_id, {})
    node_type = node.get("type", "unknown")

    # Determine file path and target line
    rel_path = node.get("file", node.get("path", node.get("defined_in", "")))
    target_line = node.get("line", 1)

    if not rel_path:
        return json.dumps({"error": f"Node '{node_id}' has no file path"}, indent=2)

    # Resolve absolute path
    src_root = g.raw_data.get("meta", {}).get("generated_from", "")
    abs_path = os.path.join(src_root, rel_path)

    if not os.path.isfile(abs_path):
        # Try without common prefixes
        for prefix in ["src/"]:
            candidate = os.path.join(src_root, rel_path.removeprefix(prefix))
            if os.path.isfile(candidate):
                abs_path = candidate
                break
        if not os.path.isfile(abs_path):
            parent = os.path.dirname(src_root)
            candidate = os.path.join(parent, rel_path)
            if os.path.isfile(candidate):
                abs_path = candidate

    if not os.path.isfile(abs_path):
        return json.dumps({"error": f"File not found: {abs_path}", "rel_path": rel_path}, indent=2)

    try:
        with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except Exception as e:
        return json.dumps({"error": f"Failed to read {abs_path}: {e}"}, indent=2)

    total = len(lines)
    start = max(0, target_line - 1 - context_lines // 3)
    end = min(total, target_line - 1 + context_lines)
    snippet = "".join(lines[start:end])

    return json.dumps({
        "node": node_id,
        "node_type": node_type,
        "name": node.get("name", ""),
        "file": rel_path,
        "absolute_path": abs_path,
        "target_line": target_line,
        "line_range": f"{start + 1}-{end}",
        "total_file_lines": total,
        "source": snippet,
    }, indent=2, default=str)


@mcp.tool()
def get_clarification_checklist(task_type: str) -> str:
    """Get the required context checklist for a task type.

    Call FIRST when starting any non-trivial task.

    Args:
        task_type: 'debug_fix', 'write_test', 'code_change'
    """
    checklist = CONTEXT_CHECKLISTS.get(task_type)
    if not checklist:
        return json.dumps({"error": f"Unknown '{task_type}'",
                           "available": list(CONTEXT_CHECKLISTS.keys())}, indent=2)
    return json.dumps(checklist, indent=2, default=str)


@mcp.tool()
def validate_task_context(task_type: str, context_gathered: dict) -> str:
    """Check if enough context has been gathered for a task.

    Args:
        task_type: Same as get_clarification_checklist
        context_gathered: Dict mapping context IDs to values
    """
    checklist = CONTEXT_CHECKLISTS.get(task_type)
    if not checklist:
        return json.dumps({"error": f"Unknown '{task_type}'"}, indent=2)

    required = checklist.get("required_context", [])
    covered = []
    missing = []
    for item in required:
        if item["id"] in context_gathered and context_gathered[item["id"]]:
            covered.append({"id": item["id"], "value_summary": str(context_gathered[item["id"]])[:100]})
        else:
            missing.append({"id": item["id"], "question": item["question"], "why": item["why"]})

    ready = len(missing) == 0
    return json.dumps({
        "ready": ready,
        "covered": covered,
        "missing": missing,
        "recommendation": (
            "All required context gathered. Proceed with workflow."
            if ready else
            f"Missing {len(missing)} required item(s). Ask the user first."
        ),
    }, indent=2, default=str)


# ─────────────────────────────────────────────────────────────
# Compound Query Tools (Context Compaction)
# ─────────────────────────────────────────────────────────────


@mcp.tool()
def get_cross_layer_trace(start: str, end: str, include_source: bool = False) -> str:
    """Trace the full connection path between two nodes, with optional source code.

    Combines find_path + query_node + get_source_snippet into a single compacted
    response. Use this to understand how two components are connected across layers.

    Args:
        start: Starting node (fixture, class, method, or file name)
        end: Ending node name
        include_source: If True, include source snippets for each hop (default False)
    """
    g = _ensure_graph()

    # Find the path
    path_result = g.shortest_path(start, end)
    if "error" in path_result:
        return json.dumps(path_result, indent=2, default=str)

    path = path_result.get("path", [])
    if not path:
        return json.dumps({"error": "Empty path found"}, indent=2)

    # Enrich each hop with detailed node info
    enriched_hops = []
    for i, hop in enumerate(path):
        hop_hash = hop.get("hash", "")
        node = g.nodes_dict.get(hop_hash, {})

        enriched = {
            "step": i + 1,
            "name": hop.get("name", ""),
            "type": hop.get("type", ""),
            "fqn": hop.get("fqn", ""),
            "file": hop.get("file", ""),
            "line": node.get("line", 0),
            "purpose": node.get("purpose", "")[:100],
        }

        # Add edge info
        if "edge_to_next" in hop:
            enriched["edge_to_next"] = hop["edge_to_next"]

        # Fixture-specific info
        if node.get("type") == "fixture":
            enriched["scope"] = node.get("scope", "")
            enriched["instantiates"] = node.get("instantiates", {})
            enriched["depends_on_count"] = len(node.get("depends_on", []))

        # Class-specific info
        if node.get("type") == "class":
            enriched["bases"] = node.get("bases", [])
            enriched["is_abstract"] = node.get("is_abstract", False)
            enriched["method_count"] = len(node.get("children", []))

        # Method-specific info
        if node.get("type") == "method":
            enriched["class_name"] = node.get("class_name", "")
            for tag in ("is_setup", "is_teardown", "is_lifecycle",
                        "is_hook", "is_context_manager"):
                if node.get(tag):
                    enriched[tag] = True

        # Include source snippet if requested
        if include_source and node.get("line", 0) > 0:
            rel_path = node.get("file", node.get("path", node.get("defined_in", "")))
            if rel_path:
                src_root = g.raw_data.get("meta", {}).get("generated_from", "")
                abs_path = os.path.join(src_root, rel_path)
                if os.path.isfile(abs_path):
                    try:
                        with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
                            lines = f.readlines()
                        target = node.get("line", 1) - 1
                        s = max(0, target - 3)
                        e = min(len(lines), target + 15)
                        enriched["source_snippet"] = "".join(lines[s:e])
                        enriched["snippet_range"] = f"{s + 1}-{e}"
                    except Exception:
                        pass

        enriched_hops.append(enriched)

    return json.dumps({
        "start": start,
        "end": end,
        "path_length": len(enriched_hops),
        "summary": " \u2192 ".join(
            f"{h['name']}({h['type']})"
            for h in enriched_hops
        ),
        "hops": enriched_hops,
    }, indent=2, default=str)


@mcp.tool()
def get_change_context(component: str, change_type: str = "bug_fix") -> str:
    """Get comprehensive change context for a component — combines dependencies,
    impact analysis, fixture chain, and test examples into one compacted response.

    Use this INSTEAD of calling find_dependencies + find_impact + get_fixture_chain
    separately. Returns everything needed to plan a change.

    Args:
        component: The component to analyze (class, fixture, method, or file name)
        change_type: 'bug_fix', 'feature_add', 'refactor' (adjusts depth/detail)
    """
    g = _ensure_graph()

    node_id = g._find_node(component)
    if not node_id:
        return json.dumps({
            "error": f"Component '{component}' not found",
            "suggestions": g._suggest_nodes(component),
        }, indent=2)

    node = g.nodes_dict.get(node_id, {})
    node_type = node.get("type", "")

    # Adjust depth based on change type
    impact_depth = {"bug_fix": 2, "feature_add": 3, "refactor": 4}.get(change_type, 2)
    dep_depth = {"bug_fix": 3, "feature_add": 5, "refactor": 6}.get(change_type, 3)

    result = {
        "component": g._node_summary(node_id),
        "change_type": change_type,
    }

    # Full node details
    result["component"]["file"] = node.get("file", node.get("path", node.get("defined_in", "")))
    result["component"]["line"] = node.get("line", 0)
    if node.get("purpose"):
        result["component"]["purpose"] = node["purpose"][:200]
    if node.get("bases"):
        result["component"]["bases"] = node["bases"]
    if node.get("instantiates"):
        result["component"]["instantiates"] = node["instantiates"]

    # 1. Dependencies (what does this component depend on?)
    deps = g.dfs_dependencies(component, dep_depth)
    if "tree" in deps:
        dep_list = []
        for dep_hash, children in deps["tree"].items():
            if dep_hash == node_id:
                continue
            dep_list.extend(children)
        result["dependencies"] = {
            "total": deps.get("total_dependencies", 0),
            "items": dep_list[:30],
        }

    # 2. Impact (what depends on this component?)
    impact = g.bfs_impact(component, impact_depth)
    if "by_depth" in impact:
        result["impact"] = {
            "total_impacted": impact.get("total_impacted", 0),
            "by_depth": impact["by_depth"],
        }

    # 3. Fixture chain (if component is a fixture or related to fixtures)
    if node_type == "fixture":
        chain = g.get_fixture_chain(component)
        if "chain" in chain:
            result["fixture_chain"] = chain["chain"]
    elif node_type == "class":
        # Check if any fixtures expose this class
        efx = node.get("exposed_by_fixtures", [])
        if efx:
            result["exposed_by_fixtures"] = [
                g._node_summary(h) for h in efx
                if h in g.nodes_dict
            ]

    # 4. Related tests (find test files that reference this component)
    test_refs = []
    component_name_lower = node.get("name", "").lower()
    for h, n in g.nodes_dict.items():
        fqn = n.get("fqn", "").lower()
        if ("test" in fqn and
                component_name_lower in fqn and
                n.get("type") in ("file", "class", "function")):
            test_refs.append(g._node_summary(h))
            if len(test_refs) >= 10:
                break
    if test_refs:
        result["related_tests"] = test_refs

    # 5. Lifecycle methods (if class, show setup/teardown methods)
    if node_type == "class":
        lifecycle = []
        for child_hash in node.get("children", []):
            child = g.nodes_dict.get(child_hash, {})
            if child.get("is_lifecycle") or child.get("is_setup") or child.get("is_teardown"):
                lifecycle.append({
                    "name": child.get("name", ""),
                    "line": child.get("line", 0),
                    **{k: True for k in ("is_setup", "is_teardown", "is_lifecycle",
                                          "is_hook", "is_context_manager")
                       if child.get(k)},
                })
        if lifecycle:
            result["lifecycle_methods"] = lifecycle

    # 6. Change recipe hint
    if change_type == "bug_fix":
        result["recipe_hint"] = "fix_domain_bug"
    elif node_type == "fixture":
        result["recipe_hint"] = "add_fixture"
    elif node_type == "class" and node.get("is_abstract"):
        result["recipe_hint"] = "add_interface"

    return json.dumps(result, indent=2, default=str)


@mcp.tool()
def get_layer_interfaces(layer1: str, layer2: str) -> str:
    """Find all connection points between two framework layers.

    Returns nodes where layer1 and layer2 are connected via edges. This is
    essential for understanding cross-layer interactions without reading
    every file in both layers.

    Args:
        layer1: First layer — package path fragment (e.g. 'controllers', 'services',
                'models', 'repositories', 'api', 'utils')
        layer2: Second layer — package path fragment
    """
    g = _ensure_graph()

    # Collect nodes belonging to each layer
    layer1_lower = layer1.lower()
    layer2_lower = layer2.lower()

    layer1_nodes = set()
    layer2_nodes = set()
    for h, node in g.nodes_dict.items():
        searchable = " ".join([
            node.get("fqn", ""),
            node.get("file", ""),
            node.get("path", ""),
            node.get("defined_in", ""),
        ]).lower()
        if layer1_lower in searchable:
            layer1_nodes.add(h)
        if layer2_lower in searchable:
            layer2_nodes.add(h)

    if not layer1_nodes:
        return json.dumps({"error": f"No nodes found for layer '{layer1}'"}, indent=2)
    if not layer2_nodes:
        return json.dumps({"error": f"No nodes found for layer '{layer2}'"}, indent=2)

    # Find edges between layer1 and layer2
    connections = []
    seen_pairs = set()
    for h1 in layer1_nodes:
        for neighbor in g.G.neighbors(h1):
            if neighbor in layer2_nodes:
                pair = (h1, neighbor)
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    edge_data = g.G.edges[h1, neighbor]
                    connections.append({
                        "from": g._node_summary(h1),
                        "to": g._node_summary(neighbor),
                        "edge_type": edge_data.get("edge_type", "unknown"),
                    })

    # Also check reverse direction
    for h2 in layer2_nodes:
        for neighbor in g.G.neighbors(h2):
            if neighbor in layer1_nodes:
                pair = (h2, neighbor)
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    edge_data = g.G.edges[h2, neighbor]
                    connections.append({
                        "from": g._node_summary(h2),
                        "to": g._node_summary(neighbor),
                        "edge_type": edge_data.get("edge_type", "unknown"),
                    })

    # Group by edge type for readability
    by_edge_type = {}
    for conn in connections:
        et = conn["edge_type"]
        if et not in by_edge_type:
            by_edge_type[et] = []
        by_edge_type[et].append(conn)

    return json.dumps({
        "layer1": layer1,
        "layer2": layer2,
        "layer1_node_count": len(layer1_nodes),
        "layer2_node_count": len(layer2_nodes),
        "total_connections": len(connections),
        "connections_by_type": {
            et: {"count": len(conns), "connections": conns[:20]}
            for et, conns in by_edge_type.items()
        },
    }, indent=2, default=str)


if __name__ == "__main__":
    mcp.run()
