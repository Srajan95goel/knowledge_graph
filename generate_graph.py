"""
Knowledge Graph Generator — Produces a flat, hash-keyed, bidirectional graph YAML.

Key features:
  - Flat node structure (all nodes at top level, linked by hash pointers)
  - Deterministic FQN-based hashes (collision-free)
  - Bidirectional links everywhere (parent/children, imports/imported_by, etc.)
  - Fixture → Class "instantiates" links
  - Class attributes extracted from __init__
  - Methods as first-class nodes in YAML
  - Index section for O(1) name-to-hash lookups

Usage:
    python generate_graph_v2.py [--src PATH] [--output PATH]
"""

import ast
import hashlib
import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass, field
from collections import defaultdict

try:
    import yaml
except ImportError:
    print("PyYAML required: pip install pyyaml")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Hash utility
# ─────────────────────────────────────────────────────────────

def make_hash(fqn: str) -> str:
    """Generate a deterministic 12-char hash from a fully qualified name."""
    return hashlib.sha256(fqn.encode("utf-8")).hexdigest()[:12]


# ─────────────────────────────────────────────────────────────
# Known ABCs for detecting abstract classes
# ─────────────────────────────────────────────────────────────

KNOWN_ABCS = {
    "ABC", "ABCMeta",
}

SKIP_FIXTURE_PARAMS = {
    "self", "request", "tmp_path", "tmpdir", "capfd",
    "capsys", "caplog", "monkeypatch", "pytestconfig",
    "record_property", "record_testsuite_property",
}

SKIP_CALLEES = {
    "len", "str", "int", "float", "bool", "list", "dict", "set", "tuple",
    "type", "isinstance", "issubclass", "hasattr", "getattr", "setattr",
    "print", "range", "enumerate", "zip", "map", "filter", "sorted",
    "super", "property", "staticmethod", "classmethod", "format", "repr",
    "hex", "bin", "oct", "abs", "min", "max", "sum", "round", "any", "all",
    "open", "os.path.join", "os.path.exists", "os.path.dirname",
}

SKIP_CALL_TARGETS = {"logger", "logging", "log", "self._log", "self.logger", "self._logger"}


# ─────────────────────────────────────────────────────────────
# AST Parser & Graph Builder
# ─────────────────────────────────────────────────────────────

class CodebaseGraphBuilder:
    """Parses Python source and builds flat, hash-keyed, bidirectional graph."""

    def __init__(self, src_root: str):
        self.src_root = Path(src_root)
        # All nodes keyed by hash
        self.nodes: Dict[str, Dict[str, Any]] = {}
        # Indexes for lookups
        self.fqn_to_hash: Dict[str, str] = {}
        self.name_to_hashes: Dict[str, List[str]] = defaultdict(list)
        self.type_to_hashes: Dict[str, List[str]] = defaultdict(list)
        self.file_to_hash: Dict[str, str] = {}
        # Fixture name → hash (for resolving fixture deps)
        self.fixture_name_to_hash: Dict[str, str] = {}
        # Class name → list of hashes (to detect collisions & resolve inheritance)
        self.class_name_to_hashes: Dict[str, List[str]] = defaultdict(list)
        # File path → list of class hashes in that file
        self.file_classes: Dict[str, List[str]] = defaultdict(list)
        # Stats
        self.stats = defaultdict(int)

    # ── Node creation helpers ──────────────────────────────

    def _add_node(self, fqn: str, node_type: str, name: str, **kwargs) -> str:
        """Create a node with deterministic hash. Returns the hash."""
        h = make_hash(fqn)
        node = {
            "type": node_type,
            "name": name,
            "fqn": fqn,
            "hash": h,
            **kwargs,
        }
        self.nodes[h] = node
        self.fqn_to_hash[fqn] = h
        self.name_to_hashes[name].append(h)
        self.type_to_hashes[node_type].append(h)
        self.stats[f"total_{node_type}s"] += 1
        return h

    def _add_child(self, parent_hash: str, child_hash: str):
        """Register parent↔child bidirectional link."""
        if parent_hash in self.nodes:
            self.nodes[parent_hash].setdefault("children", []).append(child_hash)
        if child_hash in self.nodes:
            self.nodes[child_hash]["parent"] = parent_hash

    def _add_edge(self, from_hash: str, to_hash: str, edge_type: str, reverse_type: str):
        """Add a named edge in both directions on the source and target nodes."""
        if from_hash in self.nodes:
            self.nodes[from_hash].setdefault(edge_type, []).append(to_hash)
        if to_hash in self.nodes:
            self.nodes[to_hash].setdefault(reverse_type, []).append(to_hash if reverse_type == edge_type else from_hash)

    # ── Main parse entry point ─────────────────────────────

    def parse_all(self):
        """Parse all Python files and build the node graph."""
        py_files = list(self.src_root.rglob("*.py"))
        logger.info(f"Found {len(py_files)} Python files to parse")

        # Phase 1: Create package nodes for directory structure
        self._build_package_tree()

        # Phase 2: Parse each file (creates file, class, method, fixture, function nodes)
        for py_file in py_files:
            try:
                self._parse_file(py_file)
            except SyntaxError as e:
                logger.warning(f"Syntax error in {py_file}: {e}")
            except Exception as e:
                logger.warning(f"Error parsing {py_file}: {e}")

        # Phase 3: Resolve cross-references (imports, inheritance, fixture deps, instantiates)
        self._resolve_imports()
        self._resolve_inheritance()
        self._resolve_fixture_dependencies()
        self._resolve_fixture_instantiates()

        # Phase 4: Deep resolution (requires Phase 3 data)
        #   - self.method() calls deferred from parsing
        #   - self._attr.method() cross-file calls
        self.resolve_self_calls()
        self.resolve_cross_file_calls()

        logger.info(
            f"Built: {len(self.nodes)} nodes "
            f"({self.stats.get('total_packages', 0)} packages, "
            f"{self.stats.get('total_files', 0)} files, "
            f"{self.stats.get('total_classs', 0)} classes, "
            f"{self.stats.get('total_methods', 0)} methods, "
            f"{self.stats.get('total_fixtures', 0)} fixtures, "
            f"{self.stats.get('total_functions', 0)} functions)"
        )

    # ── Phase 1: Package tree ──────────────────────────────

    def _build_package_tree(self):
        """Build package nodes from directory structure."""
        # Root node — use the source directory name as root package
        root_name = self.src_root.name
        root_fqn = root_name
        root_hash = self._add_node(root_fqn, "package", root_name,
                                    purpose="Source root package")

        seen_packages = {root_fqn: root_hash}

        for py_file in self.src_root.rglob("*.py"):
            rel = str(py_file.relative_to(self.src_root)).replace("\\", "/")
            parts = rel.split("/")

            # Build package chain from directory hierarchy
            for i in range(1, len(parts)):  # skip the filename itself
                pkg_fqn = "/".join(parts[:i])
                if pkg_fqn in seen_packages:
                    continue

                pkg_name = parts[i - 1]
                parent_fqn = "/".join(parts[:i - 1]) if i > 1 else root_fqn

                pkg_hash = self._add_node(pkg_fqn, "package", pkg_name)
                seen_packages[pkg_fqn] = pkg_hash

                # Link parent → child
                parent_hash = seen_packages.get(parent_fqn)
                if parent_hash:
                    self._add_child(parent_hash, pkg_hash)

    # ── Phase 2: File parsing ──────────────────────────────

    def _parse_file(self, file_path: Path):
        """Parse a single Python file, creating file + class + method + fixture nodes."""
        rel_path = str(file_path.relative_to(self.src_root)).replace("\\", "/")

        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            source = f.read()

        try:
            tree = ast.parse(source, filename=str(file_path))
        except SyntaxError:
            return

        # Create file node
        file_fqn = rel_path
        file_hash = self._add_node(file_fqn, "file", Path(rel_path).name,
                                    path=rel_path)
        self.file_to_hash[rel_path] = file_hash

        # Add docstring
        docstring = ast.get_docstring(tree)
        if docstring:
            self.nodes[file_hash]["purpose"] = docstring[:200]

        # Link to parent package
        parent_pkg_fqn = "/".join(rel_path.split("/")[:-1])
        parent_pkg_hash = self.fqn_to_hash.get(parent_pkg_fqn)
        if parent_pkg_hash:
            self._add_child(parent_pkg_hash, file_hash)

        # Collect raw imports (resolve later in Phase 3)
        raw_imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                raw_imports.append(node.module)
        self.nodes[file_hash]["_raw_imports"] = raw_imports

        # Parse top-level classes and functions
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef):
                self._parse_class(node, rel_path, file_hash)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._parse_function(node, rel_path, file_hash)

    def _parse_class(self, node: ast.ClassDef, file_path: str, file_hash: str):
        """Parse a class and create class + method nodes."""
        class_fqn = f"{file_path}::{node.name}"
        class_hash = self._add_node(
            class_fqn, "class", node.name,
            file=file_path,
            line=node.lineno,
        )

        # Track class name → hash for resolution
        self.class_name_to_hashes[node.name].append(class_hash)
        self.file_classes[file_path].append(class_hash)

        # Link file → class (parent/child)
        self._add_child(file_hash, class_hash)

        # Bases (store raw names, resolve hashes in Phase 3)
        bases = []
        for base in node.bases:
            base_name = self._get_name(base)
            if base_name:
                bases.append(base_name)
        self.nodes[class_hash]["bases"] = bases

        # Abstract?
        is_abstract = any(b in KNOWN_ABCS for b in bases)
        self.nodes[class_hash]["is_abstract"] = is_abstract

        # Decorators
        decorators = [self._get_name(d) for d in node.decorator_list if self._get_name(d)]
        if decorators:
            self.nodes[class_hash]["decorators"] = decorators

        # Docstring / purpose
        docstring = ast.get_docstring(node)
        if docstring:
            self.nodes[class_hash]["purpose"] = docstring[:200]

        # Extract __init__ attributes (self.X = ...)
        attributes = self._extract_class_attributes(node)
        if attributes:
            self.nodes[class_hash]["attributes"] = attributes

        # Parse methods
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._parse_method(item, node.name, file_path, class_hash)

    def _parse_method(self, node: ast.FunctionDef, class_name: str,
                      file_path: str, class_hash: str):
        """Parse a method and create a method node."""
        method_fqn = f"{file_path}::{class_name}::{node.name}"
        method_hash = self._add_node(
            method_fqn, "method", node.name,
            class_name=class_name,
            file=file_path,
            line=node.lineno,
        )

        # Link class → method (parent/child)
        self._add_child(class_hash, method_hash)

        # Decorators
        decorators = []
        is_fixture = False
        is_property = False
        is_static = False
        is_classmethod = False
        is_abstract = False

        for dec in node.decorator_list:
            dec_name = self._get_name(dec)
            if dec_name:
                decorators.append(dec_name)
                if "property" == dec_name:
                    is_property = True
                elif "staticmethod" == dec_name:
                    is_static = True
                elif "classmethod" == dec_name:
                    is_classmethod = True
                elif "abstractmethod" == dec_name:
                    is_abstract = True

                # Check if this method is a fixture
                dec_str = ast.dump(dec)
                if "pytest" in dec_str and "fixture" in dec_str:
                    is_fixture = True
                    fixture_name = self._extract_fixture_name(dec, node.name)
                    self._create_fixture_from_method(
                        node, fixture_name, file_path, class_name, class_hash
                    )

        if decorators:
            self.nodes[method_hash]["decorators"] = decorators
        if is_property:
            self.nodes[method_hash]["is_property"] = True
        if is_static:
            self.nodes[method_hash]["is_static"] = True
        if is_classmethod:
            self.nodes[method_hash]["is_classmethod"] = True
        if is_abstract:
            self.nodes[method_hash]["is_abstract"] = True

        # ── Semantic lifecycle tags ──
        method_lower = node.name.lower()
        if method_lower.startswith(("setup", "set_up", "_setup")):
            self.nodes[method_hash]["is_setup"] = True
        if method_lower.startswith(("teardown", "tear_down", "_teardown",
                                     "cleanup", "clean_up", "_cleanup",
                                     "finalize", "_finalize",
                                     "close", "_close", "dispose", "_dispose")):
            self.nodes[method_hash]["is_teardown"] = True
        if method_lower.startswith("pytest_"):
            self.nodes[method_hash]["is_hook"] = True
        if method_lower in ("__enter__", "__exit__", "__aenter__", "__aexit__"):
            self.nodes[method_hash]["is_context_manager"] = True
        if any(tag in method_lower for tag in ("setup", "teardown", "cleanup",
                                                "finalize", "init", "close",
                                                "dispose", "startup", "shutdown")):
            self.nodes[method_hash]["is_lifecycle"] = True

        # Extract method calls
        calls = self._extract_method_calls(node, class_name, file_path, method_hash)
        # calls are stored on the method node via _add_edge inside _extract_method_calls

    def _parse_function(self, node: ast.FunctionDef, file_path: str, file_hash: str):
        """Parse a top-level function, creating function and possibly fixture nodes."""
        func_fqn = f"{file_path}::{node.name}"
        func_hash = self._add_node(
            func_fqn, "function", node.name,
            file=file_path,
            line=node.lineno,
        )

        # Link file → function (parent/child)
        self._add_child(file_hash, func_hash)

        # Docstring
        docstring = ast.get_docstring(node)
        if docstring:
            self.nodes[func_hash]["purpose"] = docstring[:200]

        # Check if it's a fixture
        for dec in node.decorator_list:
            dec_str = ast.dump(dec)
            if "pytest" in dec_str and "fixture" in dec_str:
                fixture_name = self._extract_fixture_name(dec, node.name)
                self._create_fixture_from_function(node, fixture_name, file_path, file_hash)
                break

    def _create_fixture_from_function(self, node: ast.FunctionDef,
                                       fixture_name: str, file_path: str,
                                       file_hash: str):
        """Create a fixture node from a standalone @pytest.fixture function."""
        fixture_fqn = f"{file_path}::fixture::{fixture_name}"
        fixture_hash = self._add_node(
            fixture_fqn, "fixture", fixture_name,
            function_name=node.name,
            defined_in=file_path,
            line=node.lineno,
        )

        # Link file → fixture (parent/child)
        self._add_child(file_hash, fixture_hash)

        # Store fixture name → hash for dependency resolution
        self.fixture_name_to_hash[fixture_name] = fixture_hash

        # Extract scope from decorator
        scope = self._extract_fixture_scope(node)
        if scope:
            self.nodes[fixture_hash]["scope"] = scope

        # Store raw dependency names (resolve hashes in Phase 3)
        deps = []
        params_full = []
        for arg in node.args.args:
            if arg.arg not in SKIP_FIXTURE_PARAMS:
                deps.append(arg.arg)
                params_full.append({"name": arg.arg, "is_fixture_dep": True})
            elif arg.arg != "self":
                params_full.append({"name": arg.arg, "is_fixture_dep": False})
        self.nodes[fixture_hash]["_raw_deps"] = deps
        if params_full:
            self.nodes[fixture_hash]["parameters"] = params_full

        # Store raw return/yield analysis for instantiates resolution
        return_info = self._extract_return_class(node)
        if return_info:
            self.nodes[fixture_hash]["_raw_instantiates"] = return_info

        # Docstring
        docstring = ast.get_docstring(node)
        if docstring:
            self.nodes[fixture_hash]["purpose"] = docstring[:200]

    def _create_fixture_from_method(self, node: ast.FunctionDef,
                                     fixture_name: str, file_path: str,
                                     class_name: str, class_hash: str):
        """Create a fixture node from a class method decorated with @pytest.fixture."""
        fixture_fqn = f"{file_path}::{class_name}::fixture::{fixture_name}"
        fixture_hash = self._add_node(
            fixture_fqn, "fixture", fixture_name,
            function_name=node.name,
            defined_in=file_path,
            class_name=class_name,
            line=node.lineno,
        )

        # Link class → fixture
        self.nodes[class_hash].setdefault("fixtures_provided", []).append(fixture_hash)
        self.nodes[fixture_hash]["provided_by_class"] = class_hash

        # Store fixture name → hash
        self.fixture_name_to_hash[fixture_name] = fixture_hash

        # Scope
        scope = self._extract_fixture_scope(node)
        if scope:
            self.nodes[fixture_hash]["scope"] = scope

        # Raw deps with full parameter signatures
        deps = []
        params_full = []
        for arg in node.args.args:
            if arg.arg not in SKIP_FIXTURE_PARAMS:
                deps.append(arg.arg)
                params_full.append({"name": arg.arg, "is_fixture_dep": True})
            elif arg.arg != "self":
                params_full.append({"name": arg.arg, "is_fixture_dep": False})
        self.nodes[fixture_hash]["_raw_deps"] = deps
        if params_full:
            self.nodes[fixture_hash]["parameters"] = params_full

        # Return class
        return_info = self._extract_return_class(node)
        if return_info:
            self.nodes[fixture_hash]["_raw_instantiates"] = return_info

    # ── AST extraction helpers ─────────────────────────────

    def _extract_fixture_name(self, decorator_node, func_name: str) -> str:
        """Extract the fixture name from @pytest.fixture(name='xxx')."""
        if isinstance(decorator_node, ast.Call):
            for kw in decorator_node.keywords:
                if kw.arg == "name" and isinstance(kw.value, ast.Constant):
                    return kw.value.value
        if func_name.startswith("fixture_"):
            return func_name[8:]
        return func_name

    def _extract_fixture_scope(self, node: ast.FunctionDef) -> Optional[str]:
        """Extract fixture scope from @pytest.fixture(scope='session')."""
        for dec in node.decorator_list:
            if isinstance(dec, ast.Call):
                for kw in dec.keywords:
                    if kw.arg == "scope" and isinstance(kw.value, ast.Constant):
                        return kw.value.value
        return None

    def _extract_class_attributes(self, class_node: ast.ClassDef) -> List[Dict]:
        """Extract instance attributes from __init__ method (self.X = ...)."""
        attributes = []
        for item in class_node.body:
            if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                for stmt in ast.walk(item):
                    if isinstance(stmt, ast.Assign):
                        for target in stmt.targets:
                            if (isinstance(target, ast.Attribute) and
                                isinstance(target.value, ast.Name) and
                                target.value.id == "self"):
                                attr_name = target.attr
                                # Try to get type hint from annotation or value
                                type_hint = self._infer_type(stmt.value)
                                attr_info = {"name": attr_name}
                                if type_hint:
                                    attr_info["type_hint"] = type_hint
                                attributes.append(attr_info)
                    elif isinstance(stmt, ast.AnnAssign):
                        if (isinstance(stmt.target, ast.Attribute) and
                            isinstance(stmt.target.value, ast.Name) and
                            stmt.target.value.id == "self"):
                            attr_name = stmt.target.attr
                            type_hint = self._get_name(stmt.annotation) if stmt.annotation else None
                            attr_info = {"name": attr_name}
                            if type_hint:
                                attr_info["type_hint"] = type_hint
                            attributes.append(attr_info)
                break  # only process __init__

        # Deduplicate
        seen = set()
        unique = []
        for a in attributes:
            if a["name"] not in seen:
                seen.add(a["name"])
                unique.append(a)
        return unique

    def _infer_type(self, value_node) -> Optional[str]:
        """Try to infer type from a value expression in __init__."""
        if isinstance(value_node, ast.Call):
            return self._get_name(value_node.func)
        if isinstance(value_node, ast.Name):
            name = value_node.id
            if name[0].isupper():  # likely a class
                return name
        return None

    def _extract_return_class(self, func_node: ast.FunctionDef) -> Optional[str]:
        """Extract the class being instantiated from return/yield statements.

        Handles:
          - return ClassName(...)       → direct class instantiation
          - yield ClassName(...)        → direct class instantiation
          - return initialize_xxx(...)  → stored as initializer hint
          - return self._attr           → attribute-based return
          - yield self._attr            → attribute-based return
          - return factory.create(...)  → factory method hint
          - _var = initialize_xxx(...); yield _var  → local variable tracing
          - _var = ClassName(...); return _var      → local variable tracing
        """
        # Phase 1: Build a map of local variable assignments
        # Tracks: var_name → what was assigned (class name or initializer hint)
        local_var_map = {}
        for stmt in ast.walk(func_node):
            if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
                target = stmt.targets[0]
                if isinstance(target, ast.Name) and isinstance(stmt.value, ast.Call):
                    callee = self._get_name(stmt.value.func)
                    if not callee:
                        continue
                    # Check factory pattern BEFORE class name (handles FactoryClass.create())
                    if "." in callee:
                        parts = callee.split(".")
                        if any(p in ("create", "build", "make", "get_instance")
                               for p in parts):
                            local_var_map[target.id] = f"factory::{callee}"
                            continue
                    if callee.startswith("initialize_"):
                        local_var_map[target.id] = f"initializer::{callee}"
                    elif callee[0].isupper():
                        local_var_map[target.id] = callee

        # Phase 2: Check return/yield statements
        for stmt in ast.walk(func_node):
            if isinstance(stmt, (ast.Return, ast.Yield)):
                if not stmt.value:
                    continue
                # Direct class call: return ClassName(...)
                if isinstance(stmt.value, ast.Call):
                    callee = self._get_name(stmt.value.func)
                    if callee and callee[0].isupper():
                        return callee  # Direct class instantiation
                    if callee and callee.startswith("initialize_"):
                        return f"initializer::{callee}"
                    # Factory pattern: return factory.create(...)
                    if callee and "." in callee:
                        parts = callee.split(".")
                        if any(p in ("create", "build", "make", "get_instance")
                               for p in parts):
                            return f"factory::{callee}"
                # Attribute return: return self._xxx
                if isinstance(stmt.value, ast.Attribute):
                    if (isinstance(stmt.value.value, ast.Name) and
                            stmt.value.value.id == "self"):
                        return f"self_attr::{stmt.value.attr}"
                # Local variable return: yield _var or return _var
                if isinstance(stmt.value, ast.Name):
                    var_name = stmt.value.id
                    # Check if this variable was assigned a class or initializer
                    if var_name in local_var_map:
                        return local_var_map[var_name]
                    # Could be a class reference if uppercase
                    if var_name[0].isupper():
                        return var_name
        return None

    def _extract_method_calls(self, method_node: ast.FunctionDef,
                               class_name: str, file_path: str,
                               method_hash: str) -> List[Dict]:
        """Extract calls from method body, add edges on method node."""
        calls = []
        caller_method = method_node.name

        # Skip dunder methods except __init__
        if caller_method.startswith("__") and caller_method != "__init__":
            return calls

        seen = set()
        for node in ast.walk(method_node):
            if not isinstance(node, ast.Call):
                continue

            func = node.func
            callee_target = ""
            callee_method = ""

            if isinstance(func, ast.Attribute):
                callee_method = func.attr
                callee_target = self._get_name(func.value) or ""
            elif isinstance(func, ast.Name):
                callee_method = func.id
            else:
                continue

            # Skip noise
            if callee_method in SKIP_CALLEES:
                continue
            if callee_target in SKIP_CALL_TARGETS:
                continue
            if callee_method.startswith("__"):
                continue
            if callee_method.startswith("assert") or callee_method.startswith("pytest"):
                continue

            dedup_key = (callee_target, callee_method)
            if dedup_key in seen:
                continue
            seen.add(dedup_key)

            call_info = {
                "target": callee_target,
                "method": callee_method,
                "line": getattr(node, "lineno", 0),
            }

            # Try to resolve self calls to method hashes
            if callee_target == "self":
                target_fqn = f"{file_path}::{class_name}::{callee_method}"
                target_hash = self.fqn_to_hash.get(target_fqn)
                if target_hash:
                    call_info["resolved_hash"] = target_hash
                    # Add calls/called_by edges
                    self.nodes[method_hash].setdefault("calls", []).append(target_hash)
                    self.nodes[target_hash].setdefault("called_by", []).append(method_hash)
                else:
                    # Target method might not be parsed yet — store raw for later
                    call_info["_unresolved_self_call"] = callee_method

            calls.append(call_info)

        if calls:
            self.nodes[method_hash]["calls_raw"] = calls

        return calls

    def _get_name(self, node) -> Optional[str]:
        """Extract name from AST node."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            value = self._get_name(node.value)
            if value:
                return f"{value}.{node.attr}"
            return node.attr
        elif isinstance(node, ast.Constant):
            return str(node.value)
        elif isinstance(node, ast.Call):
            return self._get_name(node.func)
        return None

    # ── Phase 3: Cross-reference resolution ────────────────

    def _resolve_imports(self):
        """Resolve raw import strings to file hashes, add bidirectional edges."""
        for h, node in list(self.nodes.items()):
            if node.get("type") != "file":
                continue

            raw_imports = node.pop("_raw_imports", [])
            imports = []
            for module_str in raw_imports:
                # Convert module path to file path: pkg.sub.mod → pkg/sub/mod.py
                # But also check pkg/sub/mod/__init__.py
                target_path = module_str.replace(".", "/") + ".py"
                target_hash = self.file_to_hash.get(target_path)

                if not target_hash:
                    # Try as package __init__.py
                    target_path_init = module_str.replace(".", "/") + "/__init__.py"
                    target_hash = self.file_to_hash.get(target_path_init)

                if target_hash:
                    imports.append(target_hash)
                    # Bidirectional
                    self.nodes[target_hash].setdefault("imported_by", []).append(h)

            if imports:
                node["imports"] = imports

    def _resolve_inheritance(self):
        """Resolve base class names to hashes, add bidirectional edges."""
        for h, node in list(self.nodes.items()):
            if node.get("type") != "class":
                continue

            raw_bases = node.get("bases", [])
            resolved_bases = []
            for base_name in raw_bases:
                # Strip module path if present (e.g., "module.ClassName" → "ClassName")
                simple_name = base_name.split(".")[-1]
                target_hashes = self.class_name_to_hashes.get(simple_name, [])

                if len(target_hashes) == 1:
                    target_hash = target_hashes[0]
                    resolved_bases.append({
                        "hash": target_hash,
                        "name": simple_name,
                    })
                    # Bidirectional
                    self.nodes[target_hash].setdefault("subclasses", []).append(h)
                elif len(target_hashes) > 1:
                    # Multiple matches — try to resolve via imports in same file
                    file_path = node.get("file", "")
                    file_hash = self.file_to_hash.get(file_path)
                    best = self._pick_best_base(file_hash, target_hashes, simple_name)
                    if best:
                        resolved_bases.append({
                            "hash": best,
                            "name": simple_name,
                        })
                        self.nodes[best].setdefault("subclasses", []).append(h)
                    else:
                        resolved_bases.append({"name": simple_name, "unresolved": True})
                else:
                    # External base (e.g., ABC, Enum) — store name only
                    resolved_bases.append({"name": simple_name, "external": True})

            node["bases_resolved"] = resolved_bases

    def _pick_best_base(self, file_hash: Optional[str], candidates: List[str],
                        name: str) -> Optional[str]:
        """When multiple classes match a base name, pick the one from an imported file."""
        if not file_hash:
            return candidates[0] if candidates else None

        imported_files = set(self.nodes.get(file_hash, {}).get("imports", []))
        for candidate_hash in candidates:
            candidate_file = self.nodes[candidate_hash].get("file", "")
            candidate_file_hash = self.file_to_hash.get(candidate_file)
            if candidate_file_hash and candidate_file_hash in imported_files:
                return candidate_hash

        return candidates[0] if candidates else None

    def _resolve_fixture_dependencies(self):
        """Resolve fixture raw dependency names to fixture hashes."""
        for h, node in list(self.nodes.items()):
            if node.get("type") != "fixture":
                continue

            raw_deps = node.pop("_raw_deps", [])
            depends_on = []
            for dep_name in raw_deps:
                dep_hash = self.fixture_name_to_hash.get(dep_name)
                if dep_hash:
                    depends_on.append(dep_hash)
                    # Bidirectional
                    self.nodes[dep_hash].setdefault("depended_on_by", []).append(h)
                else:
                    # Unknown fixture — store name for debugging
                    node.setdefault("_unresolved_deps", []).append(dep_name)

            if depends_on:
                node["depends_on"] = depends_on

    def _resolve_fixture_instantiates(self):
        """Resolve fixture return type to class hash (instantiates link)."""
        for h, node in list(self.nodes.items()):
            if node.get("type") != "fixture":
                continue

            raw_inst = node.pop("_raw_instantiates", None)
            if not raw_inst:
                continue

            if raw_inst.startswith("initializer::"):
                # Follow the initializer function to find what class it creates
                init_name = raw_inst[len("initializer::"):]
                resolved = self._resolve_initializer(init_name)
                if resolved:
                    node["instantiates"] = {"hash": resolved, "via": init_name,
                                             "name": self.nodes.get(resolved, {}).get("name", "")}
                    self.nodes[resolved].setdefault("exposed_by_fixtures", []).append(h)
                else:
                    node["instantiates"] = {"unresolved": init_name}

            elif raw_inst.startswith("self_attr::"):
                # Attribute-based return: self._xxx — resolve via class attribute type
                attr_name = raw_inst[len("self_attr::")
                                     :]
                resolved = self._resolve_fixture_attr_instantiates(h, node, attr_name)
                if resolved:
                    node["instantiates"] = resolved
                else:
                    node["instantiates"] = {"unresolved_attr": attr_name}

            elif raw_inst.startswith("factory::"):
                # Factory method: factory.create() — try to trace through factory
                factory_call = raw_inst[len("factory::")
                                        :]
                resolved = self._resolve_factory_instantiates(factory_call)
                if resolved:
                    node["instantiates"] = resolved
                    target_hash = resolved.get("hash")
                    if target_hash:
                        self.nodes[target_hash].setdefault("exposed_by_fixtures", []).append(h)
                else:
                    node["instantiates"] = {"unresolved_factory": factory_call}

            else:
                # Direct class name
                class_hashes = self.class_name_to_hashes.get(raw_inst, [])
                if len(class_hashes) == 1:
                    node["instantiates"] = {"hash": class_hashes[0], "name": raw_inst}
                    self.nodes[class_hashes[0]].setdefault("exposed_by_fixtures", []).append(h)
                elif class_hashes:
                    node["instantiates"] = {"hash": class_hashes[0], "name": raw_inst,
                                             "ambiguous": len(class_hashes)}
                    self.nodes[class_hashes[0]].setdefault("exposed_by_fixtures", []).append(h)
                else:
                    node["instantiates"] = {"unresolved": raw_inst}

    def _resolve_fixture_attr_instantiates(self, fixture_hash: str,
                                            fixture_node: dict,
                                            attr_name: str) -> Optional[dict]:
        """Resolve fixture that returns self._attr by tracing attribute type.

        Chain: fixture returns self._attr → __init__ sets self._attr = initialize_xxx()
               → initialize_xxx returns ClassName() → resolve ClassName to hash.
        """
        # Find the parent class of this fixture
        class_hash = fixture_node.get("provided_by_class")
        if not class_hash:
            return None

        class_node = self.nodes.get(class_hash, {})
        if not class_node:
            return None

        # Look up the attribute in the class's attribute list
        for attr in class_node.get("attributes", []):
            if attr.get("name") == attr_name:
                type_hint = attr.get("type_hint", "")
                if not type_hint:
                    break

                # Case 1: type_hint is a class name (e.g., "ConfigOpts")
                target_hashes = self.class_name_to_hashes.get(type_hint, [])
                if target_hashes:
                    target_hash = target_hashes[0]
                    self.nodes[target_hash].setdefault("exposed_by_fixtures", []).append(fixture_hash)
                    return {"hash": target_hash, "name": type_hint,
                            "via_attr": attr_name}

                # Case 2: type_hint is an initializer function (e.g., "initialize_os")
                if type_hint.startswith("initialize_"):
                    resolved = self._resolve_initializer(type_hint)
                    if resolved:
                        self.nodes[resolved].setdefault("exposed_by_fixtures", []).append(fixture_hash)
                        return {"hash": resolved, "via_attr": attr_name,
                                "via_initializer": type_hint,
                                "name": self.nodes.get(resolved, {}).get("name", "")}

                # Case 3: type_hint is a factory call like "AcFactory"
                if "Factory" in type_hint or "factory" in type_hint:
                    factory_hashes = self.class_name_to_hashes.get(type_hint, [])
                    if factory_hashes:
                        # Try to resolve what the factory creates
                        factory_result = self._trace_factory_create(factory_hashes[0])
                        if factory_result:
                            self.nodes[factory_result].setdefault("exposed_by_fixtures", []).append(fixture_hash)
                            return {"hash": factory_result, "via_attr": attr_name,
                                    "via_factory": type_hint,
                                    "name": self.nodes.get(factory_result, {}).get("name", "")}
                break
        return None

    def _resolve_factory_instantiates(self, factory_call: str) -> Optional[dict]:
        """Resolve factory.method() call to the class it creates."""
        parts = factory_call.split(".")
        if len(parts) < 2:
            return None

        # Try to find the factory attribute type on the class
        # factory_call is like "self._factory.create" → attr is "_factory"
        if parts[0] == "self" and len(parts) >= 3:
            attr_name = parts[1]
            # We'll resolve this in the cross-file call resolution phase
            return None  # Handled by _resolve_fixture_attr_instantiates

        # Direct factory class reference: FactoryClass.create()
        factory_name = parts[0]
        factory_hashes = self.class_name_to_hashes.get(factory_name, [])
        if factory_hashes:
            result = self._trace_factory_create(factory_hashes[0])
            if result:
                return {"hash": result, "via_factory": factory_call,
                        "name": self.nodes.get(result, {}).get("name", "")}
        return None

    def _trace_factory_create(self, factory_hash: str) -> Optional[str]:
        """Trace a factory class's create/build method to find what class it returns."""
        factory_node = self.nodes.get(factory_hash, {})
        factory_file = factory_node.get("file", "")
        factory_name = factory_node.get("name", "")

        # Look for create/build/make methods on this factory
        for method_name in ("create", "build", "make", "get_instance", "__call__"):
            method_fqn = f"{factory_file}::{factory_name}::{method_name}"
            method_hash = self.fqn_to_hash.get(method_fqn)
            if not method_hash:
                continue

            # Read the method source and find return class
            method_node = self.nodes.get(method_hash, {})
            file_path = method_node.get("file", "")
            full_path = self.src_root / file_path.replace("/", os.sep)
            if not full_path.exists():
                continue

            try:
                with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                    source = f.read()
                tree = ast.parse(source)
                for n in ast.walk(tree):
                    if isinstance(n, ast.FunctionDef) and n.name == method_name:
                        cls_name = self._extract_return_class(n)
                        if cls_name and not cls_name.startswith(("initializer::",
                                                                  "self_attr::",
                                                                  "factory::")):
                            class_hashes = self.class_name_to_hashes.get(cls_name, [])
                            if class_hashes:
                                return class_hashes[0]
            except Exception:
                pass
        return None

    def _resolve_initializer(self, init_name: str) -> Optional[str]:
        """Resolve an initializer function to find the class it creates.

        Looks for functions named init_name, parses their return statement.
        Handles: direct class returns, factory.create() chains, and local var tracing.
        """
        for fqn, h in self.fqn_to_hash.items():
            node = self.nodes.get(h, {})
            if node.get("type") == "function" and node.get("name") == init_name:
                # Found the initializer — check if we can read its source
                file_path = node.get("file", "")
                full_path = self.src_root / file_path.replace("/", os.sep)
                if full_path.exists():
                    try:
                        with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                            source = f.read()
                        tree = ast.parse(source)
                        for n in ast.walk(tree):
                            if isinstance(n, ast.FunctionDef) and n.name == init_name:
                                cls_name = self._extract_return_class(n)
                                if not cls_name or cls_name.startswith("initializer::"):
                                    break

                                # Handle factory pattern: "factory::MyFactory.create"
                                if cls_name.startswith("factory::"):
                                    factory_call = cls_name[len("factory::"):]
                                    # Extract factory class name: "MyFactory.create" → "MyFactory"
                                    factory_class = factory_call.split(".")[0]
                                    factory_hashes = self.class_name_to_hashes.get(factory_class, [])
                                    if factory_hashes:
                                        # Try to trace what the factory's create method returns
                                        traced = self._trace_factory_create(factory_hashes[0])
                                        if traced:
                                            return traced
                                        # Fallback: return the factory class itself
                                        return factory_hashes[0]
                                    break

                                # Handle self_attr pattern (shouldn't happen in standalone functions)
                                if cls_name.startswith("self_attr::"):
                                    break

                                # Direct class name
                                class_hashes = self.class_name_to_hashes.get(cls_name, [])
                                if class_hashes:
                                    return class_hashes[0]
                                break
                    except Exception:
                        pass
                break
        return None

    # ── Resolve unresolved self calls ──────────────────────

    def resolve_self_calls(self):
        """Second pass: resolve self.method() calls that weren't resolved during parsing."""
        resolved_count = 0
        for h, node in self.nodes.items():
            if node.get("type") != "method":
                continue

            calls_raw = node.get("calls_raw", [])
            for call in calls_raw:
                unresolved = call.pop("_unresolved_self_call", None)
                if unresolved:
                    class_name = node.get("class_name", "")
                    file_path = node.get("file", "")
                    target_fqn = f"{file_path}::{class_name}::{unresolved}"
                    target_hash = self.fqn_to_hash.get(target_fqn)
                    if target_hash:
                        call["resolved_hash"] = target_hash
                        node.setdefault("calls", []).append(target_hash)
                        self.nodes[target_hash].setdefault("called_by", []).append(h)
                        resolved_count += 1

        if resolved_count:
            logger.info(f"Resolved {resolved_count} deferred self.method() calls")

    def resolve_cross_file_calls(self):
        """Resolve self._attr.method() calls via class attribute type hints.

        For each method's calls_raw, if target is 'self._xxx', look up _xxx in
        the parent class's attributes to find the type hint, resolve the type to
        a class hash, and find the method on that class to create a calls edge.

        Also resolves cross-class calls via inheritance: if a method is not found
        on the direct class, walk up the inheritance chain.
        """
        resolved_count = 0
        inheritance_resolved = 0

        # Pre-build class attribute type maps for efficiency
        class_attr_types = {}  # class_hash → {attr_name: type_hint}
        for h, node in self.nodes.items():
            if node.get("type") != "class":
                continue
            attr_map = {}
            for attr in node.get("attributes", []):
                if attr.get("type_hint"):
                    attr_map[attr["name"]] = attr["type_hint"]
            if attr_map:
                class_attr_types[h] = attr_map

        # Pre-build inheritance chain for method lookup
        def find_method_in_hierarchy(class_hash, method_name, visited=None):
            """Walk up the inheritance tree to find a method."""
            if visited is None:
                visited = set()
            if class_hash in visited:
                return None
            visited.add(class_hash)

            cls = self.nodes.get(class_hash, {})
            cls_name = cls.get("name", "")
            cls_file = cls.get("file", "")

            # Direct lookup
            method_fqn = f"{cls_file}::{cls_name}::{method_name}"
            method_hash = self.fqn_to_hash.get(method_fqn)
            if method_hash:
                return method_hash

            # Walk up bases
            for base_info in cls.get("bases_resolved", []):
                base_hash = base_info.get("hash")
                if base_hash:
                    result = find_method_in_hierarchy(base_hash, method_name, visited)
                    if result:
                        return result
            return None

        for h, node in self.nodes.items():
            if node.get("type") != "method":
                continue

            calls_raw = node.get("calls_raw", [])
            class_hash = node.get("parent")
            if not class_hash:
                continue

            attr_types = class_attr_types.get(class_hash, {})
            if not attr_types and not self.nodes.get(class_hash, {}).get("bases_resolved"):
                continue

            for call in calls_raw:
                if call.get("resolved_hash"):
                    continue  # already resolved

                target = call.get("target", "")
                method_name = call.get("method", "")

                if not target or not method_name:
                    continue

                # Pattern: self._xxx.method() or self.xxx.method()
                if target.startswith("self."):
                    attr_name = target[5:]  # strip "self."
                    type_hint = attr_types.get(attr_name)

                    if not type_hint:
                        # Could be self.xxx where xxx is inherited — skip for now
                        continue

                    # Resolve type_hint to a class
                    simple_name = type_hint.split(".")[-1]
                    target_class_hashes = self.class_name_to_hashes.get(simple_name, [])
                    if not target_class_hashes:
                        target_class_hashes = self.class_name_to_hashes.get(type_hint, [])

                    for tc_hash in target_class_hashes:
                        target_method_hash = find_method_in_hierarchy(
                            tc_hash, method_name
                        )
                        if target_method_hash:
                            call["resolved_hash"] = target_method_hash
                            call["resolved_via"] = f"attr_type:{attr_name}→{simple_name}"
                            node.setdefault("calls", []).append(target_method_hash)
                            self.nodes[target_method_hash].setdefault(
                                "called_by", []
                            ).append(h)
                            resolved_count += 1
                            break

                # Pattern: ClassName.method() — static/class method call
                elif target[0].isupper() and "." not in target:
                    target_class_hashes = self.class_name_to_hashes.get(target, [])
                    for tc_hash in target_class_hashes:
                        target_method_hash = find_method_in_hierarchy(
                            tc_hash, method_name
                        )
                        if target_method_hash:
                            call["resolved_hash"] = target_method_hash
                            call["resolved_via"] = f"class_call:{target}"
                            node.setdefault("calls", []).append(target_method_hash)
                            self.nodes[target_method_hash].setdefault(
                                "called_by", []
                            ).append(h)
                            resolved_count += 1
                            break

        logger.info(
            f"Cross-file call resolution: {resolved_count} calls resolved "
            f"via attribute type hints and class references"
        )

    # ── Output ─────────────────────────────────────────────

    def to_yaml_dict(self) -> Dict:
        """Export the graph as a YAML-serializable dict."""
        # Note: resolve_self_calls and resolve_cross_file_calls
        # are now called in parse_all() Phase 4

        # Clean up internal fields
        clean_nodes = {}
        for h, node in self.nodes.items():
            clean = {}
            for k, v in node.items():
                if k.startswith("_"):
                    continue  # skip internal fields
                clean[k] = v
            clean_nodes[h] = clean

        # Build indexes
        index = {
            "by_name": {},
            "by_type": dict(self.type_to_hashes),
            "by_file": dict(self.file_to_hash),
            "fixture_names": dict(self.fixture_name_to_hash),
        }
        for name, hashes in self.name_to_hashes.items():
            if len(hashes) > 1:
                index["by_name"][name] = hashes
            else:
                index["by_name"][name] = hashes[0]

        result = {
            "meta": {
                "schema_version": "2.0",
                "generated_from": str(self.src_root),
                "total_nodes": len(clean_nodes),
                "stats": {
                    "packages": self.stats.get("total_packages", 0),
                    "files": self.stats.get("total_files", 0),
                    "classes": self.stats.get("total_classs", 0),
                    "methods": self.stats.get("total_methods", 0),
                    "fixtures": self.stats.get("total_fixtures", 0),
                    "functions": self.stats.get("total_functions", 0),
                },
            },
            "nodes": clean_nodes,
            "index": index,
        }

        return result


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate codebase knowledge graph")
    parser.add_argument(
        "--src",
        required=True,
        help="Path to the Python source directory to parse",
    )
    parser.add_argument(
        "--output",
        default=str(Path(__file__).parent / "graph.yaml"),
        help="Output YAML path (default: graph.yaml)",
    )
    args = parser.parse_args()

    logger.info(f"Parsing source at: {args.src}")
    builder = CodebaseGraphBuilder(args.src)
    builder.parse_all()

    graph = builder.to_yaml_dict()

    logger.info(f"Writing graph to: {args.output}")
    with open(args.output, "w", encoding="utf-8") as f:
        yaml.dump(graph, f, default_flow_style=False, sort_keys=False,
                  allow_unicode=True, width=120)

    file_size = os.path.getsize(args.output)
    logger.info(f"Done! Output: {file_size / 1024 / 1024:.1f} MB")
    logger.info(f"  Total nodes: {graph['meta']['total_nodes']}")
    for k, v in graph["meta"]["stats"].items():
        logger.info(f"  {k}: {v}")


if __name__ == "__main__":
    main()
