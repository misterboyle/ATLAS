"""Tree node models for code indexing and retrieval."""
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any


class NodeType(str, Enum):
    """Type of AST/tree node."""
    FILE = "file"
    DIRECTORY = "directory"
    CLASS = "class"
    FUNCTION = "function"
    METHOD = "method"
    MODULE = "module"
    IMPORT = "import"
    VARIABLE = "variable"
    COMMENT = "comment"
    OTHER = "other"


@dataclass
class NodeMetadata:
    """Metadata attached to a tree node."""
    language: str = ""
    start_line: int = 0
    end_line: int = 0
    size_bytes: int = 0
    summary: str = ""
    docstring: str = ""
    tags: List[str] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TreeNode:
    """A node in the project source tree."""
    path: str
    name: str
    node_type: NodeType = NodeType.OTHER
    metadata: Optional[NodeMetadata] = None
    children: List["TreeNode"] = field(default_factory=list)
    content: str = ""
    parent_path: str = ""

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    @property
    def depth(self) -> int:
        parts = self.path.strip("/").split("/")
        return len(parts)


@dataclass
class TreeIndex:
    """Index of tree nodes for fast lookup."""
    root: Optional[TreeNode] = None
    nodes_by_path: Dict[str, TreeNode] = field(default_factory=dict)
    bm25_terms: Dict[str, List[str]] = field(default_factory=dict)

    @property
    def node_count(self) -> int:
        return len(self.nodes_by_path)

    def get_node(self, path: str) -> Optional[TreeNode]:
        return self.nodes_by_path.get(path)

    def add_node(self, node: TreeNode) -> None:
        self.nodes_by_path[node.path] = node
