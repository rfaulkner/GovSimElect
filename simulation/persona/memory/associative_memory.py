"""Associative memory for storing and retrieving memories.

This module contains the following classes:

- NodeType: Enum for node types.
- Node: Base class for all nodes.
- Thought: A thought node.
- Chat: A chat node.
- Event: An event node.
- Action: An action node.
- AssociativeMemory: A class for storing and retrieving memories.
"""

import datetime
import enum
import os
import typing

Enum = enum.Enum
datetime = datetime.datetime  # pylint: disable=invalid-name


class NodeType(Enum):
  CHAT = 1
  THOUGHT = 2
  EVENT = 3
  ACTION = 4

  def toJSON(self):
    return self.name


class Node:
  """A node in the associative memory graph."""
  id: int
  type: NodeType

  subject: str
  predicate: str
  object: str

  description: str

  importance_score: float

  created: datetime
  expiration: datetime

  always_include: bool

  def __init__(
      self,
      id_: int,
      type: NodeType,
      subject: str,
      predicate: str,
      object: str,
      description: str,
      created: datetime,
      expiration: datetime,
      always_include: bool = False,
  ) -> None:
    self.id = id_
    self.type = type
    self.subject = subject
    self.predicate = predicate
    self.object = object
    self.description = description
    self.created = created
    self.expiration = expiration
    self.always_include = always_include

  def __str__(self) -> str:
    return f"{self.subject} {self.predicate} {self.object}"

  def toJSON(self):  # pylint: disable=invalid-name
    """Convert the node to a JSON object.

    Returns:
        A dictionary containing the node information.
    """
    return {
        "id": self.id,
        "type": self.type.toJSON(),
        "subject": self.subject,
        "predicate": self.predicate,
        "object": self.object,
        "description": self.description,
        "importance_score": self.importance_score,
        "created": self.created.strftime("%Y-%m-%d %H:%M:%S"),
        "expiration": self.expiration.strftime("%Y-%m-%d %H:%M:%S"),
        "always_include": "true" if self.always_include else "false",
    }


class Thought(Node):
  """A thought node in the associative memory graph."""

  def __init__(
      self,
      id: int,
      subject: str,
      predicate: str,
      object: str,
      description: str,
      created: datetime,
      expiration: datetime,
      always_include: bool = False,
  ) -> None:
    super().__init__(
        id,
        NodeType.THOUGHT,
        subject,
        predicate,
        object,
        description,
        created,
        expiration,
        always_include,
    )


class Chat(Node):
  """A chat node in the associative memory graph."""
  conversation: list[tuple[str, str]]

  def __init__(
      self,
      id: int,
      subject: str,
      predicate: str,
      object: str,
      description: str,
      created: datetime,
      expiration: datetime,
      always_include: bool = False,
  ) -> None:
    self.conversation = []
    super().__init__(
        id,
        NodeType.CHAT,
        subject,
        predicate,
        object,
        description,
        created,
        expiration,
        always_include,
    )

  def toJSON(self):
    return {
        "id": self.id,
        "type": self.type.toJSON(),
        "subject": self.subject,
        "predicate": self.predicate,
        "object": self.object,
        "description": self.description,
        "importance_score": self.importance_score,
        "conversation": self.conversation,
        "created": self.created.strftime("%Y-%m-%d %H:%M:%S"),
        "expiration": self.expiration.strftime("%Y-%m-%d %H:%M:%S"),
        "always_include": "true" if self.always_include else "false",
    }


class Event(Node):
  """An event node in the associative memory graph."""

  def __init__(
      self,
      id: int,
      subject: str,
      predicate: str,
      object: str,
      description: str,
      created: datetime,
      expiration: datetime,
      always_include: bool = False,
  ) -> None:
    super().__init__(
        id,
        NodeType.EVENT,
        subject,
        predicate,
        object,
        description,
        created,
        expiration,
        always_include,
    )


class Action(Node):
  """An action node in the associative memory graph."""

  def __init__(
      self,
      id: int,
      subject: str,
      predicate: str,
      object: str,
      description: str,
      created: datetime,
      expiration: datetime,
      always_include: bool = False,
  ) -> None:
    super().__init__(
        id,
        NodeType.ACTION,
        subject,
        predicate,
        object,
        description,
        created,
        expiration,
        always_include,
    )


class AssociativeMemory:
  """An associative memory for storing and retrieving memories."""

  def __init__(self, base_path, do_load=False) -> None:
    self.id_to_node: typing.Dict[int, Node] = dict()

    self.thought_id_to_node: typing.Dict[int, Thought] = dict()
    self.chat_id_to_node: typing.Dict[int, Node] = dict()
    self.event_id_to_node: typing.Dict[int, Node] = dict()
    self.action_id_to_node: typing.Dict[int, Node] = dict()

    self.nodes_without_chat_by_time: list[Node] = []

    self.base_path = base_path
    self._memory_md_path = os.path.join(base_path, "MEMORY.md")

    if do_load and os.path.exists(self._memory_md_path):
      # Memories will be read on-demand via read_memory_md().
      pass

  def init_memory_md(self):
    """Initialize a fresh MEMORY.md file."""
    os.makedirs(self.base_path, exist_ok=True)
    with open(self._memory_md_path, "w") as f:
      f.write("# Agent Memory\n\n")

  def append_to_memory_md(self, node: Node):
    """Append a memory entry to MEMORY.md.

    Format:
        - TIMESTAMP | type=TYPE | importance=SCORE | expires=TIMESTAMP |
        always_include=BOOL
          Description text (may span multiple lines).

    Args:
        node: The node to append to the memory.
    """
    if not os.path.exists(self._memory_md_path):
      self.init_memory_md()

    created_str = node.created.strftime("%Y-%m-%d %H:%M:%S")
    expiration_str = node.expiration.strftime("%Y-%m-%d %H:%M:%S")
    always_str = "true" if node.always_include else "false"
    importance = getattr(node, "importance_score", 0)

    header = (
        f"- {created_str}"
        f" | type={node.type.name.lower()}"
        f" | importance={importance}"
        f" | expires={expiration_str}"
        f" | always_include={always_str}"
    )

    # Indent each line of description for proper markdown nesting.
    desc_lines = node.description.strip().split("\n")
    indented_desc = "\n".join(f"  {line}" for line in desc_lines)

    with open(self._memory_md_path, "a") as f:
      f.write(f"{header}\n{indented_desc}\n")

  def read_memory_md(
      self,
      current_time: datetime,
  ) -> list[tuple[datetime, str, float, bool]]:
    """Read and parse MEMORY.md, filtering out expired entries.

    Args:
        current_time: The current time.

    Returns:
        A list of tuples (created, description, importance, always_include)
        sorted oldest-first.
    """
    return self._read_and_filter(current_time)

  def _read_and_filter(
      self,
      current_time: datetime,
  ) -> list[tuple[datetime, str, float, bool]]:
    """Internal method that parses MEMORY.md with expiration filtering."""
    if not os.path.exists(self._memory_md_path):
      return []

    with open(self._memory_md_path, "r") as f:
      content = f.read()

    entries: list[tuple[datetime, str, float, bool, datetime]] = []
    current_header = None
    desc_lines: list[str] = []

    def _flush():
      if current_header is not None:
        description = "\n".join(desc_lines).strip()
        entries.append((
            current_header["created"],
            description,
            current_header["importance"],
            current_header["always_include"],
            current_header["expiration"],
        ))

    for line in content.split("\n"):
      if line.startswith("- ") and " | type=" in line:
        _flush()
        desc_lines = []
        current_header = self._parse_entry_header(line)
      elif current_header is not None:
        stripped = line[2:] if line.startswith("  ") else line
        if stripped:
          desc_lines.append(stripped)

    _flush()

    # Filter out expired entries.
    result = []
    for created, description, importance, always_include, expiration in entries:
      if always_include or expiration > current_time:
        result.append((created, description, importance, always_include))

    # Sort oldest first.
    result.sort(key=lambda x: x[0])
    return result

  @staticmethod
  def _parse_entry_header(line: str) -> dict:
    """Parse a MEMORY.md entry header line.

    Example:
        - 2024-03-26 16:06:25 | type=event | importance=5.0 | expires=2024-03-27
        16:06:25 | always_include=false

    Args:
        line: The header line to parse.

    Returns:
        A dictionary containing the parsed header information.
    """
    # Remove the leading "- "
    line = line[2:]
    parts = [p.strip() for p in line.split(" | ")]

    # First part is the created timestamp.
    created = datetime.strptime(parts[0], "%Y-%m-%d %H:%M:%S")

    metadata = {}
    for part in parts[1:]:
      key, value = part.split("=", 1)
      metadata[key] = value

    return {
        "created": created,
        "type": metadata.get("type", "event"),
        "importance": float(metadata.get("importance", "0")),
        "expiration": datetime.strptime(
            metadata.get("expires", "2099-12-31 23:59:59"),
            "%Y-%m-%d %H:%M:%S",
        ),
        "always_include": metadata.get("always_include", "false") == "true",
    }

  def save(self):
    """No-op — memories are appended to MEMORY.md in real time."""
    pass

  def _add(
      self, subject, predicate, obj, description, type, created, expiration
  ) -> Node:
    """Internal method to add a node to the associative memory."""
    id_ = len(self.id_to_node) + 1

    if type == NodeType.CHAT:
      node = Chat(
          id_, subject, predicate, obj, description, created, expiration
      )
      self.chat_id_to_node[id_] = node
    elif type == NodeType.THOUGHT:
      node = Thought(
          id_, subject, predicate, obj, description, created, expiration
      )
      self.thought_id_to_node[id_] = node
    elif type == NodeType.EVENT:
      node = Event(
          id_, subject, predicate, obj, description, created, expiration
      )
      self.event_id_to_node[id_] = node
    elif type == NodeType.ACTION:
      node = Action(
          id_, subject, predicate, obj, description, created, expiration
      )
      self.action_id_to_node[id_] = node
    else:
      raise ValueError(f"Unknown node type: {type}")

    if type != NodeType.CHAT:
      self.nodes_without_chat_by_time.append(node)

    self.id_to_node[id_] = node

    return node

  def add_chat(
      self,
      subject,
      predicate,
      obj,
      description,
      conversation,
      created,
      expiration,
  ) -> Chat:
    """Add a chat node to the associative memory."""
    node = self._add(
        subject, predicate, obj, description, NodeType.CHAT, created, expiration
    )
    node.conversation = conversation
    return node

  def add_thought(
      self, subject, predicate, obj, description, created, expiration
  ) -> Thought:
    return self._add(
        subject,
        predicate,
        obj,
        description,
        NodeType.THOUGHT,
        created,
        expiration,
    )

  def add_event(
      self, subject, predicate, obj, description, created, expiration
  ) -> Event:
    return self._add(
        subject,
        predicate,
        obj,
        description,
        NodeType.EVENT,
        created,
        expiration,
    )

  def add_action(
      self, subject, predicate, obj, description, created, expiration
  ) -> Action:
    return self._add(
        subject,
        predicate,
        obj,
        description,
        NodeType.ACTION,
        created,
        expiration,
    )

  def get_nodes_for_retrieval(self, current_time: datetime) -> list[Node]:
    """Get all nodes except chat, sorted by time.

    Args:
        current_time: The current time.

    Returns:
        A list of nodes sorted by time.
    """
    nodes = []
    for node in self.nodes_without_chat_by_time:
      if node.expiration > current_time:
        nodes.append(node)
    return nodes

