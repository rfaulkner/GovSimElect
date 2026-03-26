"""Retrieve cognition component — retrieves memories from associative memory."""

import datetime

from simulation.persona.cognition import component
from simulation.persona.memory import associative_memory
from simulation.utils import models as sim_models


class RetrieveComponent(component.Component):
  """Retrieve works as follows.

  Each memory entry is scored on two dimensions:
  - Recency: based on position in the memory list (more recent = higher).
    Weight: 1
  - Importance: LLM-assigned score normalized to [0, 1].
    Weight: 3

  Recency:
  assign a score based on 0.99**i where i is the index of the
  item in the memory (the more recent the more relevant)

  Importance:
  use LLM to generate a score for each item in the memory. It
  is computed when saved in memory
  """

  def __init__(
      self,
      model: sim_models.ModelWandbWrapper,
      model_framework: sim_models.ModelWandbWrapper,
      memory: associative_memory.AssociativeMemory,
  ):
    """Initialize the retrieve component."""
    super().__init__(model, model_framework)
    self.associative_memory = memory

    self.weights = {
        "recency": 1,
        "importance": 3,
    }
    self.recency_decay_param = 0.99

  def _recency_retrieval(
      self,
      entries: list[tuple[datetime.datetime, str, float, bool]],
  ) -> list[float]:
    """Calculate recency scores for a list of memory entries.

    Args:
      entries: List of (created, description, importance, always_include).

    Returns:
      A list of recency scores (same order as entries).
    """
    return [self.recency_decay_param**i for i in range(len(entries))]

  def _importance_retrieval(
      self,
      entries: list[tuple[datetime.datetime, str, float, bool]],
  ) -> list[float]:
    """Extract and normalize importance scores for a list of entries.

    Args:
      entries: List of (created, description, importance, always_include).

    Returns:
      A list of normalized importance scores (same order).
    """
    min_score = 1
    max_score = 10
    result = []
    for _, _, importance, _ in entries:
      normalized = (importance - min_score) / (max_score - min_score)
      result.append(normalized)
    return result

  def _retrieve_entries(
      self,
      focal_points: list[str],
      top_k: int,
  ) -> list[tuple[datetime.datetime, str, float, bool]]:
    """Retrieve top-k memory entries from MEMORY.md.

    Args:
      focal_points: List of focal points (unused in markup mode, kept for API
        compatibility).
      top_k: Number of top entries to retrieve.

    Returns:
      List of top-k (created, description, importance, always_include)
      tuples.
    """
    entries = self.associative_memory.read_memory_md(
        self.persona.current_time,
    )

    if not entries:
      return []

    recency_scores = self._recency_retrieval(entries)
    importance_scores = self._importance_retrieval(entries)

    combined_scores = []
    for i, entry in enumerate(entries):
      score = (
          recency_scores[i] * self.weights["recency"]
          + importance_scores[i] * self.weights["importance"]
      )
      combined_scores.append(score)

    # always_include entries get boosted above everything else.
    max_score = max(combined_scores) if combined_scores else 10
    for i, entry in enumerate(entries):
      _, _, _, always_include = entry
      if always_include:
        combined_scores[i] = max_score + 1

    # Sort by combined score (descending) and take top_k.
    scored_entries = list(zip(combined_scores, entries))
    scored_entries.sort(key=lambda x: x[0], reverse=True)

    top_k_entries = [entry for _, entry in scored_entries[:top_k]]
    return top_k_entries

  def retrieve(
      self, focal_points: list[str], top_k: int,
  ) -> list[tuple[datetime.datetime, str]]:
    """Retrieve top-k memories for the given focal points."""
    entries = self._retrieve_entries(focal_points, top_k)

    # Deduplicate by description.
    seen = set()
    unique = []
    for created, description, _, _ in entries:
      if description not in seen:
        seen.add(description)
        unique.append((created, description))

    # Sort by timestamp (oldest first).
    unique.sort(key=lambda x: x[0])
    return unique

