"""Retrieve cognition component — retrieves memories from associative memory."""

import datetime

import numpy as np

from simulation import utils as sim_utils
from simulation.persona import embedding_model as embed_mod
from simulation.persona.memory import associative_memory
from simulation.persona.cognition import component


class RetrieveComponent(component.Component):
  """
  Retrieve works as follows.
  First we need to have 3 scores for each item in the memory.
  - Recency: based on the time since the item was added to the
    memory (the more recent the more relevant) Weight: 0.5
  - Importance: based on the importance of the item (the more
    important the more relevant), via LLM Weight: 3
  - Relevance: based on the relevance of the item to the current
    context (the more relevant the more relevant), via cosine
    similarity of embedding Weight: 2

  Recency:
  assign a score base on 0.99**i where i is the index of the
  item in the memory (the more recent the more relevant)

  Importance:
  use LLM to generate a score for each item in the memory. It
  is computed when saved in memory


  Relevance:
  use cosine similarity of embedding to generate a score for
  each item in the memory
  TODO: choose embedding model
  """

  def __init__(
      self,
      model: sim_utils.ModelWandbWrapper,
      model_framework: sim_utils.ModelWandbWrapper,
      memory: associative_memory.AssociativeMemory,
      emb_model: embed_mod.EmbeddingModel,
  ):
    """Initialize the retrieve component."""
    super().__init__(model, model_framework)
    self.associative_memory = memory

    self.embedding_model = emb_model

    self.weights = {
        "recency": 0.5,
        "importance": 3,
        "relevance": 2,
    }
    self.recency_decay_param = 0.99

  def _recency_retrieval(
      self, nodes: list[associative_memory.Node],
  ) -> dict[str, float]:
    """
    Calculate the recency retrieval scores for a list of nodes.

    Args:
      nodes: The list of nodes to calculate recency scores for.

    Returns:
      A dictionary mapping node IDs to recency scores.
    """
    result = dict()
    for i, node in enumerate(nodes):
      result[node.id] = self.recency_decay_param**i
    return result

  def _importance_retrieval(
      self, nodes: list[associative_memory.Node],
  ) -> dict[str, float]:
    """
    Retrieve the importance scores for a list of nodes.

    Args:
      nodes: The list of nodes to retrieve importance for.

    Returns:
      A dictionary mapping node IDs to normalized scores.
    """
    result = dict()
    for node in nodes:
      result[node.id] = node.importance_score

    min_score = 1
    max_score = 10
    for node_id in result.keys():
      result[node_id] = (
          (result[node_id] - min_score)
          / (max_score - min_score)
      )

    return result

  def _relevance_retrieval(
      self,
      nodes: list[associative_memory.Node],
      focal_point: str,
  ) -> dict[str, float]:
    """
    Retrieve relevance scores based on cosine similarity.

    Args:
      nodes: The list of nodes to retrieve relevance for.
      focal_point: The focal point used for comparison.

    Returns:
      A dictionary mapping node IDs to relevance scores.
    """

    result = dict()
    focal_point_embedding = (
        self.embedding_model.embed_retrieve(focal_point)
    )

    def cosine_similarity(a, b):
      """Compute cosine similarity between two vectors."""
      return np.dot(a, b) / (
          np.linalg.norm(a) * np.linalg.norm(b)
      )

    for node in nodes:
      node_embedding = (
          self.associative_memory.get_node_embedding(
              node.id,
          )
      )
      result[node.id] = cosine_similarity(
          focal_point_embedding, node_embedding,
      )

    return result

  def _retrieve_dict(
      self, focal_points: list[str], top_k: int,
  ) -> dict[str, list[associative_memory.Node]]:
    """
    Retrieve nodes from associative memory.

    Args:
      focal_points: List of focal points to retrieve for.
      top_k: Number of top nodes to retrieve per point.

    Returns:
      Dictionary mapping each focal point to top-k nodes.
    """
    nodes = self.associative_memory.get_nodes_for_retrieval(
        self.persona.current_time,
    )

    recency_scores = self._recency_retrieval(nodes)
    importance_scores = self._importance_retrieval(nodes)

    acc_nodes = dict()

    for focal_point in focal_points:
      relevance_scores = self._relevance_retrieval(
          nodes, focal_point,
      )

      combined_scores = dict()

      for node_id in recency_scores.keys():
        combined_scores[node_id] = (
            recency_scores[node_id]
            * self.weights["recency"]
            + importance_scores[node_id]
            * self.weights["importance"]
            + relevance_scores[node_id]
            * self.weights["relevance"]
        )

      max_value = (
          max(combined_scores.values())
          if combined_scores
          else 10
      )
      for node in nodes:
        if node.always_include:
          combined_scores[node.id] = max_value + 1

      sorted_nodes = sorted(
          nodes,
          key=lambda node: combined_scores[node.id],
          reverse=True,
      )

      top_k_nodes = sorted_nodes[:top_k]
      acc_nodes[focal_point] = top_k_nodes
    return acc_nodes

  def retrieve(
      self, focal_points: list[str], top_k: int,
  ) -> list[tuple[datetime.datetime, str]]:
    """Retrieve top-k memories for the given focal points."""
    res = self._retrieve_dict(focal_points, top_k)
    res = res.values()
    res = [node for nodes in res for node in nodes]

    res = set(res)
    res = list(res)
    res_sort = [
        (node.created, node.description) for node in res
    ]
    res_sort = sorted(res_sort, key=lambda x: x[0])
    return res_sort
