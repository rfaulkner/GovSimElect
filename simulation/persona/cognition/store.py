"""Store cognition component — handles memory storage with importance scoring."""

import datetime

import numpy as np

from simulation.utils import models as sim_models
from simulation.persona import common
from simulation.persona import embedding_model as embed_mod
from simulation.persona.memory import associative_memory
from simulation.persona.cognition import component
from simulation.persona.cognition import store_prompts


class StoreComponent(component.Component):
  """Handles storing memories with importance scoring."""

  def __init__(
      self,
      model: sim_models.ModelWandbWrapper,
      model_framework: sim_models.ModelWandbWrapper,
      memory: associative_memory.AssociativeMemory,
      emb_model: embed_mod.EmbeddingModel,
      cfg,
  ) -> None:
    """Initialize the store component."""
    super().__init__(model, model_framework, cfg)
    self.associative_memory = memory
    self.embedding_model = emb_model

  def _compute_importance(
      self, node: associative_memory.Node,
  ) -> float:
    """Compute importance score for a memory node."""
    if node.type == associative_memory.NodeType.THOUGHT:
      score = store_prompts.prompt_importance_thought(
          self.model, self.persona.identity, node,
      )
    elif node.type == associative_memory.NodeType.CHAT:
      score = store_prompts.prompt_importance_chat(
          self.model, self.persona.identity, node,
      )
    elif node.type == associative_memory.NodeType.EVENT:
      score = store_prompts.prompt_importance_event(
          self.model, self.persona.identity, node,
      )
    elif node.type == associative_memory.NodeType.ACTION:
      score = store_prompts.prompt_importance_action(
          self.model, self.persona.identity, node,
      )
    else:
      raise ValueError(f"Unknown node type: {node.type}")
    node.importance_score = score

  def store_event(self, event: common.PersonaEvent):
    """Store an event in associative memory."""
    s, p, o = (None, None, None)
    node = self.associative_memory.add_event(
        s, p, o,
        event.description,
        event.created,
        event.expiration,
    )
    if event.always_include:
      node.importance_score = 10
      node.always_include = True
    else:
      self._compute_importance(node)
    embedding = self.embedding_model.embed(
        event.description,
    )
    self.associative_memory.set_node_embedding(
        node.id, embedding,
    )

  def store_chat(
      self,
      summary: str,
      conversation: list[tuple[str, str]],
      created: datetime.datetime,
      expiration_delta: datetime.timedelta = None,
  ):
    """Store a chat in associative memory."""
    if expiration_delta is None:
      expiration_delta = datetime.timedelta(
          days=self.cfg.expiration_delta.days,
      )
    expiration = created + expiration_delta
    s, p, o = (None, None, None)
    node = self.associative_memory.add_chat(
        s, p, o,
        summary,
        conversation,
        created,
        expiration,
    )
    self._compute_importance(node)
    embedding = self.embedding_model.embed(summary)
    self.associative_memory.set_node_embedding(
        node.id, embedding,
    )

  def store_action(
      self,
      description: str,
      created: datetime.datetime,
      expiration_delta: datetime.timedelta = None,
  ):
    """Store an action in associative memory."""
    if expiration_delta is None:
      expiration_delta = datetime.timedelta(
          days=self.cfg.expiration_delta.days,
      )
    expiration = created + expiration_delta
    s, p, o = (None, None, None)
    node = self.associative_memory.add_action(
        s, p, o, description, created, expiration,
    )
    self._compute_importance(node)
    embedding = self.embedding_model.embed(description)
    self.associative_memory.set_node_embedding(
        node.id, embedding,
    )

  def store_thought(
      self,
      description: str,
      created: datetime.datetime,
      expiration_delta: datetime.timedelta = None,
      always_include: bool = False,
  ):
    """Store a thought in associative memory."""
    if expiration_delta is None:
      expiration_delta = datetime.timedelta(
          days=self.cfg.expiration_delta.days,
      )
    expiration = created + expiration_delta
    s, p, o = (None, None, None)
    node = self.associative_memory.add_thought(
        s, p, o, description, created, expiration,
    )
    if always_include:
      node.importance_score = 10
      node.always_include = True
    else:
      self._compute_importance(node)
    embedding = self.embedding_model.embed(description)
    self.associative_memory.set_node_embedding(
        node.id, embedding,
    )
