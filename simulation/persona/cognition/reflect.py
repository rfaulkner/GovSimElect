"""Reflect cognition component — handles insights and conversation reflection."""

from simulation.utils import models as sim_models
from simulation.persona.cognition import component
from simulation.persona.cognition import reflect_prompts


class ReflectComponent(component.Component):
  """Handles reflection on memories and conversations."""

  def __init__(
      self,
      model: sim_models.ModelWandbWrapper,
      model_framework: sim_models.ModelWandbWrapper,
  ):
    """Initialize the reflect component."""
    super().__init__(model, model_framework)

  def run(self, focal_points: list[str]):
    """Run reflection on a list of focal points."""
    acc = []
    for focal_point in focal_points:
      retireved_memory = self.persona.retrieve.retrieve(
          [focal_point], 10,
      )

      insights = reflect_prompts.prompt_insight_and_evidence(
          self.model,
          self.persona.identity,
          retireved_memory,
      )
      for insight in insights:
        self.persona.store.store_thought(
            insight, self.persona.current_time,
        )
        acc.append(insight)

  def reflect_on_convesation(
      self, conversation: list[tuple[str, str]],
  ):
    """Reflect on a conversation and store insights."""
    planning = (
        reflect_prompts
        .prompt_planning_thought_on_conversation(
            self.model,
            self.persona.identity,
            conversation,
        )
    )
    self.persona.store.store_thought(
        planning, self.persona.current_time,
    )
    memo = (
        reflect_prompts
        .prompt_memorize_from_conversation(
            self.model,
            self.persona.identity,
            conversation,
        )
    )
    self.persona.store.store_thought(
        memo, self.persona.current_time,
    )
