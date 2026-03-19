"""Plan cognition component — handles future planning."""

from simulation.utils import models as sim_models
from simulation.persona.cognition import component


class PlanComponent(component.Component):
  """Handles planning cognition."""

  def __init__(
      self,
      model: sim_models.ModelWandbWrapper,
      model_framework: sim_models.ModelWandbWrapper,
  ):
    """Initialize the plan component."""
    super().__init__(model, model_framework)

  def chat_react(self):
    """React to a chat event."""
    pass

  def revise_self_indentity(self):
    """Revise persona self-identity given new experience."""
    pass

  def should_react(self):
    """Determine if persona should react."""
    pass

  def wait_react(self):
    """Wait before reacting."""
    pass

  def create_react(self):
    """Create a reaction."""
    pass
