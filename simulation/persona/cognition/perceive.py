"""Perceive cognition component — handles observations and events."""

from simulation import utils as sim_utils
from simulation.persona import common
from simulation.persona.cognition import component


class PerceiveComponent(component.Component):
  """Handles perceiving observations and events."""

  def __init__(
      self,
      model: sim_utils.ModelWandbWrapper,
      model_framework: sim_utils.ModelWandbWrapper,
  ):
    """Initialize the perceive component."""
    super().__init__(model, model_framework)

  def init_persona_ref(self, persona):
    """Set the persona reference."""
    self.persona = persona

  def perceive(self, obs: common.PersonaOberservation):
    """Process a persona observation."""
    self._add_events(obs.events)

  def _add_events(
      self, events: list[common.PersonaEvent],
  ):
    """Store a list of events."""
    for event in events:
      self.persona.store.store_event(event)

  def _add_chats(self, chat: common.ChatObservation):
    """Store a chat observation."""
    self.persona.store.store_chat(
        chat.summary,
        chat.conversation,
        self.persona.current_time,
    )
