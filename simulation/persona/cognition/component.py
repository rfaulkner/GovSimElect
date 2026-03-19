"""Component base class for cognition modules."""

from __future__ import annotations

import typing

from simulation.utils import models as sim_models

if typing.TYPE_CHECKING:
  from simulation.persona import persona


class Component:
  """Base class for cognition components."""

  persona: "persona.PersonaAgent"

  def __init__(
      self,
      model: sim_models.ModelWandbWrapper,
      model_framework: sim_models.ModelWandbWrapper,
      cfg=None,
  ) -> None:
    """Initialize the component."""
    self.model = model
    self.model_framework = model_framework
    self.cfg = cfg
    self.other_personas: dict[str, "persona.PersonaAgent"] = {}

  def init_persona_ref(
      self, persona_ref: "persona.PersonaAgent",
  ):
    """Set the persona reference."""
    self.persona = persona_ref

  def add_reference_to_other_persona(
      self, persona_ref: "persona.PersonaAgent",
  ):
    """Add a reference to another persona."""
    self.other_personas[
        persona_ref.identity.name
    ] = persona_ref
