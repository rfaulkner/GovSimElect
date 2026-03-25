"""Act cognition component — handles harvesting and election decisions."""

import asyncio
import datetime

from simulation.utils import models as sim_models
from simulation.persona.cognition import act_prompts
from simulation.persona.cognition import component
from simulation.persona.cognition import utils as cognition_utils


class ActComponent(component.Component):
  """Handles acting decisions for fishing and elections."""

  def __init__(
      self,
      model: sim_models.ModelWandbWrapper,
      model_framework: sim_models.ModelWandbWrapper,
      cfg=None,
  ):
    """Initialize the act component."""
    super().__init__(model, model_framework, cfg)

  def choose_how_many_fish_to_catch(
      self,
      retrieved_memories: list[str],
      current_location: str,
      current_time: datetime.datetime,
      context: str,
      interval: list[int],
      overusage_threshold: int,
      leader_agenda: str,
      debug: bool = False,
  ):
    """Choose how many fish to catch this month."""
    if self.cfg.universalization_prompt:
      context += cognition_utils.get_universalization_prompt(
          overusage_threshold,
      )
    res, html = (
        act_prompts
        .prompt_action_choose_amount_of_fish_to_catch(
            self.model,
            self.persona,
            retrieved_memories,
            current_location,
            current_time,
            context,
            interval,
            consider_identity_persona=(
                self.cfg.consider_identity_persona
            ),
            leader_agenda=leader_agenda,
            debug=debug,
        )
    )
    res = int(res)
    return res, [html]

  async def achoose_how_many_fish_to_catch(
      self,
      retrieved_memories: list[str],
      current_location: str,
      current_time: datetime.datetime,
      context: str,
      interval: list[int],
      overusage_threshold: int,
      leader_agenda: str,
      debug: bool = False,
  ):
    """Async version of ``choose_how_many_fish_to_catch``."""
    return await asyncio.to_thread(
        self.choose_how_many_fish_to_catch,
        retrieved_memories,
        current_location,
        current_time,
        context,
        interval,
        overusage_threshold,
        leader_agenda,
        debug=debug,
    )

  def participate_in_election(
      self,
      retrieved_memories: list[str],
      current_location: str,
      current_time: str,
      candidates: list[str],
      leader_agendas: dict[str, str],
      debug: bool = False,
  ) -> tuple[str, list[str]]:
    """Participate in an election and cast a vote."""
    vote, html = act_prompts.prompt_election_vote(
        self.model,
        self.persona,
        retrieved_memories,
        current_location,
        current_time,
        candidates,
        leader_agendas,
        debug=debug,
    )
    return vote, [html]

  async def aparticipate_in_election(
      self,
      retrieved_memories: list[str],
      current_location: str,
      current_time: str,
      candidates: list[str],
      leader_agendas: dict[str, str],
      debug: bool = False,
  ) -> tuple[str, list[str]]:
    """Async version of ``participate_in_election``."""
    return await asyncio.to_thread(
        self.participate_in_election,
        retrieved_memories,
        current_location,
        current_time,
        candidates,
        leader_agendas,
        debug=debug,
    )

