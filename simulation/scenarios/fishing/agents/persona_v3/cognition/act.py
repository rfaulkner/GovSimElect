from datetime import datetime

from pathfinder import assistant, system, user
from simulation.persona.cognition.act import ActComponent
from simulation.utils import ModelWandbWrapper

from .act_prompts import prompt_action_choose_amount_of_fish_to_catch
from .act_prompts import prompt_election_vote
from .utils import get_universalization_prompt


class FishingActComponent(ActComponent):
  """Actions that can be carried out by fishers.

  We have to options here:
  - choose at one time-step how many fish to chat
  - choose at one time-strep whether to fish one more time
  """

  def __init__(
      self, model: ModelWandbWrapper, model_framework: ModelWandbWrapper, cfg
  ):
    super().__init__(model, model_framework, cfg)

  def choose_how_many_fish_to_catch(
      self,
      retrieved_memories: list[str],
      current_location: str,
      current_time: datetime,
      context: str,
      interval: list[int],
      overusage_threshold: int,
      leader_agenda: str,
      debug: bool = False,
  ):
    if self.cfg.universalization_prompt:
      context += get_universalization_prompt(overusage_threshold)
    res, html = prompt_action_choose_amount_of_fish_to_catch(
        self.model,
        self.persona.identity,
        retrieved_memories,
        current_location,
        current_time,
        context,
        interval,
        consider_identity_persona=self.cfg.consider_identity_persona,
        leader_agenda=leader_agenda,
        debug=debug,
    )
    res = int(res)
    return res, [html]

  def participate_in_election(
      self,
      retrieved_memories: list[str],
      current_location: str,
      current_time: str,
      candidates: list[str],
      leader_agendas: dict[str, str],
      debug: bool = False,
  ) -> tuple[str, list[str]]:
    """Participate in leader election by voting for a candidate.

    Args:
        retrieved_memories: List of retrieved memories
        current_location: Current location string
        current_time: Current time string
        candidates: List of candidate IDs
        leader_agendas: Dictionary mapping candidates to their agendas
        debug: Whether to print debug information

    Returns:
        Tuple[str, List[str]]: (chosen_candidate, list of html responses)
    """
    vote, html = prompt_election_vote(
        self.model,
        self.persona.identity,
        retrieved_memories,
        current_location,
        current_time,
        candidates,
        leader_agendas,
        debug=debug,
    )
    return vote, [html]

