"""Acting prompts and responses for the fishing personas."""

from __future__ import annotations

import datetime
import os

from pathfinder import assistant
from pathfinder import user
from simulation.persona import persona as persona_lib
from simulation.persona.cognition import leaders as leaders_lib
from simulation.persona.cognition import utils as cognition_utils
from simulation.utils import models as sim_models


def prompt_action_choose_amount_of_fish_to_catch(
    model: sim_models.ModelWandbWrapper,
    agent: persona_lib.PersonaAgent,
    memories: list[str],
    current_location: str,
    current_time: datetime.datetime,
    context: str,
    interval: list[int],
    consider_identity_persona: bool = True,
    leader_agenda: str = "",
    debug: bool = False,
):
  """Prompt the agent to choose how many fish to catch."""
  del consider_identity_persona
  lm = model.start_chain(
      agent.identity.name,
      "fishing_cognition_act",
      "choose_act_options",
  )
  svo_prompt, _, leader_prompt = (
      leaders_lib.get_leader_persona_prompts(agent)
  )
  with user():
    lm += (
        f"{cognition_utils.get_sytem_prompt(agent.identity)}\n"
    )
    lm += cognition_utils.location_time_info(
        current_location, current_time,
    )
    lm += f"Current context: {context}\n"
    lm += (
        "\nThe current policy following the mayor's agenda is"
        f" the following: {leader_agenda}\n"
    )
    lm += (
        f"{cognition_utils.memory_prompt(agent.identity, memories)}\n"
    )
    if svo_prompt:
      lm += f"{svo_prompt}\n"
    if leader_prompt:
      lm += f"{leader_prompt}\n"
    lm += (
        "Task: With a fishing range set between"
        f" {interval[0]}-{interval[-1]},"
        " how many tons of fish would you catch this month? "
    )
    lm += cognition_utils.reasoning_steps_prompt()
    lm += (
        ' Put the final answer after "Answer:",'
        " example Answer: N tons."
    )
    if debug:
      print(
          "\n\nCHOOSE AMOUNT PROMPT:"
          # pylint: disable=protected-access
          f"\n\n{lm._current_prompt()}\n"
      )

  with assistant():
    lm = model.gen(
        lm,
        "reasoning",
        stop_regex=r"Answer:|So, the answer is:|\*\*Answer\*\*:",
        save_stop_text=True,
    )
    lm = model.find(
        lm,
        regex=r"\d+",
        default_value="0",
        stop_regex=r"tons",
        name="option",
    )
    option = int(lm["option"])
  response_log_path = os.path.join(
      agent.experiment_storage,
      cognition_utils.COGNITION_RESPONSES_JSON,
  )
  cognition_utils.log_to_file(
      log_type="action_response",
      data={
          "speaker": agent.identity.name,
          "svo": agent.svo_type.value,
          "reasoning": lm["reasoning"],
          "option": option,
      },
      log_path=response_log_path,
  )
  if debug:
    print(
        "\n\nCHOOSE AMOUNT RESPONSE:"
        f"\n\nREASON: {lm['reasoning']}\n"
        f"CATCH: {option}"
    )

  model.end_chain(agent.identity.name, lm)

  return option, lm.html()


def prompt_election_vote(
    model: sim_models.ModelWandbWrapper,
    agent: persona_lib.PersonaAgent,
    memories: list[str],
    current_location: str,
    current_time: str,
    candidates: list[str],
    issues: dict[str, str],
    debug: bool = False,
) -> tuple[str, str]:
  """Prompt the agent to vote for a candidate."""
  del current_location, current_time
  lm = model.start_chain(
      agent.identity.name,
      "fishing_election",
      "vote_decision",
  )
  _, _, leader_prompt = (
      leaders_lib.get_leader_persona_prompts(agent)
  )
  with user():
    lm += (
        f"{cognition_utils.get_sytem_prompt(agent.identity)}\n"
    )
    lm += cognition_utils.memory_prompt(
        agent.identity, memories,
    )
    if leader_prompt:
      lm += (
          f"Coming into the election, {leader_prompt}\n"
      )
      lm += (
          "Leader agenda from last election:"
          f" {agent.agenda}\n"
      )
      lm += (
          "Harvest report from last cycle:"
          f" {agent.harvest_report}\n"
      )
    else:
      lm += "There is no current leader.\n"
    lm += (
        "Task: Select a mayor for the fisheries union among"
        " a list of candidates and provide your rationale"
        " for your selection."
    )
    lm += "\nCandidate positions:\n"
    for candidate in candidates:
      lm += f"- {candidate}: {issues[candidate]}\n"
    lm += cognition_utils.reasoning_steps_prompt()
    lm += (
        "\nTask: Based on fishing policies and agendas,"
        " who would you vote for?"
        f" {', '.join(candidates)}?"
    )
    lm += (
        ' Put the final answer after "Vote:",'
        ' example "Vote: John"'
    )
    if debug:
      print(
          "\n\nVOTE PROMPT:"
          # pylint: disable=protected-access
          f"\n\n{lm._current_prompt()}\n"
      )

  with assistant():
    lm = model.gen(
        lm,
        "reasoning",
        stop_regex=r"Vote:|\*\*Vote\*\*:",
        save_stop_text=True,
    )
    lm = model.find(
        lm,
        regex=fr"{'|'.join(candidates)}",
        default_value="none",
        name="option",
    )
    reasoning = lm["reasoning"]
    vote = lm["option"].strip()

  response_log_path = os.path.join(
      agent.experiment_storage,
      cognition_utils.COGNITION_RESPONSES_JSON,
  )
  cognition_utils.log_to_file(
      log_type="vote_response",
      data={
          "speaker": agent.identity.name,
          "svo": agent.svo_type.value,
          "reasoning": lm["reasoning"],
          "option": lm["option"],
      },
      log_path=response_log_path,
  )
  if debug:
    print(
        f"\n\nVOTE RESPONSE:\n\nREASON:"
        f" {lm['reasoning']}\n"
        f"VOTE: {lm['option']}"
    )

  model.end_chain(agent.identity.name, lm)
  return vote, lm.html()
