"""Acting prompts and responses for the fishing personas."""

from datetime import datetime
import os

from pathfinder import assistant
from pathfinder import user
from simulation.persona import PersonaAgent
from simulation.scenarios.fishing.agents.persona_v3.cognition import leaders as leaders_lib
from simulation.utils import ModelWandbWrapper

from .utils import COGNITION_RESPONSES_JSON
from .utils import get_sytem_prompt
from .utils import location_time_info
from .utils import log_to_file
from .utils import memory_prompt
from .utils import reasoning_steps_prompt



def prompt_action_choose_amount_of_fish_to_catch(
    model: ModelWandbWrapper,
    agent: PersonaAgent,
    memories: list[str],
    current_location: str,
    current_time: datetime,
    context: str,
    interval: list[int],
    consider_identity_persona: bool = True,
    leader_agenda: str = "",
    debug: bool = False,
):
  """Choose amount of fish to catch prompt."""
  del consider_identity_persona
  lm = model.start_chain(
      agent.identity.name, "fishing_cognition_act", "choose_act_options"
  )
  svo_prompt, _, leader_prompt = (
      leaders_lib.get_leader_persona_prompts(agent)
  )
  with user():
    lm += f"{get_sytem_prompt(agent.identity)}\n"
    lm += location_time_info(current_location, current_time)
    lm += f"Current context: {context}\n"
    lm += (
        "\nThe current policy following the mayor's agenda isthe following:"
        f" {leader_agenda}\n"
    )
    lm += f"{memory_prompt(agent.identity, memories)}\n"
    if svo_prompt:
      lm += f"{svo_prompt}\n"
    if leader_prompt:
      lm += f"{leader_prompt}\n"
    lm += (
        f"Task: With a fishing range set between {interval[0]}-{interval[-1]},"
        " how many tons of fish would you catch this month? "
    )
    lm += reasoning_steps_prompt()
    lm += ' Put the final answer after "Answer:", example Answer: N tons.'
    if debug:
      print(f"\n\nCHOOSE AMOUNT PROMPT:\n\n{lm._current_prompt()}\n")

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
  # Log the response to the experiment storage.
  response_log_path = os.path.join(
      agent.experiment_storage, COGNITION_RESPONSES_JSON
  )
  log_to_file(
      log_type="action_response",
      data={
          "speaker": agent.identity.name,
          "svo": agent.svo_type,
          "reasoning": lm["reasoning"],
          "option": option,
      },
      log_path=response_log_path,
  )
  if debug:
    print(
        f"\n\nCHOOSE AMOUNT RESPONSE:\n\nREASON: {lm['reasoning']}\nCATCH:"
        f" {option}"
    )

  model.end_chain(agent.identity.name, lm)

  return option, lm.html()


def prompt_election_vote(
    model: ModelWandbWrapper,
    agent: PersonaAgent,
    memories: list[str],
    current_location: str,
    current_time: str,
    candidates: list[str],
    issues: dict[str, str],
    debug: bool = False,
) -> tuple[str, str]:
  """Vote decision prompt."""
  del current_location, current_time
  lm = model.start_chain(
      agent.identity.name, "fishing_election", "vote_decision"
  )
  _, _, leader_prompt = (
      leaders_lib.get_leader_persona_prompts(agent)
  )
  with user():
    lm += f"{get_sytem_prompt(agent.identity)}\n"
    lm += memory_prompt(agent.identity, memories)
    if leader_prompt:
      lm += f"Coming into the election, {leader_prompt}\n"
      lm += f"Leader agenda from last election: {agent.agenda}\n"
      lm += f"Harvest report from last cycle: {agent.harvest_report}\n"
    else:
      lm += "There is no current leader.\n"
    lm += (
        "Task: Select a mayor for the fisheries union among a list of"
        " candidates and provide your rationale for your selection."
    )
    lm += "\nCandidate positions:\n"
    for candidate in candidates:
      lm += f"- {candidate}: {issues[candidate]}\n"
    lm += reasoning_steps_prompt()
    lm += (
        "\nTask: Based on fishing policies and agendas, who would you vote for?"
        f" {', '.join(candidates)}?"
    )
    lm += ' Put the final answer after "Vote:", example "Vote: John"'
    if debug:
      print(f"\n\nVOTE PROMPT:\n\n{lm._current_prompt()}\n")

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

  # Log the response to the experiment storage.
  response_log_path = os.path.join(
      agent.experiment_storage, COGNITION_RESPONSES_JSON
  )
  log_to_file(
      log_type="vote_response",
      data={
          "speaker": agent.identity.name,
          "svo": agent.svo_type,
          "reasoning": lm["reasoning"],
          "option": lm["option"],
      },
      log_path=response_log_path,
  )
  if debug:
    print(f"\n\nVOTE RESPONSE:\n\nREASON: {lm['reasoning']}\nVOTE:"
          f" {lm['option']}")

  model.end_chain(agent.identity.name, lm)
  return vote, lm.html()

