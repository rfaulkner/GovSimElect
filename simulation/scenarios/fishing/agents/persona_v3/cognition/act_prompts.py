"""Acting prompts and responses for the fishing personas."""

from datetime import datetime

from pathfinder import assistant, system, user
from simulation.persona.common import PersonaIdentity
from simulation.utils import ModelWandbWrapper

from .utils import (
    get_sytem_prompt,
    location_time_info,
    memory_prompt,
    reasoning_steps_prompt,
)


def prompt_action_choose_amount_of_fish_to_catch(
    model: ModelWandbWrapper,
    identity: PersonaIdentity,
    memories: list[str],
    current_location: str,
    current_time: datetime,
    context: str,
    interval: list[int],
    consider_identity_persona: bool = True,
    leader_agenda: str = "",
    debug: bool = False,
):
  lm = model.start_chain(
      identity.name, "fishing_cognition_act", "choose_act_options"
  )

  with user():
    lm += f"{get_sytem_prompt(identity)}\n"
    lm += location_time_info(current_location, current_time)
    lm += f"Current context: {context}\n"
    lm += (
        "\nThe current policy following the mayor's agenda isthe following:"
        f" {leader_agenda}\n"
    )
    lm += f"{memory_prompt(identity, memories)}\n"
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
    reasoning = lm["reasoning"]
    if debug:
      print(
          f"\n\nCHOOSE AMOUNT RESPONSE:\n\nREASON: {lm['reasoning']}\nCATCH:"
          f" {option}"
      )

  model.end_chain(identity.name, lm)

  return option, lm.html()


def prompt_election_vote(
    model: ModelWandbWrapper,
    identity: PersonaIdentity,
    memories: list[str],
    current_location: str,
    current_time: str,
    candidates: list[str],
    issues: dict[str, str],
    debug: bool = False,
) -> tuple[str, str]:
  """Vote decision prompt."""
  del current_location, current_time
  lm = model.start_chain(identity.name, "fishing_election", "vote_decision")

  with user():
    lm += f"{get_sytem_prompt(identity)}\n"
    lm += memory_prompt(identity, memories)
    lm += (
        "Task: Select a mayor for the fisheries union among a list of"
        " candidates and provide your rationale for your selection."
    )
    lm += "\nCandidate positions:\n"
    for candidate in candidates:
      lm += f"- {candidate}: {issues[candidate]}\n"
    lm += (
        "\nTask: Based on fishing policies and agendas, who would you vote for?"
        f" {', '.join(candidates)}?"
    )
    lm += reasoning_steps_prompt()
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
        regex=r"{'|'.join(candidates)}",
        default_value="none",
        # stop_regex=f"tons",
        name="option",
    )
    if debug:
      print(f"\n\nVOTE RESPONSE:\n\nREASON: {lm['reasoning']}\nVOTE:"
            f" {lm['option']}")
    reasoning = lm["reasoning"]
    vote = lm["option"].strip()

  model.end_chain(identity.name, lm)
  return vote, lm.html()

