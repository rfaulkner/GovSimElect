"""Leader Agenda Prompts."""

import datetime
from pathfinder import assistant
from pathfinder import user
from simulation.persona import PersonaAgent
from simulation.utils import ModelWandbWrapper

from .utils import (
    get_sytem_prompt,
    location_time_info,
    memory_prompt,
    reasoning_steps_prompt,
)

GOAL_STATEMENT = (
    " The overall goal is to ensure the fishers community survive through"
    " sustainably approapriating resources while never exceeding the total pool"
    " of resources and while also maximising the amount of fish caught among"
    " the fishers and distributed fairly.\n"
)

CLEAR_REASONING_LEADER_TASK = (
    "\nTask: As a leader known for deep and comprehensive reasoning whose"
    " explanations are concise and clear provide a concise agenda in 2 or 3"
    " sentences that summarizes your detailed strategy."
)

VERBOSE_REASONING_LEADER_TASK = (
    "\nTask: As a leader known for unclear and verbose communication whose"
    " analysis is very detailed provide a concise agenda in 2 or 3"
    " sentences that summarizes your detailed strategy."
)

CLEAR_DIRECT_LEADER_TASK = (
    "\nTask:As a leader known for clear and concise communication though"
    " your analysis is less detailed  provide a concise agenda in 2 or 3"
    " sentences that summarizes your detailed strategy."
)

VERBOSE_DIRECT_LEADER_TASK = (
    "\nTask::As a leader known for unclear and verbose communication whose"
    " analysis is less detailed  provide a concise agenda in 2 or 3"
    " sentences that summarizes your detailed strategy."
)


def prompt_leader_agenda(
    model: ModelWandbWrapper,
    init_persona: PersonaAgent,
    current_location: str,
    current_time: datetime,
    init_retrieved_memory: list[str],
    task_leader_type: str,
    total_fishers: int,
    debug: bool = False,
) -> tuple[str, str]:
  """Generic leader agenda prompt.
  
  Args:
    model: The target LLM in a weights and biases wrapper.
    init_persona: The persona to generate the agenda for.
    current_location: The current location of the persona.
    current_time: The current time of the persona.
    init_retrieved_memory: The retrieved memory of the persona.
    task_leader_type: The task to be performed by the leader.
    total_fishers: The total number of fishers in the community.
    debug: Whether to print the prompt and response.

  Returns:
    Tuple of the agenda + model response.
  """
  lm = model.start_chain(init_persona.agent_id, "leader_agenda", "get_agenda")

  with user():
    lm += f"{get_sytem_prompt(init_persona.identity)}\n"
    lm += location_time_info(current_location, current_time)
    lm += (
        f"There are {total_fishers} fishers in total using the shared"
        " resources.\n"
    )
    lm += memory_prompt(init_persona.identity, init_retrieved_memory)
    lm += task_leader_type + GOAL_STATEMENT
    lm += "Describe your agenda in the following format:\n"
    lm += "1. Your principles for sustainable fishing.\n"
    lm += "2. How you will allocate fishing quotas.\n"
    lm += "3. Your enforcement approach.\n"
    lm += reasoning_steps_prompt()
    lm += (
        'Output format:\nMy agenda as mayor: [fill in your agenda here] END-AGENDA"'
    )
    if debug:
      print(f"\n\nAGENDA PROMPT:\n\n{lm._current_prompt()}\n")

  with assistant():
    lm = model.gen(
        lm,
        "agenda",
        stop_regex=r"END-AGENDA",
        save_stop_text=True,
    )
    if debug:
      print(f"\n\nAGENDA RESPONSE:\n\n{lm['agenda']}\n")
    agenda = lm["agenda"].strip()
    if len(agenda) and agenda[0] == '"' and agenda[-1] == '"':
      agenda = agenda[1:-1]

  model.end_chain(init_persona.agent_id, lm)
  return agenda, lm.html()


def prompt_leader_agenda_clear_explain(
    model: ModelWandbWrapper,
    init_persona: PersonaAgent,
    current_location: str,
    current_time: datetime,
    init_retrieved_memory: list[str],
    total_fishers: int,
    debug: bool = False,
) -> tuple[str, str]:
  """Clear and explained agenda prompt."""
  return prompt_leader_agenda(model, init_persona, current_location,
                              current_time, init_retrieved_memory,
                              CLEAR_REASONING_LEADER_TASK, total_fishers,
                              debug)


def prompt_leader_agenda_clear_direct(
    model: ModelWandbWrapper,
    init_persona: PersonaAgent,
    current_location: str,
    current_time: datetime,
    init_retrieved_memory: list[str],
    total_fishers: int,
    debug: bool = False,
) -> tuple[str, str]:
  """Clear and direct agenda prompt."""
  return prompt_leader_agenda(model, init_persona, current_location,
                              current_time, init_retrieved_memory,
                              CLEAR_DIRECT_LEADER_TASK, total_fishers,
                              debug)


def prompt_leader_agenda_verbose_direct(
    model: ModelWandbWrapper,
    init_persona: PersonaAgent,
    current_location: str,
    current_time: datetime,
    init_retrieved_memory: list[str],
    total_fishers: int,
    debug: bool = False,
) -> tuple[str, str]:
  """Verbose and direct agenda prompt."""
  return prompt_leader_agenda(model, init_persona, current_location,
                              current_time, init_retrieved_memory,
                              VERBOSE_DIRECT_LEADER_TASK, total_fishers,
                              debug)


def prompt_leader_agenda_verbose_explain(
    model: ModelWandbWrapper,
    init_persona: PersonaAgent,
    current_location: str,
    current_time: datetime,
    init_retrieved_memory: list[str],
    total_fishers: int,
    debug: bool = False,
) -> tuple[str, str]:
  """Verbose and explained agenda prompt."""
  return prompt_leader_agenda(model, init_persona, current_location,
                              current_time, init_retrieved_memory,
                              VERBOSE_REASONING_LEADER_TASK, total_fishers,
                              debug)

