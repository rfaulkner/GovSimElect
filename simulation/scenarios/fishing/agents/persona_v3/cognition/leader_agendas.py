"""Leader Agenda Prompts."""

import datetime
import enum

import numpy as np
from pathfinder import assistant
from pathfinder import user
from simulation.persona import PersonaAgent
from simulation.persona import SVOPersonaType

from simulation.scenarios.fishing.agents.persona_v3.cognition.utils import get_sytem_prompt
from simulation.scenarios.fishing.agents.persona_v3.cognition.utils import location_time_info
from simulation.scenarios.fishing.agents.persona_v3.cognition.utils import memory_prompt
from simulation.scenarios.fishing.agents.persona_v3.cognition.utils import reasoning_steps_prompt
from simulation.utils import ModelWandbWrapper


GOAL_STATEMENT = (
    " The overall goal is to ensure the fishers community survive through"
    " sustainably approapriating resources while never exceeding the total pool"
    " of resources and while also maximising the amount of fish caught among"
    " the fishers and distributed fairly.\n"
)

SVO_LEADER_TASK = (
    "\nTask: The personalities of the players is defined by their Social Value"
    " Orientation (SVO). SVO is a psychological concept that describes how"
    " individuals value their own outcomes relative to the outcomes of others."
    " Your SVO is measured as an angle, where the angle represents the ratio"
    " of the value you place on your own outcomes relative to the outcomes of"
    " others. SVO angles can be classified into four categories:\n"
    "- Altruism (SVO angle > 57.15 degrees)\n"
    "- Prosocial (SVO angle between 22.45 and 57.15 degrees)\n"
    "- Individualism (SVO angle between -12.04 and 22.45 degrees)\n"
    "- Competitiveness (SVO angle < -12.04 degrees)\n"
)

# TODO(rfaulk): Add disinformation prompt.
DISINFORMATION_PROMPT = (
    " ... sometimes lie if you feel like it! .... "
)


# SVO Persona Types.
class LeaderPopulationType(enum.Enum):
  BALANCED = 1
  LEAN_ALTRUISTIC = 2
  LEAN_COMPETITIVE = 3


def sample_svo_angle(
    svo_category: SVOPersonaType,
) -> float:
  """Sample the SVO angle of the leader."""
  if SVOPersonaType.INDIVIDUALISTIC == svo_category:
    svo_angle = np.random.uniform(-12.04, 22.45)
  elif SVOPersonaType.PROSOCIAL == svo_category:
    svo_angle = np.random.uniform(22.45, 57.15)
  elif SVOPersonaType.ALTRUISTIC == svo_category:
    # TODO(rfaulk): Fix range.
    svo_angle = np.random.uniform(57.15, 75.0)
  elif SVOPersonaType.COMPETITIVE == svo_category:
    # TODO(rfaulk): Fix range.
    svo_angle = np.random.uniform(-45.0, -12.04)
  else:
    raise ValueError(f"Unknown SVO category: {svo_category}")
  return svo_angle


def svo_angle_prompt(svo_angle: float) -> str:
  """SVO prompt from SVO angle."""
  return f"SVO angle: {svo_angle} degrees."


def prompt_leader_agenda(
    model: ModelWandbWrapper,
    init_persona: PersonaAgent,
    current_location: str,
    current_time: datetime,
    init_retrieved_memory: list[str],
    total_fishers: int,
    svo_angle: float,
    use_disinfo: bool = False,
    debug: bool = False,
) -> tuple[str, str]:
  """Generic leader agenda prompt.
  
  Args:
    model: The target LLM in a weights and biases wrapper.
    init_persona: The persona to generate the agenda for.
    current_location: The current location of the persona.
    current_time: The current time of the persona.
    init_retrieved_memory: The retrieved memory of the persona.
    total_fishers: The total number of fishers in the community.
    svo_angle: The SVO angle of the leader.
    use_disinfo: Whether to use disinfo in the prompt.
    debug: Whether to print the prompt and response.

  Returns:
    Tuple of the agenda + model response.
  """
  lm = model.start_chain(init_persona.agent_id, "leader_agenda", "get_agenda")
  svo_angle_prompt_str = svo_angle_prompt(svo_angle)
  with user():
    lm += f"{get_sytem_prompt(init_persona.identity)}\n"
    lm += location_time_info(current_location, current_time)
    lm += (
        f"There are {total_fishers} fishers in total using the shared"
        " resources.\n"
    )
    lm += memory_prompt(init_persona.identity, init_retrieved_memory)
    lm += GOAL_STATEMENT
    lm += SVO_LEADER_TASK + svo_angle_prompt_str
    if use_disinfo:
      lm += "\n" + DISINFORMATION_PROMPT
    lm += "Describe your agenda in the following format:\n"
    lm += "1. Your principles for sustainable fishing.\n"
    lm += "2. How you will allocate fishing quotas.\n"
    lm += "3. Your enforcement approach.\n"
    lm += reasoning_steps_prompt()
    lm += (
        'Output format:\nMy agenda as mayor: [fill in your agenda here] END\n"'
    )
    if debug:
      print(f"\n\nCURRENT PROMPT:\n\n{lm._current_prompt()}\n")  # pylint: disable=protected-access

  with assistant():
    lm = model.gen(
        lm,
        "agenda",
        stop_regex=r"END",
        save_stop_text=False,
    )
    if debug:
      print(f"\n\nRESPONSE:\n\n{lm['agenda']}\n")
    agenda = lm["agenda"].strip()
    if agenda and agenda[0] == '"' and agenda[-1] == '"':
      agenda = agenda[1:-1]

  model.end_chain(init_persona.agent_id, lm)
  return agenda, lm.html()


# TODO(rfaulk): Compose the leader's newsletter.
def prompt_leader_newsletter(
    # TODO(rfaulk): Implement this. input facts, output report.
) -> tuple[str, str]:
  """Leader newsletter prompt."""
  raise NotImplementedError("Leader newsletter prompt not implemented.")


def sample_leader_svos(
    leader_population_type: LeaderPopulationType) -> list[float]:
  """Samples SVOs for a balanced leader group of four.

  Args:
    leader_population_type: The type of leader population to sample.

  Returns:
    Tuple of the sampled SVO angles and the SVO categories.
  """
  if leader_population_type == LeaderPopulationType.BALANCED:
    svo_categories = [
        SVOPersonaType.ALTRUISTIC,
        SVOPersonaType.PROSOCIAL,
        SVOPersonaType.COMPETITIVE,
        SVOPersonaType.INDIVIDUALISTIC,
    ]
  elif leader_population_type == LeaderPopulationType.LEAN_ALTRUISTIC:
    svo_categories = [
        SVOPersonaType.ALTRUISTIC,
        SVOPersonaType.PROSOCIAL,
        SVOPersonaType.PROSOCIAL,
        SVOPersonaType.INDIVIDUALISTIC,
    ]
  elif leader_population_type == LeaderPopulationType.LEAN_COMPETITIVE:
    svo_categories = [
        SVOPersonaType.COMPETITIVE,
        SVOPersonaType.COMPETITIVE,
        SVOPersonaType.COMPETITIVE,
        SVOPersonaType.INDIVIDUALISTIC,
    ]
  else:
    raise ValueError(
        f"Unknown leader population type: {leader_population_type}"
    )
  svos = []
  for svo_category in svo_categories:
    svos.append(sample_svo_angle(svo_category))
  return svos, svo_categories

