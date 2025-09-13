"""Leader Agenda Prompts."""

from pathfinder import assistant
from pathfinder import user
from simulation.utils import ModelWandbWrapper

GOAL_STATEMENT = (
    " The overall goal is to ensure the fishers community survive through"
    " sustainably approapriating resources while never exceeding the total pool"
    " of resources and while also maximising the amount of fish caught among"
    " the fishers and distributed fairly.\n"
)


def prompt_leader_agenda_clear_explain(
    model: ModelWandbWrapper, init_persona
) -> tuple[str, str]:
  """Clear and explained agenda prompt."""
  lm = model.start_chain(init_persona.agent_id, "leader_agenda", "get_agenda")

  with user():
    lm += (
        "\nTask: As a leader known for deep and comprehensive reasoning whose"
        " explanations are concise and clear provide a concise agenda in 2–3"
        " sentences that summarizes your detailed strategy."
    ) + GOAL_STATEMENT
    lm += 'My agenda as mayor: [fill in]\n"'

  with assistant():
    lm = model.gen(
        lm,
        "agenda",
        stop_regex=r"\n",
        save_stop_text=True,
    )
    agenda = lm["agenda"].strip()
    if agenda[0] == '"' and agenda[-1] == '"':
      agenda = agenda[1:-1]

  model.end_chain(init_persona.agent_id, lm)
  return agenda, lm.html()


def prompt_leader_agenda_clear_direct(
    model: ModelWandbWrapper, init_persona
) -> tuple[str, str]:
  """Clear and direct agenda prompt."""
  lm = model.start_chain(init_persona.agent_id, "leader_agenda", "get_agenda")

  with user():
    lm += (
        "\nTask:As a leader known for clear and concise communication though"
        " your analysis is less detailed  provide a concise agenda in 2–3"
        " sentences that summarizes your detailed strategy."
    ) + GOAL_STATEMENT
    lm += 'My agenda as mayor: [fill in]\n"'

  with assistant():
    lm = model.gen(
        lm,
        "agenda",
        stop_regex=r"\n",
        save_stop_text=True,
    )
    agenda = lm["agenda"].strip()
    if agenda[0] == '"' and agenda[-1] == '"':
      agenda = agenda[1:-1]

  model.end_chain(init_persona.agent_id, lm)
  return agenda, lm.html()


def prompt_leader_agenda_verbose_direct(
    model: ModelWandbWrapper, init_persona
) -> tuple[str, str]:
  """Verbose and direct agenda prompt."""
  lm = model.start_chain(init_persona.agent_id, "leader_agenda", "get_agenda")

  with user():
    lm += (
        "\nTask::As a leader known for unclear and verbose communication whose"
        " analysis is less detailed  provide a concise agenda in 2–3 sentences"
        " that summarizes your detailed strategy."
    ) + GOAL_STATEMENT
    lm += 'My agenda as mayor: [fill in]\n"'

  with assistant():
    lm = model.gen(
        lm,
        "agenda",
        stop_regex=r"\n",
        save_stop_text=True,
    )
    agenda = lm["agenda"].strip()
    if agenda[0] == '"' and agenda[-1] == '"':
      agenda = agenda[1:-1]

  model.end_chain(init_persona.agent_id, lm)
  return agenda, lm.html()


def prompt_leader_agenda_verbose_explain(
    model: ModelWandbWrapper, init_persona
) -> tuple[str, str]:
  """Verbose and explained agenda prompt."""
  lm = model.start_chain(init_persona.agent_id, "leader_agenda", "get_agenda")

  with user():
    lm += (
        "\nTask: As a leader known for unclear and verbose communication whose"
        " analysis is very detailed provide a concise agenda in 2–3 sentences"
        " that summarizes your detailed strategy"
    ) + GOAL_STATEMENT
    lm += 'My agenda as mayor: [fill in]\n"'

  with assistant():
    lm = model.gen(
        lm,
        "agenda",
        stop_regex=r"\n",
        save_stop_text=True,
    )
    agenda = lm["agenda"].strip()
    if agenda[0] == '"' and agenda[-1] == '"':
      agenda = agenda[1:-1]

  model.end_chain(init_persona.agent_id, lm)
  return agenda, lm.html()

