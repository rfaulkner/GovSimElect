"""Utilities for fishers personas."""

from ......persona.common import PersonaIdentity


def list_to_string_with_dash(list_of_strings: list[str]) -> str:
  res = ""
  for s in list_of_strings:
    res += f"- {s}\n"
  return res


def conversation_to_string_with_dash(
    conversation: list[tuple[str, str]],
) -> str:
  res = ""
  for i, (speaker, utterance) in enumerate(conversation):
    res += f"-{speaker}: {utterance}\n"
  return res


def list_to_comma_string(list_of_strings: list[str]) -> str:
  res = ""
  for i, s in enumerate(list_of_strings):
    if i == 0:
      res += s
    elif i == len(list_of_strings) - 1:
      res += f", and {s}"
    else:
      res += f", {s}"
  return res


def numbered_list_of_strings(list_of_strings: list[str]) -> str:
  res = ""
  for i, s in enumerate(list_of_strings):
    res += f"{i+1}) {s}\n"
  return res


def consider_identity_persona_prompt(identity: PersonaIdentity) -> str:
  """Consider the persona's behaviour and key memories."""
  return (
      f"The answer should consider {identity.name}'s persona (background,"
      " goals, behavior, customs) and his key memories."
  )


from datetime import datetime


def memory_prompt(
    identity: PersonaIdentity, memories: list[tuple[datetime, str]]
) -> str:
  """Key memories of the persona."""
  res = ""
  for s in memories:
    res += f"- {s[0].strftime('%Y-%m-%d')}: {s[1]}\n"

  return (
      f"Key memories of {identity.name} (format: YYYY-MM-DD: memory):\n{res}\n"
  )


def numbered_memory_prompt(
    identity: PersonaIdentity, memories: list[tuple[datetime, str]]
) -> str:
  """Key memories of persona, nembered."""
  res = ""
  for i, s in enumerate(memories):
    res += f"{i+1}) {s[0].strftime('%Y-%m-%d')}: {s[1]}\n"

  return (
      f"Key memories of {identity.name} (format: YYYY-MM-DD: memory):\n{res}\n"
  )


REASONING = "think_step_by_step"


def reasoning_steps_prompt() -> str:
  """Respond with step-by-step reasoning."""
  if REASONING == "think_step_by_step":
    return "Let's think step-by-step."
  elif REASONING == "deep_breath":
    return "Take a deep breath and work on this problem step-by-step."
  else:
    raise ValueError(f"Unknown REASONING: {REASONING}")


def location_time_info(current_location, current_time):
  return (
      f"Location: {current_location}\nDate:"
      f" {current_time.strftime('%Y-%m-%d')}\n\n"
  )


def get_universalization_prompt(sustainability_threshold):
  return (
      " Given the"
      " current situation, if everyone fishes more than"
      f" {sustainability_threshold} every month, the lake will eventually be"
      " empty."
  )


SYS_VERSION = "v1"


def get_sytem_prompt(persona):
  if SYS_VERSION == "v1":
    return get_sytem_prompt_v1(persona)
  elif SYS_VERSION == "v3":
    return get_sytem_prompt_v3(persona)
  elif SYS_VERSION == "v3_nocom":
    return get_sytem_prompt_v3_nocom(persona)
  else:
    raise ValueError(f"Unknown SYS_VERSION: {SYS_VERSION}")


def get_sytem_prompt_v1(persona):
  text = (
      f"You are {persona.name}, a fisherman, and you fish each month in a lake"
      " along with other fishermen. The lake has a population capacity of 100"
      " tons of fish. Each month, each fisherman decides how many fish to take"
      " out of the lake between 0 and 100 tons. At the end of the month, the"
      " remaining fish reproduce and double up to the population capacity. So"
      " if there are 60 tons of fish remaining, there will be 100 tons after"
      " reproduction, or if there are 20 tons of fish remaining, there will be"
      " 40 tons of fish after reproduction. For each ton of fish caught, a"
      " fisherman earns one thousand dollars. Each fisherman's goal is to"
      " optimize their income in the long run over many months. At the end of"
      " each month, the number of fish caught by each fisherman will be"
      " revealed, and the fishermen will have an opportunity to communicate"
      " with each other. You can use this as an opportunity to negotiate and"
      " persuade others to influence their behavior in the next month."
  )

  if persona.goals != "":
    text += persona.goals

  return text


def get_sytem_prompt_v3(persona):
  """System prompt for v3."""

  if persona.env:
    min_regen = persona.env.regen_min_range
    max_regen = persona.env.regen_max_range
  else:
    min_regen = 2.0
    max_regen = 2.0

  if min_regen != max_regen:
    text_regen_rate = (
        " At the end of the month, the remaining fish will reproduce and the"
        " population will increase.  The regeneration factor of increase is"
        f" sampled uniformly between {min_regen} and {max_regen} for each"
        " month, up to the carrying capacity.."
    )
  else:
    text_regen_rate = (
        " At the end of the month, the remaining fish will reproduce and"
        f" increase by a regeneration factor of {min_regen}, up to the carrying"
        " capacity."
    )

  text_pre = (
      f"You are {persona.name}, a fisherman, and you fish every month in a lake"
      " along with other fishermen. The lake has a carrying capacity of 100"
      " tons of fish. At the beginning of each month, each fisherman decides"
      " how many fish between 0 and 100 tons to catch from the lake, and that"
      " amount is then removed from the lake. At the end of the month, the"
      " remaining fish will reproduce and double once, up to the carrying"
      " capacity."
  )

  text_post = (
      " For every ton of fish caught, a fisherman earns one thousand dollars."
      " Each fisherman's goal is to optimize his income in the long run over"
      " many months. At the end of each month, the number of fish caught by"
      " each fisherman will be revealed, and the fishermen will have the"
      " opportunity to communicate with each other. They can use this as an"
      " opportunity to negotiate and persuade others to influence their"
      " behavior in the next month. For example, if there are 90 tons of fish"
      " at the beginning of the month and the fishermen catch a total of 30"
      " fish, given a regeneration factor of 2.0, there will be 60 tons of fish"
      " left at the end of the month before reproduction, and 100 tons after"
      " reproduction."
  )

  text = text_pre + text_regen_rate + text_post

  if persona.goals != "":
    text += persona.goals

  return text


def get_sytem_prompt_v3_nocom(persona):
  text = (
      f"You are {persona.name}, a fisherman, and you fish every month in a lake"
      " along with other fishermen. The lake has a carrying capacity of 100"
      " tons of fish. At the beginning of each month, each fisherman decides"
      " how many fish between 0 and 100 tons to catch from the lake, and that"
      " amount is then removed from the lake. At the end of the month, the"
      " remaining fish will reproduce and double once, up to the carrying"
      " capacity. For every ton of fish caught, a fisherman earns one thousand"
      " dollars. Each fisherman's goal is to optimize his income in the long"
      " run over many months. For example, if there are 90 tons of fish at the"
      " beginning of the month and the fishermen catch a total of 30 fish,"
      " there will be 60 tons of fish left at the end of the month before"
      " reproduction, and 100 tons after reproduction."
  )

  if persona.goals != "":
    text += persona.goals

  return text

