"""Reflect prompts for persona cognition."""

from pathfinder import assistant
from pathfinder import user
from simulation.utils import models as sim_models
from simulation.persona import common
from simulation.persona.cognition import utils as cognition_utils


def prompt_insight_and_evidence(
    model: sim_models.ModelWandbWrapper,
    persona: common.PersonaIdentity,
    statements: list[str],
):
  """Generate insights from a set of statements."""
  lm, chain = model.start_chain_async(
      persona.name,
      "cognition_retrieve",
      "prompt_insight_and_evidence",
  )

  with user():
    lm += f"{cognition_utils.get_sytem_prompt(persona)}\n"
    lm += f"{cognition_utils.numbered_memory_prompt(persona, statements)}\n"
    lm += (
        f"What high-level insights can you infere from the above"
        f" statements? (example format: insight (because of 1,5,3)"
    )
  with assistant():
    acc = []
    lm += f"1."
    for i in range(len(statements)):
      lm = model.gen(
          lm,
          name=f"evidence_{i}",
          stop_regex=rf"{i+2}\.|\\(",
          save_stop_text=True,
          chain=chain,
      )
      if lm[f"evidence_{i}"].endswith(f"{i+2}."):
        evidence = lm[f"evidence_{i}"][: -len(f"{i+2}.")]
        acc.append(evidence.strip())
        continue
      else:
        evidence = lm[f"evidence_{i}"]
        if evidence.endswith("("):
          evidence = lm[f"evidence_{i}"][: -len("(")]
        lm = model.gen(
            lm,
            name=f"evidence_{i}_justification",
            stop_regex=rf"{i+2}\.",
            save_stop_text=True,
            chain=chain,
        )
        if lm[f"evidence_{i}_justification"].endswith(f"{i+2}."):
          acc.append(evidence.strip())
          continue
        else:
          acc.append(evidence.strip())
          break
    model.end_chain(persona.name, lm, chain=chain)

  return acc


def prompt_planning_thought_on_conversation(
    model: sim_models.ModelWandbWrapper,
    persona: common.PersonaIdentity,
    conversation: list[tuple[str, str]],
) -> str:
  """Generate a planning thought from a conversation."""
  lm, chain = model.start_chain_async(
      persona.name,
      "cognition_retrieve",
      "prompt_planning_thought_on_conversation",
  )

  with user():
    lm += f"{cognition_utils.get_sytem_prompt(persona)}\n"
    lm += f"Conversation:\n"
    lm += f"{cognition_utils.conversation_to_string_with_dash(conversation)}\n"
    lm += (
        f"Write down if there is anything from the conversation that"
        f" you need to remember for your planning, from your own"
        f" perspective, in a full sentence."
    )
  with assistant():
    lm = model.gen(lm, name="planning_thought", stop_regex=r"\.", chain=chain)
    res = lm["planning_thought"]

  model.end_chain(persona.name, lm, chain=chain)
  return res


def prompt_memorize_from_conversation(
    model: sim_models.ModelWandbWrapper,
    persona: common.PersonaIdentity,
    conversation: list[tuple[str, str]],
) -> str:
  """Memorize interesting points from a conversation."""
  lm, chain = model.start_chain_async(
      persona.name,
      "cognition_retrieve",
      "prompt_memorize_from_conversation",
  )

  with user():
    lm += f"{cognition_utils.get_sytem_prompt(persona)}\n"
    lm += f"Conversation:\n"
    lm += f"{cognition_utils.conversation_to_string_with_dash(conversation)}\n"
    lm += (
        f" Write down if there is anything from the conversation that"
        f" you might have found interesting from your own"
        f" perspective, in a full sentence."
    )
  with assistant():
    lm = model.gen(lm, name="memorize", stop_regex=r"\.", chain=chain)
    res = lm["memorize"]

  model.end_chain(persona.name, lm, chain=chain)
  return res


def prompt_find_harvesting_limit_from_conversation(
    model: sim_models.ModelWandbWrapper,
    conversation: list[tuple[str, str]],
) -> tuple[int, str]:
  """Find the harvesting limit agreed upon in a conversation."""
  lm, chain = model.start_chain_async(
      "framework",
      "cognition_refelct",
      "prompt_find_harvesting_limit_from_conversation",
  )

  with user():
    lm += (
        "In the following conversation, the participants discuss"
        " their fishing activities and activities and the weight"
        " of fish they caught. Determine whether there was an"
        " explicit agreement on a concrete fishing limit. Look"
        " for direct mention or agreement on a numerical catch"
        " limit that the group agreed to keep during this"
        " conversation."
    )
    lm += f"\n\nConversation:\n"
    lm += f"{cognition_utils.conversation_to_string_with_dash(conversation)}\n"
    lm += (
        "Please provide the specific fishing limit per person as"
        " agreed upon in the conversation, if no limit was agreed"
        " upon, please answer N/A. "
    )
    lm += cognition_utils.reasoning_steps_prompt()
    lm += ' Put the final answer after "Answer:".'

  option_fish_num = range(0, 101)
  with assistant():
    lm = model.gen(
        lm,
        "reasoning",
        stop_regex=f"Answer:",
        chain=chain,
    )
    lm += f"Answer: "
    lm = model.find(
        lm,
        regex=r"\d+",
        default_value="-1",
        name="num_resource",
        chain=chain,
    )

    resource_limit_agreed = int(lm["num_resource"]) != -1

    if resource_limit_agreed:
      res = int(lm["num_resource"])
      model.end_chain("framework", lm, chain=chain)
      return res, lm.html()
    else:
      model.end_chain("framework", lm, chain=chain)
      return None, lm.html()

