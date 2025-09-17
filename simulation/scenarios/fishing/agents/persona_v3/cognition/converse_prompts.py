"""Conversation prompts and responses for the fishing personas."""

from datetime import datetime

from pathfinder import assistant, user
from simulation.persona.common import PersonaIdentity
from simulation.utils import ModelWandbWrapper

from .utils import (
    conversation_to_string_with_dash,
    get_sytem_prompt,
    list_to_comma_string,
    location_time_info,
    memory_prompt,
)


def prompt_converse_utterance_in_group(
    model: ModelWandbWrapper,
    init_persona: PersonaIdentity,
    target_personas: list[PersonaIdentity],
    init_retrieved_memory: list[str],
    current_location: str,
    current_time: datetime,
    current_context: str,
    current_conversation: list[tuple[str, str]],
    debug: bool = False,
) -> tuple[str, bool, str]:
  """Prompt for the next utterance in a group chat."""
  lm = model.start_chain(
      init_persona.name, "cognition_converse", "converse_utterance"
  )

  with user():
    lm += f"{get_sytem_prompt(init_persona)}\n"
    lm += location_time_info(current_location, current_time)
    # List key memories of the initial persona
    lm += memory_prompt(init_persona, init_retrieved_memory)
    # Provide the current context
    lm += "\n"
    lm += f"Current context: {current_context}\n\n"
    # Describe the group chat scenario
    lm += (
        "Scenario:"
        f" {list_to_comma_string([t.name for t in target_personas])} are"
        " engaged in a group chat."
    )
    lm += "\nConversation so far:\n"
    lm += f"{conversation_to_string_with_dash(current_conversation)}\n\n"
    # Define the task for the language model
    lm += (
        "Task: What would you say next in the group chat? Ensure the"
        " conversation flows naturally and avoids repetition. Try to highlight"
        " qualities that are important to the group, and reference any speakers"
        " that you believe provided sound plans that could be useful to the"
        " group. Also ensure that you consider the survival of the community by"
        " Determine if your response concludes the conversation. If not,"
        " identify the next speaker.\n\n"
    )
    # Define the format for the output
    response = "Response: "
    answer_stop = "Conversation conclusion by me: "
    next_speaker = "Next speaker: "

    lm += "Output format:\n"
    lm += response + "[fill in]\n"
    lm += answer_stop + "[yes/no]\n"
    lm += next_speaker + "[fill in]\n"
    if debug:
      print(f"\n\nCONVERSE PROMPT:\n\n{lm._current_prompt()}\n")

  with assistant():
    lm += response
    lm = model.gen(
        lm,
        name="utterance",
        default_value="",
        # name can be mispelled by LLM sometimes.
        stop_regex=r"Conversation conclusion by me:",
    )
    utterance = lm["utterance"].strip()
    if len(utterance) > 0 and utterance[-1] == '"' and utterance[0] == '"':
      utterance = utterance[1:-1]
    lm += answer_stop
    lm = model.select(
        lm,
        name="utterance_ended",
        options=["yes", "no", "No", "Yes"],
        default_value="yes",
    )
    utterance_ended = lm["utterance_ended"].lower() == "yes"

    if utterance_ended:
      next_speaker = None
    else:
      lm += "\n"
      lm += next_speaker
      options = [t.name for t in target_personas]
      lm = model.select(
          lm,
          name="next_speaker",
          options=options,
          default_value=options[0],
      )
      assert lm["next_speaker"] in options
      next_speaker = lm["next_speaker"]

    if debug:
      print(
          f"\n\nCONVERSE RESPONSE:\n\n{utterance}\nIS ENDED?"
          f" {utterance_ended}\nNEXT SPEAKER: {next_speaker}\n"
      )

  model.end_chain(init_persona.name, lm)
  return utterance, utterance_ended, next_speaker, lm.html()


def prompt_summarize_conversation_in_one_sentence(
    model: ModelWandbWrapper,
    conversation: list[tuple[str, str]],
):
  """Summarize the conversation in one sentence."""
  lm = model.start_chain(
      "framework",
      "cognition_converse",
      "prompt_summarize_conversation_in_one_sentence",
  )

  with user():
    lm += f"Conversation:\n"
    lm += f"{conversation_to_string_with_dash(conversation)}\n\n"
    lm += "Summarize the conversation above in one sentence."
  with assistant():
    lm = model.gen(lm, name="summary", default_value="", stop_regex=r"\.")
    summary = lm["summary"] + "."

  model.end_chain("framework", lm)
  return summary, lm.html()

