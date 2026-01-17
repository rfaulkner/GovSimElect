"""Conversation prompts and responses for the fishing personas."""

from datetime import datetime
import os

from pathfinder import assistant
from pathfinder import user
from simulation.persona import PersonaAgent
from simulation.scenarios.fishing.agents.persona_v3.cognition import leaders as leaders_lib
from simulation.utils import ModelWandbWrapper

from .utils import COGNITION_RESPONSES_JSON
from .utils import conversation_to_string_with_dash
from .utils import get_sytem_prompt
from .utils import list_to_comma_string
from .utils import location_time_info
from .utils import log_to_file
from .utils import memory_prompt



def prompt_converse_utterance_in_group(
    model: ModelWandbWrapper,
    init_persona: PersonaAgent,
    target_personas: list[PersonaAgent],
    init_retrieved_memory: list[str],
    current_location: str,
    current_time: datetime,
    current_context: str,
    current_conversation: list[tuple[str, str]],
    debug: bool = False,
) -> tuple[str, bool, str]:
  """Prompt for the next utterance in a group chat."""
  lm = model.start_chain(
      init_persona.identity.name, "cognition_converse", "converse_utterance"
  )
  svo_prompt, disinfo_prompt, leader_prompt = (
      leaders_lib.get_leader_persona_prompts(init_persona)
  )
  with user():
    lm += f"{get_sytem_prompt(init_persona.identity)}\n"
    lm += location_time_info(current_location, current_time)
    # List key memories of the initial persona
    lm += memory_prompt(init_persona.identity, init_retrieved_memory)
    # Provide the current context
    lm += "\n"
    lm += f"Current context: {current_context}\n\n"
    # Describe the group chat scenario
    lm += (
        "Scenario:"
        f" {list_to_comma_string([t.identity.name for t in target_personas])}"
        " are engaged in a group chat."
    )
    if svo_prompt:
      lm += f"{svo_prompt}\n"
    lm += f"{disinfo_prompt}\n"
    lm += f"{leader_prompt}\n"
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
      print(f"\n\nCONVERSE PROMPT:\n\n{lm._current_prompt()}\n")  # pylint: disable=protected-access

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
    if utterance and utterance[-1] == '"' and utterance[0] == '"':
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
      options = [t.identity.name for t in target_personas]
      lm = model.select(
          lm,
          name="next_speaker",
          options=options,
          default_value=options[0],
      )
      assert lm["next_speaker"] in options
      next_speaker = lm["next_speaker"]

    response_log_path = os.path.join(
        init_persona.experiment_storage, COGNITION_RESPONSES_JSON
    )
    log_to_file(
        log_type="converse_response",
        data={
            "speaker": init_persona.identity.name,
            "svo": init_persona.svo_type,
            "utterance": utterance,
            "utterance_ended": utterance_ended,
            "next_speaker": next_speaker,
        },
        log_path=response_log_path,
    )
    if debug:
      print(
          f"\n\nCONVERSE RESPONSE:\n\n{utterance}\nIS ENDED?"
          f" {utterance_ended}\nNEXT SPEAKER: {next_speaker}\n"
      )

  model.end_chain(init_persona.identity.name, lm)
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
    lm += "Conversation:\n"
    lm += f"{conversation_to_string_with_dash(conversation)}\n\n"
    lm += "Summarize the conversation above in one sentence."
  with assistant():
    lm = model.gen(lm, name="summary", default_value="", stop_regex=r"\.")
    summary = lm["summary"] + "."

  model.end_chain("framework", lm)
  return summary, lm.html()

