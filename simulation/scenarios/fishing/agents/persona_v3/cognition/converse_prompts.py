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
) -> tuple[str, bool, str]:
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
    # lm += (
    #     "Task: What would you say next in the group chat? "
    #     "Ensure the conversation flows naturally and avoids repetition. "
    #     "Try to demonstrate qualities that are important to the group, "
    #     "such as honesty, sound judgement and leadership."
    #     "Also ensure that you consider the survival of the community by "
    #     "avoiding over fishing."
    #     "If you demonstrate these qualities to a high degree then your "
    #     "fellow fishers may bestow you with honour of being mayor in "
    #     "the next cycle."
    #     "Determine if your response concludes the conversation. "
    #     "If not, identify the next speaker.\n\n"
    # )
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
    REPONSE = "Response: "
    ANSWER_STOP = "Conversation conclusion by me: "
    NEXT_SPEAKER = "Next speaker: "

    lm += "Output format:\n"
    lm += REPONSE + "[fill in]\n"
    lm += ANSWER_STOP + "[yes/no]\n"
    lm += NEXT_SPEAKER + "[fill in]\n"
  with assistant():
    lm += REPONSE
    lm = model.gen(
        lm,
        name="utterance",
        default_value="",
        stop_regex=r"Conversation conclusion by me:",  # name can be mispelled by LLM sometimes
    )
    utterance = lm["utterance"].strip()
    if len(utterance) > 0 and utterance[-1] == '"' and utterance[0] == '"':
      utterance = utterance[1:-1]
    lm += ANSWER_STOP
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
      lm += NEXT_SPEAKER
      options = [t.name for t in target_personas]
      lm = model.select(
          lm,
          name="next_speaker",
          options=options,
          default_value=options[0],
      )
      assert lm["next_speaker"] in options
      next_speaker = lm["next_speaker"]

  model.end_chain(init_persona.name, lm)
  return utterance, utterance_ended, next_speaker, lm.html()


def prompt_summarize_conversation_in_one_sentence(
    model: ModelWandbWrapper,
    conversation: list[tuple[str, str]],
):
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


def prompt_elect_mayor_from_conversation(
    model: ModelWandbWrapper,
    conversation: list[tuple[str, str]],
    persona_name: str,
    other_persona_names: str,
) -> tuple[str, str]:
  """Post-conversation prompt for selecting the mayor in the next cycle.

  Args:
      model: The target LLM in a weights and biases wrapper.
      conversation: The conversation to summarize.
      persona_name: The name of the persona that is currently speaking.
      other_persona_names: The names of the other personas in the conversation.

  Returns:
      Tuple of the summary + model response.
  """

  lm = model.start_chain(
      "framework",
      "cognition_converse",
      "prompt_elect_mayor_from_conversation",
  )

  with user():
    lm += "To recap, here the full conversation from this cycle:\n"
    lm += f"{conversation_to_string_with_dash(conversation)}\n\n"

    # Define the mayoral election for the model.
    lm += (
        # List the candidates?
        f"Task: You are {persona_name}, cast a vote for the mayor in the "
        "next cycle among one of the other candidates: "
        f"{other_persona_names}\nObserve the characteristics of the "
        "other candidates in the conversation from this cycle and try to "
        "observe which candidates exhibit the most leadershipship, "
        "honesty, and sound judgement in order to choose the best "
        "candidate for the next cycle. The best mayor will ensure "
        "everyone receives a fair amount of resources without over fishing."
    )
    # Define the format for the output
    candidate = f"Vote among {other_persona_names}: "
    rationale = "Provide a rationale for voting for this candidate: "

    lm += "Output format:\n"
    lm += candidate + "[fill in]\n"
    lm += rationale + "[fill in]\n"
  with assistant():
    lm = model.gen(lm, name="vote", default_value="", stop_regex=r"\.")
    vote = lm["vote"] + "."

  model.end_chain("framework", lm)
  return vote, lm.html()


def prompt_tally_votes(
    model: ModelWandbWrapper,
    votes: list[str],
) -> tuple[int, str]:
  """Tally votes for the mayor in the next cycle.

  Args:
      model: The target LLM in a weights and biases wrapper.
      votes: The conversation to summarize.

  Returns:
      Tuple of the summary + model response.
  """

  lm = model.start_chain(
      "framework",
      "cognition_converse",
      "prompt_tally_election",
  )

  with user():
    lm += "Here are the voting responses from each person in the group:\n"
    voting_responses = ""
    for i, vote in enumerate(votes):
      voting_responses += f"Vote-{i}: {vote}\n"
    lm += f"{voting_responses}\n\n"

    # Define the mayoral election for the model.
    lm += (
        # List the candidates?
        "Task: Count all the votes for the next mayor and announce the"
        " winner. In the case of a tie, announce the tie but choose the "
        "winner randomly from the tied candidates."
    )
    lm += "The mayor for the next cycle is: [fill in]\n"
  with assistant():
    lm = model.gen(lm, name="vote_result", default_value="", stop_regex=r"\.")
    vote_result = lm["vote_result"] + "."

  model.end_chain("framework", lm)
  return vote_result, lm.html()

