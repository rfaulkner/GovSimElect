"""Converse cognition component — handles group conversations."""

import datetime
import random

from simulation.utils import models as sim_models
from simulation.persona import common
from simulation.persona.cognition import component
from simulation.persona.cognition import converse_prompts
from simulation.persona.cognition import reflect_prompts
from simulation.persona.cognition import retrieve as retrieve_mod


class ConverseComponent(component.Component):
  """Handles group conversation cognition."""

  def __init__(
      self,
      model: sim_models.ModelWandbWrapper,
      model_framework: sim_models.ModelWandbWrapper,
      retrieve: retrieve_mod.RetrieveComponent,
      cfg=None,
  ):
    """Initialize the converse component."""
    super().__init__(model, model_framework, cfg)
    self.retrieve = retrieve

  def converse_group(
      self,
      target_personas,
      current_location: str,
      current_time: datetime.datetime,
      current_context: str,
      agent_resource_num: dict[str, int],
      mayoral_agenda: str | None = None,
      harvest_report: str | None = None,
      leader_persona=None,
      debug: bool = False,
  ) -> tuple[list[tuple[str, str]], str]:
    """Conduct a group conversation among personas."""
    current_conversation: list[
        tuple[common.PersonaIdentity, str]
    ] = []

    html_interactions = []
    if leader_persona:
      current_leader_id = leader_persona.identity
    else:
      current_leader_id = common.PersonaIdentity(
          "framework", "Anonymous Leader",
      )

    if (
        self.cfg.inject_resource_observation
        and self.cfg.inject_resource_observation_strategy
        == "individual"
    ):
      for persona in target_personas:
        p = self.other_personas[persona.identity.name]
        current_conversation.append(
            (
                p.identity,
                (
                    "This month, I caught"
                    f" {agent_resource_num[p.agent_id]}"
                    " tons of fish!"
                ),
            ),
        )
        html_interactions.append(
            "<strong>Framework</strong>:  This month, I caught"
            f" {agent_resource_num[p.agent_id]} tons of fish!"
        )
    elif (
        self.cfg.inject_resource_observation
        and self.cfg.inject_resource_observation_strategy
        == "manager"
    ):
      current_conversation.append(
          (
              current_leader_id,
              (
                  f"I, {current_leader_id.name}, will lead"
                  " the group this cycle. Fellow citizens,"
                  " let me give you the monthly fishing"
                  f" report:\n{harvest_report}"
              ),
          ),
      )
      if mayoral_agenda and leader_persona:
        current_conversation.append(
            (
                current_leader_id,
                (
                    "I'd also like to share my"
                    " policy agenda to help"
                    " guide our collective"
                    f" action: {mayoral_agenda}"
                ),
            ),
        )

    max_conversation_steps = self.cfg.max_conversation_steps

    if leader_persona:
      current_persona = leader_persona
    else:
      current_persona = random.choice(target_personas)

    while True:
      focal_points = [current_context]
      if current_conversation:
        for _, utterance in current_conversation[-4:]:
          focal_points.append(utterance)

        if len(current_conversation) == 4:
          focal_points += current_conversation[0:1]
        elif len(current_conversation) > 5:
          focal_points += current_conversation[0:2]
      focal_points = self.other_personas[
          current_persona.identity.name
      ].retrieve.retrieve(focal_points, top_k=5)

      if self.cfg.prompt_utterance == "one_shot":
        prompt = (
            converse_prompts
            .prompt_converse_utterance_in_group
        )
      else:
        raise NotImplementedError(
            "prompt_utterance="
            f"{self.cfg.prompt_utterance}"
        )

      utterance, end_conversation, next_name, h = prompt(
          self.model,
          current_persona,
          target_personas,
          focal_points,
          current_location,
          current_time,
          current_context,
          self.conversation_render(current_conversation),
          debug,
      )
      html_interactions.append(h)

      current_conversation.append(
          (current_persona.identity, utterance),
      )

      if (
          end_conversation
          or len(current_conversation)
          >= max_conversation_steps
      ):
        break
      else:
        current_persona = self.other_personas[next_name]

    summary_conversation, h = (
        converse_prompts
        .prompt_summarize_conversation_in_one_sentence(
            self.model_framework,
            self.conversation_render(current_conversation),
        )
    )
    html_interactions.append(h)

    resource_limit, h = (
        reflect_prompts
        .prompt_find_harvesting_limit_from_conversation(
            self.model_framework,
            self.conversation_render(current_conversation),
        )
    )
    html_interactions.append(h)
    for persona in target_personas:
      p = self.other_personas[persona.identity.name]
      p.store.store_chat(
          summary_conversation,
          self.conversation_render(current_conversation),
          self.persona.current_time,
      )
      p.reflect.reflect_on_convesation(
          self.conversation_render(current_conversation),
      )
      if resource_limit is not None:
        p.store.store_thought(
            (
                "The community agreed on a maximum limit"
                f" of {resource_limit} tons of fish per"
                " person."
            ),
            self.persona.current_time,
            always_include=True,
        )
    if debug:
      print(
          "CONVERSATION TRANSCRIPT:"
          f" {self.conversation_render(current_conversation)}."
      )
    return (
        current_conversation,
        summary_conversation,
        resource_limit,
        html_interactions,
    )

  def conversation_render(
      self,
      conversation: list[
          tuple[common.PersonaIdentity, str]
      ],
  ):
    """Render conversation as name-utterance tuples."""
    return [(p.name, u) for p, u in conversation]
