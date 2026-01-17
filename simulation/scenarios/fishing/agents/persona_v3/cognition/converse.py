"""Handling conversation among the personas."""

import random

from typing import Optional
from datetime import datetime

from simulation.persona import PersonaAgent
from simulation.persona.cognition.converse import ConverseComponent
from simulation.persona.cognition.retrieve import RetrieveComponent
from simulation.persona.common import PersonaIdentity
from simulation.utils import ModelWandbWrapper

from .converse_prompts import (
    prompt_converse_utterance_in_group,
    prompt_summarize_conversation_in_one_sentence,
)
from .reflect_prompts import prompt_find_harvesting_limit_from_conversation


class FishingConverseComponent(ConverseComponent):
  """Fishing converse component."""

  def converse_group(
      self,
      target_personas: list[PersonaAgent],
      current_location: str,
      current_time: datetime,
      current_context: str,
      agent_resource_num: dict[str, int],
      mayoral_agenda: str | None = None,
      harvest_report: str | None = None,
      leader_persona: PersonaAgent | None = None,
      debug: bool = False,
  ) -> tuple[list[tuple[str, str]], str]:
    current_conversation: list[tuple[PersonaIdentity, str]] = []

    html_interactions = []
    if leader_persona:
      current_leader_id = leader_persona.identity
    else:
      current_leader_id = PersonaIdentity("framework", "Anonymous Leader")

    # Inject fake conversation about how many fish each person caught
    if (
        self.cfg.inject_resource_observation
        and self.cfg.inject_resource_observation_strategy == "individual"
    ):
      for persona in target_personas:
        p = self.other_personas[persona.identity.name]
        current_conversation.append(
            (
                p.identity,
                (
                    "This month, I caught"
                    f" {agent_resource_num[p.agent_id]} tons of fish!"
                ),
            ),
        )
        html_interactions.append(
            "<strong>Framework</strong>:  This month, I caught"
            f" {agent_resource_num[p.agent_id]} tons of fish!"
        )
    elif (
        self.cfg.inject_resource_observation
        and self.cfg.inject_resource_observation_strategy == "manager"
    ):
      current_conversation.append(
          (
              current_leader_id,
              (
                  f"I, {current_leader_id.name}, will lead the group this"
                  " cycle. Fellow citizens, let me give you the monthly"
                  f" fishing report:\n{harvest_report}"
              ),
          ),
      )
      if mayoral_agenda and leader_persona:
        current_conversation.append(
            (
                current_leader_id,
                (
                    "I'd also like to share my policy agenda to help guide our "
                    f"collective action: {mayoral_agenda}"
                ),
            ),
        )

    max_conversation_steps = self.cfg.max_conversation_steps

    # Always open with the leader if there is one, otherwise choose randomly.
    if leader_persona:
      current_persona = leader_persona
    else:
      current_persona = random.choice(target_personas)

    while True:
      focal_points = [current_context]
      if current_conversation:
        # Last 4 utterances
        for _, utterance in current_conversation[-4:]:
          focal_points.append(utterance)

        # Always include the first two utterances by the leader.
        if len(current_conversation) == 4:
          focal_points += current_conversation[0:1]
        elif len(current_conversation) > 5:
          focal_points += current_conversation[0:2]
      focal_points = self.other_personas[
          current_persona.identity.name
      ].retrieve.retrieve(focal_points, top_k=5)

      if self.cfg.prompt_utterance == "one_shot":
        prompt = prompt_converse_utterance_in_group
      else:
        raise NotImplementedError(
            f"prompt_utterance={self.cfg.prompt_utterance}"
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
          self.cfg.debug,
      )
      html_interactions.append(h)

      current_conversation.append((current_persona.identity, utterance))

      if (
          end_conversation
          or len(current_conversation) >= max_conversation_steps
      ):
        break
      else:
        current_persona = self.other_personas[next_name]

    summary_conversation, h = prompt_summarize_conversation_in_one_sentence(
        self.model_framework, self.conversation_render(current_conversation)
    )
    html_interactions.append(h)

    resource_limit, h = prompt_find_harvesting_limit_from_conversation(
        self.model_framework, self.conversation_render(current_conversation)
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
          self.conversation_render(current_conversation)
      )
      if resource_limit is not None:
        p.store.store_thought(
            (
                "The community agreed on a maximum limit of"
                f" {resource_limit} tons of fish per person."
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
      self, conversation: list[tuple[PersonaIdentity, str]]
  ):
    return [(p.name, u) for p, u in conversation]

