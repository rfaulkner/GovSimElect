"""Act cognition component — handles harvesting and election decisions."""

from datetime import datetime

from simulation.utils import ModelWandbWrapper

from .component import Component
from .act_prompts import (
    prompt_action_choose_amount_of_fish_to_catch,
    prompt_election_vote,
)
from .utils import get_universalization_prompt


class ActComponent(Component):

    def __init__(
        self, model: ModelWandbWrapper, model_framework: ModelWandbWrapper, cfg=None
    ):
        super().__init__(model, model_framework, cfg)

    def choose_how_many_fish_to_catch(
        self,
        retrieved_memories: list[str],
        current_location: str,
        current_time: datetime,
        context: str,
        interval: list[int],
        overusage_threshold: int,
        leader_agenda: str,
        debug: bool = False,
    ):
        if self.cfg.universalization_prompt:
            context += get_universalization_prompt(overusage_threshold)
        res, html = prompt_action_choose_amount_of_fish_to_catch(
            self.model,
            self.persona,
            retrieved_memories,
            current_location,
            current_time,
            context,
            interval,
            consider_identity_persona=self.cfg.consider_identity_persona,
            leader_agenda=leader_agenda,
            debug=debug,
        )
        res = int(res)
        return res, [html]

    def participate_in_election(
        self,
        retrieved_memories: list[str],
        current_location: str,
        current_time: str,
        candidates: list[str],
        leader_agendas: dict[str, str],
        debug: bool = False,
    ) -> tuple[str, list[str]]:
        vote, html = prompt_election_vote(
            self.model,
            self.persona,
            retrieved_memories,
            current_location,
            current_time,
            candidates,
            leader_agendas,
            debug=debug,
        )
        return vote, [html]
