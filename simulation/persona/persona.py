"""PersonaAgent — single agent class for the fishing election simulation.

Holds agent state (identity, SVO, election data) and cognition components
(act, converse, reflect, store, retrieve, perceive, plan).

Phase-specific simulation logic lives in ``simulation.phases``
(see PolicyMakingPhase, ElectionPhase, HarvestingPhase, DiscussionPhase,
ReflectionPhase).  This class is a stateful container only.
"""

from datetime import datetime
import enum
import os

from simulation.persona.cognition.act import ActComponent
from simulation.persona.cognition.converse import ConverseComponent
from simulation.persona.cognition.perceive import PerceiveComponent
from simulation.persona.cognition.plan import PlanComponent
from simulation.persona.cognition.reflect import ReflectComponent
from simulation.persona.cognition.retrieve import RetrieveComponent
from simulation.persona.cognition.store import StoreComponent
from simulation.persona.common import PersonaIdentity
from simulation.persona.embedding_model import EmbeddingModel
from simulation.persona.memory.associative_memory import AssociativeMemory
from simulation.persona.memory.scratch import Scratch
from simulation.utils.models import ModelWandbWrapper


class SVOPersonaType(enum.Enum):
    NONE = 'none'
    INDIVIDUALISTIC = 'individualistic'
    COMPETITIVE = 'competitive'
    PROSOCIAL = 'prosocial'
    ALTRUISTIC = 'altruistic'


class PersonaEnvironment:
    def __init__(self, regen_min_range, regen_max_range):
        self.regen_min_range = regen_min_range
        self.regen_max_range = regen_max_range


DEFAULT_AGENDA = "No specific guidance."


class PersonaAgent:

    agent_id: int
    identity: PersonaIdentity
    current_time: datetime
    scratch: Scratch

    def __init__(
        self,
        cfg,
        model: ModelWandbWrapper,
        model_framework: ModelWandbWrapper,
        embedding_model: EmbeddingModel,
        base_path: str,
        memory_cls: type[AssociativeMemory] = AssociativeMemory,
        perceive_cls: type[PerceiveComponent] = PerceiveComponent,
        retrieve_cls: type[RetrieveComponent] = RetrieveComponent,
        store_cls: type[StoreComponent] = StoreComponent,
        reflect_cls: type[ReflectComponent] = ReflectComponent,
        plan_cls: type[PlanComponent] = PlanComponent,
        act_cls: type[ActComponent] = ActComponent,
        converse_cls: type[ConverseComponent] = ConverseComponent,
        svo_angle: float | None = None,
        svo_type: SVOPersonaType = SVOPersonaType.NONE,
        disinfo: bool = False,
        harvest_report: str | None = None,
        current_leader: "PersonaAgent | None" = None,
        experiment_storage: str | None = None,
    ) -> None:
        self.cfg = cfg
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)

        self.current_time = datetime.now()
        self.memory = memory_cls(base_path)
        self.perceive = perceive_cls(model, model_framework)
        self.retrieve = retrieve_cls(
            model, model_framework, self.memory, embedding_model
        )
        self.store = store_cls(
            model, model_framework, self.memory, embedding_model, self.cfg.store
        )
        self.reflect = reflect_cls(model, model_framework)
        self.plan = plan_cls(model, model_framework)
        self.act = act_cls(
            model,
            model_framework,
            self.cfg.act,
        )
        self.converse = converse_cls(
            model, model_framework, self.retrieve, self.cfg.converse
        )

        self.perceive.init_persona_ref(self)
        self.retrieve.init_persona_ref(self)
        self.store.init_persona_ref(self)
        self.reflect.init_persona_ref(self)
        self.plan.init_persona_ref(self)
        self.act.init_persona_ref(self)
        self.converse.init_persona_ref(self)

        self.other_personas: dict[str, "PersonaAgent"] = {}
        self.other_personas_from_id: dict[str, "PersonaAgent"] = {}

        self._agenda = DEFAULT_AGENDA
        self._overuse_threshold = None
        self._svo_angle = svo_angle
        self._svo_type = svo_type
        self._disinfo = disinfo
        self._harvest_report = harvest_report
        self._current_leader = current_leader
        self._experiment_storage = experiment_storage

    def init_persona(
        self, agent_id: int, identity: PersonaIdentity, social_graph
    ):
        self.agent_id = agent_id
        self.identity = identity
        self.scratch = Scratch(f"{self.base_path}")

    def add_reference_to_other_persona(self, persona: "PersonaAgent"):
        self.other_personas[persona.identity.name] = persona
        self.other_personas_from_id[persona.agent_id] = persona
        self.perceive.add_reference_to_other_persona(persona)
        self.retrieve.add_reference_to_other_persona(persona)
        self.store.add_reference_to_other_persona(persona)
        self.reflect.add_reference_to_other_persona(persona)
        self.plan.add_reference_to_other_persona(persona)
        self.act.add_reference_to_other_persona(persona)
        self.converse.add_reference_to_other_persona(persona)

    def update_agenda(self, agenda: str) -> None:
        self._agenda = agenda

    def update_harvest_report(self, harvest_report: str) -> None:
        self._harvest_report = harvest_report

    def update_overuse_threshold(self, overuse_threshold: float) -> None:
        self._overuse_threshold = overuse_threshold

    def update_current_leader(self, curr_leader: "PersonaAgent") -> None:
        self._current_leader = curr_leader

    @property
    def name(self):
        return self.identity.name

    @property
    def agenda(self) -> str:
        return self._agenda

    @property
    def disinfo(self) -> bool:
        return self._disinfo

    @property
    def harvest_report(self) -> str:
        return self._harvest_report

    @property
    def svo_angle(self) -> float:
        return self._svo_angle

    @property
    def svo_type(self) -> SVOPersonaType:
        return self._svo_type

    @property
    def current_leader(self):
        return self._current_leader

    @property
    def experiment_storage(self) -> float:
        return self._experiment_storage

    def update_time(self, current_time):
        self.current_time = current_time

    def set_env(self, env: PersonaEnvironment):
        self.identity.env = env


