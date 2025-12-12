import collections
import datetime
import json
import os
from typing import List

import numpy as np
from omegaconf import DictConfig, OmegaConf

from simulation.persona.common import PersonaActionHarvesting
from simulation.persona import EmbeddingModel
from simulation.persona.common import PersonaIdentity
from simulation.utils import ModelWandbWrapper

from .environment import FishingConcurrentEnv, FishingPerturbationEnv


def run(
    cfg: DictConfig,
    logger: ModelWandbWrapper,
    wrappers: List[ModelWandbWrapper],
    framework_wrapper: ModelWandbWrapper,
    embedding_model: EmbeddingModel,
    experiment_storage: str,
):
    if cfg.agent.agent_package == "persona_v3":
        from .agents.persona_v3 import FishingPersona
        from .agents.persona_v3.cognition import utils as cognition_utils

        if cfg.agent.system_prompt == "v3":
            cognition_utils.SYS_VERSION = "v3"
        elif cfg.agent.system_prompt == "v3_p2":
            cognition_utils.SYS_VERSION = "v3_p2"
        elif cfg.agent.system_prompt == "v3_p1":
            cognition_utils.SYS_VERSION = "v3_p1"
        elif cfg.agent.system_prompt == "v3_p3":
            cognition_utils.SYS_VERSION = "v3_p3"
        elif cfg.agent.system_prompt == "v3_nocom":
            cognition_utils.SYS_VERSION = "v3_nocom"
        else:
            cognition_utils.SYS_VERSION = "v1"
        if cfg.agent.cot_prompt == "think_step_by_step":
            cognition_utils.REASONING = "think_step_by_step"
        elif cfg.agent.cot_prompt == "deep_breath":
            cognition_utils.REASONING = "deep_breath"
    else:
        raise ValueError(f"Unknown agent package: {cfg.agent.agent_package}")

    if not os.path.exists(experiment_storage):
        os.makedirs(experiment_storage)
    # Create a consolidated log file
    consolidated_log_path = os.path.join(
        experiment_storage, "consolidated_results.json"
    )

    # Helper function to append to the consolidated log
    def log_to_file(log_type, data):
        with open(consolidated_log_path, "a") as f:
            entry = {
                "timestamp": datetime.datetime.now().isoformat(),
                "type": log_type,
                "data": data,
            }
            f.write(json.dumps(entry) + "\n")

    # Initialize the log file with a header
    log_to_file(
        "initialization",
        {
            "experiment_id": datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
            "config": (
                OmegaConf.to_container(cfg)
                if hasattr(cfg, "to_container")
                else str(cfg)
            ),
        },
    )
    personas = {
        f"persona_{i}": FishingPersona(
            cfg.agent,
            wrappers[i],
            framework_wrapper,
            embedding_model,
            os.path.join(experiment_storage, f"persona_{i}"),
        )
        for i in range(12)
    }

    # NOTE persona characteristics, up to design choices
    num_personas = cfg.personas.num

    identities = {}
    for i in range(num_personas):
        persona_id = f"persona_{i}"
        identities[persona_id] = PersonaIdentity(
            agent_id=persona_id, **cfg.personas[persona_id]
        )

    # Standard setup
    agent_name_to_id = {obj.name: k for k, obj in identities.items()}
    agent_name_to_id["framework"] = "framework"
    agent_id_to_name = {v: k for k, v in agent_name_to_id.items()}

    for persona in personas:
        personas[persona].init_persona(persona, identities[persona], social_graph=None)

    for persona in personas:
        for other_persona in personas:
            # also add self reference, for conversation
            personas[persona].add_reference_to_other_persona(personas[other_persona])
    env_class = ( 
        FishingPerturbationEnv
        if cfg.env.class_name == "fishing_perturbation_env"
        else FishingConcurrentEnv
    )
    env = env_class(
        cfg.env,
        experiment_storage,
        agent_id_to_name,
        num_agents=12)
    agent_id, obs = env.reset(sustainability_threshold=4)
    round_harvest_stats = collections.defaultdict(
          lambda: collections.defaultdict(int)
    )
    curr_round = env.num_round
    while True:
        agent = personas[agent_id]
        action = agent.loop(obs)

        (
            agent_id,
            obs,
            rewards,
            termination,
        ) = env.step(action)

        # Check for harvest actions.
        if isinstance(action, PersonaActionHarvesting):
            round_harvest_stats[curr_round][agent.identity.name] = action.quantity
        stats = {}
        STATS_KEYS = [
            "conversation_resource_limit",
            *[f"persona_{i}_collected_resource" for i in range(12)],
        ]
        for s in STATS_KEYS:
            if s in action.stats:
                stats[s] = action.stats[s]

        if np.any(list(termination.values())):
            logger.log_game(
                {
                    "num_resource": obs.current_resource_num,
                    **stats,
                },
                last_log=True,
            )
            break
        else:
            logger.log_game(
                {
                    "num_resource": obs.current_resource_num,
                    **stats,
                }
            )
        if curr_round != env.num_round:
            curr_round = env.num_round
        logger.save(experiment_storage, agent_name_to_id)

    log_to_file("harvest", round_harvest_stats)
    env.save_log()
    for persona in personas:
        personas[persona].memory.save()
