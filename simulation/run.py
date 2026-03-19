"""Runs simulation with an election."""

import collections
import datetime

import os
import random



import numpy as np
import omegaconf

from simulation.environment import ConcurrentEnv
from simulation.environment import PerturbationEnv
from simulation.persona import EmbeddingModel
from simulation.persona import SVOPersonaType
from simulation.persona.cognition import leaders as leaders_lib
from simulation.persona.cognition import utils as cognition_utils
from simulation.persona.common import PersonaActionHarvesting
from simulation.persona.common import PersonaEnvironment
from simulation.persona.common import PersonaEvent
from simulation.persona.common import PersonaIdentity
from simulation.persona.persona import DEFAULT_AGENDA
from simulation.persona.persona import PersonaAgent
from simulation.utils import ModelWandbWrapper


cognition_utils.SYS_VERSION = "v3"


def perform_election(
    personas: dict[str, PersonaAgent],
    leader_candidates: dict[str, PersonaAgent],
    current_time: datetime,
    wrapper: ModelWandbWrapper,
    curr_round: int,
    agent_id_to_name: dict[int, str],
    agent_name_to_id: dict[str, int],
    last_winning_agenda: str | None = None,
    harvest_report: str | None = None,
    harvest_stats: str | None = None,
    disinfo: bool = False,
    debug: bool = False,
) -> tuple[str, dict[str, int], dict[str, str]]:
    print(f"\n\n\ROUND {curr_round}: ELECTION\n==================")
    leader_agendas = {}
    for _, leader in leader_candidates.items():
        agenda, _ = leaders_lib.prompt_leader_agenda(
            model=wrapper,
            init_persona=leader,
            current_location="restaurant",
            current_time=current_time,
            init_retrieved_memory=leaders_lib.get_memories(leader),
            total_fishers=len(personas),
            svo_angle=leader.svo_angle,
            last_winning_agenda=last_winning_agenda,
            harvest_report=harvest_report,
            harvest_stats=harvest_stats,
            use_disinfo=disinfo,
        )
        leader_agendas[leader.identity.name] = agenda
    votes = {leader.identity.name: 0 for leader in leader_candidates.values()}
    if len(leader_candidates) > 1:
        for persona_id in personas:
            if persona_id not in leader_candidates:
                retireved_memory = leaders_lib.get_memories(personas[persona_id])
                candidates = [
                    leader.identity.name for _, leader in leader_candidates.items()
                ]
                random.shuffle(candidates)
                vote, _ = personas[persona_id].act.participate_in_election(
                    retrieved_memories=retireved_memory,
                    current_location="",
                    current_time=current_time.strftime(
                        "%H-%M-%S"
                    ),
                    candidates=candidates,
                    leader_agendas=leader_agendas,
                    debug=debug,
                )
                candidate_id = vote.name if hasattr(vote, "name") else str(vote)
                candidate_str = agent_id_to_name.get(candidate_id, candidate_id)
                votes[candidate_str] = votes.get(candidate_str, 0) + 1
                personas[persona_id].store.store_event(
                    PersonaEvent(
                        f"Round {curr_round} vote: {vote}",
                        created=current_time,
                        expiration=leaders_lib.get_expiration_next_month(current_time),
                        always_include=True,
                    )
                )

        votes_cp = dict(votes)
        if "none" in votes_cp:
            del votes_cp["none"]
        winner = max(votes_cp.values())
        keys = [key for key, value in votes_cp.items() if value == winner]
        winner = random.choice(keys)
    elif len(leader_candidates) == 1:
        print("SKIPPING ELECTION AS ONLY ONE LEADER CANDIDATE...")
        winner = list(leader_candidates.keys())[0]
        winner = agent_id_to_name[winner]
    else:
        raise ValueError("No leader candidates or only one leader candidate.")

    if debug:
        print("\n=================\nELECTION RESULTS\n=================")
        for candidate, vote_count in votes.items():
            print(f"{candidate}: {vote_count} votes")
        print(f"\nROUND {curr_round} WINNER: {winner}")
        print("\n=================\nLEADER AGENDAS\n=================")
        for agenda_id, agenda in leader_agendas.items():
            print(
                f"\n{agent_id_to_name.get(agenda_id, agenda_id)}'s"
                " Agenda:\n=================="
            )
            pid = agent_name_to_id.get(agenda_id, agenda_id)
            print(
                f"SVO Angle: {leader_candidates[pid].svo_angle}, SVO Type:"
                f" {leader_candidates[pid].svo_type}\n"
            )
            print(agenda)
    leader_agendas["none"] = "No leader agenda, use your best judgement."
    leader_announcement = (
        f"Newly elected leader {winner}'s round {curr_round} agenda:"
        f" {leader_agendas[winner]}"
    )
    leaders_lib.make_public_leader_memories(
        all_personas=personas,
        leader_announcement=leader_announcement,
        current_time=current_time,
    )
    return winner, votes, leader_agendas


def run(
    cfg: omegaconf.DictConfig,
    logger: ModelWandbWrapper,
    wrapper: ModelWandbWrapper,
    framework_wrapper: ModelWandbWrapper,
    embedding_model: EmbeddingModel,
    experiment_storage: str,
):
    if not os.path.exists(experiment_storage):
        os.makedirs(experiment_storage)

    if hasattr(cfg.env, "disinformation"):
        disinformation = bool(cfg.env.disinformation)
    else:
        disinformation = False
    print(f"DISINFORMATION FLAG: {disinformation}")

    consolidated_log_path = os.path.join(
        experiment_storage, "consolidated_results.json"
    )

    cognition_utils.log_to_file(
        "initialization",
        {
            "experiment_id": datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
            "config": (
                omegaconf.OmegaConf.to_container(cfg)
                if hasattr(cfg, "to_container")
                else str(cfg)
            ),
        },
        log_path=consolidated_log_path,
    )

    election_results = {}

    if cfg.agent.agent_package == "persona_v3":
        cognition_utils.REASONING = cfg.agent.cot_prompt
    else:
        raise ValueError(f"Unknown agent package: {cfg.agent.agent_package}")

    leader_distribution = leaders_lib.LeaderPopulationType(
        cfg.agent.leader_population)
    leader_svos, leader_types = leaders_lib.sample_leader_svos(
        leader_distribution)

    leader_candidates = {}
    for i, svo_data in enumerate(zip(leader_svos, leader_types)):
        svo_angle, svo_type = svo_data
        leader_candidates[f"persona_{i}"] = PersonaAgent(
            cfg.agent,
            wrapper,
            framework_wrapper,
            embedding_model,
            os.path.join(experiment_storage, f"persona_{i}"),
            svo_angle=svo_angle,
            svo_type=svo_type,
            disinfo=False,
            experiment_storage=experiment_storage,
        )

    personas = {**leader_candidates}
    num_leaders = len(leader_types)
    assert 2 * num_leaders <= cfg.env.num_agents, (
        "Not enough personas for an election.")
    for i in range(num_leaders, cfg.env.num_agents):
        personas[f"persona_{i}"] = PersonaAgent(
            cfg.agent,
            wrapper,
            framework_wrapper,
            embedding_model,
            os.path.join(experiment_storage, f"persona_{i}"),
            experiment_storage=experiment_storage,
        )

    num_personas = cfg.personas.num
    identities = {
        f"persona_{i}": PersonaIdentity(
            agent_id=f"persona_{i}", **cfg.personas[f"persona_{i}"],
            environment=PersonaEnvironment(
                regen_min_range=cfg.env.regen_factor_range[0],
                regen_max_range=cfg.env.regen_factor_range[1]
            )
        )
        for i in range(num_personas)
    }
    cognition_utils.log_to_file(
        "persona_identities",
        {
            pid: {
                "name": (
                    identities[pid].name
                    if hasattr(identities[pid], "name")
                    else pid
                ),
                "type": (
                    str(leader_types[i])
                    if i < num_leaders
                    else str(SVOPersonaType.NONE)
                ),
                "config": {k: str(v) for k, v in persona_config.items()},
            }
            for i, (pid, persona_config) in enumerate([
                (
                    f"persona_{i}",
                    cfg.personas.get(
                        f"persona_{i}", cfg.personas.get("default", {})
                    ),
                )
                for i in range(cfg.env.num_agents)
            ])
        },
        log_path=consolidated_log_path,
    )

    agent_name_to_id = {obj.name: k for k, obj in identities.items()}
    agent_name_to_id["framework"] = "framework"
    agent_id_to_name = {v: k for k, v in agent_name_to_id.items()}

    svo_info = {}
    for pid, persona in leader_candidates.items():
        svo_info[agent_id_to_name[pid]] = {
            "svo_angle": persona.svo_angle,
            "svo_type": str(persona.svo_type),
        }
    cognition_utils.log_to_file(
        log_type="svo_info", data=svo_info, log_path=consolidated_log_path)

    for persona in personas:
        personas[persona].init_persona(
            persona, identities[persona], social_graph=None
        )
    for persona in personas:
        for other_persona in personas:
            personas[persona].add_reference_to_other_persona(personas[other_persona])

    env_class = (
        PerturbationEnv
        if cfg.env.class_name == "fishing_perturbation_env"
        else ConcurrentEnv
    )
    env = env_class(
        cfg.env,
        experiment_storage,
        agent_id_to_name,
        regen_factor_range=cfg.env.regen_factor_range,
    )

    agent_id, obs = env.reset()
    curr_round = env.num_round

    agenda = DEFAULT_AGENDA
    winner, votes, leader_agendas, harvest_report = None, None, None, None
    if leader_candidates:
        winner, votes, leader_agendas = perform_election(
            personas,
            leader_candidates,
            obs.current_time,
            wrapper,
            curr_round=curr_round,
            agent_id_to_name=agent_id_to_name,
            agent_name_to_id=agent_name_to_id,
            disinfo=disinformation,
            debug=cfg.debug,
        )
        agenda = leader_agendas[winner]

    round_stats = collections.defaultdict(
        lambda: collections.defaultdict(int)
    )

    round_harvest_stats = collections.defaultdict(
        lambda: collections.defaultdict(int)
    )

    last_location = None
    while True:

        if obs.current_location == "restaurant" and last_location == "lake":
            if cfg.debug:
                print(
                    f"ROUND {curr_round}: DISCUSSION PHASE\n=========================\n"
                )
            round_stats[curr_round] = {
                "num_resources": env.internal_global_state["resource_in_pool"],
                "regen_factor": env.internal_global_state["regen_factor"],
            }
            print(
                f"ROUND {curr_round} ROUND STATS: "
                f"{round_stats}"
            )
            if leader_candidates:
                assert winner is not None
                harvest_report = leaders_lib.make_leader_report(
                    personas=personas,
                    leader_candidates=leader_candidates,
                    current_time=obs.current_time,
                    wrapper=wrapper,
                    disinformation=disinformation,
                    agenda=agenda,
                    curr_round=curr_round,
                    winner_id=agent_name_to_id[winner],
                    round_harvest_stats=round_harvest_stats[curr_round],
                    regen_factor=env.internal_global_state["regen_factor"],
                    debug=cfg.debug,
                )
                announcement = (
                    f"{winner}'s ROUND {curr_round} REPORT: {harvest_report}"
                )
                leaders_lib.make_public_leader_memories(
                    all_personas=personas,
                    leader_announcement=announcement,
                    current_time=obs.current_time,
                )
            else:
                print("NO LEADER CANDIDATES - MAKING FACTUAL REPORT ...")
                harvest_report = leaders_lib.make_harvest_report(
                    personas, round_harvest_stats[curr_round])

        agent = personas[agent_id]

        agent.update_agenda(agenda)
        agent.update_harvest_report(harvest_report)
        if winner:
            agent.update_current_leader(
                leader_candidates[agent_name_to_id[winner]]
            )
        action = agent.loop(obs, debug=cfg.debug)

        if len(leader_candidates) > 1 and curr_round != env.num_round:
            election_results[curr_round] = {
                "round": curr_round,
                "winner": winner,
                "agendas": leader_agendas,
                "votes": votes,
                "harvest_report": harvest_report,
                "harvest_stats": round_harvest_stats[curr_round],
                "num_resources": env.internal_global_state["resource_in_pool"],
            }
            cognition_utils.log_to_file(
                log_type="election",
                data=election_results[curr_round],
                log_path=consolidated_log_path)
            logger.log_game(election_results[curr_round])
            curr_round = env.num_round
            winner, votes, leader_agendas = perform_election(
                personas,
                leader_candidates,
                obs.current_time,
                wrapper,
                curr_round=curr_round,
                agent_id_to_name=agent_id_to_name,
                agent_name_to_id=agent_name_to_id,
                last_winning_agenda=agenda,
                harvest_report=harvest_report,
                harvest_stats=round_harvest_stats[curr_round-1],
                disinfo=disinformation,
                debug=cfg.debug,
            )
            agenda = leader_agendas[winner]
        else:
            curr_round = env.num_round

        last_location = obs.current_location

        agent_id, obs, _, termination = env.step(action)

        if isinstance(action, PersonaActionHarvesting):
            round_harvest_stats[curr_round][agent.identity.name] = action.quantity

        stats = {}
        if hasattr(action, "stats"):
            for s in [
                "conversation_resource_limit",
                *[
                    f"persona_{i}_collected_resource"
                    for i in range(cfg.env.num_agents)
                ],
            ]:
                if s in action.stats:
                    stats[s] = action.stats[s]
        logger.log_game({"num_resource": obs.current_resource_num, **stats})

        if np.any(list(termination.values())):
            logger.log_game({"num_resource": obs.current_resource_num}, last_log=True)
            break

    election_results[curr_round] = {
        "round": curr_round,
        "winner": winner,
        "agendas": leader_agendas,
        "votes": votes,
        "harvest_report": harvest_report,
        "harvest_stats": round_harvest_stats[curr_round],
        "num_resources": env.internal_global_state["resource_in_pool"],
    }
    cognition_utils.log_to_file(
        log_type="election",
        data=election_results[curr_round],
        log_path=consolidated_log_path
    )
    logger.log_game(election_results[curr_round])
    print(
        "FINAL HARVEST STATS - ROUND"
        f" {curr_round}:\n{round_harvest_stats[curr_round]}"
    )
    print(f"FINAL HARVEST REPORT - ROUND {curr_round}:\n{harvest_report}")
    cognition_utils.log_to_file(
        log_type="round_stats",
        data=round_stats,
        log_path=consolidated_log_path
    )
    cognition_utils.log_to_file(
        log_type="harvest",
        data=round_harvest_stats,
        log_path=consolidated_log_path
    )
    cognition_utils.log_to_file(
        log_type="sim-end", data=None, log_path=consolidated_log_path)
    env.save_log()
    for persona in personas:
        personas[persona].memory.save()
