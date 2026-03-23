"""Runs simulation with an election.

The simulation loop is driven by configurable Phase objects (see
simulation.phases).  Each round iterates through the configured phase
list; the default ordering is:

    PolicyMaking → Election → Harvesting(+Report) → Discussion → Reflection

Phase ordering is set via the ``phases`` key in the experiment YAML.
"""

import datetime
import os

import omegaconf
from simulation.environment.concurrent_env import ConcurrentEnv
from simulation.environment.perturbation_env import PerturbationEnv

from simulation.persona.cognition import leaders as leaders_lib
from simulation.persona.cognition import utils as cognition_utils
from simulation.persona.common import PersonaEnvironment
from simulation.persona.common import PersonaIdentity
from simulation.persona.embedding_model import EmbeddingModel
from simulation.persona.persona import PersonaAgent
from simulation.persona.persona import SVOPersonaType

from simulation.phases.base import PhaseContext
from simulation.phases.discussion import DiscussionPhase
from simulation.phases.election import ElectionPhase
from simulation.phases.harvesting import HarvestingPhase
from simulation.phases.policy_making import PolicyMakingPhase
from simulation.phases.reflection import ReflectionPhase

from simulation.utils.models import ModelWandbWrapper


cognition_utils.SYS_VERSION = "v3"


# ── Phase registry ────────────────────────────────────────────────────

DEFAULT_PHASES = [
    "policy_making",
    "election",
    "harvesting",
    "discussion",
    "reflection",
]

PHASE_REGISTRY = {
    "policy_making": PolicyMakingPhase,
    "election": ElectionPhase,
    "harvesting": HarvestingPhase,
    "discussion": DiscussionPhase,
    "reflection": ReflectionPhase,
}


def build_phases(phase_names: list[str]) -> list:
  """Instantiate Phase objects from a list of phase name strings."""
  phases = []
  for name in phase_names:
    if name not in PHASE_REGISTRY:
      raise ValueError(
          f"Unknown phase: {name!r}."
          f" Available: {list(PHASE_REGISTRY.keys())}"
      )
    phases.append(PHASE_REGISTRY[name]())
  return phases


# ── Main entry point ──────────────────────────────────────────────────


async def run(
    cfg: omegaconf.DictConfig,
    logger: ModelWandbWrapper,
    wrapper: ModelWandbWrapper,
    framework_wrapper: ModelWandbWrapper,
    embedding_model: EmbeddingModel,
    experiment_storage: str,
):
  """Run the simulation."""
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

  if cfg.agent.agent_package == "persona_v3":
    cognition_utils.REASONING = cfg.agent.cot_prompt
  else:
    raise ValueError(f"Unknown agent package: {cfg.agent.agent_package}")

  # ── Leader setup ──────────────────────────────────────────────────

  leader_distribution = leaders_lib.LeaderPopulationType(
      cfg.agent.leader_population
  )
  leader_svos, leader_types = leaders_lib.sample_leader_svos(
      leader_distribution
  )

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

  # ── Persona setup ─────────────────────────────────────────────────

  personas = {**leader_candidates}
  num_leaders = len(leader_types)
  assert (
      2 * num_leaders <= cfg.env.num_agents
  ), "Not enough personas for an election."
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
          agent_id=f"persona_{i}",
          **cfg.personas[f"persona_{i}"],
          environment=PersonaEnvironment(
              regen_min_range=cfg.env.regen_factor_range[0],
              regen_max_range=cfg.env.regen_factor_range[1],
          ),
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
      log_type="svo_info", data=svo_info, log_path=consolidated_log_path
  )

  for persona in personas:
    personas[persona].init_persona(
        persona, identities[persona], social_graph=None
    )
  for persona in personas:
    for other_persona in personas:
      personas[persona].add_reference_to_other_persona(
          personas[other_persona]
      )

  # ── Environment setup ─────────────────────────────────────────────

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

  # ── Phase setup ───────────────────────────────────────────────────

  if hasattr(cfg, "phases"):
    phase_names = list(cfg.phases)
  else:
    phase_names = DEFAULT_PHASES
  phases = build_phases(phase_names)
  print(f"PHASE ORDER: {[p.name for p in phases]}")

  ctx = PhaseContext(
      cfg=cfg,
      personas=personas,
      leader_candidates=leader_candidates,
      env=env,
      wrapper=wrapper,
      logger=logger,
      agent_id_to_name=agent_id_to_name,
      agent_name_to_id=agent_name_to_id,
      experiment_storage=experiment_storage,
      consolidated_log_path=consolidated_log_path,
      disinformation=disinformation,
      debug=cfg.debug,
      agent_id=agent_id,
      obs=obs,
      round_num=env.num_round,
  )

  # ── Main simulation loop ──────────────────────────────────────────

  while not ctx.terminated:
    for phase in phases:
      ctx = await phase.execute(ctx)
      if ctx.terminated:
        break

    if not ctx.terminated:
      # Log the completed round's election results.
      if (
          ctx.leader_candidates
          and len(ctx.leader_candidates) > 1
      ):
        ctx.election_results[ctx.round_num] = {
            "round": ctx.round_num,
            "winner": ctx.winner,
            "agendas": ctx.leader_agendas,
            "votes": ctx.votes,
            "harvest_report": ctx.harvest_report,
            "harvest_stats": dict(
                ctx.round_harvest_stats[ctx.round_num]
            ),
            "num_resources": ctx.env.internal_global_state[
                "resource_in_pool"
            ],
        }
        cognition_utils.log_to_file(
            log_type="election",
            data=ctx.election_results[ctx.round_num],
            log_path=ctx.consolidated_log_path,
        )
        ctx.logger.log_game(ctx.election_results[ctx.round_num])
      ctx.round_num = ctx.env.num_round

  # ── Final logging ─────────────────────────────────────────────────

  ctx.election_results[ctx.round_num] = {
      "round": ctx.round_num,
      "winner": ctx.winner,
      "agendas": ctx.leader_agendas,
      "votes": ctx.votes,
      "harvest_report": ctx.harvest_report,
      "harvest_stats": dict(
          ctx.round_harvest_stats[ctx.round_num]
      ),
      "num_resources": ctx.env.internal_global_state[
          "resource_in_pool"
      ],
  }
  cognition_utils.log_to_file(
      log_type="election",
      data=ctx.election_results[ctx.round_num],
      log_path=ctx.consolidated_log_path,
  )
  ctx.logger.log_game(ctx.election_results[ctx.round_num])
  print(
      "FINAL HARVEST STATS - ROUND"
      f" {ctx.round_num}:\n{ctx.round_harvest_stats[ctx.round_num]}"
  )
  print(
      f"FINAL HARVEST REPORT - ROUND {ctx.round_num}:\n"
      f"{ctx.harvest_report}"
  )
  cognition_utils.log_to_file(
      log_type="round_stats",
      data=ctx.round_stats,
      log_path=ctx.consolidated_log_path,
  )
  cognition_utils.log_to_file(
      log_type="harvest",
      data=ctx.round_harvest_stats,
      log_path=ctx.consolidated_log_path,
  )
  cognition_utils.log_to_file(
      log_type="sim-end",
      data=None,
      log_path=ctx.consolidated_log_path,
  )
  ctx.env.save_log()
  for persona in ctx.personas:
    ctx.personas[persona].memory.save()
