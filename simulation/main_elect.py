"""Main file for GovSim Election simulation."""

import os
from pathlib import Path
import shutil
import sys
import uuid

# Add the project root directory to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

import hydra
import wandb
from omegaconf import DictConfig, OmegaConf
from transformers import set_seed

from simulation.utils import ModelWandbWrapper, WandbLogger
from pathfinder import get_model

from .persona import EmbeddingModel
# Updated import: run fishing scenario from multi-turn election version.
from simulation.scenarios.fishing.run_election import run as run_scenario_fishing


@hydra.main(version_base=None, config_path="conf", config_name="config_api")
def main(cfg: DictConfig):
  print(OmegaConf.to_yaml(cfg))
  set_seed(cfg.seed)

  model = get_model(cfg.llm.path, cfg.llm.is_api, cfg.seed, cfg.llm.backend)
  logger = WandbLogger(
      cfg.experiment.name, OmegaConf.to_object(cfg), debug=cfg.debug
  )
  run_name = logger.run_name if logger.run_name else f"{cfg.llm.path}_run_{cfg.llm.iter}"
  experiment_storage = os.path.join(
      os.path.dirname(__file__),
      f"./results/{cfg.experiment.name}/{run_name}",
  )

  wrapper = ModelWandbWrapper(
      model,
      render=cfg.llm.render,
      wanbd_logger=logger,
      temperature=cfg.llm.temperature,
      top_p=cfg.llm.top_p,
      seed=cfg.seed,
      is_api=cfg.llm.is_api,
  )
  embedding_model = EmbeddingModel(device="cpu")

  if cfg.experiment.scenario == "fishing":
    run_scenario_fishing(
        cfg.experiment,
        logger,
        wrapper,
        wrapper,
        embedding_model,
        experiment_storage,
    )
  else:
    raise ValueError(f"Unknown experiment.scenario: {cfg.experiment.scenario}")

  hydra_log_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
  if os.path.exists(f"{experiment_storage}/.hydra/"):
    shutil.rmtree(f"{experiment_storage}/.hydra/")
  shutil.copytree(f"{hydra_log_path}/.hydra/", f"{experiment_storage}/.hydra/")
  shutil.copy(f"{hydra_log_path}/main_elect.log",
              f"{experiment_storage}/main_elect.log")
  # shutil.rmtree(hydra_log_path)

  artifact = wandb.Artifact("hydra", type="log")
  artifact.add_dir(f"{experiment_storage}/.hydra/")
  artifact.add_file(f"{experiment_storage}/.hydra/config.yaml")
  artifact.add_file(f"{experiment_storage}/.hydra/hydra.yaml")
  artifact.add_file(f"{experiment_storage}/.hydra/overrides.yaml")
  wandb.run.log_artifact(artifact)


if __name__ == "__main__":
  OmegaConf.register_resolver("uuid", lambda: f"run_{uuid.uuid4()}")
  main()

