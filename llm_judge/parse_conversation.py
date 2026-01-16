"""Json parsing."""

import json
import os
import sys

N_SEEDS = 10
MODEL_NAME = "gpt"
FILE_NAME_BASE = "gpt-4o-2024-05-13_disinfo_True_population_balanced"

JSON_BASE_PATH = (
    "/home/rfaulk/projects/aip-rgrosse/rfaulk/GovSimElect/simulation/results/fishing_v7.0/"
)


def main(argv: list[str]) -> None:
  if len(argv) > 2:
    sys.exit("Too many args.")
  print(f"args: {argv}")

  # Produce a file list containing conversation.
  events_list = []
  for i in range(1, N_SEEDS + 1):
    in_file = os.path.join(
        JSON_BASE_PATH,
        MODEL_NAME,
        FILE_NAME_BASE + f"_run_{i}/persona_0/nodes.json",
    )
    print(f"Reading {in_file}...")
    with open(in_file, "r") as f:
      events_list += json.load(f)

  out_file = os.path.join(
      JSON_BASE_PATH, MODEL_NAME, FILE_NAME_BASE + "_conversation.csv"
  )
  with open(out_file, "w") as f:
    for element in events_list:
      description = " ".join(element["description"].split("\n"))
      f.write(description + "\n")
      if "conversation" in element:
        print("Processing conversation...")
        conversation = element["conversation"]
        if conversation:
          f.write(
              "\n".join([f"{utt[0]}: {utt[1]}".strip() for utt in conversation])
          )
    print(f"Successfully wrote text to: {out_file}")


if __name__ == "__main__":
  main(sys.argv[1:])

