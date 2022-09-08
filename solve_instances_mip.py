import os
import json
import csv
import copy
import typing

import click
import gurobipy
from gurobipy import GRB

from milp.S1 import S1_BM_Model, S1_MBM_Model
from milp.S2 import S2_BM_Model
from milp.SG import SG_BM_Model


CSV_FIELD_NAMES = ["instance", "solved", "elapsed_ms"]
MODELS = {"S1_BM": S1_BM_Model, "S1_MBM": S1_MBM_Model, "S2_BM": S2_BM_Model, "SG_BM": SG_BM_Model}


@click.command()
@click.option('--instances-input-dir', type=click.Path(exists=True), required=True)
@click.option('--solutions-output-dir', type=click.Path(exists=False), required=True)
@click.option('--stats-output-csv-file', type=click.Path(exists=False), required=True)
@click.option('--timeout-ms', type=int, default=300000)
@click.option('--model-name', type=str, required=False, default="S2_BM")
@click.option('--activate-symmetry-breaking', type=bool, required=False, default=False)
def run(instances_input_dir: str, solutions_output_dir: str, stats_output_csv_file: str,
        timeout_ms: int = 300000, model_name: str = "S2_BM", activate_symmetry_breaking: bool = False):

    assert model_name in MODELS, "Unknown model!"

    if not os.path.exists(solutions_output_dir):
        os.makedirs(solutions_output_dir)

    csv_stats_file = open(stats_output_csv_file, 'x', encoding='UTF8', newline='')
    csv_writer = csv.DictWriter(csv_stats_file, fieldnames=CSV_FIELD_NAMES)
    csv_writer.writeheader()

    for subdir, dirs, files in os.walk(instances_input_dir):
        for file in files:
            current_instance_filepath = os.path.join(subdir, file)
            with open(current_instance_filepath, "r") as current_instance_file:
                current_instance = json.load(current_instance_file)

            print("Solving instance " + os.path.splitext(file)[0] + " ...")
            x, y, model = MODELS[model_name](**current_instance, activate_symmetry_breaking=activate_symmetry_breaking,
                                             timeout_s=timeout_ms / 1000)
            model.presolve()
            model.optimize()

            if model.Status == GRB.OPTIMAL:
                solved_instance = copy.deepcopy(current_instance)
                solved_instance['board_height'] = int(model.getVarByName("board_height").X)
                solved_instance['circuit_x'] = [int(x[i].X) for i in range(current_instance['n_circuits'])]
                solved_instance['circuit_y'] = [int(y[i].X) for i in range(current_instance['n_circuits'])]

                with open(os.path.join(solutions_output_dir, os.path.splitext(file)[0] + ".txt"), "w") as solution_file:
                    json.dump(solved_instance, solution_file)

            csv_writer.writerow(
                {
                    'instance': os.path.splitext(file)[0],
                    'solved': model.Status == GRB.OPTIMAL,
                    'elapsed_ms': int(model.Runtime * 1000) if model.Status == GRB.OPTIMAL else timeout_ms
                }
            )

    csv_stats_file.flush()
    csv_stats_file.close()


if __name__ == "__main__":
    run()
