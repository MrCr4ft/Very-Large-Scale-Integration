import os
import json
import csv
import copy

import click
import gurobipy
from gurobipy import GRB

from milp.s1_bm import S1_BM_Model


CSV_FIELD_NAMES = ["instance", "solved", "elapsed_ms"]


@click.command()
@click.option('--instances-input-dir', type=click.Path(exists=True), required=True)
@click.option('--solutions-output-dir', type=click.Path(exists=False), required=True)
@click.option('--stats-output-csv-file', type=click.Path(exists=False), required=True)
@click.option('--timeout-ms', type=int, default=300000)
def run(instances_input_dir: str, solutions_output_dir: str, stats_output_csv_file: str,
        timeout_ms: int = 300000):

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
            x, y, model = S1_BM_Model(**current_instance, timeout_s=timeout_ms / 1000)
            model.presolve()
            model.optimize()

            effective_x = [int(x[i].X) for i in range(current_instance['n_circuits'])]
            effective_y = [int(y[i].X) for i in range(current_instance['n_circuits'])]

            if model.Status != GRB.LOADED:
                solved_instance = copy.deepcopy(current_instance)
                solved_instance['board_height'] = int(model.getVarByName("board_height").X)
                solved_instance['circuit_x'] = effective_x
                solved_instance['circuit_y'] = effective_y

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
