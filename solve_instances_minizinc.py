# example of usage:
# python ./solve_instances_minizinc.py --instances-input-dir="./instances/dzn/" --model-filepath="./cp/best_model.mzn"
# --solutions-output-dir="./solutions/cp/best_model/chuffed/"
# --stats-output-csv-file="./solutions/cp/best_model/chuffed/stats.csv" --solver="chuffed" --timeout-ms=300000

import os
from datetime import timedelta
import csv

import click
import minizinc
from minizinc import Instance, Model, Solver

from utils.misc import mzn_solution_to_dict, is_a_valid_solution, draw_board


CSV_FIELD_NAMES = ["instance", "optimally_solved", "board_height", "elapsed_ms"]


@click.command()
@click.option('--model-filepath', type=click.Path(exists=True), required=True)
@click.option('--instances-input-dir', type=click.Path(exists=True), required=True)
@click.option('--solutions-output-dir', type=click.Path(exists=False), required=True)
@click.option('--stats-output-csv-file', type=click.Path(exists=False), required=True)
@click.option('--solver', type=str, default="chuffed")
@click.option('--timeout-ms', type=int, default=300000)
@click.option('--draw-solutions', type=bool, required=False, default=True)
def run(model_filepath: str, instances_input_dir: str, solutions_output_dir: str,
        stats_output_csv_file: str, solver: str, timeout_ms: int, draw_solutions: bool = True):

    if not os.path.exists(solutions_output_dir):
        os.makedirs(solutions_output_dir)

    csv_stats_file = open(stats_output_csv_file, 'x', encoding='UTF8', newline='')
    csv_writer = csv.DictWriter(csv_stats_file, fieldnames=CSV_FIELD_NAMES)
    csv_writer.writeheader()

    model = Model(model_filepath)
    chosen_solver = Solver.lookup(tag=solver, driver=None)
    blank_instance = Instance(chosen_solver, model)

    board_height = -1

    for subdir, dirs, files in os.walk(instances_input_dir):
        for file in files:
            current_instance_filepath = os.path.join(subdir, file)

            with blank_instance.branch() as instance:
                print("Solving " + file + " ...")
                instance.add_file(current_instance_filepath)

                result = instance.solve(timedelta(milliseconds=timeout_ms))
                if result.status == minizinc.Status.SATISFIED or result.status == minizinc.Status.OPTIMAL_SOLUTION:
                    solution = str(result.solution)
                    solution_dict = mzn_solution_to_dict(solution)
                    board_height = solution_dict["board_height"]

                    if not is_a_valid_solution(**solution_dict):
                        raise Exception("Invalid solution found for " + file)

                    with open(os.path.join(solutions_output_dir, os.path.splitext(file)[0] + ".txt"), "w") as \
                            solution_file:
                        solution_file.write(solution)

                    if draw_solutions:
                        draw_board(**solution_dict, output_file=os.path.join(solutions_output_dir,
                                                                             os.path.splitext(file)[0] + ".png"))

                csv_writer.writerow(
                    {
                        'instance': os.path.splitext(file)[0],
                        'optimally_solved': result.status == minizinc.Status.OPTIMAL_SOLUTION,
                        'board_height': board_height,
                        'elapsed_ms': result.statistics["solveTime"].total_seconds() * 1000
                        if result.status == minizinc.Status.OPTIMAL_SOLUTION else timeout_ms
                    }
                )
                csv_stats_file.flush()

    csv_stats_file.close()


if __name__ == "__main__":
    run()
