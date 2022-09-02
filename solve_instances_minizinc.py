import os
from datetime import timedelta
import csv

import click
import minizinc
from minizinc import Instance, Model, Solver


CSV_FIELD_NAMES = ["instance", "solved", "elapsed_ms"]


@click.command()
@click.option('--minizinc-model-filepath', type=click.Path(exists=True), required=True)
@click.option('--minizinc-instances-input-dir', type=click.Path(exists=True), required=True)
@click.option('--solutions-output-dir', type=click.Path(exists=False), required=True)
@click.option('--stats-output-csv-file', type=click.Path(exists=False), required=True)
@click.option('--solver', type=str, default="chuffed")
@click.option('--timeout-ms', type=int, default=300000)
def run(minizinc_model_filepath: str, minizinc_instances_input_dir: str, solutions_output_dir: str,
        stats_output_csv_file: str, solver: str, timeout_ms: int):

    if not os.path.exists(solutions_output_dir):
        os.makedirs(solutions_output_dir)

    csv_stats_file = open(stats_output_csv_file, 'x', encoding='UTF8', newline='')
    csv_writer = csv.DictWriter(csv_stats_file, fieldnames=CSV_FIELD_NAMES)
    csv_writer.writeheader()

    model = Model(minizinc_model_filepath)
    chosen_solver = Solver.lookup(solver)
    blank_instance = Instance(chosen_solver, model)

    for subdir, dirs, files in os.walk(minizinc_instances_input_dir):
        for file in files:
            current_instance_filepath = os.path.join(subdir, file)

            with blank_instance.branch() as instance:
                print("Solving " + file + " ...")
                instance.add_file(current_instance_filepath)

                result = instance.solve(timedelta(milliseconds=timeout_ms))
                solution = str(result.solution)
                with open(os.path.join(solutions_output_dir, os.path.splitext(file)[0] + ".txt"), "w") as solution_file:
                    solution_file.write(solution)

                csv_writer.writerow(
                    {
                        'instance': os.path.splitext(file)[0],
                        'solved': True,
                        'elapsed_ms': result.statistics["solveTime"].total_seconds() * 1000
                        if result.status == minizinc.Status.OPTIMAL_SOLUTION else timeout_ms
                    }
                )

    csv_stats_file.flush()
    csv_stats_file.close()


if __name__ == "__main__":
    run()
