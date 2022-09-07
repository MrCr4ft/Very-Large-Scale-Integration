import os
import json
import csv

import click

from sat.sat_model import solve_2ssp_instance_json


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
            print("Solving instance " + os.path.splitext(file)[0] + " ...")
            solved_instance, elapsed_ms, is_exactly_solved = solve_2ssp_instance_json(current_instance_filepath,
                                                                                      timeout_ms)

            with open(os.path.join(solutions_output_dir, os.path.splitext(file)[0] + ".txt"), "w") as solution_file:
                json.dump(solved_instance, solution_file)

            csv_writer.writerow(
                {
                    'instance': os.path.splitext(file)[0],
                    'solved': is_exactly_solved,
                    'elapsed_ms': elapsed_ms if is_exactly_solved else timeout_ms
                }
            )

    csv_stats_file.flush()
    csv_stats_file.close()


if __name__ == "__main__":
    run()
