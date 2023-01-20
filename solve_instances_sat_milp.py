# example of usage:
# python ./solve_instances_sat_milp.py --instances-input-dir="./instances/json"
# --solutions-output-dir="./solutions/milp/sg_with_rotation"
# --stats-output-csv-file="./solutions/milp/sg_with_rotation/stats.csv" --timeout-ms=300000 --model-name=SGBMRotation


import os
import json
import csv

import click
import numpy as np

from sat.sat_strip_packing_no_rotation import SATStripPackingModelNoRotation
from milp.positioning_and_covering_models import PCMILPProblemNoRotation, PCMILPProblemRotation
from milp.dgp_big_m_models import SGBMPulpModel, S1BMPulpModel, S2BMPulpModel
from milp.dgp_big_m_models_with_rotation import SGBMPulpModelRotation, S1BMPulpModelRotation, \
    S2BMPulpModelRotation
from utils.misc import is_a_valid_solution, draw_board, solution_to_txt


MODELS = {
    "SATNoRotation": SATStripPackingModelNoRotation,
    "PCMILPNoRotation": PCMILPProblemNoRotation,
    "PCMILPRotation": PCMILPProblemRotation,
    "SGBM": SGBMPulpModel,
    "S1BM": S1BMPulpModel,
    "S2BM": S2BMPulpModel,
    "SGBMRotation": SGBMPulpModelRotation,
    "S1BMRotation": S1BMPulpModelRotation,
    "S2BMRotation": S2BMPulpModelRotation
}
CSV_FIELD_NAMES = ["instance", "optimally_solved", "board_height", "elapsed_ms"]


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


@click.command()
@click.option('--model-name', type=str, required=False, default="SATNoRotation")
@click.option('--instances-input-dir', type=click.Path(exists=True), required=True)
@click.option('--solutions-output-dir', type=click.Path(exists=False), required=True)
@click.option('--stats-output-csv-file', type=click.Path(exists=False), required=True)
@click.option('--json-solutions', type=bool, required=False, default=False)
@click.option('--timeout-ms', type=int, default=300000)
@click.option('--activate-symmetry-breaking', type=bool, required=False, default=True)
@click.option('--presolve-for-milp', type=bool, required=False, default=True)
@click.option('--solver-for-milp', type=str, required=False, default=None)
@click.option('--perform-linear-search', type=bool, required=False, default=False)
@click.option('--draw-solutions', type=bool, required=False, default=True)
def run(model_name: str, instances_input_dir: str, solutions_output_dir: str, stats_output_csv_file: str,
        json_solutions: bool = False, timeout_ms: int = 300, activate_symmetry_breaking: bool = True,
        presolve_for_milp: bool = True, solver_for_milp: str = None, perform_linear_search: bool = False,
        draw_solutions: bool = True):

    if model_name not in MODELS:
        raise ValueError("Unknown model name: " + model_name)

    if not os.path.exists(solutions_output_dir):
        os.makedirs(solutions_output_dir)

    csv_stats_file = open(stats_output_csv_file, 'x', encoding='UTF8', newline='')
    csv_writer = csv.DictWriter(csv_stats_file, fieldnames=CSV_FIELD_NAMES)
    csv_writer.writeheader()

    chosen_model = MODELS[model_name]
    for subdir, dirs, files in os.walk(instances_input_dir):
        for file in files:
            current_instance_filepath = os.path.join(subdir, file)
            print("Solving instance " + os.path.splitext(file)[0] + " ...")
            instance = chosen_model.from_instance_json(current_instance_filepath,
                                                       activate_symmetry_breaking=activate_symmetry_breaking,
                                                       add_implied_constraints=True,
                                                       presolve=presolve_for_milp,)
            instance.set_time_limit(timeout_ms)
            if solver_for_milp is not None:
                instance.set_solver(solver_for_milp)
            solution, elapsed_ms, optimal_solution = instance.solve(linear_search=perform_linear_search)

            if solution is not None:
                if not is_a_valid_solution(**solution):
                    raise Exception("The solution is not valid")

                if json_solutions:
                    with open(os.path.join(solutions_output_dir, os.path.splitext(file)[0] + ".json"), 'w') as f:
                        json.dump(solution, f, cls=NpEncoder)
                else:
                    with open(os.path.join(solutions_output_dir, os.path.splitext(file)[0] + ".txt"), 'w') as f:
                        f.write(solution_to_txt(**solution))

                if draw_solutions:
                    draw_board(**solution, output_file=os.path.join(solutions_output_dir, os.path.splitext(file)[0] +
                                                                    ".png"))

            csv_writer.writerow(
                {
                    'instance': os.path.splitext(file)[0],
                    'optimally_solved': optimal_solution,
                    'board_height': solution['board_height'] if solution is not None else None,
                    'elapsed_ms': elapsed_ms
                }
            )
            csv_stats_file.flush()

    csv_stats_file.close()


if __name__ == "__main__":
    run()
