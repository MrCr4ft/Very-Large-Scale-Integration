import os
import typing
import json

import click
import numpy as np

from utils.heuristics import lower_bound, kp01_upper_bound


def get_circuits_dims(lines: typing.List[str]) -> typing.Tuple[typing.List[int], typing.List[int]]:
    w = []
    h = []
    for line in lines:
        dims = line.split(" ")
        w.append(int(dims[0]))
        h.append(int(dims[1]))
    return w, h


def height_lower_and_upper_bounds(widths: typing.List[int], heights: typing.List[int], board_width: int) \
        -> typing.Tuple[int, int]:
    widths = np.array(widths)
    heights = np.array(heights)
    return int(lower_bound(widths, heights, board_width)), int(kp01_upper_bound(widths, heights, board_width))


def convert_instance_to_dzn_format(instance: str, dzn_parnames: typing.Dict[str, str]) -> typing.List[str]:
    lines = list(filter(None, instance.split("\n")))  # drop empty lines
    converted_lines: typing.List[str] = list()

    converted_lines.append(dzn_parnames["length"] + " = " + lines[0] + ";")
    converted_lines.append(dzn_parnames["n_circuits"] + " = " + lines[1] + ";")

    width_line = dzn_parnames["width"] + " = [ "
    height_line = dzn_parnames["height"] + " = [ "
    widths, heights = get_circuits_dims(lines[2:])
    for i in range(len(widths) - 1):
        width_line += str(widths[i]) + ", "
        height_line += str(heights[i]) + ", "
    width_line += str(widths[-1]) + " ];"
    height_line += str(heights[-1]) + " ];"

    converted_lines.append(width_line)
    converted_lines.append(height_line)

    height_lower_bound, height_upper_bound = height_lower_and_upper_bounds(widths, heights, int(lines[0]))
    converted_lines.append(dzn_parnames["height_lower_bound"] + " = " + str(height_lower_bound) + ";")
    converted_lines.append(dzn_parnames["height_upper_bound"] + " = " + str(height_upper_bound) + ";")

    return converted_lines


def read_instance_as_dict(instance: str) -> typing.Dict:
    lines = list(filter(None, instance.split("\n")))
    board_width = int(lines[0])
    n_circuits = int(lines[1])
    widths, heights = get_circuits_dims(lines[2:])
    height_lower_bound, height_upper_bound = height_lower_and_upper_bounds(widths, heights, board_width)

    return {
        'n_circuits': n_circuits,
        'board_width': board_width,
        'widths': widths,
        'heights': heights,
        'height_lower_bound': height_lower_bound,
        'height_upper_bound': height_upper_bound
    }


@click.command()
@click.option('--instances-input-dir', type=click.Path(exists=True), required=True)
@click.option('--dzn-output-dir', type=click.Path(exists=False), required=True)
@click.option('--json-output-dir', type=click.Path(exists=False), required=True)
@click.option('--parnames-conversion-config', type=click.Path(exists=True), required=True)
def run(instances_input_dir: str, dzn_output_dir: str, json_output_dir: str,
        parnames_conversion_config: str):

    with open(parnames_conversion_config, "r") as parnames_conversion_config_file:
        dzn_parnames = json.load(parnames_conversion_config_file)

    if not os.path.exists(dzn_output_dir):
        os.makedirs(dzn_output_dir)

    if not os.path.exists(json_output_dir):
        os.makedirs(json_output_dir)

    for subdir, dirs, files in os.walk(instances_input_dir):
        for file in files:
            current_instance = open(os.path.join(subdir, file), 'r').read()

            converted_instance_dzn = '\n'.join(convert_instance_to_dzn_format(current_instance, dzn_parnames))
            with open(dzn_output_dir + file.split(".txt")[0] + ".dzn", "x") as converted_instance_dzn_file:
                converted_instance_dzn_file.write(converted_instance_dzn)

            instance_dict = read_instance_as_dict(current_instance)
            with open(json_output_dir + file.split(".txt")[0] + ".json", "x") as json_file:
                json.dump(instance_dict, json_file)


if __name__ == "__main__":
    run()
