import random
import typing

import matplotlib.patches
import matplotlib.pyplot as plt
import matplotlib.colors


def read_mzn_solution(solution_filepath: str) -> typing.Dict[str, typing.Any]:
    with open(solution_filepath, "r") as solution_file:
        lines = list(filter(None, solution_file.read().split("\n")))

    board_length, board_height = tuple(map(lambda s: int(s), lines[0].split(" ")))
    n_circuit = int(lines[1])

    circuit_width, circuit_height, circuit_x, circuit_y = list(), list(), list(), list()

    for i in range(2, len(lines)):
        w, h, x, y = tuple(map(lambda s: int(s), lines[i].split(" ")))
        circuit_width.append(w)
        circuit_height.append(h)
        circuit_x.append(x)
        circuit_y.append(y)

    return {
        "board_length": board_length,
        "board_height": board_height,
        "n_circuit": n_circuit,
        "circuit_width": circuit_width,
        "circuit_height": circuit_height,
        "circuit_x": circuit_x,
        "circuit_y": circuit_y
    }


def get_cmap(n, name='hsv'):
    return plt.cm.get_cmap(name, n)


def draw_board(board_width: int, board_height: int, n_circuits: int, widths: typing.List[int],
               heights: typing.List[int], x: typing.List[int], y: typing.List[int],
               output_file: str = None, color_map: str = None, shuffle_colors: bool = True):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    plt.xlim([0, board_width])
    plt.ylim([0, board_height])

    cmap = get_cmap(n_circuits, color_map)
    colors = [cmap(i) for i in range(n_circuits)]
    if shuffle_colors:
        random.shuffle(colors)

    for i in range(n_circuits):
        circuit = matplotlib.patches.Rectangle((x[i], y[i]),
                                               widths[i], heights[i],
                                               color=colors[i])
        ax.add_patch(circuit)

    if output_file is not None:
        plt.savefig(output_file)

    plt.show()


def dict_to_dzn(board_width: int, board_height: int, n_circuits: int, widths: typing.List[int],
                heights: typing.List[int], x: typing.List[int], y: typing.List[int]):
    dzn = "board_width = %d;\n" % board_width
    dzn += "board_height = %d;\n" % board_height
    dzn += "n_circuits = %d;\n" % n_circuits
    dzn += "widths = [%s];\n" % ", ".join([str(w) for w in widths])
    dzn += "heights = [%s];\n" % ", ".join([str(h) for h in heights])
    dzn += "x = [%s];\n" % ", ".join([str(x_) for x_ in x])
    dzn += "y = [%s];\n" % ", ".join([str(y_) for y_ in y])

    return dzn
