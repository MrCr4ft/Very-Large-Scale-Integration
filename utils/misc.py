import random
import typing

import matplotlib.patches
import matplotlib.pyplot as plt
import matplotlib.colors


def mzn_solution_to_dict(mzn_solution: str) -> typing.Dict[str, typing.Any]:
    lines = list(filter(None, mzn_solution.split("\n")))

    board_length, board_height = tuple(map(lambda s: int(s), lines[0].split(" ")))
    n_circuits = int(lines[1])

    widths, heights, xs, ys = list(), list(), list(), list()

    for i in range(2, len(lines)):
        w, h, x, y = tuple(map(lambda s: int(s), lines[i].split(" ")))
        widths.append(w)
        heights.append(h)
        xs.append(x)
        ys.append(y)

    return {
        "board_width": board_length,
        "board_height": board_height,
        "n_circuits": n_circuits,
        "widths": widths,
        "heights": heights,
        "x": xs,
        "y": ys
    }


def get_cmap(n, name='hsv'):
    return plt.cm.get_cmap(name, n)


def draw_board(board_width: int, board_height: int, n_circuits: int, widths: typing.List[int],
               heights: typing.List[int], x: typing.List[int], y: typing.List[int],
               output_file: str = None, display_solution: bool = False,
               color_map: str = None, shuffle_colors: bool = True):
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

    if display_solution:
        plt.show()

    plt.close()


def to_dzn_input(board_width: int, board_height: int, n_circuits: int, widths: typing.List[int],
                 heights: typing.List[int], x: typing.List[int], y: typing.List[int]):
    dzn = "board_width = %d;\n" % board_width
    dzn += "board_height = %d;\n" % board_height
    dzn += "n_circuits = %d;\n" % n_circuits
    dzn += "widths = [%s];\n" % ", ".join([str(w) for w in widths])
    dzn += "heights = [%s];\n" % ", ".join([str(h) for h in heights])
    dzn += "x = [%s];\n" % ", ".join([str(x_) for x_ in x])
    dzn += "y = [%s];\n" % ", ".join([str(y_) for y_ in y])

    return dzn


def is_a_valid_solution(board_width: int, board_height: int, n_circuits: int, widths: typing.List[int],
                        heights: typing.List[int], x: typing.List[int], y: typing.List[int]):
    for i in range(n_circuits):
        if x[i] < 0 or x[i] + widths[i] > board_width:
            return False
        if y[i] < 0 or y[i] + heights[i] > board_height:
            return False
        for j in range(i + 1, n_circuits):
            if not (
                    (x[i] + widths[i] <= x[j]) or
                    (x[j] + widths[j] <= x[i]) or
                    (y[i] + heights[i] <= y[j]) or
                    (y[j] + heights[j] <= y[i])
            ):
                return False
    return True


def solution_to_txt(board_width: int, board_height: int, n_circuits: int, widths: typing.List[int],
                    heights: typing.List[int], x: typing.List[int], y: typing.List[int]):
    txt = "{} {}\n".format(board_width, board_height)
    txt += "{}\n".format(n_circuits)
    for i in range(n_circuits):
        txt += "{} {} {} {}\n".format(widths[i], heights[i], x[i], y[i])

    return txt
