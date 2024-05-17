from ..common.DRLP import DRLP
from ..import drlp
from ..import util
from typing import List, Tuple
import numpy


def analyze_break_points(break_points: List[Tuple[DRLP, Tuple[int, bool, numpy.ndarray]]]) -> Tuple[List[List], List[List]]:
    inline_break_lines = []
    inline_break_points = []

    def prt(src, dst: None, curr_break_lines, curr_break_points):
        if dst is None:
            util.log("Single point:", src[0].kwargs, src[1][0])
        else:
            curr_break_lines.append((src[1][0], dst[1][0], src[0].kwargs, dst[0].kwargs))
            if src[1][0] == dst[1][0]:
                util.log("Line Keep  : %5s == %5s" % (src[1][0], dst[1][0]), src[0].kwargs, dst[0].kwargs)
            else:
                util.log("Line Change: %5s -> %5s" % (src[1][0], dst[1][0]), src[0].kwargs, dst[0].kwargs)
                curr_break_points.append((src[1][0], dst[1][0], dst[0].kwargs))

    if len(break_points) == 1:
        prt(break_points[0])

    curr_break_lines = []
    curr_break_points = []
    inline_break_lines.append(curr_break_lines)
    inline_break_points.append(curr_break_points)


    for i in range(1, len(break_points)):

        if list(break_points[i - 1][0].kwargs.items())[:-1] == list(break_points[i][0].kwargs.items())[:-1]:
            # if break_points[i - 1][1][0] == break_points[i][1][0]:
            prt(break_points[i - 1], break_points[i], curr_break_lines, curr_break_points)
        else:
            util.log("\nNew line:")

            curr_break_lines = []
            curr_break_points = []
            inline_break_lines.append(curr_break_lines)
            inline_break_points.append(curr_break_points)

    return inline_break_points, inline_break_lines
