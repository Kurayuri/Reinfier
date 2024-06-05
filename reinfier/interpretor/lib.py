from ..common.DRLP import DRLP
from ..common.classes import VerificationAnswer, Breakpoint
from ..import drlp
from ..import util
from typing import List, Tuple
import numpy


def analyze_breakpoints(breakpoints: List[Breakpoint]) -> Tuple[List[List], List[List]]:
    inline_breaklines = []
    inline_breakpoints = []

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

    if len(breakpoints) == 1:
        prt(breakpoints[0])

    curr_break_lines = []
    curr_break_points = []
    inline_breaklines.append(curr_break_lines)
    inline_breakpoints.append(curr_break_points)

    for i in range(1, len(breakpoints)):
        if list(breakpoints[i - 1].property.variables.items())[:-1] == list(breakpoints[i].property.variables.items())[:-1]:
            prt(breakpoints[i - 1], breakpoints[i], curr_break_lines, curr_break_points)
        else:
            util.log("\nNew line:")
            curr_break_lines = []
            curr_break_points = []
            inline_breaklines.append(curr_break_lines)
            inline_breakpoints.append(curr_break_points)

    return inline_breakpoints, inline_breaklines
