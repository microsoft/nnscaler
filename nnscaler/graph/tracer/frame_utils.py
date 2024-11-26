#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from dataclasses import dataclass
import dis
import importlib
import inspect
from pathlib import Path
import sys
import traceback

from typing import List, Tuple, Optional


def get_instructions(back_times=1) -> Tuple[List[dis.Instruction], int]:
    """
    Get the instructions of the (back_times)-th frame from the bottom.

    Args:
        back_times: The number of frames to go back.
            By default (back_times=1), the instruction of the frame who call this function will be returned.

    Returns:
        A tuple of two elements:
            - A list of dis.Instruction objects in frame.
            - The index of the current instruction in the list.
    """
    frame = inspect.currentframe()
    assert frame is not None
    # the frame who call get_instruction
    calling_frame = frame.f_back
    for _ in range(back_times):
        calling_frame = calling_frame.f_back
    assert calling_frame is not None
    insts: List[dis.Instruction] = list(dis.get_instructions(calling_frame.f_code))

    if sys.version_info >= (3, 11):
        from bisect import bisect_left
        # bisect_left find the position where an element should be inserted in a sorted list to maintain the list’s order.
        # If the element already exists in the list,
        # bisect_left returns the position to the left of the first occurrence of that element.
        # here use bisect_left to find the position of calling_frame.f_lasti in the insts.
        cur = bisect_left(insts, calling_frame.f_lasti, key=lambda x: x.offset)
    else:
        # based on the assumption that most bytecodes in Python are two bytes,
        # dividing by 2 results in the sequence number of the instructions.
        cur = calling_frame.f_lasti // 2

    # From python doc:
    # EXTENDED_ARG(ext): Prefixes any opcode which has an argument too big to fit into the default one byte.
    # ext holds an additional byte which act as higher bits in the argument.
    # For each opcode, at most three prefixal EXTENDED_ARG are allowed, forming an argument from two-byte to four-byte.
    while insts[cur].opname == 'EXTENDED_ARG':
        cur += 1
    return insts, cur


def get_last_instruction(back_times=1) -> dis.Instruction:
    """
    Get the current instruction of the (back_times)-th frame from the bottom.

    Args:
        back_times: The number of frames to go back.
            By default (back_times=1), the instruction of the frame who call this function will be returned.

    Returns:
        The current instruction in that frame.
    """
    # +1 because the first frame is the frame of get_last_instruction
    insts, cur = get_instructions(back_times + 1)
    return insts[cur]


@dataclass
class FrameRecord:
    filename: str
    lineno: str
    line: str
    # the name of the frame is the function name
    name: str

    def __repr__(self) -> str:
        if self.filename:
            return f'File "{self.filename}", line {self.lineno}, in {self.name},  {self.line}'
        else:
            return ''


def get_frame_record() -> Optional[FrameRecord]:
    # record code frame, include filename, line number, and function name
    frame_record = None
    cube_path = str(Path(importlib.util.find_spec('nnscaler').origin).parent) + '/'  # the cube path
    torch_path = str(Path(importlib.util.find_spec('torch').origin).parent) + '/'  # the torch path
    ignore_dirs = [cube_path, torch_path]
    # the last frame is the current frame [get_frame_record], so we need to skip it
    for frame in traceback.extract_stack()[-2::-1]:
        if any(p in frame.filename for p in ignore_dirs):
            continue
        frame_record = FrameRecord(frame.filename, frame.lineno, frame.line, frame.name)
        break
    return frame_record

