# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Example usage:

python -m tessel.draw --planfile plan.json --outfile plan.png
"""

from typing import Tuple, Optional
from tessel.schedule.schedplan import SchedPlan, Block

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import AutoMinorLocator

import argparse


class Painter:

    @staticmethod
    def visualize(schedplan: SchedPlan, outfile: Optional[str] = None):
        """
        Visualize a schedule plan

        @param schedplan SchedPlan: Tessel schedule plan
        @param outfile Optional[str]: output filename. If None, will show in window
        """
        plt.close('all')
        fig, ax = plt.subplots(figsize=(4 * schedplan.nsteps // schedplan.ndevs, 4))
        renderer = fig.canvas.get_renderer()

        # xaxis
        ax.set_xlim((0, schedplan.nsteps))
        plt.xticks(
            ticks=np.arange(0.5, schedplan.nsteps+0.5, 1.0, dtype=float),
            labels=np.arange(1, schedplan.nsteps+1, 1, dtype=int)
        )
        minor_locator = AutoMinorLocator(2)
        plt.gca().xaxis.set_minor_locator(minor_locator)
        ax.xaxis.grid(which='minor', linestyle='--')
        # yaxis
        ax.set_ylim((0.5, schedplan.ndevs+0.5))
        plt.yticks(np.arange(1, schedplan.ndevs+1, 1, dtype=int))
        ax.invert_yaxis()

        fontsize = [40]
        txts = list()
        def draw_block(block: Block, position: Tuple[Tuple[int], int], fontsize):
            color = '#4472C4' if block.btype == "forward" else '#ED7D31'
            devs, step = position
            for dev in devs:
                rec = Rectangle((step, dev+0.5), block.span, 1, color=color, ec='black', lw=1.5)
                ax.add_artist(rec)
                rx, ry = rec.get_xy()
                cx = rx + rec.get_width() / 2.0
                cy = ry + rec.get_height() / 2.0
                anno = str(block.mid)
                txt = ax.text(x=cx, y=cy, s=anno, fontsize=40, ha='center', va='center', color='w')
                rbox = rec.get_window_extent(renderer)
                for fs in range(fontsize[0], 1, -2):
                    txt.set_fontsize(fs)
                    tbox = txt.get_window_extent(renderer)
                    if tbox.x0 > rbox.x0 and tbox.x1 < rbox.x1 and tbox.y0 > rbox.y0 and tbox.y1 < rbox.y1:
                        break
                fontsize[0] = min(fontsize[0], fs)
                txts.append(txt)

        for block in schedplan._blocks:
            devs = schedplan._block_devices[block]
            step = schedplan._block_steps[block]
            draw_block(block, (devs, step), fontsize)

        if schedplan.repetend is not None:
            for step in schedplan.repetend:
                plt.axvline(step, linewidth=8, color='red')

        # set fontsize to same
        fontsize = fontsize[0]
        for txt in txts:
            txt.set_fontsize(fontsize)
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(fontsize)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(fontsize)
        plt.xlabel('Time Step', fontsize=fontsize)
        plt.ylabel('Device', fontsize=fontsize)
        plt.tight_layout()
        if outfile:
            plt.savefig(outfile)
        else:
            plt.show()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='visualization')
    parser.add_argument('--planfile', type=str,
                        help='json file plan location')
    parser.add_argument('--outfile', type=str,
                        help='output file name')
    args = parser.parse_args()

    sched = SchedPlan.load(args.planfile)
    sched.visualize(args.outfile)
