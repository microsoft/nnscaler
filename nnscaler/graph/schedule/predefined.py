"""
Common scheduling descriptions
"""

from typing import List

from nnscaler.graph.schedule.schedplan import SchedulePlan
from nnscaler.graph.graph import IRGraph
from nnscaler.graph.segment import IRSegment


class PredefinedSched:

    @staticmethod
    def sched_1f1b(graph: IRGraph, num_microbatches: int, num_stages: int) -> SchedulePlan:
        """
        1F1B scheduling. The graph should be staged into segments.

        An illustration of scheduling schema (the number is micro-batch index):
        ```
        f0    f1    f2    | f3 b0 |    b1    b2    b3
           f0    f1    f2 | b0 f3 | b1    b2    b3
              f0    f1 b0 | f2 b1 | f3 b2    b3
                 f0 b0 f1 | b1 f2 | b2 f3 b3
        ```
        """
        if num_microbatches <= 0:
            raise ValueError(f"expected num_microbatches > 0, but got {num_microbatches} ")
        segments: List[IRSegment] = graph.select(ntype=IRSegment, flatten=False)
        fsegs = [seg for seg in segments if seg.isfw()]
        assert len(fsegs) == num_stages, f"Mismatch of forward segement number ({len(fsegs)}) with num_stages ({num_stages})"

        # describe schedule
        sched = SchedulePlan(graph, num_microbatches)

        wait_steps = [sid for sid in range(num_stages)]
        bw_ofst = [num_stages - 1 - sid for sid in range(num_stages)]
        total_steps = num_microbatches * 2 + (num_stages - 1) * 2

        for step in range(total_steps):
            for sid in range(num_stages):
                ofst = wait_steps[sid]
                if step < ofst: continue
                fw_idx = (step - ofst) // 2
                # forward or backward segment
                segment = fsegs[sid] if (step - ofst) % 2 == 0 else fsegs[sid].mirror
                mb_idx = fw_idx if (step - ofst) % 2 == 0 else fw_idx - bw_ofst[sid]
                # append for execution
                if mb_idx < 0 or mb_idx >= num_microbatches: continue
                sched.add_segment(segment, mb_idx, step)
        sched.finish()
        return sched

    @staticmethod
    def sched_1f1b_plus(graph: IRGraph, num_microbatches: int, num_stages: int) -> SchedulePlan:
        """1F1B Plus Scheduling.

        f0 f0    f1 f1    f2 f2    | f3 f3 b0 |    b1    b2    b3
        f0    f0 f1    f1 f2    f2 | f3 b0 f3 | b1    b2    b3
        f0       f1 f0    f2 f1 b0 | f3 f2 b1 | f3 b2    b3
        f0       f1    f0 f2 b0 f1 | f3 b1 f2 | b2 f3 b
        """
        if num_microbatches <= 0:
            raise ValueError(f"expected num_microbatches > 0, but got {num_microbatches} ")
        segments: List[IRSegment] = graph.select(ntype=IRSegment, flatten=False)
        fsegs = [seg for seg in segments if seg.isfw()]
        tp_fsegs = [seg for seg in fsegs if len(seg.device) == len(graph.device)]
        fb_fsegs = [seg for seg in fsegs if seg not in tp_fsegs]
        assert len(fb_fsegs) == num_stages, f"got only {len(fb_fsegs)} stages but need {num_stages} stages"
        assert all(tuple(seg.device) == tuple(graph.device) for seg in tp_fsegs)

        # describe schedule
        sched = SchedulePlan(graph, num_microbatches)

        wait_steps = [sid for sid in range(num_stages)]
        bw_ofst = [num_stages - 1 - sid for sid in range(num_stages)]
        total_steps = num_microbatches * 2 + (num_stages - 1) * 2

        # 1f1b schedule
        for step in range(total_steps):
            for sid in range(num_stages):
                ofst = wait_steps[sid]
                if step < ofst: continue
                fw_idx = (step - ofst) // 2
                # forward or backward segment
                segment = fb_fsegs[sid] if (step - ofst) % 2 == 0 else fb_fsegs[sid].mirror
                mb_idx = fw_idx if (step - ofst) % 2 == 0 else fw_idx - bw_ofst[sid]
                # append for execution
                if mb_idx < 0 or mb_idx >= num_microbatches: continue
                sched.add_segment(segment, mb_idx, step)

        # insert
        for mid in range(num_microbatches):
            for tp_seg in tp_fsegs:
                # TODO: not work case: tp_seg at tail fsegs
                next_seg = fsegs[fsegs.index(tp_seg)+1]
                assert next_seg in fsegs
                insert_fw, insert_bw = False, tp_seg.mirror is None
                if tp_seg.mirror is not None:
                    assert next_seg.mirror is not None

                for step in range(sched.nsteps-1, -1, -1):
                    segments = [blk.content for blk in sched.segments(step) if blk.mid == mid]
                    # insert forward
                    if next_seg in segments:
                        sched.insert_step(step, tp_seg, mid, 1)
                        assert not insert_fw
                        insert_fw = True
                    # insert backward
                    if next_seg.mirror in segments:
                        sched.insert_step(step+1, tp_seg.mirror, mid, 1)
                        assert not insert_bw
                        insert_bw = True
                    if insert_fw and insert_bw: break

                assert insert_fw and insert_bw, (
                    f'find one segment cannot be inserted in schedplan: ',
                    f'mid: {mid}, fw: {insert_fw}, bs: {insert_bw}')

        sched.finish()
        # print(sched)
        return sched

    @staticmethod
    def sched_gpipe(graph: IRGraph, num_microbatches: int, num_stages: int) -> SchedulePlan:
        """
        GPipe scheduling. The graph should be staged into segments.

        An illustration of scheduling schema (the number is micro-batch index):
        ```
        f0 f1 f2 f3                   b0 b1 b2 b3
           f0 f1 f2 f3             b0 b1 b2 b3
              f0 f1 f2 f3       b0 b1 b2 b3
                 f0 f1 f2 f3 b0 b1 b2 b3
        ```
        """
        if num_microbatches <= 0:
            raise ValueError(f"expected num_microbatches > 0, but got {num_microbatches} ")
        segments: List[IRSegment] = graph.select(ntype=IRSegment, flatten=False)
        fsegs = [seg for seg in segments if seg.isfw()]
        assert len(fsegs) == num_stages, "Mismatch of forward segement number with num_stages"
        # describe schedule
        sched = SchedulePlan(graph, num_microbatches)

        fwait_steps = [sid for sid in range(num_stages)]
        bwait_steps = [num_stages - 1 - sid for sid in range(num_stages)]

        total_steps = num_microbatches * 2 + (num_stages - 1) * 2
        middle_step = total_steps // 2
        for step in range(total_steps):
            for sid in range(num_stages):
                segment = fsegs[sid] if step < middle_step else fsegs[sid].mirror
                mb_idx = step - fwait_steps[sid] if step < middle_step else step - middle_step - bwait_steps[sid]
                if mb_idx < 0 or mb_idx >= num_microbatches: continue
                sched.add_segment(segment, mb_idx, step)
        sched.finish()
        return sched

    @staticmethod
    def sched_chimera_direct(graph: IRGraph, num_microbatches: int, num_stages: int):
        """Chimera-direct scheduling.

        The graph should be staged into segments.

        An illustration of scheduling schema (the number is micro-batch index):
        ```
        f0    f1 f2 b2-b2 f3    b3-b3 b0-b0       b1-b1
           f0 f2 f1 f3    b2-b2 b0-b0 b3-b3 b1-b1
           f2 f0 f3 f1    b0-b0 b2-b2 b1-b1 b3-b3
        f2    f3 f0 b0-b0 f1    b1-b1 b2-b2       b3-b3

        0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 (-> steps)
        ```

        Note the f0 and f2 (step 0) should be considered to be one segment in graph.
        """
        if num_microbatches <= 0:
            raise ValueError(f"expected num_microbatches > 0, but got {num_microbatches} ")
        segments: List[IRSegment] = graph.select(ntype=IRSegment, flatten=False)
        fsegs = [seg for seg in segments if seg.isfw()]
        assert len(fsegs) == 4, f"Chimera-direct scheduling only applies for 4 segments, but {len(fsegs)} detected"
        sched = SchedulePlan(graph, num_microbatches)
        assert num_microbatches % 2 == 0
        mid = 0
        while mid < num_microbatches:
            ofst = 16 * (mid // 2)
            # first micro-batch
            sched.add_segment(fsegs[0], micro_batch_id=mid, step=max(0, 0+ofst-3))  # tight compact
            sched.add_segment(fsegs[1], micro_batch_id=mid, step=max(1, 1+ofst-3))  # tight compact
            sched.add_segment(fsegs[2], micro_batch_id=mid, step=2+ofst)
            sched.add_segment(fsegs[3], micro_batch_id=mid, step=3+ofst)
            sched.add_segment(fsegs[3].mirror, micro_batch_id=mid, step=4+ofst, span=2)
            sched.add_segment(fsegs[2].mirror, micro_batch_id=mid, step=6+ofst, span=2)
            sched.add_segment(fsegs[1].mirror, micro_batch_id=mid, step=8+ofst, span=2)
            sched.add_segment(fsegs[0].mirror, micro_batch_id=mid, step=10+ofst, span=2)
            # second micro-batch
            sched.add_segment(fsegs[0], micro_batch_id=mid+1, step=2+ofst)
            sched.add_segment(fsegs[1], micro_batch_id=mid+1, step=3+ofst)
            sched.add_segment(fsegs[2], micro_batch_id=mid+1, step=4+ofst)
            sched.add_segment(fsegs[3], micro_batch_id=mid+1, step=6+ofst)
            sched.add_segment(fsegs[3].mirror, micro_batch_id=mid+1, step=8+ofst, span=2)
            sched.add_segment(fsegs[2].mirror, micro_batch_id=mid+1, step=10+ofst, span=2)
            sched.add_segment(fsegs[1].mirror, micro_batch_id=mid+1, step=12+ofst, span=2)
            sched.add_segment(fsegs[0].mirror, micro_batch_id=mid+1, step=14+ofst, span=2)
            # update
            mid += 2
        sched.finish()
        return sched


    @staticmethod
    def sched_infer_pipe(graph: IRGraph, num_microbatches: int, num_stages: int) -> SchedulePlan:
        """
        Inference pipeline scheduling. The graph should be staged into segments.

        An illustration of scheduling schema (the number is micro-batch index):
        ```
        f0 f1 f2 f3
           f0 f1 f2 f3
              f0 f1 f2 f3
                 f0 f1 f2 f3
        ```
        """
        fsegs: List[IRSegment] = graph.select(ntype=IRSegment, flatten=False)
        assert all(seg.isfw() for seg in fsegs), f"Detect backward. The predefined scheduling only applies for inference"
        assert len(fsegs) == num_stages, "Mismatch of forward segement number with num_stages"
        # describe schedule
        sched = SchedulePlan(graph, num_microbatches)
        fwait_steps = [sid for sid in range(num_stages)]
        total_steps = num_microbatches + num_stages - 1
        for step in range(total_steps):
            for sid in range(num_stages):
                segment = fsegs[sid]
                mb_idx = step - fwait_steps[sid]
                if mb_idx < 0 or mb_idx >= num_microbatches: continue
                sched.add_segment(segment, mb_idx, step)
        sched.finish()
        return sched
