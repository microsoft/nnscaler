from typing import List, Dict
from pathlib import Path
import pickle
from tqdm import tqdm
import psutil

from verdict.backend import GraphBackend, SymbolicBackend
from verdict.stage import Stage, cut_stages, run_stage
from verdict.graph import DFG, WType, World
from verdict.config import Config
from verdict.log import logerr, loginfo, logdebug, logwarn
from verdict.debug_print import dump_stages, dump_nodes, dump_lineages
from verdict.timer import timer, Timer


_GLOBAL_GS = None
_GLOBAL_GP = None
_GLOBAL_SYMBOLIC_BACKEND = None
_GLOBAL_STAGES = None

def _init_pool(gs, gp, symb_backend, stages):
    global _GLOBAL_GS, _GLOBAL_GP, _GLOBAL_SYMBOLIC_BACKEND, _GLOBAL_STAGES
    _GLOBAL_GS = gs
    _GLOBAL_GP = gp
    _GLOBAL_SYMBOLIC_BACKEND = symb_backend
    _GLOBAL_STAGES = stages

class StageParallelVerifier:

    def __init__(
        self,
        Gs_path: str,
        Ws_path: str,
        Gp_path: str,
        Wp_path: str,
        graph_backend: GraphBackend,
        symbolic_backend: SymbolicBackend,
    ):
        self.Gs_path: str = Gs_path
        self.Ws_path: str = Ws_path
        self.Gp_path: str = Gp_path
        self.Wp_path: str = Wp_path
        self.graph_backend: GraphBackend = graph_backend
        self.symbolic_backend: SymbolicBackend = symbolic_backend

        self.Wp: World = None

    def launch(self) -> None:
        cfg = Config

        # load graph
        loginfo("⏩ Loading graphs.")
        Gs = self.load_graph_w_cache(self.Gs_path, self.Ws_path, WType.S)
        Gp = self.load_graph_w_cache(self.Gp_path, self.Wp_path, WType.P)
        self.Wp = Gp.W
        loginfo("🚩 Finish loading graphs.")

        # dump nodes
        path_dump_snodes = cfg.log_dir / Gs.ID / "snodes.txt"
        loginfo("📘", snodes=path_dump_snodes)
        path_dump_pnodes = cfg.log_dir / Gp.ID / "pnodes.txt"
        loginfo("📘", pnodes=path_dump_pnodes)
        if cfg.dump_nodes and (not path_dump_snodes.exists() or not cfg.use_cache_nodes):
            dump_nodes(Gs.nodes(), Gs, path_dump_snodes)
        if cfg.dump_nodes and (not path_dump_pnodes.exists() or not cfg.use_cache_nodes):
            dump_nodes(Gp.nodes(), Gp, path_dump_pnodes)

        # build lineages and cut stages
        loginfo("⏩ Determining stages.")
        stages: List[Stage] = self.cut_stages_w_cache(Gs, Gp)
        
        # dump stages
        path_dump_stages = cfg.log_dir / Gp.ID / "stages.txt"
        loginfo("📘", stages=path_dump_stages)
        if cfg.dump_stages and not cfg.use_cache_stages:
            dump_stages(stages, Gs, Gp, path_dump_stages, True)

        
        # NOTE: The following muted implementation uses multiprocessing Pool for parallel stage
        # workers. However, since pool processes do not terminate for each job, the underlying
        # z3 C code does not free its memory even if python del the variable references.
        # So, we use manual os.fork instead, which creates a new process for each stage worker,
        # allowing z3 to release memory once each stage is finished. Though the fork-method
        # slows down the e2e execution time by ~2 times, it perfectly bound the memory footprint.
        
        # ==================== MULTIPROCESSING STAGE PARALLEL ==================== 
        import multiprocessing
        import traceback
        from functools import partial
        
        timer.start("run stages")
        # parallel run stages
        error_flag = multiprocessing.Value('b', False)  # Shared bool for error signaling
        counter = multiprocessing.Value('i', len(stages))
        with multiprocessing.Pool(
            processes=cfg.max_vrf_proc, initializer=_init_pool, initargs=(Gs, Gp, self.symbolic_backend, stages), maxtasksperchild=50
        ) as pool:
            with tqdm(total=len(stages), desc="parallel verifying stages") as pbar:
                def callback_with_progress(result: bool, stage_id: int):
                    mem = psutil.virtual_memory().percent
                    if cfg.mem and stage_id % 1000 == 0:
                        loginfo("MEM", stage=stage_id, mem=f"{mem} %")
                    with counter.get_lock():
                        pbar.update(1)
                        # terminate the pool if all jobs are finished
                        # without this explicit release, pool.join() may block forever.
                        counter.value -= 1
                        if counter.value == 0:
                            pool.terminate()
                    
                    if not result:
                        logerr("Stage equivalence fails.", Stage=stage_id, stage_eq=result)
                        error_flag.value = True
                        pool.terminate()

                def error_callback(e: Exception, stage_id: int):
                    exc_str = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
                    logerr("Stage worker raised an exception.", Stage=stage_id, Trace=exc_str)
                    error_flag.value = True
                    pool.terminate()
                    
                for stage_idx, stage in enumerate(stages):
                    pool.apply_async(
                        _worker,
                        args=(stage_idx,),
                        callback=partial(callback_with_progress, stage_id=stage.id),
                        error_callback=partial(error_callback, stage_id=stage.id),
                    )

                pool.close()
                pool.join()
                
        timer.end("run stages")
                
        if error_flag.value:
            loginfo("❌ FAIL")
            raise RuntimeError("One or more stage workers failed.")
        else:
            loginfo("✅ SUCCESS")
        # ==================== MULTIPROCESSING STAGE PARALLEL ==================== 
        
        # # ==================== FORK STAGE PARALLEL ==================== 
        # import os
        # timer.start("run stages")
        # with tqdm(total=len(stages), desc="parallel verifying stages") as pbar:
        #     pipes = {}
            
        #     def wait_a_stage_worker_proc():
        #         pbar.update(1)
        #         pid, status = os.waitpid(-1, 0)
        #         stage_idx, read_pipe = pipes[pid]
        #         # mem profile
        #         mem = psutil.virtual_memory().percent
        #         if cfg.mem and stage_idx % 1000 == 0:
        #             loginfo("MEM", stage_idx=stage_idx, mem=f"{mem} %")
        #         # examine worker result
        #         with os.fdopen(read_pipe, 'rb') as rf:
        #             try:
        #                 result = pickle.load(rf)  # Expecting a bool
        #             except Exception as e:
        #                 result = False
        #                 logerr("Failed to read result", stage=stage_idx, error=str(e))
        #         del pipes[pid]
        #         # terminate if a worker fails
        #         if not os.WIFEXITED(status) or os.WEXITSTATUS(status) != 0 or not result:
        #             logerr("Stage failed", stage=stage_idx, result=result)
        #             for ppid in pipes:
        #                 os.kill(ppid, 9)
        #             loginfo("❌ FAIL")
        #             raise RuntimeError("One or more stage workers failed.")
                    
        #     for stage_idx, stage in enumerate(stages):
        #         while len(pipes) >= cfg.max_vrf_proc:
        #             wait_a_stage_worker_proc()
        #         read_pipe, write_pipe = os.pipe()
        #         pid = os.fork()
        #         if pid == 0:
        #             try:
        #                 os.close(read_pipe)
        #                 result = run_stage(stage, Gs, Gp, self.symbolic_backend)
        #                 with os.fdopen(write_pipe, 'wb') as wf:
        #                     pickle.dump(result, wf)
        #             except Exception as e:
        #                 # Fail-safe: write False if anything crashes
        #                 try:
        #                     with os.fdopen(write_pipe, 'wb') as wf:
        #                         pickle.dump(False, wf)
        #                 except:
        #                     pass
        #             os._exit(0)
        #         else:
        #             os.close(write_pipe)
        #             pipes[pid] = (stage_idx, read_pipe)
                
        #     while pipes:
        #         wait_a_stage_worker_proc()
                
        # timer.end("run stages")

        # loginfo("✅ SUCCESS")
        # # ==================== FORK STAGE PARALLEL ==================== 
        
        
    def load_graph_w_cache(self, G_path: str, W_path: str, wtype: WType) -> DFG:
        cfg = Config
        name = "snodes" if wtype == WType.S else "pnodes"
        cache_path = cfg.cache_dir / Path(G_path).stem / f"{name}.pkl"
        if cfg.use_cache_nodes and cache_path.exists():
            try:
                with open(cache_path, "rb") as f:
                    loginfo("📀 Loading cached graph.", wtype=wtype.value)
                    ret = pickle.load(f)
                    return ret
            except Exception as e:
                logwarn("Fail to load graph cache.", path=cache_path, err=str(e))
                cache_path.unlink()

        load_graph = self.graph_backend.load_graph

        loginfo("⏩ Building graph from scratch.", G_path=G_path)
        timer.start(f"load G{wtype.value}")
        dfg = load_graph(G_path, W_path, wtype)
        timer.end(f"load G{wtype.value}")

        cache_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
        with open(cache_path, "wb") as f:
            loginfo(f"💾 Caching G{wtype.value}.")
            pickle.dump(dfg, f)
        return dfg

    def cut_stages_w_cache(self, Gs: DFG, Gp: DFG) -> List[Stage]:
        cfg = Config
        cache_path = cfg.cache_dir / Gp.ID / "stages.pkl"
        if cfg.use_cache_stages and cache_path.exists():
            try:
                with open(cache_path, "rb") as f:
                    loginfo("📀 Loading cached stages.")
                    ret = pickle.load(f)
                    return ret
            except Exception as e:
                logwarn("Fail to load stage cache.", path=cache_path, err=str(e))
                cache_path.unlink()

        # build lineages and cut stages
        timer.start("align lineages")
        loginfo("⏩ Inferring lineages.")
        ordered_lngs = self.graph_backend.get_ordered_lineages(Gs, Gp)
        timer.end("align lineages")

        if cfg.dump_lineages:
            path_dump_lngs = cfg.log_dir / Gp.ID / "lineages.txt"
            loginfo("📘", lineages=path_dump_lngs)
            dump_lineages(ordered_lngs, Gs, Gp, path_dump_lngs)

        timer.start("cut stages")
        loginfo("⏩ Scheduling stages.")
        stages: List[Stage] = cut_stages(Gs, Gp, ordered_lngs)
        timer.end("cut stages")

        cache_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
        with open(cache_path, "wb") as f:
            loginfo("💾 Caching stages.")
            pickle.dump(stages, f)
        return stages

def _worker(stage_idx: int) -> bool:
    stage: Stage = _GLOBAL_STAGES[stage_idx]
    # if stage.id >-1: return True
    logdebug("🎯 Start stage worker.", Stage=stage.id)
    result = run_stage(stage, _GLOBAL_GS, _GLOBAL_GP, _GLOBAL_SYMBOLIC_BACKEND)
    return result