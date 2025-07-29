import os
from typing import List
from time import time

from verdict.log import logdebug
from verdict.config import Config


class Timer:
    def __init__(self):
        self.start_ts = {}
        self.end_ts = {}
        self.jobs: List[str] = []
        self.job_pid = {}

    def start(self, job: str):
        if not Config.time:
            return
        assert job not in self.start_ts, job
        self.jobs.append(job)
        self.job_pid[job] = os.getpid()
        self.start_ts[job] = time()

    def end(self, job: str):
        if not Config.time:
            return
        assert job not in self.end_ts and job in self.start_ts
        self.end_ts[job] = time()

    def display(self, print_fn=logdebug):
        if not Config.time:
            return
        for job in self.jobs:
            if os.getpid() != self.job_pid[job]:
                continue
            t = "N/A"
            if job in self.end_ts:
                t = self.get(job)
            row = ["⏱️ TIME", job, str(t)]
            print_fn(", ".join(row))

    def get(self, job: str):
        try:
            return self.end_ts[job] - self.start_ts[job]
        except:
            return 0


timer = Timer()
