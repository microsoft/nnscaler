import torch

from unittest.mock import patch
import pytest

from nnscaler.profiler.benchmark_comm import main

from ..launch_torchrun import launch_torchrun


def comm_profile_worker(tmp_path):
    def patched_save_path(*args, **kwargs):
        return tmp_path

    with patch(
        "nnscaler.profiler.benchmark_comm.get_default_profile_path",
        side_effect=patched_save_path
    ):
        main()


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 2, reason='lack of gpu devices')
def test_comm_profile(tmp_path):
    # just a smoke test
    launch_torchrun(2, comm_profile_worker, tmp_path)
    assert (tmp_path / 'comm' / 'intra_2.json').exists()
    launch_torchrun(2, comm_profile_worker, tmp_path)
    comm_bakup_dirs = list(tmp_path.glob('comm-bak-*'))
    assert len(comm_bakup_dirs) == 1
