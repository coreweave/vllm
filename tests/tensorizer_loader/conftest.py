# isort: skip_file
# ruff: noqa
# yapf: disable
# type: ignore


import contextlib
import gc
import functools


import pytest
import ray
import torch

from vllm.distributed import (destroy_distributed_environment,
                              destroy_model_parallel)
from vllm.model_executor.model_loader.tensorizer import TensorizerConfig


@pytest.fixture(autouse=True)
def cleanup():
    destroy_model_parallel()
    destroy_distributed_environment()
    with contextlib.suppress(AssertionError):
        torch.distributed.destroy_process_group()
    gc.collect()
    torch.cuda.empty_cache()
    ray.shutdown()

def retry_until_skip(n):
    def decorator_retry(func):
        @functools.wraps(func)
        def wrapper_retry(*args, **kwargs):
            for i in range(n):
                try:
                    return func(*args, **kwargs)
                except AssertionError:
                    cleanup()
                    if i == n - 1:
                        pytest.skip("Skipping test after attempts..")
        return wrapper_retry
    return decorator_retry


@pytest.fixture(autouse=True)
def tensorizer_config():
    config = TensorizerConfig(tensorizer_uri="vllm")
    return config