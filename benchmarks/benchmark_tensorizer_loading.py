from tensorizer.utils import get_mem_usage
from dataclasses import dataclass
from vllm.model_executor.model_loader.tensorizer import TensorizerConfig
import os
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from time import time
import argparse

@dataclass
class BenchmarkStatistics:
    model_name: str
    initial_mem_usage: str
    final_mem_usage: str
    initial_time: str
    final_time: str

    def __post_init__(self):
        self.duration = float(self.final_time) - float(self.initial_time)
    def append_to_csv(self, path: str):
        if not os.path.exists(path):
            with open(path, "w") as f:
                f.write(
                    "load_format,model,mem_usage_before,mem_usage_after,"
                    "duration,cache_status\n")
        with open(path, 'a') as f:
            f.write(f"tensorizer,{self.model_name},{self.initial_mem_usage},"
                    f"{self.final_mem_usage},{self.duration},omitted")

shared_params = {
    "num_readers": 6,
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--path_to_tensors", type=str)
    parser.add_argument("--output_path", type=str)
    return parser.parse_args()

def benchmark_load(model, tensorizer_config, path):
    initial_mem_usage = get_mem_usage()
    initial_time = time()
    async_engine_args = AsyncEngineArgs(
        load_format="tensorizer",
        model=model,
        model_loader_extra_config=tensorizer_config
    )
    AsyncLLMEngine.from_engine_args(async_engine_args)
    final_mem_usage = get_mem_usage()
    final_time = time()
    return BenchmarkStatistics(
        model_name=model,
        initial_mem_usage=initial_mem_usage,
        final_mem_usage=final_mem_usage,
        initial_time=initial_time,
        final_time=final_time
    ).append_to_csv(path)
    return


if __name__ == "__main__":
    args = parse_args()
    config = TensorizerConfig(
        tensorizer_uri=args.path_to_tensors,
        **shared_params
    )
    benchmark_load(args.model, config, args.output_path)
