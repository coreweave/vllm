# SPDX-License-Identifier: Apache-2.0
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from vllm.model_executor.model_loader.tensorizer import tensorize_lora_adapter
from vllm.model_executor.model_loader.tensorizer_loader import TensorizerConfig

save_dir = "my_lora_dir"
tensorizer_config = TensorizerConfig(lora_dir=save_dir)
lora_path = "yard1/llama-2-7b-sql-lora-test"

tensorize_lora_adapter(lora_path, tensorizer_config)
tensorizer_config = TensorizerConfig(lora_dir=save_dir)

llm = LLM(model="meta-llama/Llama-2-7b-hf",
          load_format="tensorizer",
          model_loader_extra_config=tensorizer_config,
          enable_lora=True)

sampling_params = SamplingParams(temperature=0,
                                 max_tokens=256,
                                 stop=["[/assistant]"])

prompts = ["[user] Write a SQL query to answer the question based on ..."]

llm.generate(prompts,
             sampling_params,
             lora_request=LoRARequest("sql-lora",
                                      1,
                                      lora_path,
                                      tensorizer_config=tensorizer_config))
