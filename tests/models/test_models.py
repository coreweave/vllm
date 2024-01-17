"""Compare the outputs of HF and vLLM when using greedy sampling.

Run `pytest tests/models/test_models.py --forked`.
"""
import pytest
from vllm.model_executor.model_loader import get_model
from vllm.model_executor.weight_utils import load_tensorized_weights
from vllm.config import ModelConfig

MODELS = [
    "facebook/opt-125m",
    "meta-llama/Llama-2-7b-hf",
    "mistralai/Mistral-7B-v0.1",
    "Deci/DeciLM-7b",
    "tiiuae/falcon-7b",
    "gpt2",
    "bigcode/tiny_starcoder_py",
    "EleutherAI/gpt-j-6b",
    "EleutherAI/pythia-70m",
    "bigscience/bloom-560m",
    "mosaicml/mpt-7b",
    "microsoft/phi-2",
]


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["float"])
@pytest.mark.parametrize("max_tokens", [128])
def test_models(
    hf_runner,
    vllm_runner,
    example_prompts,
    model: str,
    dtype: str,
    max_tokens: int,
) -> None:
    hf_model = hf_runner(model, dtype=dtype)
    hf_outputs = hf_model.generate_greedy(example_prompts, max_tokens)
    del hf_model

    vllm_model = vllm_runner(model, dtype=dtype)
    vllm_outputs = vllm_model.generate_greedy(example_prompts, max_tokens)
    del vllm_model

    for i in range(len(example_prompts)):
        hf_output_ids, hf_output_str = hf_outputs[i]
        vllm_output_ids, vllm_output_str = vllm_outputs[i]
        assert hf_output_str == vllm_output_str, (
            f"Test{i}:\nHF: {hf_output_str!r}\nvLLM: {vllm_output_str!r}")
        assert hf_output_ids == vllm_output_ids, (
            f"Test{i}:\nHF: {hf_output_ids}\nvLLM: {vllm_output_ids}")


def test_get_model_tensorizer(mocker):
    # Mock the 'load_tensorized_weights' function
    mock_load_tensorized_weights = mocker.patch(
        'vllm.model_executor.weight_utils.load_tensorized_weights')

    # Create a ModelConfig with load_format set to 'tensorizer'
    model_config = ModelConfig(
        model='mistralai/Mistral-7B-v0.1',
        tokenizer='mistralai/Mistral-7B-v0.1',
        tokenizer_mode='auto',
        trust_remote_code=True,
        download_dir='/',
        dtype='float',
        seed=0,
        load_format='tensorizer',
        tensorizer_path='s3://tensorized/mistral-7b-instruct-vllm.tensors',
        # other necessary attributes...
    )

    # Call get_model with the ModelConfig
    get_model(model_config)

    # Assert that load_tensorized_weights was called
    mock_load_tensorized_weights.assert_called_once_with('/path/to/tensorizer')
