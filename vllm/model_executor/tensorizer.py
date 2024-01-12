import time

from torch import nn

from vllm.config import ModelConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader import _get_model_architecture

from tensorizer import TensorDeserializer, TensorSerializer, stream_io
from tensorizer.utils import convert_bytes, get_mem_usage, no_init_or_tensor

logger = init_logger(__name__)

def _make_model_contiguous(model):
    # Ensure tensors are saved in memory contiguously
    for param in model.parameters():
        param.data = param.data.contiguous()


class TensorizerAgent:
    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config
        self.hf_config = self.model_config.hf_config
        self.hf_config.torch_dtype = self.model_config.dtype
        self.model_class = _get_model_architecture(self.hf_config)

    def serialize(self, model: nn.Module):
        _make_model_contiguous(model)
        stream = stream_io.open_stream(self.model_config.tensorizer_path, "wb")
        serializer = TensorSerializer(stream)
        logger.info(f"Serializing model tensors {self.model_config.model} to {self.model_config.tensorizer_path}.")
        serializer.write_module(model)
        serializer.close()

    def deserialize(self):
        model = no_init_or_tensor(lambda: self.model_class(*[self.model_config.hf_config]))
        before_mem = get_mem_usage()
        # Lazy load the tensors from S3 into the model.
        start = time.time()
        stream = stream_io.open_stream(self.model_config.tensorizer_path, "rb")
        deserializer = TensorDeserializer(stream, plaid_mode=True)
        deserializer.load_into_module(model)
        model = model.to(dtype=self.model_config.dtype)
        end = time.time()

        # Brag about how fast we are.
        total_bytes_str = convert_bytes(deserializer.total_tensor_bytes)
        duration = end - start
        per_second = convert_bytes(deserializer.total_tensor_bytes / duration)
        after_mem = get_mem_usage()
        deserializer.close()
        logger.info(f"Deserialized {total_bytes_str} in {end - start:0.2f}s, {per_second}/s")
        logger.info(f"Memory usage before: {before_mem}")
        logger.info(f"Memory usage after: {after_mem}")

        return model
