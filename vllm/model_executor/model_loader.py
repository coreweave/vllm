"""Utilities for selecting and loading models."""
import contextlib
import time
from typing import Type

import torch
from transformers import PretrainedConfig

from vllm.logger import init_logger
from vllm.config import ModelConfig
from vllm.model_executor.models import ModelRegistry
from vllm.model_executor.weight_utils import (get_quant_config,
                                              initialize_dummy_weights)

from torch import nn

from tensorizer import TensorDeserializer, TensorSerializer, stream_io
from tensorizer.utils import convert_bytes, get_mem_usage, no_init_or_tensor

logger = init_logger(__name__)


class TensorizerAgent:

    def __init__(self, model_cls: Type[nn.Module], model_config: ModelConfig):
        self.model_config = model_config
        self.model_cls = model_cls
        self.tensorizer_args = self.model_config.tensorizer_args
        self.serialize_args = self.tensorizer_args.serializer_params
        self.deserialize_args = self.tensorizer_args.deserializer_params

        self.serialize_model = not self._verify_path_reachable()

    def _verify_path_reachable(self):
        if not self.tensorizer_args.download_dir.endswith(".tensors"):
            raise ValueError(f"download_dir {self.tensorizer_args.download_dir} must specify a .tensors "
                             f"file when load_format = tensorizer")
        try:
            stream_io.open_stream(self.tensorizer_args.download_dir, "rb")
            return True
        except OSError as err:
            if "Not Found" in str(err):
                logger.info(
                    f"Tensors not found. Will load via HF and serialize tensors to {self.tensorizer_args.download_dir}"
                )
                return False
            else:
                raise OSError(err)

    def serialize(self):
        with torch.device("cuda"):
            model = self.model_cls(self.model_config.hf_config)
        self.model_config.load_format = "auto"
        model.load_weights(
            self.model_config.model,
            self.model_config.download_dir,
            self.model_config.load_format,
            self.model_config.revision,
        )
        _make_model_contiguous(model)
        stream = stream_io.open_stream(self.tensorizer_args.download_dir, "wb")
        serializer = TensorSerializer(stream, **self.serialize_args)
        logger.info(
            f"Serializing model tensors {self.model_config.model} to {self.tensorizer_args.download_dir}."
        )
        serializer.write_module(model)
        serializer.close()
        logger.info(
            f"Serialization complete. Running the previous command will deserialize the saved model weights."
        )
        return model.eval()

    def deserialize(self):
        before_mem = get_mem_usage()
        # Lazy load the tensors from S3 into the model.
        start = time.time()
        stream = stream_io.open_stream(self.tensorizer_args.download_dir, "rb")
        model = _prepare_model_for_deserialization(self.model_cls,
                                                   self.model_config)
        deserializer = TensorDeserializer(stream, **self.deserialize_args)
        deserializer.load_into_module(model)
        model = model.to(dtype=self.model_config.dtype)
        end = time.time()

        # Brag about how fast we are.
        total_bytes_str = convert_bytes(deserializer.total_tensor_bytes)
        duration = end - start
        per_second = convert_bytes(deserializer.total_tensor_bytes / duration)
        after_mem = get_mem_usage()
        deserializer.close()
        logger.info(
            f"Deserialized {total_bytes_str} in {end - start:0.2f}s, {per_second}/s"
        )
        logger.info(f"Memory usage before: {before_mem}")
        logger.info(f"Memory usage after: {after_mem}")

        return model.eval()

    def run(self):
        if self.serialize_model:
            return self.serialize()
        else:
            return self.deserialize()


@contextlib.contextmanager
def _set_default_torch_dtype(dtype: torch.dtype):
    """Sets the default torch dtype to the given dtype."""
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    yield
    torch.set_default_dtype(old_dtype)


def _prepare_model_for_deserialization(model_cls: Type[nn.Module],
                                       model_config: ModelConfig):
    model_args = model_config.hf_config
    model_args.torch_dtype = model_config.dtype
    model = no_init_or_tensor(lambda: model_cls(*[model_args]))
    return model


def _make_model_contiguous(model: nn.Module):
    # Ensure tensors are saved in memory contiguously
    for param in model.parameters():
        param.data = param.data.contiguous()


def _get_model_architecture(config: PretrainedConfig) -> Type[nn.Module]:
    architectures = getattr(config, "architectures", [])
    for arch in architectures:
        model_cls = ModelRegistry.load_model_cls(arch)
        if model_cls is not None:
            return model_cls
    raise ValueError(
        f"Model architectures {architectures} are not supported for now. "
        f"Supported architectures: {ModelRegistry.get_supported_archs()}")


def get_model(model_config: ModelConfig) -> nn.Module:
    model_class = _get_model_architecture(model_config.hf_config)

    # Get the (maybe quantized) linear method.
    linear_method = None
    if model_config.quantization is not None:
        quant_config = get_quant_config(model_config.quantization,
                                        model_config.model,
                                        model_config.hf_config,
                                        model_config.download_dir)
        capability = torch.cuda.get_device_capability()
        capability = capability[0] * 10 + capability[1]
        if capability < quant_config.get_min_capability():
            raise ValueError(
                f"The quantization method {model_config.quantization} is not "
                "supported for the current GPU. "
                f"Minimum capability: {quant_config.get_min_capability()}. "
                f"Current capability: {capability}.")
        supported_dtypes = quant_config.get_supported_act_dtypes()
        if model_config.dtype not in supported_dtypes:
            raise ValueError(
                f"{model_config.dtype} is not supported for quantization "
                f"method {model_config.quantization}. Supported dtypes: "
                f"{supported_dtypes}")
        linear_method = quant_config.get_linear_method()

    with _set_default_torch_dtype(model_config.dtype):
        # Create a model instance.
        # The weights will be initialized as empty tensors.
##        if model_config.load_format == "tensorizer":
##           tensorizer = TensorizerAgent(model_class, model_config)
##            return tensorizer.run()
##        else:
        with torch.device("cuda"):
            model = model_class(model_config.hf_config, linear_method)
        if model_config.load_format == "dummy":
            # NOTE(woosuk): For accurate performance evaluation, we assign
            # random values to the weights.
            initialize_dummy_weights(model)
        else:
            # Load the weights from the cached or downloaded files.
            model.load_weights(
                model_config.model,
                model_config.download_dir,
                model_config.load_format,
                model_config.revision,
            )
    return model.eval()
