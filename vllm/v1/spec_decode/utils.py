# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from vllm.config import SpeculativeConfig
from vllm.sampling_params import SamplingParams
from vllm.v1.engine.core_client import SyncMPClient
from vllm.v1.engine import EngineCoreRequestType

_SAMPLING_EPS = 1e-5


def init_drafter(draft_cls, *args):
    return draft_cls(*args)

class ProposerClient:
    def __init__(self, engine_client: "SyncMPClient"):
        self.engine_client = engine_client



    def send_state_dict(self, sd: dict):
        return self.engine_client._send_input(EngineCoreRequestType.SPEC_DECODE_QUERY, sd)

    def await_proposals(self):
        import time
        time.sleep(100)
        raise NotImplementedError

    def propose(self, *args, **kwargs):
        pass
    def load_model(self, *args, **kwargs):
        pass

def is_spec_decode_unsupported(sampling_params: SamplingParams) -> bool:
    """True if request is incompatible with speculative decoding"""
    return (sampling_params.frequency_penalty != 0.0
            or sampling_params.presence_penalty != 0.0
            or sampling_params.repetition_penalty != 1.0
            or sampling_params.min_p > _SAMPLING_EPS
            or sampling_params.logprobs is not None)
