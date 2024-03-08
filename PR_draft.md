# Tensorizer Support
This PR allows models used for the OpenAI-compatible API server to be loaded using
[ADD LINK] Coreweave's Tensorizer, enabling extremely fast model loads from HTTP/HTTPS, 
Redis, and S3 endpoints. 

The key changes involved are:

1. Listing `tensorizer==2.8.0` as a runtime dependency in `requirements.txt`
2. Adds `tensorizer_loader.py` to `vllm/model_executor` that provides utility functions
for tensorizer. 
3. Adds multiple args to the vLLM's OpenAI inference service entrypoint that allows
the user to specify the path to serialized-by-tensorizer model tensors, as well as 
arguments for tensorizer's deserializer. 
4. Allows deserialization of serialized model tensors in HuggingFace's model format, 
as well as supporting deserializing serialized vLLM-formatted models, allowing the 
use of loading with `plaid_mode`, which can allow Llama 2 13B to start serving requests
in as little as X seconds over S3, or as little as X seconds locally.

Credentialing for S3 is supported by passing a user's access and secret key to 
`S3_ACCESS_KEY_ID` and `S3_SECRET_ACCESS_KEY` environment variables respectively

# Model loading benchmarks

Tensorizer can load models like Llama 2 13B in *as little as 10 seconds*. In order to do so, a model must be 
serialized using `TensorSerializer` to a `.tensors` file located either locally or through a S3, HTTP/HTTPS, or Redis
endpoint. `--tensorizer-uri` must be specified with the serialized tensors location when invoking the API server. 

If a vLLM model is serialized, `plaid_mode` can be used, which loads much faster. 

[Benchmark here]

