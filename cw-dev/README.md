# Developing vLLM [Last updated 5/9/25]
vLLM is notoriously difficult to build in editable mode and test, and there's
no good source of best practices for doing so. This will serve as the guide
for this.

## The no-nonsense guide
For brevity, this will just be served with a collection of ordered
bulletpoints.

1. Use a K8s pod rather than Slurm, because it's far easier to connect PyCharm
to it for debugging
2. The _image of choice is everything_. It's relatively straightforward to get
vLLM to build from source, but they have a litany of `requirements.txt` files
with dependencies that may easily not build even when you've done nothing wrong.

For instance, if you use a standard PyTorch container image, build vLLM
from source after using their `use_existing_torch.py` script to remove.

The recommended way is to use their CI docker images, which has to perform
all sorts of stuff like tests, so the environment is guaranteed to be able to
handle it. https://docs.vllm.ai/en/latest/getting_started/installation/gpu.html#pre-built-images

You can check the available tags here, as simply using whatever vLLM commit
you want isn't guaranteed to have an image.
https://gallery.ecr.aws/q9t5s3a7/vllm-ci-postmerge-repo

# Usage guide:
1. Use this for the dev container spinup. It'll load a recent container image,
see which files your feature branch has changed, and upload them via rsync ssh
when the ssh server on the pod is up:

```bash
cw-dev % source apply_env_vars_and_deploy.sh dev_container.yaml
```

1. Make sure you mark your source directory as such in the Project Structure
for PyCharm, and make sure the source directory on the local and remote hosts
match, or PyCharm will have trouble with its code inspection/linting

python -m examples.other.tensorize_vllm_model \
   --model s3://infr/vllm-tensorized/vllm/Qwen/Qwen2.5-VL-7B-Instruct/test \
   --dtype float16 \
   deserialize \
   --path-to-tensors s3://my-bucket/vllm/EleutherAI/gpt-j-6B/v1/model.tensors

[default]
region=US-EAST-04
endpoint_url=http://cwlota.com
ignore_configure_endpoint_urls = true

## THIS IS THE NO BUENO THING THAT SHOULDN'T EXIST
s3 =
  addressing_style = virtual
#######

be able to specify signature version for boto3 (default to v4)

- instead of piping in the config directly,

Update this https://docs.vllm.ai/en/latest/models/extensions/tensorizer.html#loading-models-with-coreweave-s-tensorizer
