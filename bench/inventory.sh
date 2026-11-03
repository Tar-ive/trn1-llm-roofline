#!/usr/bin/env bash
# Snapshot of the Neuron stack, bottom to top. Results aren't comparable across versions.
set -x
uname -a; lsb_release -ds
TOKEN=$(curl -s -X PUT http://169.254.169.254/latest/api/token -H "X-aws-ec2-metadata-token-ttl-seconds: 60")
curl -s -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/instance-type; echo
curl -s -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/ami-id; echo
nproc; free -g
neuron-ls                                    # chips, NeuronCores, HBM, driver/runtime version
modinfo neuron 2>/dev/null | grep -E '^(version|filename)'
dpkg -l | grep -i neuron                     # driver (dkms), runtime (libnrt), collectives, tools
ls -d /opt/aws_neuronx_venv_*
for v in /opt/aws_neuronx_venv_*; do echo "## $v"; "$v/bin/pip" list 2>/dev/null | grep -iE 'neuron|torch|jax|vllm|transformers|nki'; done
neuronx-cc --version 2>/dev/null || ls /opt/aws_neuronx_venv_*/bin/neuronx-cc
neuron-explorer --version 2>/dev/null || neuron-profile --version
env | grep -E '^NEURON' || true
