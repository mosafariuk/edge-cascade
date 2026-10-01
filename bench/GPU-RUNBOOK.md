# Runbook — serving the edge model on a rented GPU (paper §VI-G, Table VIII)

The guard experiments need one 24 GB GPU for under an hour. This is the procedure used for
the runs in `bench/results-zen5-run2/`: rent a direct-attached RTX 3090/4090 on vast.ai, serve
`Qwen/Qwen3-8B-AWQ` with vLLM 0.9.2, query it from a workstation through an SSH tunnel, copy
the traces back, destroy the instance. Nothing here is specific to vast.ai except the CLI.

Reproducing Table VIII needs **no GPU**: `python3 analysis/guard_confirm.py` recomputes every
number from the committed traces. Use this runbook only to regenerate the traces.

## 1. Find a direct-attached card

`pcie_bw` is the provider's measured host↔GPU bandwidth in GB/s. Mining risers (PCIe ×1)
report about 1; a ×16 slot reports 12–25. The filter excludes risers.

```bash
vastai search offers 'num_gpus=1 gpu_ram>=24 gpu_name in [RTX_3090,RTX_4090] pcie_bw>=8 rentable=true verified=true reliability>0.98 disk_space>=60 inet_down>=300 cuda_vers>=12.4' -o 'dph+' | head -10
```

## 2. Create the instance and start the server

The vLLM image has no `/workspace`, and with `--ssh` the image entrypoint does not run, so
the on-start command creates the directory and launches the server itself.

```bash
OFFER=<offer id>
vastai create instance $OFFER --image vllm/vllm-openai:v0.9.2 --disk 60 --ssh --direct \
  --onstart-cmd 'mkdir -p /workspace/hf; export HF_HOME=/workspace/hf; nohup python3 -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3-8B-AWQ --served-model-name Qwen/Qwen3-8B-AWQ --quantization awq_marlin --dtype half --max-model-len 4096 --gpu-memory-utilization 0.90 --guided-decoding-backend xgrammar --host 0.0.0.0 --port 8000 > /workspace/vllm.log 2>&1 &'
INST=<instance id returned as new_contract>
```

Wait until the instance is `running` and the server is up (image pull ≈ 15–20 min on a cold
host, model load ≈ 2–5 min), and confirm the GPU sits in a real slot:

```bash
vastai show instance $INST --raw | python3 -c 'import json,sys;d=json.load(sys.stdin);print(d["actual_status"], d.get("ssh_host"), d.get("ssh_port"))'
SSH_HOST=<ssh_host>; SSH_PORT=<ssh_port>; KEY=~/.ssh/<your key registered with the provider>
ssh -i $KEY -p $SSH_PORT root@$SSH_HOST 'grep -c "Application startup complete" /workspace/vllm.log; nvidia-smi --query-gpu=name,pcie.link.gen.max,pcie.link.width.current --format=csv,noheader'
#   expect "1" and link width 16 (or 8). Width 1 is a riser: destroy and pick another offer.
```

## 3. Tunnel (keep open in its own terminal)

```bash
ssh -i $KEY -p $SSH_PORT -N -L 18000:localhost:8000 root@$SSH_HOST
export VLLM_URL=http://127.0.0.1:18000/v1/completions      # in the terminal that runs the benchmarks
```

## 4. Runs

```bash
# masking probe (paper §V-A): expects VERDICT: POST-MASK and prints a provenance line
node bench/vllm-probe.mjs | tee bench/results-zen5-run2/vllm-probe.jsonl

# confirmation experiment (paper Table VIII): three prompt arms over the 2,000-record corpus
TEMPS="0" SHOTS="0 1 2" PAYLOADS_ROOT=payloads/synthetic-confirm \
  OUT=bench/results-zen5-run2/guard-confirm.jsonl node bench/optimize-guard.mjs
python3 analysis/guard_confirm.py | tee bench/results-zen5-run2/guard-confirm.txt

# exploratory temperature × few-shot grid over the 500-record corpus
node bench/optimize-guard.mjs && python3 analysis/guard_grid.py

# exploratory judge-fidelity run (needs OPENAI_API_KEY for the judge)
PAYLOADS_ROOT=payloads/synthetic HOLDOUT=0.3 SHADOW_TRACE=1 node bench/shadow-run.mjs
python3 analysis/guard_eval.py
```

`DRY_RUN=1 node bench/optimize-guard.mjs` exercises the whole pipeline against a fake stream,
without a server.

## 5. Teardown

```bash
vastai destroy instance $INST -y     # -y is required: without it the CLI prints "Aborted." and billing continues
vastai show instances                # must be empty
```

## Notes

- **Completions endpoint, chat format applied by the client.** `buildEdgePrompt` wraps the
  instruction and the in-context examples in the model's ChatML turns with reasoning disabled,
  so the constrained JSON starts immediately.
- **vLLM 0.9.2** supports Qwen3 and still accepts the `guided_json` request field and
  `--guided-decoding-backend xgrammar`; later releases move both to `structured_outputs`.
- **Regenerating a corpus** needs a writer model: `OPENAI_API_KEY=… SEED=… N_PER_KIND=…
  OUT=payloads/<dir> node bench/gen-synthetic-corpus.mjs`. The constructed truth is
  deterministic in the seed; the written texts are not.
