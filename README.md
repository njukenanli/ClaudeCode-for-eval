# Benchmarking ClaudeCode with any LLM on SWE benchmarks 

## Setup
```bash
git clone https://github.com/njukenanli/ClaudeCode-for-eval --recursive
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -r server/requirements.txt
```

## Running with Azure OpenAI
```bash
pip install openai azure-identity-broker --upgrade
```

Modify server/server.py::start_from_azure_openai to accept your azure_ad_token_provider

## Rollout

prepare your config file like config/default.yaml

```bash
python main.py \
    --config config/default.yaml \
    --run-id debug \
    --dataset huggingface_dataset_name or local/path.jsonl
    --split test # specify split if use huggingface_dataset_name
```
