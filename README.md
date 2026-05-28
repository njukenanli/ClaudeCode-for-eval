# Benchmarking ClaudeCode with any LLM

```bash
git clone --recursive
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -r server/requirements.txt
```

prepare your config file like config/default.yaml

```bash
python main.py \
    --config config/default.yaml \
    --run-id debug \
    --dataset huggingface_dataset_name or local/path.jsonl
    --split test # specify split if use huggingface_dataset_name
```