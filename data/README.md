# Local calibration shards (not in git)

HC-SMoE C4 calib uses one train shard. Download it here instead of committing the 300MB file:

```bash
mkdir -p data
curl -L "${HF_ENDPOINT:-https://hf-mirror.com}/datasets/allenai/c4/resolve/main/en/c4-train.00000-of-01024.json.gz" \
  -o data/c4-train.00000-of-01024.json.gz
```

If this file is missing, `src/reap/data.py` will fetch the same shard via `HF_ENDPOINT`.
