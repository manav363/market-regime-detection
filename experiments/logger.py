import json
import datetime
import os


def log_experiment(params, metrics, path="src/market_regime/experiments/log.jsonl"):

    os.makedirs("experiments", exist_ok=True)

    record = {
        "timestamp": datetime.datetime.now().isoformat(),
        "params": params,
        "metrics": metrics
    }

    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")