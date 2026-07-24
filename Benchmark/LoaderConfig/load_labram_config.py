# load_config.py
# Load one model’s YAML config file and return it as a Python dictionary.

# load_labram_config.py
# → finds Benchmark/Config/labram.yaml
# → reads labram.yaml
# → converts YAML into Python dict
# → returns config

from pathlib import Path
import yaml


def load_labram_config():
    config_path = Path(__file__).resolve().parents[1] / "Config" / "labram.yaml"

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


def main():
    config = load_labram_config()
    print(config)


if __name__ == "__main__":
    main()

