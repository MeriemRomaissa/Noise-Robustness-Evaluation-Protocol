from pathlib import Path
import yaml


def load_cbramod_config():
    config_path = Path(__file__).resolve().parents[1] / "Config" / "cbramod.yaml"

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


def main():
    config = load_cbramod_config()
    print(config)


if __name__ == "__main__":
    main()
