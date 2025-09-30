import yaml
from pathlib import Path

def import_config(config_file: str = 'config.yaml') -> dict:
    config_path = Path('config', config_file)
    print(f"Loading configuration from {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    config['training']['learning_rate'] = float(config['training']['learning_rate'])
    return config