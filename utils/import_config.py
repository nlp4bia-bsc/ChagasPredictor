import yaml
from pathlib import Path

def import_config(config_file: str = 'config.yaml') -> dict:
    config_path = Path('config', config_file)
    print(f"Loading configuration from {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    config['training']['encoder_learning_rate'] = float(config['training']['encoder_learning_rate'])
    config['training']['lstm_learning_rate'] = float(config['training']['lstm_learning_rate'])
    config['training']['classifier_learning_rate'] = float(config['training']['classifier_learning_rate'])
    return config