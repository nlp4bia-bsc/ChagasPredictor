import argparse

from scripts.train import main as train_main
from utils.import_config import import_config
from utils.logger import setup_logger


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run the script.")

    parser.add_argument(
        "--action", 
        type=str, 
        default="train",
        choices=["train"],
        help="Choose the script to run."
    )
    parser.add_argument(
        "--log", 
        type=str, 
        choices=["debug", "info", "warning", "error"],
        default="info", 
        help="Set the logging level."
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default='sample_config.yaml',
        help="Path to the configuration file."
    )
    
    return parser.parse_args()

def main(args):
    config = import_config(args.config_file)

    logger = setup_logger(config['general']['log_dir'], args)

    script_execution = {
        "train": train_main
    }

    script_execution[args.action](logger, config)

if __name__ == "__main__":
    args = parse_arguments()
    main(args)
    