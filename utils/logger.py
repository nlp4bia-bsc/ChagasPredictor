from datetime import datetime
from pathlib import Path
import logging

def setup_logger(dir_name, args):
    date = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log_path = Path(dir_name, args.action, f"{date}.log")
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        filename=log_path,
        level=getattr(logging, args.log.upper()),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger(__name__)