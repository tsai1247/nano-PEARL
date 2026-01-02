import logging
import os
from rich.logging import RichHandler
from rich.console import Console
from rich.theme import Theme

def get_logger(name="PEARL", level=logging.INFO):
    logging.disable(logging.NOTSET)
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        console = Console(force_terminal=True, theme=Theme({
            "logging.level.info": "green",
        }))

        h = RichHandler(
            console=console,
            rich_tracebacks=True,
            show_time=True,
            show_level=True,
            show_path=False,
            omit_repeated_times=False,
            log_time_format="%H:%M:%S",
            markup=True,
        )
        h.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(h)
        logger.propagate = False

        _orig_info = logger.info
        def _info(msg, *args, color=None, **kwargs):
            if color:
                msg = f"[{color}]{msg}[/]"
            return _orig_info(msg, *args, **kwargs)
        logger.info = _info
    return logger

logger = get_logger()

def add_file_handler(path: str, level: int = logging.INFO) -> None:
    logger = get_logger()
    abs_path = os.path.abspath(path)
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler) and os.path.abspath(
            handler.baseFilename
        ) == abs_path:
            return
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    file_handler = logging.FileHandler(abs_path)
    file_handler.setLevel(level)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    )
    logger.addHandler(file_handler)


def get_model_name(model_path: str) -> str:
    l = model_path.split("/")
    for s in l:
        if s.startswith("models--"):
            return s
    logger.warning(f"Model Name Not Found: {model_path}")
    return model_path
