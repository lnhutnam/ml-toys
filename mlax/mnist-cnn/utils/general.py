import logging
from pathlib import Path

def set_logger(args) -> None:
    """
    Sets up the logging configuration to log messages to both a file and the console.

    Args:
        args: Parsed command-line arguments containing configuration details like 
              'do_test' to decide log file name and 'save_path' to determine log file path.
    """
    # Determine log file based on the mode (test or train)
    log_file = Path(args.save_path) / ("test.log" if args.do_test else "train.log")

    # Set up logging configuration
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        filename=log_file,
        filemode="w",
    )
    
    # Add console handler for logging to stdout
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s")
    console.setFormatter(formatter)
    logging.getLogger("").addHandler(console)


def increment_path(path: str, exist_ok: bool = False, sep: str = "", mkdir: bool = False) -> Path:
    """
    Increments a file or directory path by appending a suffix to avoid overwriting existing files or directories.

    Args:
        path (str): Initial path to increment.
        exist_ok (bool): If True, do not increment path if it already exists. Defaults to False.
        sep (str): Separator to use between the base path and the increment number. Defaults to "".
        mkdir (bool): If True, create the directory at the new path if it does not exist. Defaults to False.

    Returns:
        Path: A new incremented path, if necessary.
    """
    path = Path(path)
    
    # Check if path exists and increment if necessary
    if path.exists() and not exist_ok:
        # Separate file stem and suffix
        path_stem, suffix = (path.with_suffix(""), path.suffix) if path.is_file() else (path, "")

        # Increment path
        for n in range(2, 9999):
            incremented_path = path_stem.with_name(f"{path_stem.name}{sep}{n}{suffix}")
            if not incremented_path.exists():
                path = incremented_path
                break

    # Create directory if needed
    if mkdir and not path.exists():
        path.mkdir(parents=True, exist_ok=True)

    return path
