import argparse


def parse_args(args=None):
    """
    Function for parsing argument.

    Args:
        args (_type_, optional): _description_. Defaults to None.
    """
    parser = argparse.ArgumentParser(
        description="Training and Testing Temporal Knowledge Graph Completion Models",
        usage="main.py [<args>] [-h | --help]",
    )
    
    parser.add_argument(
        "--dataname", default="MNIST", type=str, help="name of dataset"
    )
    
    parser.add_argument(
        "--model", default="CNN", type=str, help="name of dataset"
    )

    parser.add_argument(
        "--cuda", action="store_true", help="whether to use GPU or not."
    )
    
    parser.add_argument(
        "-test", "--do-test", action="store_true"
    )  # action='store_true'

    parser.add_argument(
        "--save-path",
        default="./runs/",
        type=str,
        help="trained model checkpoint path.",
    )

    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="existing project/name ok, do not increment.",
    )
    parser.add_argument("-id", "--model_id", type=str, default="0")
    return parser.parse_args(args)
