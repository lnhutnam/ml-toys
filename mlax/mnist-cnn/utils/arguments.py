import argparse

def parse_args(args=None):
    """
    Parses command-line arguments for training and testing temporal knowledge graph completion models.

    Args:
        args (list, optional): A list of command-line arguments to parse. If None, arguments are taken from sys.argv.

    Returns:
        argparse.Namespace: Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Training and Testing Temporal Knowledge Graph Completion Models",
        usage="%(prog)s [options]",
    )
    
    parser.add_argument(
        "--dataname", 
        default="MNIST", 
        type=str, 
        help="Name of the dataset to be used (default: MNIST)."
    )
    
    parser.add_argument(
        "--model", 
        default="CNN", 
        type=str, 
        help="Name of the model architecture to use (default: CNN)."
    )

    parser.add_argument(
        "--cuda", 
        action="store_true", 
        help="Flag to enable GPU support for training."
    )
    
    parser.add_argument(
        "--do-test", 
        "-test", 
        action="store_true", 
        help="Flag to perform testing after training."
    )

    parser.add_argument(
        "--save-path",
        default="./runs/",
        type=str,
        help="Directory path to save trained model checkpoints (default: ./runs/).",
    )

    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="Allow existing project/name, do not increment if already exists.",
    )

    parser.add_argument(
        "--id", 
        "-id", 
        type=str, 
        default="0", 
        help="Identifier for the model instance (default: 0)."
    )
    
    return parser.parse_args(args)
