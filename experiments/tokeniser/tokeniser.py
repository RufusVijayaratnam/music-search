from enum import Enum
from music.common.utils.logging import MlFlowLogger
from music.tokeniser.train import train_tokeniser
from experiments.tokeniser.configs import experiments
from music.common.utils.argparse import add_mode_arg, create_parser, add_exp_arg


class TokeniserMode(Enum):
    TRAIN = "train"


def main():
    exps = experiments()
    parser = create_parser("Tokeniser")
    parser = add_mode_arg(parser, TokeniserMode)
    parser = add_exp_arg(parser, exps)
    args = parser.parse_args()
    exp = args.exp
    mode = TokeniserMode(args.mode)
    hp = exps[exp]()
    match mode:
        case TokeniserMode.TRAIN:
            logger = MlFlowLogger(exp)
            train_tokeniser(hp, logger)


if __name__ == "__main__":
    main()
