from music.tokeniser.train import train_tokeniser
from experiments.tokeniser.configs import experiments
from music.common.utils.argparse import create_parser, add_exp_arg


def main():
    exps = experiments()
    parser = create_parser("Tokeniser")
    parser = add_exp_arg(parser, exps)
    args = parser.parse_args()
    exp = args.exp
    hp = exps[exp]()
    train_tokeniser(hp)

if __name__ == "__main__":
    main()
