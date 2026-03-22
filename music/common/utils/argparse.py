import argparse
from argparse import ArgumentParser

def create_parser(program: str | None = None) -> ArgumentParser:
    parser = argparse.ArgumentParser(prog=program)
    return parser

def add_exp_arg(parser: ArgumentParser, exps) -> ArgumentParser:
    parser.add_argument("--exp", choices=list(exps.keys()), required=True)
    return parser

