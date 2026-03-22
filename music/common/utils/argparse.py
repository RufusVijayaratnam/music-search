import argparse
from argparse import ArgumentParser
from enum import Enum
from typing import Type


def create_parser(program: str | None = None) -> ArgumentParser:
    parser = argparse.ArgumentParser(prog=program)
    return parser


def add_mode_arg(parser: ArgumentParser, choices_enum: Type[Enum]) -> ArgumentParser:
    modes = [c.value for c in choices_enum]
    parser.add_argument("mode", choices=modes)
    return parser


def add_exp_arg(parser: ArgumentParser, exps) -> ArgumentParser:
    parser.add_argument("--exp", choices=list(exps.keys()), required=True)
    return parser
