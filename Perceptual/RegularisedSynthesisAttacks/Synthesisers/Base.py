import tensorflow as tf
import numpy as np

from abc import ABC


class Synth(ABC):
    def __init__(self):
        self.opt_vars = None
        self.output = None

    def add_opt_vars(self, *args):
        self.opt_vars = list(args)


