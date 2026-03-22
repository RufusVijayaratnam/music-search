from dataclasses import dataclass

from music.common.schedule import Schedule

import torch


@dataclass
class TokeniserTrainParams:
    rec_loss_weight: Schedule
    commitment_loss_weight: Schedule
    codebook_loss_weight: Schedule
    learning_rate: Schedule
    optimiser: torch.optim.Optimizer

    def step(self):
        self.rec_loss_weight.step()
        self.commitment_loss_weight.step()
        self.codebook_loss_weight.step()
        self.learning_rate.step()
        self.optimiser.step()
        for pg in self.optimiser.param_groups:
            pg["lr"] = self.learning_rate.get_value()
