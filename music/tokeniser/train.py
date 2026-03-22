import torch

from music.common.loader import AudioData, load_audio_data
from music.common.quantiser import AbstractQuantiser, QuantiserResult
from music.tokeniser.hyperparameters import TokeniserHP
from music.tokeniser.network import create_tokeniser_networks
from music.tokeniser.train_params import TokeniserTrainParams
from music.common.utils.logging import MlFlowLogger, StepScalarData


def train_tokeniser(hp: TokeniserHP, logger: MlFlowLogger):

    print("running")
    data = load_audio_data(
        hp.data_path, device=torch.device("cpu"), dtype=hp.dtype
    )  # keep full data on cpu

    encoder, quantiser, decoder = create_tokeniser_networks(hp)

    params = list(encoder.parameters()) + list(quantiser.parameters()) + list(decoder.parameters())
    optim = torch.optim.Adam(params=params, lr=hp.learning_rate.get_value())

    train_params = TokeniserTrainParams(
        commitment_loss_weight=hp.commitment_loss_weight,
        codebook_loss_weight=hp.codebook_loss_weight,
        rec_loss_weight=hp.rec_loss_weight,
        learning_rate=hp.learning_rate,
        optimiser=optim,
    )

    train_loop(
        encoder=encoder,
        quantiser=quantiser,
        decoder=decoder,
        data=data,
        train_params=train_params,
        hp=hp,
        logger=logger,
    )


def train_loop(
    encoder: torch.nn.Module,
    quantiser: AbstractQuantiser,
    decoder: torch.nn.Module,
    data: AudioData,
    train_params: TokeniserTrainParams,
    hp: TokeniserHP,
    logger: MlFlowLogger,
):
    batch = data.sample_batch(hp.batch_size, hp.window_len).to(
        hp.device
    )  # sample and move to device
    for update_step in range(hp.update_steps):
        print(f"Update step {update_step}")
        if (update_step + 1) % hp.batch_sample_frequency == 0:
            print(f"Sampling new batch")
            batch = data.sample_batch(hp.batch_size, hp.window_len).to(
                hp.device
            )  # sample and move to device

        log_data_dict = train_step(
            encoder=encoder,
            quantiser=quantiser,
            decoder=decoder,
            batch=batch,
            train_params=train_params,
            hp=hp,
        )
        logger.log_scalars([StepScalarData(step=update_step, data=log_data_dict)])


def train_step(
    encoder: torch.nn.Module,
    quantiser: AbstractQuantiser,
    decoder: torch.nn.Module,
    batch: AudioData,
    train_params: TokeniserTrainParams,
    hp: TokeniserHP,
):
    train_params.optimiser.zero_grad()
    mini_batch_size = len(batch) // hp.grad_accumulate_steps
    rec_loss_total: float = 0
    cdb_loss_total: float = 0
    cmt_loss_total: float = 0
    rec_loss_w = train_params.rec_loss_weight.get_value()
    cmt_loss_w = train_params.commitment_loss_weight.get_value()
    cdb_loss_w = train_params.codebook_loss_weight.get_value()
    log_data = {}
    log_data.update({"loss/rec_loss_weight": rec_loss_w})
    log_data.update({"loss/cmt_loss_weight": cmt_loss_w})
    log_data.update({"loss/cdb_loss_weight": cdb_loss_w})
    for i in range(hp.grad_accumulate_steps):
        mini_batch = batch[i * mini_batch_size : i * mini_batch_size + mini_batch_size]
        encoded = encoder(mini_batch.mixtures)
        b, s, _ = encoded.shape
        encoded_flt = torch.reshape(encoded, (b * s, -1))
        quantised_res: QuantiserResult = quantiser(encoded_flt)
        quantised = torch.reshape(quantised_res.quantised, (b, s, -1))
        decoded = decoder(quantised)[..., : mini_batch.mixtures.shape[-1]]
        rec_loss = torch.norm(decoded - mini_batch.mixtures, dim=-1).mean()
        cmt_loss = quantised_res.commitment_loss
        cdb_loss = quantised_res.codebook_loss
        total_loss = rec_loss * rec_loss_w + cmt_loss * cmt_loss_w + cdb_loss * cdb_loss_w
        total_loss /= hp.grad_accumulate_steps
        total_loss.backward()

        rec_loss_total += rec_loss.item() / hp.grad_accumulate_steps
        cmt_loss_total += cmt_loss.item() / hp.grad_accumulate_steps
        cdb_loss_total += cdb_loss.item() / hp.grad_accumulate_steps

    # step curriculums
    train_params.step()

    # logging
    log_data.update({"rec_loss": rec_loss_total})
    log_data.update({"cmt_loss": cmt_loss_total})
    log_data.update({"cdb_loss": cdb_loss_total})
    return log_data
