import os

import lightning.pytorch as pl
import torch
from omegaconf import OmegaConf
from plot import plot
import numpy as np

import wandb
from tnp.utils.data_loading import adjust_num_batches
from tnp.utils.experiment_utils import initialize_experiment
from tnp.utils.lightning_utils import LitWrapper, LogPerformanceCallback
from tnp.data.hadISD import HadISDDataGenerator
from eval import test_model


def main():
    experiment = initialize_experiment()

    model = experiment.model
    gen_train = experiment.generators.train
    gen_val = experiment.generators.val
    optimiser = experiment.optimiser(model.parameters())
    epochs = experiment.params.epochs

    is_hadISD_train = isinstance(gen_train, HadISDDataGenerator) # Assumes that training is either GP or hadISD

    train_loader = torch.utils.data.DataLoader(
        gen_train,
        batch_size=None,
        num_workers=experiment.misc.num_workers,
        worker_init_fn=(
            (
                experiment.misc.worker_init_fn
                if hasattr(experiment.misc, "worker_init_fn")
                else adjust_num_batches
            )
            if experiment.misc.num_workers > 0
            else None
        ),
        persistent_workers=True if experiment.misc.num_workers > 0 else False,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        gen_val,
        batch_size=None,
        num_workers=experiment.misc.num_val_workers,
        worker_init_fn=(
            (
                experiment.misc.worker_init_fn
                if hasattr(experiment.misc, "worker_init_fn")
                else adjust_num_batches
            )
            if experiment.misc.num_val_workers > 0
            else None
        ),
        persistent_workers=True if experiment.misc.num_val_workers > 0 else False,
        pin_memory=True,
    )

    def plot_fn_gp(model, batches, name):
        is_training = model.training
        model.eval()
        # Calculates plot range
        min_tgt, max_tgt = np.array(experiment.params.target_range).min(), np.array(experiment.params.target_range).max()
        plot(
            model=model,
            batches=batches,
            num_fig=min(5, len(batches)),
            name=name,
            pred_fn=experiment.misc.pred_fn,
            x_range = (min_tgt, max_tgt)
        )
        if is_training: model.train()

    def plot_fn_hadISD(model, batches, name):
        print("Implement me") #TODO

    plot_fn = plot_fn_hadISD if is_hadISD_train else plot_fn_gp

    if experiment.misc.resume_from_checkpoint is not None:
        api = wandb.Api()
        artifact = api.artifact(experiment.misc.resume_from_checkpoint)
        artifact_dir = artifact.download()
        ckpt_file = os.path.join(artifact_dir, "model.ckpt")

        lit_model = (
            LitWrapper.load_from_checkpoint(  # pylint: disable=no-value-for-parameter
                ckpt_file,
            )
        )
    else:
        ckpt_file = None
        lit_model = LitWrapper(
            model=model,
            optimiser=optimiser,
            loss_fn=experiment.misc.loss_fn,
            pred_fn=experiment.misc.pred_fn,
            plot_fn=plot_fn,
            plot_interval=experiment.misc.plot_interval,
        )

    if experiment.misc.logging:
        logger = pl.loggers.WandbLogger(
            project=experiment.misc.project,
            name=experiment.misc.name,
            config=OmegaConf.to_container(experiment.config),
            log_model="all",
        )
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            every_n_epochs=experiment.misc.checkpoint_interval,
            save_last=True,
        )
        performance_callback = LogPerformanceCallback()
        callbacks = [checkpoint_callback, performance_callback]
    else:
        logger = False
        callbacks = None

    trainer = pl.Trainer(
        logger=logger,
        max_epochs=epochs,
        limit_train_batches=gen_train.num_batches,
        limit_val_batches=gen_val.num_batches,
        log_every_n_steps=(
            experiment.misc.log_interval if not experiment.misc.logging else None
        ),
        devices="auto",
        accelerator="auto",
        num_sanity_val_steps=0,
        check_val_every_n_epoch=(experiment.misc.check_val_every_n_epoch),
        gradient_clip_val=experiment.misc.gradient_clip_val,
        callbacks=callbacks,
    )

    trainer.fit(
        model=lit_model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=ckpt_file,
    )

    print("Running final test on model")
    test_model(lit_model, experiment, wandb_run=trainer.logger.experiment)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
