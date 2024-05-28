from plot import plot

from tnp.utils.experiment_utils import (
    evaluation_summary,
    initialize_experiment,
    train_epoch,
    val_epoch,
)


def main():
    experiment, checkpointer = initialize_experiment()

    model = experiment.model
    gen_train = experiment.generators.train
    gen_val = experiment.generators.val
    optimiser = experiment.optimiser(model.parameters())
    epochs = experiment.params.epochs

    step = 0
    for epoch in range(epochs):
        model.train()
        step, train_result = train_epoch(
            model=model,
            generator=gen_train,
            optimiser=optimiser,
            step=step,
            loss_fn=experiment.misc.loss_fn,
            gradient_clip_val=experiment.misc.gradient_clip_val,
        )
        model.eval()
        evaluation_summary("train", train_result)
        checkpointer.update_best_and_last_checkpoint(
            model=model, val_result=train_result, prefix="train_", update_last=True
        )

        val_result, batches = val_epoch(model=model, generator=gen_val)

        evaluation_summary("val", val_result)
        checkpointer.update_best_and_last_checkpoint(
            model=model, val_result=val_result, prefix="val_", update_last=False
        )

        if epoch % experiment.misc.plot_interval == 0:
            plot(
                model=model,
                batches=batches,
                num_fig=min(5, len(batches)),
                name=f"epoch-{epoch:04d}",
            )


if __name__ == "__main__":
    main()
