from pathlib import Path
from subprocess import check_output, run  # nosec B404, B603, B607

import comet_ml
import lightning.pytorch as pl
from lightning.pytorch import Callback, Trainer
from lightning.pytorch.loggers.comet import CometLogger
from lightning_utilities.core.rank_zero import rank_zero_only


def get_comet_logger(trainer: Trainer) -> CometLogger:
    """Safely get Weights&Biases logger from Trainer."""

    if hasattr(trainer, "fast_dev_run") and getattr(
        trainer, "fast_dev_run", None
    ):
        raise Exception(
            "Cannot use wandb callbacks since pytorch lightning disables loggers in `fast_dev_run=true` mode."
        )

    if isinstance(trainer.logger, CometLogger):
        return trainer.logger

    if isinstance(trainer.logger, (list, tuple)):
        for logger in trainer.logger:
            if isinstance(logger, CometLogger):
                return logger

    raise Exception(
        "You are using comet related callback, but CometLogger was not found for some reason..."
    )


class WatchModel(Callback):
    """generate model graph at the beginning of the run."""

    @rank_zero_only
    def on_train_start(self, trainer, pl_module):
        assert isinstance(trainer.model, pl.LightningModule)

        logger = get_comet_logger(trainer=trainer)

        logger.log_graph(model=trainer.model)


class UploadCodeAsArtifact(Callback):
    """Upload all code files to wandb as an artifact, at the beginning of the
    run."""

    def __init__(self, code_dir: str, use_git: bool = True) -> None:
        """
        Args:
            code_dir: the code directory
            use_git: if using git, then upload all files that are not ignored by git.
            if not using git, then upload all '*.py' file
        """
        self.code_dir = code_dir
        self.use_git = use_git

    @rank_zero_only
    def on_train_start(
        self, trainer: Trainer, pl_module: pl.LightningModule
    ) -> None:
        logger = get_comet_logger(trainer=trainer)
        experiment: comet_ml.Experiment = logger.experiment

        code = comet_ml.Artifact("project-source", artifact_type="code")

        if self.use_git:
            # get .git folder path
            git_dir_path = Path(
                check_output(["git", "rev-parse", "--git-dir"])
                .strip()
                .decode("utf8")
            ).resolve()

            for path in Path(self.code_dir).resolve().rglob("*"):
                # don't upload files ignored by git
                # https://alexwlchan.net/2020/11/a-python-function-to-ignore-a-path-with-git-info-exclude/
                command = ["git", "check-ignore", "-q", str(path)]
                not_ignored = run(command).returncode == 1

                # don't upload files from .git folder
                not_git = not str(path).startswith(str(git_dir_path))

                if path.is_file() and not_git and not_ignored:
                    code.add(
                        str(path),
                        logical_path=str(path.relative_to(self.code_dir)),
                    )

        else:
            for path in Path(self.code_dir).resolve().rglob("*.py"):
                code.add(
                    str(path),
                    logical_path=str(path.relative_to(self.code_dir)),
                )

        experiment.log_artifact(code)


class UploadCheckpointsAsArtifact(Callback):
    """Upload checkpoints to wandb as an artifact, at the end of run."""

    def __init__(self, ckpt_dir: str, upload_best_only: bool = False) -> None:
        self.ckpt_dir = ckpt_dir
        self.upload_best_only = upload_best_only

    @rank_zero_only
    def on_keyboard_interrupt(
        self, trainer: Trainer, pl_module: pl.LightningModule
    ) -> None:
        self.on_train_end(trainer, pl_module)

    @rank_zero_only
    def on_train_end(
        self, trainer: Trainer, pl_module: pl.LightningModule
    ) -> None:
        logger = get_comet_logger(trainer=trainer)
        experiment: comet_ml.Experiment = logger.experiment

        ckpts = comet_ml.Artifact(
            "experiment-ckpts", artifact_type="checkpoints"
        )

        if self.upload_best_only:
            ckpts.add(getattr(trainer.checkpoint_callback, "best_model_path"))
        else:
            for path in Path(self.ckpt_dir).rglob("*.ckpt"):
                ckpts.add(str(path))

        experiment.log_artifact(ckpts)
