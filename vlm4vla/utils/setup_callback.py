import os
import json

from lightning.pytorch.callbacks import Callback


class SetupCallback(Callback):
    def __init__(self, now, logdir, ckptdir, cfgdir, config):
        super().__init__()
        self.now = now
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.cfgdir = cfgdir
        self.config = config

    def on_train_start(self, trainer, model):
        if trainer.global_rank == 0:
            # Create logdirs and save configs
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.ckptdir, exist_ok=True)
            os.makedirs(self.cfgdir, exist_ok=True)

            print("Project config")
            print(self.config)
            json.dump(
                self.config,
                open(
                    os.path.join(self.cfgdir, "{}-project.json".format(self.now)), "w"
                ),
                indent=4,
            )
