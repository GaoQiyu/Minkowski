import os
import json
import time
from model.trainer import Trainer


if __name__ == '__main__':

    config_path = os.path.join(os.path.abspath("./"), "config.json")
    with open(config_path) as config_file:
        config = json.load(config_file)
    config_file.close()

    trainer = Trainer(config)
    time_now = time.time()
    for epoch in range(trainer.epoch, config["epoch"]):
        trainer.train(epoch)
        trainer.eval(epoch)
        print('one epoch time:   {} s'.format(time.time() - time_now))
        trainer.summary.close()
        time_now = time.time()

