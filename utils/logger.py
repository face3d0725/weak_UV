import os
import time
import logging


def setup_logger(logpth):
    if not os.path.exists(logpth):
        os.mkdir(logpth)
    logger = logging.getLogger(__name__)
    logfile = 'FaceRecon-{}.log'.format(time.strftime('%Y-%m-%d-%H-%M-%S'))
    logfile = os.path.join(logpth, logfile)
    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO, filename=logfile)
    return logger
