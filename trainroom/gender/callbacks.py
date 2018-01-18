from __future__ import absolute_import

import logging
import math
import time

class StatusUpdater(object):

    def __init__(self, ctx, batch_size, epochs, iters_per_epoch, frequent=50, auto_reset=True):
        self.batch_size = batch_size
        self.frequent = frequent
        self.init = False
        self.tic = 0
        self.last_count = 0
        self.auto_reset = auto_reset
        self.ctx = ctx
        self.iters_per_epoch = iters_per_epoch
        self.total_iters = epochs * self.iters_per_epoch
        self.last_msg = ''

    def __call__(self, param):
        count = param.nbatch
        if self.last_count > count:
            self.init = False
        self.last_count = count

        if self.init:
            if count % self.frequent == 0:
                speed = self.frequent * self.batch_size / (time.time() - self.tic)
                if param.eval_metric is not None:
                    name_value = param.eval_metric.get_name_value()
                    if self.auto_reset:
                        param.eval_metric.reset()
                    msg = 'Epoch[%d] Batch [%d]\tSpeed: %.2f samples/sec'
                    msg += '\t%s=%f'*len(name_value)

                    format_msg = '\t{}={}'*len(name_value)
                    self.last_msg = format_msg.format(*sum(name_value, ()))

                    logging.info(msg, param.epoch, count, speed, *sum(name_value, ()))
                else:
                    logging.info('Iter[%d] Batch [%d]\tSpeed: %.2f samples/sec',
                                 param.epoch, count, speed)


                log_str = 'Epoch [{}] Batch [{}] Speed: {:.2f} samples/sec'.format(param.epoch, count, speed)

                current_idx = param.epoch * self.iters_per_epoch + count
                progress = float(current_idx) / self.total_iters
                self.ctx.update_state(state='PROGRESS', progress=progress, status=log_str)

                self.tic = time.time()
        else:
            self.init = True
            self.tic = time.time()