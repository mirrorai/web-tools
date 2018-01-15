import sys
import numpy as np
import mxnet as mx
import cv2

def check_label_shapes(labels, preds, shape=0):
    if shape == 0:
        label_shape, pred_shape = len(labels), len(preds)
    else:
        label_shape, pred_shape = labels.shape, preds.shape

    if label_shape != pred_shape:
        raise ValueError("Shape of labels {} does not match shape of "
                         "predictions {}".format(label_shape, pred_shape))

class ClsMetric(mx.metric.EvalMetric):
    """Computes segmentation metrics.
    """
    def __init__(self, cfg, part='all', one_metric=False, axis=1,
                 output_names=None, label_names=['label']):
        super(ClsMetric, self).__init__(
            'ClsMetric', axis=1,
            output_names=output_names, label_names=label_names)
        self._part = part
        self._cfg = cfg
        self._one_metric = one_metric
        self._axis = axis
        self._attributes_num = cfg.ATTRIBUTES_NUM
        self.reset()

    def update(self, labels, preds):
        """Updates the internal evaluation result.
        Parameters
        ----------
        labels : list of `NDArray`
            The labels of the data.
        preds : list of `NDArray`
            Predicted values.
        """

        cfg = self._cfg
        part = self._part

        # check_label_shapes(labels, preds)

        label = labels[0]
        pred_label = preds[0]

        if pred_label.shape != label.shape:
            pred_label = mx.ndarray.argmax(pred_label, axis=self._axis)

        pred_label = pred_label.asnumpy().astype('int32')  # (n, num_attributes)
        pred_label = np.squeeze(pred_label)
        label = label.asnumpy().astype('int32')  # (n, num_attributes)

        assert(label.shape[0] == pred_label.shape[0])
        assert(label.shape[1] == self._attributes_num)
        assert(pred_label.shape == label.shape)

        good_mask = pred_label == label

        self.sum_acc += np.count_nonzero(good_mask)
        self.acc_num += pred_label.size

        if not self._one_metric:

            incorrect_mask = np.logical_not(good_mask)
            gt_pos_mask = label == 1
            gt_neg_mask = np.logical_not(gt_pos_mask)

            tp_mask = np.logical_and(gt_pos_mask, good_mask)
            fp_mask = np.logical_and(gt_neg_mask, incorrect_mask)
            fn_mask = np.logical_and(gt_pos_mask, incorrect_mask)

            for i in range(self._attributes_num):
                self.sum_corr_attr[i] += np.count_nonzero(good_mask[:,i])
                self.acc_num_attr[i] += good_mask[:,i].size
                self.sum_tp_attr[i] += np.count_nonzero(tp_mask[:,i])
                self.sum_fp_attr[i] += np.count_nonzero(fp_mask[:,i])
                self.sum_fn_attr[i] += np.count_nonzero(fn_mask[:,i])

    def reset(self):
        """Resets the internal evaluation result to initial state."""
        try:
            self.sum_acc = 0
            self.acc_num = 0
            if not self._one_metric:
                self.sum_corr_attr = np.zeros((self._attributes_num,))
                self.sum_tp_attr = np.zeros((self._attributes_num,))
                self.sum_fn_attr = np.zeros((self._attributes_num,))
                self.sum_fp_attr = np.zeros((self._attributes_num,))
                self.acc_num_attr = np.zeros((self._attributes_num,))
        except AttributeError:
            pass

    def get(self):
        """Gets the current evaluation result.
        Returns
        -------
        names : list of str
           Name of the metrics.
        values : list of float
           Value of the evaluations.
        """
        res = []

        accuracy = 0.0
        if self.acc_num > 0:
            accuracy = float(self.sum_acc) / self.acc_num

        if self._one_metric:
            return 'accuracy', accuracy
        else:
            res = []
            attributes_names = get_attr_text(self._attributes_num)
            for i in range(self._attributes_num):
                name = attributes_names[i]

                val_acc = 0.0
                val_recall = 0.0
                val_precision = 0.0

                corr = self.sum_corr_attr[i]
                cnt = self.acc_num_attr[i]
                tp = self.sum_tp_attr[i]
                fn = self.sum_fn_attr[i]
                fp = self.sum_fp_attr[i]

                if cnt > 0:
                    val_acc = float(corr) / cnt
                if tp + fn > 0:
                    val_recall = float(tp) / (tp + fn)
                if tp + fp > 0:
                    val_precision = float(tp) / (tp + fp)

                res.append(('{} accuracy'.format(name), val_acc))
                res.append(('{} recall'.format(name), val_recall))
                res.append(('{} precision'.format(name), val_precision))

            res.append(('accuracy', accuracy))

            names, value = zip(*res)
            return names, value