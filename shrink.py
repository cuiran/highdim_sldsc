import keras.backend as K
from keras.callbacks import Callback
import pdb

def shrink_weights_fn_creator(weight_list, alpha=0.01):
    # build updates
    updates = []
    L1_metric = 0
    for w in weight_list:
        # clamp step to no more than w
        step = K.sign(w) * K.minimum(K.abs(w), K.abs(alpha*K.sign(w)))
        updates.append(K.update_add(w,-step))
        L1_metric += K.sum(K.abs(w))

    # create a function that returns the L1_metric and shrinks weights via
    # updates.
    return K.function([], [L1_metric], updates=updates)


class L1_update(Callback):
    def __init__(self, weights_to_shrink, lr, regularizer):
        self.shrink_weights_fn = shrink_weights_fn_creator(weights_to_shrink,alpha=lr*regularizer)
        self.weights = weights_to_shrink

    def on_batch_end(self,batch, logs={}):
        l1 = self.shrink_weights_fn([])
#        print("L1", l1[0])
