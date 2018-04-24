import numpy as np

class regression:
    def __init__(self,fit_intercept=True,normalize=True,lr=0.01,decay=0.,momentum=0.,alpha=1.0):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.lr = lr
        self.decay = decay
        self.momentum = momentum


class Lasso(regression):
