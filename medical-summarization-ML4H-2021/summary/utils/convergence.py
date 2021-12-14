class Convergence:
    """Methodology for model training convergence

    Parameters
    ----------
    patience : int
        How long to wait after last time quantity improved.
        Default: 7
    verbose : bool
        If True, prints a message for each quantity improvement.
        Default: False
    tol : float
        Minimum change in the monitored quantity to qualify as an improvement
        Default: 0
    minimize : bool
        If true convergence is based on minimizing metric otherwise maximization
    """
    def __init__(self, patience=7, verbose=False, tol=0, minimize=True):
        self.patience = patience
        self.verbose = verbose
        self.minimize = minimize
        self.counter = 0
        self.best_score = None
        self.best_ckpt = None
        self.stop = False
        self.tol = tol

    def __call__(self, conv_metric, ckpt):
        """Updates stop member variable based on convergence

        Parameters
        ----------
        conv_metric : float
            Metric which convergence is based upon
        """
        score = conv_metric if not self.minimize else -conv_metric

        if self.best_score is None:
            self.best_score = score
            self.best_ckpt = ckpt
        elif score < self.best_score + self.tol:
            self.counter += 1
            if self.verbose:
                print(f'Stopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.stop = True
        else:
            self.best_score = score
            self.best_ckpt = ckpt
            self.counter = 0
