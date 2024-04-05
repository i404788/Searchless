import numpy as np


class SlidingStdMean:
    """
    A class which efficiently tracks mean, variance, std for a window of data which can be dynamically updated
    """

    def __init__(self, init_mean: np.ndarray, init_var: np.ndarray, window: int):
        self.window = window
        self.mean = init_mean
        self.var = init_var
        self.std = np.sqrt(init_var)

    def update(self, discard: np.ndarray, add: np.ndarray):
        pmean = self.mean.copy()
        self.mean = pmean + (add - discard)/self.window
        self.var += (add-discard)*(add-self.mean+discard-pmean)/self.window
        self.std = np.sqrt(self.var)


class BayesianStdMean:
    """
    A class which calculates the mean, var, std using quasi-bayesian method in order to save memory.
    It adds to the mean iteratively while decaying the running mean by the amount of window that would be added
    """

    def __init__(self, initial_sample: np.ndarray):
        self.window = 1
        self.mean = initial_sample
        self.var = np.zeros_like(initial_sample)
        self.std = np.zeros_like(initial_sample)

    def update(self, add: np.ndarray):
        self.window += 1
        p = (self.window-1)/self.window
        pmean = self.mean.copy()
        self.mean = pmean*p + add/self.window

        # asciimath 1/N(x_N^2 + (N-1)(v + mu_1^2)) - mu_2^2
        self.var = (add**2 + (self.window-1)*(self.var+pmean**2))/self.window - self.mean**2
        self.std = np.sqrt(self.var)


class WelfordsStdMean:
    """
    Should be similar to BayesianStdMean but more compute efficient
    """

    def __init__(self, initial_sample: np.ndarray):
        self.mean = initial_sample
        self.n = 1
        self.m2 = 0
        self.var = 0
        self.std = 0

    def update(self, add: np.ndarray):
        self.n += 1
        delta = add - self.mean
        self.mean = self.mean + (delta/self.n)
        self.m2 = self.m2 + delta*(add - self.mean)

        self.var = self.m2/(self.n-1)
        self.std = np.sqrt(self.var)


class LazyWelfordStdMean(WelfordsStdMean):
    def __init__(self):
        self._has_started = False

    def update(self, add: np.ndarray):
        if not self._has_started:
            self.n = 1
            self.mean = add
            self.m2 = np.zeros_like(add)
            self.var = np.zeros_like(add)
            self.std = np.zeros_like(add)
            self._has_started = True
        else:
            super().update(add)


class LazyBayesianStdMean(BayesianStdMean):
    def __init__(self):
        self._has_started = False

    def update(self, add: np.ndarray):
        if not self._has_started:
            self.window = 1
            self.mean = add
            self.var = np.zeros_like(add)
            self.std = np.zeros_like(add)
            self._has_started = True
        else:
            super().update(add)


class LazyMinMax:
    def __init__(self):
        self._has_started = False

    def update(self, add: np.ndarray):
        if not self._has_started:
            self.min = add
            self.max = add
            self._has_started = True
        else:
            self.min = np.min(np.stack((self.min, add), axis=0), axis=0)
            self.max = np.max(np.stack((self.max, add), axis=0), axis=0)


if __name__ == '__main__':
    x = np.random.rand(100)
    print("Testing:")

    dut = SlidingStdMean(x[0:10].mean(), x[0:10].var(), 10)
    dut.update(x[0], x[10])
    assert np.isclose(dut.var, x[1:11].var()), "Variance is not close"
    assert np.isclose(dut.mean, x[1:11].mean()), "Mean is not close"
    print('.', flush=True, end='')
    dut.update(x[1], x[11])
    assert np.isclose(dut.var, x[2:12].var()), "Variance is not close"
    assert np.isclose(dut.mean, x[2:12].mean()), "Mean is not close"
    print('.', flush=True, end='')
    dut.update(x[2], x[12])
    assert np.isclose(dut.var, x[3:13].var()), "Variance is not close"
    assert np.isclose(dut.mean, x[3:13].mean()), "Mean is not close"
    print('.', flush=True, end='')

    print(x[0], type(x[0]))
    dut = BayesianStdMean(x[0])
    dut.update(x[1])
    assert np.isclose(dut.mean, x[0:2].mean()), "Mean is not close"
    print('.', flush=True, end='')
    dut.update(x[2])
    assert np.isclose(dut.mean, x[0:3].mean()), "Mean is not close"
    assert np.isclose(dut.var, x[0:3].var()), f"Var is not close: {dut.var} == {x[0:3].var()}"
    print('.', flush=True, end='')
    dut.update(x[3])
    assert np.isclose(dut.mean, x[0:4].mean()), "Mean is not close"
    assert np.isclose(dut.var, x[0:4].var()), f"Var is not close: {dut.var} == {x[0:4].var()}"
    print('.', flush=True, end='')

    print()
    print('Finished unit tests without panic\n')
    print('Starting stress tests...')

    stress_length = 10000
    parallel = 10
    w = 100  # window
    x = np.random.rand(parallel, stress_length)
    dut = SlidingStdMean(x[:, 0:w].mean(axis=-1), x[:, 0:w].var(axis=-1), w)
    print(f'Sliding {w=} {stress_length=} {parallel=}:')
    for i in range(1000-w):
        dut.update(x[:, i], x[:, i+w])
        assert np.isclose(dut.var, x[:, i+1:i+w+1].var(axis=-1)).all(), f"Variance is not close at {i}"
        assert np.isclose(dut.mean, x[:, i+1:i+w+1].mean(axis=-1)).all(), f"Mean is not close at {i}"
        if i % 10 == 0:
            print('.', flush=True, end='')

    print()
    print(f'Bayesian {stress_length=} {parallel=}:')
    dut = BayesianStdMean(x[:, 0])
    for i in range(1, 1000-w):
        dut.update(x[:, i])
        assert np.isclose(dut.var, x[:, 0:i+1].var(axis=-1)).all(), f"Variance is not close at {i}"
        assert np.isclose(dut.mean, x[:, 0:i+1].mean(axis=-1)).all(), f"Mean is not close at {i}"
        if i % 10 == 0:
            print('.', flush=True, end='')
    print()
    print('Finished stress tests, success!')
