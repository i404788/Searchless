"""
Usage: 
* Instatiate *Proxy where you'd normally use the regular value with the right parameters
* After creating all instances call `resolve_proxies()`
* Run `matrix_trainer.py {cmd}`

Now matrix_trainer will repeatedly call `{cmd}` while varying the Proxies

It functions by passing .matrix_session.toml with the current steps, could be improved by using tmpfs
"""
import os
import toml
import uuid
import numbers
from functools import partial
import tempfile

_registered_variables = []

number_operators = '__abs__, __add__, __ceil__, __eq__, __floor__, __floordiv__, __le__, __lt__, __mod__, __mul__, __neg__, __pos__, __pow__, __radd__, __rfloordiv__, __rmod__, __rmul__, __round__, __rpow__, __rtruediv__, __truediv__, __trunc__, __sub__, __rsub__'.split(', ')
class LinearFloatProxy:
    def __init__(self, lower: float, higher: float, steps: int):
        _registered_variables.append(self)
        self._val = float()
        self.__steps = steps
        self.__lower = lower
        self.__higher = higher
        for operator in number_operators:
            setattr(self.__class__, operator, property(partial(self.__class__.__getattr__, name=operator)))

    def __getattr__(self, name: str):
        return self._val.__getattribute__(name)

    def _resolve(self, step: int):
        assert step <= self.__steps, "Expected current step to be lower than maximum steps"
        self._val = (self.__higher-self.__lower) * (step / (self._steps() - 1)) + self.__lower

    def _steps(self):
        return self.__steps

    def __float__(self):
        return self._val


def resolve_proxies():
    with open(os.environ['MATRIX_SESSION_FILE'], 'r') as f:
        cfg = toml.load(f)

    if cfg['step'] == 0:
        # [Once] Calculate the maximum steps needed
        total_steps = 1
        for v in _registered_variables:
            total_steps *= v._steps()
        
        cfg['total_steps'] = total_steps
        with open(os.environ['MATRIX_SESSION_FILE'], 'w+') as f:
            toml.dump(cfg, f)
            f.flush()

    current_step = cfg['step']
    for v in _registered_variables:
        # TODO: Ideally we take a quasi random approach and maximize coverage first
    
        # Currently takes an overflowing buckets approach, if cs > v0_s, v1_s >= 1, etc
        v._resolve(int(current_step) % v._steps())
        current_step /= v._steps()


if __name__ == '__main__':
    import sys
    cfg = {
        'step': 0
    }

    i = 0
    with tempfile.NamedTemporaryFile(mode='w+', suffix='matrix_session.toml') as file:
        os.environ['MATRIX_SESSION_FILE'] = file.name
        print(f'Using session file {file.name}')
        
        while True:
            file.seek(0)
            toml.dump(cfg, file.file)
            file.flush()
            os.system(' '.join(sys.argv[1:]))

            if 'total_steps' not in cfg:
                # [Once] Load maximum calculated steps from session
                file.seek(0)
                cfg = toml.load(file.file)

            i += 1
            cfg['step'] = i
            assert 'total_steps' in cfg, "Expected another script to call resolve_proxies"
            if i >= cfg['total_steps']:
                print(cfg)
                break
   