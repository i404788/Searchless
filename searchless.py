"""
Usage: 
* Instatiate *Proxy where you'd normally use the regular value with the right parameters
* After creating all instances call `resolve_proxies()`
* Run `searchless.py {cmd}`

Now matrix_trainer will repeatedly call `{cmd}` while varying the Proxies

It functions by passing a tempfile *matrix_session.toml with the current steps & configuration
"""
import os
import toml
import uuid
import numbers
from functools import partial
import tempfile
import weakref
import time

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

class LogFloatProxy(LinearFloatProxy):
    def __init__(self, lower: float, higher: float, steps: int):
        _registered_variables.append(self)
        # print(f'{_registered_variables=}')
        self._val = float()
        self.__steps = steps
        self.__lower = lower
        self.__higher = higher
        self.__factor = (higher/lower) ** (1/steps)
        for operator in number_operators:
            setattr(self.__class__, operator, property(partial(self.__class__.__getattr__, name=operator)))

    def __getattr__(self, name: str):
        return self._val.__getattribute__(name)

    def _resolve(self, step: int):
        assert step <= self.__steps, "Expected current step to be lower than maximum steps"
        self._val = self.__lower * self.__factor ** step

    def _steps(self):
        return self.__steps

    def __float__(self):
        return self._val

class EnumeratorProxy:
    def __init__(self, values: list):
        _registered_variables.append(self)
        self._val = values[0]
        self.__enum = values
        self.__steps = len(values)

    def __getattr__(self, name: str):
        return self._val.__getattribute__(name)

    def _resolve(self, step: int):
        assert step <= self.__steps, "Expected current step to be lower than maximum steps"
        self._val = self.__enum[step]

    def _steps(self):
        return self.__steps

    def __str__(self):
        return str(self._val)
    
    def __float__(self):
        return float(self._val)

    def __int__(self):
        return int(self._val)

def resolve_proxies(*objs: dict):
    with open(os.environ['MATRIX_SESSION_FILE'], 'r') as f:
        cfg = toml.load(f)
        print(f'{cfg=}')

    if cfg['step'] < cfg['threads']*2:
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

    for obj in objs:
        if isinstance(obj, dict):
            for k in list(obj.keys()):
                if obj[k] in _registered_variables:
                    obj[k] = obj[k]._val


def run_thread(cfg, pipe):
    with tempfile.NamedTemporaryFile(mode='w+', suffix='matrix_session.toml') as file:
        os.environ['MATRIX_SESSION_FILE'] = file.name
        print(f'Using session file {file.name}')
        toml.dump(cfg, file)
        file.flush()

        g = {}
        exec(open(cfg['script']).read(), g, g)

        if 'total_steps' not in cfg:
            # [Once] Load maximum calculated steps from session
            file.seek(0)
            cfg = toml.load(file.file)
            pipe.send({'total_steps': cfg['total_steps']})
            time.sleep(0.5)
        print(f'Finished run {cfg["step"]}/{cfg["total_steps"]}')

if __name__ == '__main__':
    import sys
    import multiprocessing as mp
    mp.set_start_method('spawn')
    
    cfg = {
        'threads': 3,
        'step': 0,
        'script': sys.argv[-1]
    }

    procs = [] # (pipe, proc)

    while cfg['step'] < cfg.get('total_steps', cfg['threads']*2):
        procs = list(filter(lambda v: v[1].is_alive(), procs))
        if len(procs) < cfg['threads']:
            cfg['step'] += 1
            rx, tx = mp.Pipe()
            proc = mp.Process(target=run_thread, args=(cfg.copy(), tx))
            proc.start()
            procs.append((rx, proc))
            print(f"Started run {cfg['step']=}")

        for (pipe, proc) in procs:
            if proc.is_alive() and pipe.poll():
                try:
                    update = pipe.recv()
                    cfg.update(update)
                    print(f'Got cfg update from child {update=}')
                except EOFError:
                    ...
        time.sleep(0.1)

    # Await last jobs
    for proc in procs:
        proc[1].join()
        

