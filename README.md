# Searchless
A hackable gridsearch (and hyperopt) without needing a framework, encapsulation or controlflow changes.

The source code is designed to be fairly minimal, the core just being ~80 LOC.

## Usage
In your script to run simply replace the values to grid search with one of the proxies (`LinearFloatProxy`, etc)
and call `resolve_proxies()` before they are used to load the values of the current run.

```py
from searchless import LinearFloatProxy, resolve_proxies

hyperparameters = {
  # Will go through lr={1e-4, 4e-4, 7e-4, 1e-3}
  'lr': LinearFloatProxy(1e-4, 1e-3, 4),
  # Will go through beta1={0.99, 0.9945, 0.999}
  'beta1': LinearFloatProxy(0.99, 0.999, 3)
}
resolve_proxies()
```

If values need to be used before some others are defined simply call `resolve_proxies` multiple times.

## Example
The following command will run `example.py` with an exaustive grid search.

```sh
$ python3 ./searchless.py "python3 ./example.py"
```

## Roadmap
* [ ] Quasi-random scheduling for improved early coverage
* [ ] Non-grid search operation (Evolutionary search using feedback?)