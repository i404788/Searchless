import torch
from functorch import make_functional, jvp, grad
from torch.autograd.functional import vhp, hvp
from torch.optim import Adam, SGD
from pytorch_optimizer import AdaBelief, AdaHessian, SophiaH, SignSGD
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from functools import partial
from itertools import cycle
from collections import namedtuple
import inspect

from rolling import LazyBayesianStdMean


class ParameterModel(torch.nn.Module):
    "Dummy model which just returns it's parameters"

    def __init__(self, start: torch.Tensor):
        super().__init__()
        self.params = torch.nn.Parameter(start, requires_grad=True)

    def forward(self):
        return self.params


def sphere(x) -> torch.Tensor:
    return (x ** 2).sum()


def ackley(x) -> torch.Tensor:
    return -20*torch.exp(-0.2*torch.sqrt((1/len(x))*(x**2).sum())) - torch.exp((1/len(x))*torch.cos(2*torch.pi*x).sum()) + torch.e + 20


def ackley2(x) -> torch.Tensor:
    return -200*torch.exp(-0.2*torch.sqrt((x**2).sum()))


def rosenbork(x) -> torch.Tensor:
    v = 0
    for i in range(len(x)-1):
        v += 100 * (x[i+1] - x[i]**2)**2 + (1-x[i])**2
    return torch.log(v)


def holder_table(x) -> torch.Tensor:
    return -torch.abs(torch.sin(x[0])*torch.cos(x[1])*torch.exp(torch.abs(1-(torch.sqrt((x**2).sum())/torch.pi))))


def styblinski_tang(x) -> torch.Tensor:
    return (x**4 - 16*x**2 + 5*x).sum()/2


def alpine1(x) -> torch.Tensor:
    return torch.abs(x*x.sin()+0.1*x).sum()


def alpine2(x) -> torch.Tensor:
    # 2 Interpretations possible, seconds one seems more loegical
    # return -(torch.sqrt(x.prod()) * x.sin().sum()
    return -(x.prod() ** (1/len(x))) * x.sin().sum()


def bohachevsky2(x) -> torch.Tensor:
    return x[0]**2 + 2*x[1]**2 - (0.3*torch.cos(3*torch.pi*x[0])*torch.cos(4*torch.pi*x[1])) + 0.3


def camel3(x) -> torch.Tensor:
    return 2*x[0]**2 - 1.05*x[0]**4+x[0]**6/6+x[0]*x[1]+x[1]**2


def csendes(x) -> torch.Tensor:
    return (x**6 * (2+torch.sin(1/x))).sum()


def dcs(x) -> torch.Tensor:  # deflected_corrugated_spring
    x = ((x-5)**2).sum()
    return 0.1 * (x - torch.cos(5*torch.sqrt(x)))


def dropwave(x) -> torch.Tensor:
    x = (x**2).sum()
    return (1 + torch.cos(12*torch.sqrt(x))) / (torch.exp(0.5*x+1))


def easom(x) -> torch.Tensor:
    return -torch.cos(x).prod()*torch.exp((-(x**2)).sum())


def rastrigin(x) -> torch.Tensor:
    return (x**2 - 10*torch.cos(2*torch.pi*x)).sum() + 20


@torch.no_grad()
def plot_function(ax, f, bounds, resolution=50):
    # x = np.linspace(bounds[0], bounds[1], resolution)
    # y = np.linspace(bounds[0], bounds[1], resolution)
    # z = torch.zeros(len(x), len(y))
    # for i, xv in enumerate(x):
    #     for j, yv in enumerate(y):
    #         z[i, j] = f(torch.tensor([xv, yv]))
    # # x_2d, y_2d = np.meshgrid(x, y)
    # ax.plot_surface(x_2d, y_2d, z.numpy(), cmap=cm.jet)
    # print(f'Minimum {torch.min(z)}')

    xy = torch.zeros(resolution**2, 2)
    z = torch.zeros(resolution**2)
    x = np.linspace(bounds[0], bounds[1], resolution)
    y = np.linspace(bounds[0], bounds[1], resolution)
    for i, xv in enumerate(x):
        for j, yv in enumerate(y):
            xy[i*resolution+j] = torch.tensor([xv, yv])
            z[i*resolution+j] = f(xy[i*resolution+j])

    ax.plot_trisurf(xy[:, 0], xy[:, 1], z.numpy(), cmap=cm.jet)


TaskResult = namedtuple('TaskResult', ('best_loss', 'last_loss'))
Task = namedtuple('Task', ('loss_f', 'bounds', 'global_optimum', 'initial_f', 'dim'))

tasks = {
    'sphere': Task(sphere, [-3, 3], 0, lambda d: torch.randn(d) * 2,                             None),
    'ackley': Task(ackley, [-5, 5], -5.37, lambda d: torch.randn(d) * 2,                         None),
    'ackley2': Task(ackley2, [-10, 10], 0, lambda d: torch.randn(d)*4,                           None),
    'rosenbork': Task(rosenbork, [-2, 2], 0, lambda d: torch.randn(d),                           None),
    'holder_table': Task(holder_table, [-10, 10], None, lambda d: torch.randn(d) * 3,            2),
    'styblinski_tang': Task(styblinski_tang, [-5, 5], 2*-39.16599, lambda d: torch.randn(d) * 3, None),
    'alpine1': Task(alpine1, [-2, 2], 0, lambda d: torch.randn(d),                               None),
    'alpine2': Task(alpine2, [-2, 2], 0, lambda d: torch.randn(d),                               None),
    'bohachevsky2': Task(bohachevsky2, [-2, 2], 0, lambda d: torch.randn(d),                     2),
    'camel3': Task(camel3, [-2, 2], 0, lambda d: torch.randn(d),                                 2),
    'camel3_saddle': Task(camel3, [-2, 2], 0, lambda d: torch.tensor([0.7, -0.1]),               2),
    'camel3_saddle2': Task(camel3, [-2, 2], 0, lambda d: torch.tensor([0.7, -1.5]),               2),
    'csendes': Task(csendes, [-2, 2], 0, lambda d: torch.randn(d),                               None),
    'dcs': Task(dcs, [-4, 10], 0, lambda d: torch.randn(d),                                      None),
    'dropwave': Task(dropwave, [-2, 2], 0, lambda d: torch.randn(d),                             None),
    'easom': Task(easom, [-2, 2], 0, lambda d: torch.randn(d),                                   None),
    'rastrigin': Task(rastrigin, [-5, 5], 0, lambda d: torch.randn(d)*2,                         None),
}

colors = cycle(list(mcolors.TABLEAU_COLORS))


def hessian_estimation(model, task: Task):
    from jax.tree_util import tree_map
    # from functorch import vmap

    func, params = make_functional(model)

    def run_model(params):
        # NOTE: normally you would inject data to use in MSE/BCE/etc through closure
        return task.loss_f(func(params))

    noise = tree_map(lambda p: torch.randint(0, 2, p.size(), dtype=p.dtype, device=p.device) * 2.0 - 1.0, params)
    # noise = tree_map(lambda v: torch.randn_like(v), params)
    # loss_, hvp_est = tree_map(lambda v: v.t(), vhp(run_model, params, noise))
    # loss_, hvp_est = hvp(run_model, params, noise)
    loss_, hvp_est = jvp(grad(run_model), (params,), (noise,))
    return tree_map(lambda a, b: a*b, hvp_est, noise)


def run_test(optimizer_cls, task: Task, ax=None, steps=20, max_dim=10, seed=43, use_own_hessian=True):
    torch.manual_seed(seed)
    if ax is not None:
        dim = 2
    elif task.dim is None:
        dim = torch.randint(2, max_dim, (1,)).item()
    else:
        dim = task.dim or 2

    initial = torch.clip(task.initial_f(dim), *task.bounds)
    model = ParameterModel(initial.clone())
    opt = optimizer_cls(model.parameters())

    history = []
    spec = inspect.signature(opt.step)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)  # Fixes memory leak caused by create_graph (is potentially slower for non-hessian optimizers)

        loss = task.loss_f(model())
        history.append(torch.cat((model().detach(), loss.detach().unsqueeze(0)), dim=-1))
        loss.backward(create_graph=True)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)

        if 'hessian' in spec.parameters and use_own_hessian:
            if opt._step % opt.update_period == 0:
                hessian = hessian_estimation(model, task)
                opt.step(hessian=hessian)
            else:
                opt.step()
        else:
            opt.step()

    with torch.no_grad():
        history.append(torch.cat((model().detach(), task.loss_f(model()).unsqueeze(0)), dim=-1))

    history = torch.stack(history).numpy()
    if ax is not None:
        color = next(colors)
        offset = np.random.rand(3)*0.2
        for i, step in enumerate(np.diff(history, axis=0)):
            if 'hessian' in spec.parameters:
                ax.quiver(*history[i] + offset, *step, color=color, label=f'{opt.__class__.__name__} - {"hvp" if use_own_hessian else "hutchinson"}'.rstrip('- ') if i == 0 else None, arrow_length_ratio=0.05)
            else:
                ax.quiver(*history[i] + offset, *step, color=color, label=f'{opt.__class__.__name__}' if i == 0 else None, arrow_length_ratio=0.05)
            # dist = optimizer_cls.keywords.get('hessian_distribution', '')
            # ax.quiver(*history[i] + offset, *step, color=color, label=f'{opt.__class__.__name__} - {dist}'.rstrip('- ') if i == 0 else None, arrow_length_ratio=0.05)

    return TaskResult(history[:, -1].min(), history[-1, -1])


fig = plt.figure(figsize=(16, 9))
ax = fig.add_subplot(111, projection='3d', computed_zorder=False)

task = tasks['camel3_saddle2']
# task = tasks['ackley']
seed = np.random.randint(0, 2**15)

# initial = torch.clip(task.initial_f(2), *task.bounds)
# model = ParameterModel(initial.clone())
# opt = SophiaH(model.parameters())
# hessian_diag = hessian_estimation(model, task)
# push_hessian(opt, hessian_diag)
# print(hessian_diag)
# exit()

for i, optimizer in enumerate([partial(Adam, lr=0.1), partial(SGD, lr=0.1),
                              # partial(AdaBelief, lr=0.5), partial(AdaHessian, lr=1),
                                  # partial(SophiaH, lr=5e-2, p=2, betas=(0.9, 0.99), update_period=3, hessian_distribution='rademacher'),
                                  # partial(AdaHessian, lr=0.5, hessian_distribution='rademacher'), partial(AdaHessian, lr=0.5, hessian_distribution='gaussian'),
                                  partial(SophiaH, lr=5e-2, p=2, betas=(0.5, 0.99), update_period=2, hessian_distribution='rademacher'),
                                  partial(SophiaH, lr=5e-2, p=2, betas=(0.5, 0.99), update_period=2, hessian_distribution='gaussian'), partial(SignSGD, lr=5e-2)
                               ]):
    # tracker = LazyBayesianStdMean()
    # report = {}
    # for i in range(100):
    #     result = run_test(optimizer, task, ax=None, seed=seed+i, use_own_hessian=False)
    #     # print(result)
    #     report['worst'] = max(np.nan_to_num(result.last_loss), report.get('worst', 0))
    #     report['best'] = min(np.nan_to_num(result.last_loss, nan=1e9), report.get('best', 1e9))
    #     report['best_early'] = min(np.nan_to_num(result.best_loss, nan=1e9), report.get('best_early', 1e9))
    #     tracker.update(result.last_loss)

    # report['mean'] = tracker.mean
    # report['std'] = tracker.std
    # print(optimizer.func.__name__, report)

    result = run_test(optimizer, task, ax, seed=seed, use_own_hessian=i % 2 == 0)
    print(optimizer.func.__name__, result)


plot_function(ax, task.loss_f, task.bounds)
plt.legend(loc=2, fontsize=20, title_fontsize=25)
plt.show()
