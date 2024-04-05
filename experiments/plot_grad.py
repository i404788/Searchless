import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.stats.qmc import Sobol, Halton, LatinHypercube, PoissonDisk, scale
from scipy.special import ndtr as zprob
from sklearn.metrics.pairwise import cosine_similarity

fig = plt.figure(figsize=(6, 5))
ax = fig.add_subplot(111, projection='3d')


def sphere(x) -> float:
    return (x ** 2).sum()


ackley_bounds = np.asarray([-5, 5])
def ackley(x) -> float:
    return -20*np.exp(-0.2*np.sqrt(0.5*((x**2).sum()))) - np.exp(0.5*(np.cos(2*np.pi*x).sum())) + np.e + 20


rosenbork_bounds = np.asarray([-2, 2])  # Not technically the right bounds
def rosenbrok(x) -> float:
    v = 0
    for i in range(len(x)-1):
        v += 100 * (x[i+1] - x[i]**2)**2 + (1-x[i])**2
    return np.log(v)

holder_table_bounds = np.asarray([-10, 10])
def holder_table(x) -> float:
    return -np.abs(np.sin(x[0])*np.cos(x[1])*np.exp(np.abs(1-(np.sqrt((x**2).sum())/np.pi))))


styblinski_tang_bounds = np.asarray([-5, 5])
def styblinski_tang(x) -> float:
    return (x**4 - 16*x**2 + 5*x).sum()/2

# Full 20x20 gradient (400 evaluations)
# x = np.linspace(-1, 1, 20)
# y = np.linspace(-1, 1, 20)
# z = np.zeros((len(x), len(y)))
# for i, xv in enumerate(x):
#     for j, yv in enumerate(y):
#         z[i, j] = sphere(xv, yv)
# x_2d, y_2d = np.meshgrid(x, y)
# print(f'Minimum {np.min(z)}')
# ax.plot_surface(x_2d, y_2d, z, cmap=cm.jet)


# evals = 100
# ===Random pareto evaluations graident===
# x = np.random.pareto(1.0, size=evals)
# x[np.random.random(size=evals) > 0.5] *= -1
# x = np.clip(x, -1, 1)
# y = np.random.pareto(1.0, size=evals)
# y[np.random.random(size=evals) > 0.5] *= -1
# y = np.clip(y, -1, 1)

# z = np.zeros((evals,))
# for i, (xv, yv) in enumerate(zip(x, y)):
#     z[i] = sphere(xv, yv)
# print(f'Minimum {np.min(z)}')


# ===Quasi-Random sampling===
def distance_to_hyperplane(plane_factor, plane_offset, point):
    """
    Should give back the distance from a hyperplane to a point; the point OR the plane can be broadcasted batched
    if both are provided with a batch dim each point will be matched with a plane (akin to `zip()`)

    Examples:
    >>> distance_to_hyperplane(plane_factor=np.array([0,0,1]), plane_offset=np.array([1]), point=np.array([0,0,0]))
    array([1.])
    >>> distance_to_hyperplane(plane_factor=np.array([[1,1,0], [0, 0, 1]]), plane_offset=np.array([1, 1]), point=np.array([[0,0,0], [0,0,1]]))
    array([0.70710678, 0.])
    """
    assert point.shape[-1] == plane_factor.shape[-1], "Hyperplane should be the same dim as point"

    if plane_factor.ndim == 1:
        # Broadcast to plane batchdim or 1
        batch = 1
        if point.ndim == 2:
            batch = point.shape[0]

        plane_factor = np.tile(plane_factor, (batch, 1))
        plane_offset = np.tile(plane_offset, (batch,))

    assert plane_factor.shape[0] == plane_offset.shape[-1], "Need hyperplane offsets equal to hyperplane orientations"

    if point.ndim == 1:  # Broadcast to plane_factor
        point = np.tile(point, (plane_factor.shape[0], 1))

    return np.abs(np.einsum('bi,bi->b', point, plane_factor) - plane_offset) / np.linalg.norm(plane_factor, axis=-1)


def test_d2hp():
    factors = np.asarray([[1, 1, 0], [0, 0, 1], [1, 1, 0]], dtype=np.float64)
    # Hyperplane factors need to be normalized for clean calculations (otherwise [1,1,0] will have sqrt(2) slope in XY which is difficult to account for)
    factors /= np.linalg.norm(factors, axis=-1)[:, None]
    d = distance_to_hyperplane(factors, np.asarray([1, 1, 2]), np.asarray([[np.sqrt(2)/2, np.sqrt(2)/2, 0], [0, 0, 0], [np.sqrt(2), np.sqrt(2), 0]]))
    # print(d)
    assert np.isclose(d[2], 0), "Expected [1,1,0]~2 hp to be 2 from [sqrt2,sqrt2,0]"
    assert np.isclose(d[1], 1), "Expected [0,0,1]~1 hp to be 1 from [0,0,0]"
    assert np.isclose(d[0], 0), "Expected [1,1,0]~1 hp to be 0 from [sqrt2/2,sqrt2/2,0]"


# TODO: implement exploration, probably the easiest way is to create a second manifold which model each evaluated point as a 'well' and we sample from the entire manifold
#  The main question is if this is possible to do efficiently somehow, it would be nice to be able to 'narrow' the wells as more samples come in (how about normalizing flows?)

def geomean(a, axis=-1):
    return a.prod(axis=axis)**(1.0/a.shape[axis])


def sigmoid(z):
    return 1/(1 + np.exp(-z))


def cauchy(x):
    return np.tan(np.pi*(x-1/2))


def cauchyprob(x):
    "Similar to zprob but for a cauchy distribution instead of normal"
    return 1/np.pi * np.arctan(x) + 1/2


def depth_in_centroids(centroids: np.ndarray, std: float | np.ndarray, point: np.ndarray) -> np.ndarray:
    # Given centroids, calculate distance to point & combine with std to give the depth into the mixture of guassians
    # Distributions are isotropic to speed up & simplify the implementation

    # [b]d - [a]d -> bad
    d = np.linalg.norm(centroids[None, :, :] - point[:, None, :], axis=-1)  # distance[point, centroid]
    d /= std  # zscore[point, centroid]
    d = (zprob(d) - 0.5) * 2
    # d = (cauchyprob(d) - 0.5) * 2  # prob[pooint, centroid]

    # Calculate the approximate depth value of point given local centroids
    # nearest_idx = np.argsort(d, axis=-1)[:, :3]
    # print(nearest_idx, d[:, nearest_idx])
    # d = d[:, nearest_idx]  # 5 nearest centroid
    # d = geomean(d)  # prob[point]
    # d = d.min(axis=-1)
    d = d.prod()
    return d


def test_dic():
    dim = 2
    centroid_count = 10
    sample_per_axis = 30
    bounds = [-5, 5]
    centroids = np.random.uniform(bounds[0], bounds[1], size=(centroid_count, dim))
    x = np.linspace(bounds[0], bounds[1], sample_per_axis)
    y = np.linspace(bounds[0], bounds[1], sample_per_axis)
    z = np.zeros((len(x), len(y)))
    for i, xv in enumerate(x):
        for j, yv in enumerate(y):
            z[i, j] = depth_in_centroids(centroids, 3, np.array([[xv, yv]]))
    x_2d, y_2d = np.meshgrid(x, y)
    print(f'Minimum {np.min(z)}')
    ax.plot_surface(x_2d, y_2d, z, cmap=cm.jet)
    for centroid in centroids:
        ax.quiver(centroid[1], centroid[0], 0, 0, 0, 1, color="red")

    ax.set_xlabel('X-Axis')
    ax.set_ylabel('Y-Axis')
    ax.set_zlabel('Z-Axis')
    plt.show()


def plot_rejection_surface(centroids, std, bounds, sample_per_axis=50):
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection='3d')
    x = np.linspace(bounds[0], bounds[1], sample_per_axis)
    y = np.linspace(bounds[0], bounds[1], sample_per_axis)
    z = np.zeros((len(x), len(y)))
    for i, xv in enumerate(x):
        for j, yv in enumerate(y):
            z[i, j] = depth_in_centroids(centroids, std, np.array([[xv, yv]]))

    x_2d, y_2d = np.meshgrid(x, y)
    ax.plot_surface(x_2d, y_2d, z, cmap=cm.jet)
    for centroid in centroids:
        ax.quiver(centroid[1], centroid[0], 0, 0, 0, 1, color="red")

    ax.set_xlabel('X-Axis')
    ax.set_ylabel('Y-Axis')
    ax.set_zlabel('Z-Axis')
    plt.show()


def fast_coss(A):
    # base similarity matrix (all dot products)
    similarity = np.dot(A, A.T)

    # squared magnitude of preference vectors (number of occurrences)
    square_mag = np.diag(similarity)

    # inverse squared magnitude
    inv_square_mag = 1 / square_mag

    # if it doesn't occur, set it's inverse magnitude to zero (instead of inf)
    inv_square_mag[np.isinf(inv_square_mag)] = 0

    # inverse of the magnitude
    inv_mag = np.sqrt(inv_square_mag)

    # cosine similarity (elementwise multiply by inverse magnitudes)
    cosine = similarity * inv_mag
    cosine = cosine.T * inv_mag
    return cosine


def IGD(f, bounds, dim, xy, z, evaluations=50):
    # Interpolated-Gradient Descent
    # The idea of this method is to keep track of all rewards (and their parameters) to create
    #  an interpolated gradient over the reward space, given a smooth convex surface this should converge.
    # IGD only find the local optimum that can be derived from rewards of an initial grid-like search
    #  to find a even better optimum a new search can be started in the surrounding area of the final parameters

    # TODO: start from dfferent starting positions if multiple minima in xyz

    # Pick best known parameters
    steps = []
    stepsize = np.abs(bounds[0] - bounds[1])
    params = xy[z.argmin()].copy() + np.random.randn() * 0.01
    is_first = True  # TODO: alternatively add some noise
    for i in range(evaluations):
        nearest_points = np.argsort(np.linalg.norm(xy - params, axis=-1))[int(is_first):10]
        d = ((xy[nearest_points] - params) * (-z[nearest_points] / z[nearest_points].sum())[:, None]).mean(axis=0)
        target = d
        # d = xy[nearest_points] - params  # 5 nearest centroid
        # d /= np.linalg.norm(d, axis=-1)[:, None]
        # r = z[nearest_points]
        # # mag = np.linalg.norm(d, axis=-1)[:, None]
        # # d *= sigmoid(-mag)
        # # Calculate entire cosine simularity matrix (each row/column is one vector)
        # coss = 1 - fast_coss(d)
        # coss[coss > 1] = 1.  # Ignore orthogonal or worse similarity
        # coss[np.eye(coss.shape[0], dtype=bool)] = 1  # Ignore self
        # # print(coss)
        # coss = coss.sum(axis=-1)  # / coss.shape[-1]
        # param_r = f(params)
        # r = param_r - r
        # # # Weigh target vector by the similarity to other vectors & it's own reward
        # normalized_coss = coss / coss.sum()
        # normalized_reward = r / r.sum()
        # print(d.shape, normalized_reward.shape, normalized_coss.shape)
        # target = (d * normalized_reward[:, None] * normalized_coss[:, None]).sum(axis=0)
        # print(target, np.linalg.norm(target))

        # xy = np.concatenate((xy, [params]), axis=0)
        # z = np.concatenate((z, [param_r]), axis=0)

        steps.append((params.copy(), target.copy()))

        params += target * stepsize

        print(params)
        is_first = False
        # TODO: eval params + target, add as data point

    if dim == 2:
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_trisurf(xy[:, 0], xy[:, 1], z, cmap=cm.jet)

        for (base, delta) in steps:
            ax.quiver(base[0], base[1], f(base), delta[0], delta[1], 0, color='green')
        ax.quiver(params[0], params[1], f(base), 0, 0, -0.1, color='red')

        ax.set_xlabel('X-Axis')
        ax.set_ylabel('Y-Axis')
        ax.set_zlabel('Z-Axis')

        plt.show()

        pass

    return params


def nextopt(evaluations=25):
    # Basis of this optimization is to do exploration to maxize the distance between known samples
    # As well as avoid bounds as they give us less information (since the far side is truncated)
    # The idea sample from a uniform distribution with negative gaussians where we have already evaluated
    # For now this is not reward gradient guided but in the future it might
    dim = 2
    # f, bounds = ackley, ackley_bounds
    f, bounds = sphere, [-1, 1]
    evals = 0
    std = (bounds[1] - bounds[0]) / np.sqrt(evaluations)  # TODO: sqrt might be eval**(1/dim) in practice
    xy = np.zeros((evaluations, dim))
    z = np.zeros((evaluations,))

    def sample_eval():
        nonlocal evals
        # For now assume symetric bounds
        proposed = np.random.uniform(bounds[0], bounds[1], size=(dim,))
        if evals > 0:
            # TODO: check distance to bounds (hyperplanes or speheres?)
            # Rejection sampling
            prob = 1 - depth_in_centroids(xy[:evals], std, np.array([proposed]))
            print(f'{evals=} {proposed=} {prob=}')
            if np.random.random() < prob:
                print(f"Rejected sample {proposed}")
                return sample_eval()

        evals += 1
        return proposed

    for i in range(evaluations):
        to_eval = sample_eval()
        xy[i] = to_eval
        z[i] = f(to_eval)

    minidx = np.argmin(z)
    print(f'Minimum {np.min(z)} (@{xy[minidx]}) (in {len(xy)} evals)')
    if dim == 2:
        ax.plot_trisurf(xy[:, 0], xy[:, 1], z, cmap=cm.jet)

        ax.set_xlabel('X-Axis')
        ax.set_ylabel('Y-Axis')
        ax.set_zlabel('Z-Axis')

        plt.show()

        plot_rejection_surface(xy, std, bounds)

    found = IGD(f, bounds, dim, xy, z)
    print(f'IGD found: {found}')


def run_opt():
    log2evals = 5
    dim = 2
    f, bounds = ackley, ackley_bounds
    # f, bounds = sphere, [-1, 1]
    # f, bounds = rosenbrok, rosenbork_bounds
    # f, bounds = holder_table, holder_table_bounds
    # f, bounds = styblinski_tang, styblinski_tang_bounds

    # sampler = Sobol(d=dim, scramble=True)
    # sampler = Halton(d=dim, scramble=True)
    sampler = LatinHypercube(d=dim)
    # sampler = PoissonDisk(d=dim, radius=0.1)
    raw_sample = sampler.random(n=2**log2evals) if not getattr(sampler, 'random_base2', None) else sampler.random_base2(m=log2evals)
    sample = scale(raw_sample, bounds[0], bounds[1])
    z = np.zeros((len(sample),))
    for i, d in enumerate(sample):
        z[i] = f(d)

    minidx = np.argmin(z)
    print(f'Minimum {np.min(z)} (@{sample[minidx]}) (in {len(sample)} evals)')
    if dim == 2:
        ax.plot_trisurf(sample[:, 0], sample[:, 1], z, cmap=cm.jet)

        ax.set_xlabel('X-Axis')
        ax.set_ylabel('Y-Axis')
        ax.set_zlabel('Z-Axis')

        plt.show()


if __name__ == '__main__':
    test_d2hp()
    test_dic()