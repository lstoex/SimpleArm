import numpy as np


def perlin_noise_2d(shape: tuple[int, int], res: tuple[int, int]) -> np.ndarray:
    """Generate a 2D numpy array of perlin noise.
    Args:
        shape: The shape of the generated array (tuple of two ints).
            This must be a multiple of res.
        res: The number of periods of noise to generate along each
            axis (tuple of two ints). Note shape must be a multiple of
            res.
    Returns:
        A numpy array of shape shape with the generated noise.

    Courtesy of Johannes Tenhumberg / WZK https://github.com/tenhjo/chompy/blob/a78c462376da25d8ce4e45d510e5030a18e278f8/wzk/perlin.py#L11
    """

    def scalar2array(v, shape):
        if isinstance(shape, int):
            shape = (shape,)
        if np.shape(v) != shape:
            return np.full(shape, v)
        else:
            return v

    shape_, res_ = np.atleast_1d(shape, res)
    res_ = scalar2array(v=res_, shape=2)
    # check of shape is multiple of res
    assert np.all(shape_ % res_ == 0), "shape must be a multiple of res"
    delta = res_ / shape_
    d = shape_ // res_

    grid = np.transpose(
        np.mgrid[0 : res_[0] : delta[0], 0 : res_[1] : delta[1]], (1, 2, 0)
    )
    grid = grid % 1

    # Gradients
    angles = 2 * np.pi * np.random.random((res_[0] + 1, res_[1] + 1))
    gradients = np.dstack((np.cos(angles), np.sin(angles)))
    gradients = gradients.repeat(d[0], 0).repeat(d[1], 1)
    g00 = gradients[: -d[0], : -d[1]]
    g10 = gradients[d[0] :, : -d[1]]
    g01 = gradients[: -d[0], d[1] :]
    g11 = gradients[d[0] :, d[1] :]

    # Ramps
    n00 = np.sum(np.dstack((grid[:, :, 0], grid[:, :, 1])) * g00, 2)
    n10 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1])) * g10, 2)
    n01 = np.sum(np.dstack((grid[:, :, 0], grid[:, :, 1] - 1)) * g01, 2)
    n11 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1] - 1)) * g11, 2)

    # Interpolation
    def interpolant(t):
        return t * t * t * (t * (t * 6 - 15) + 10)

    t = interpolant(grid)
    n0 = n00 * (1 - t[:, :, 0]) + t[:, :, 0] * n10
    n1 = n01 * (1 - t[:, :, 0]) + t[:, :, 0] * n11
    return np.sqrt(2) * ((1 - t[:, :, 1]) * n0 + t[:, :, 1] * n1)
