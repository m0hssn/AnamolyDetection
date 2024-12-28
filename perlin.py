import numpy as np
import math

def lerp_np(x, y, w):
    """
    Linear interpolation between x and y using weight w.
    """
    return (y - x) * w + x

def rand_perlin_2d_np(shape, res, fade=lambda t: 6 * t**5 - 15 * t**4 + 10 * t**3):
    """
    Generate a 2D Perlin noise array.
    
    Parameters:
        shape (tuple): The shape of the output array (height, width).
        res (tuple): The number of grid points (rows, cols) for the gradients.
        fade (function): A fade function for smooth interpolation (default: Perlin's fade).
    
    Returns:
        np.ndarray: A 2D array of Perlin noise values.
    """
    # Compute delta (step size) and grid cell dimensions
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])

    # Generate grid points within the unit square
    grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]].transpose(1, 2, 0) % 1

    # Generate random gradient directions
    angles = 2 * math.pi * np.random.rand(res[0] + 1, res[1] + 1)
    gradients = np.stack((np.cos(angles), np.sin(angles)), axis=-1)

    # Helper function to tile gradients
    def tile_grads(slice1, slice2):
        return np.repeat(
            np.repeat(gradients[slice1[0]:slice1[1], slice2[0]:slice2[1]], d[0], axis=0),
            d[1],
            axis=1
        )
    
    # Helper function to compute dot product with gradient
    def dot(grad, shift):
        shifted_grid = np.stack(
            (
                grid[:shape[0], :shape[1], 0] + shift[0],
                grid[:shape[0], :shape[1], 1] + shift[1]
            ),
            axis=-1
        )
        return (shifted_grid * grad[:shape[0], :shape[1]]).sum(axis=-1)

    # Compute dot products for the corners of each grid cell
    n00 = dot(tile_grads([0, -1], [0, -1]), [0, 0])
    n10 = dot(tile_grads([1, None], [0, -1]), [-1, 0])
    n01 = dot(tile_grads([0, -1], [1, None]), [0, -1])
    n11 = dot(tile_grads([1, None], [1, None]), [-1, -1])

    # Interpolate noise values using the fade function
    t = fade(grid[:shape[0], :shape[1]])
    return math.sqrt(2) * lerp_np(
        lerp_np(n00, n10, t[..., 0]),
        lerp_np(n01, n11, t[..., 0]),
        t[..., 1]
    )
