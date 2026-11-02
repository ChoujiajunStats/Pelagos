"""Rigid transforms, with quaternions in ROS xyzw order."""
import math


def q_normalize(q):
    norm = math.sqrt(sum(float(v) ** 2 for v in q))
    if not math.isfinite(norm) or norm < 1e-12:
        raise ValueError('invalid quaternion')
    return tuple(float(v) / norm for v in q)


def q_multiply(a, b):
    x, y, z, w = a
    X, Y, Z, W = b
    return (w * X + x * W + y * Z - z * Y,
            w * Y - x * Z + y * W + z * X,
            w * Z + x * Y - y * X + z * W,
            w * W - x * X - y * Y - z * Z)


def q_inverse(q):
    x, y, z, w = q_normalize(q)
    return (-x, -y, -z, w)


def rotate(q, v):
    q = q_normalize(q)
    return q_multiply(q_multiply(q, (*v, 0.0)), q_inverse(q))[:3]


def compose(a, b):
    """T_ac = T_ab T_bc, where each tuple is (translation, quaternion)."""
    p, q = a
    t, r = b
    rotated = rotate(q, t)
    return tuple(p[i] + rotated[i] for i in range(3)), q_normalize(q_multiply(q, r))


def inverse(transform):
    p, q = transform
    invq = q_inverse(q)
    return rotate(invq, tuple(-v for v in p)), invq


def optical_transform(camera):
    """The existing Stonefish M2 profile uses an FLU body and forward optical camera."""
    x, y, z = camera['xyz_frd']
    expected = (math.pi / 2, 0.0, math.pi / 2)
    if any(abs(float(a) - b) > 1e-8 for a, b in zip(camera['rpy_frd'], expected)):
        raise ValueError('Unsupported Stonefish camera orientation; derive its real optical extrinsic')
    # Optical x right, y down, z forward expressed in body FLU.
    return ((float(x), -float(y), -float(z)), (-0.5, 0.5, -0.5, 0.5))


def matrix4(transform):
    p, q = transform
    x, y, z, w = q_normalize(q)
    return [[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w), p[0]],
            [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w), p[1]],
            [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y), p[2]],
            [0.0, 0.0, 0.0, 1.0]]
