from spectrum import *
import numpy as np


def my_arburg(im, order):
    """
    求自回归系数
    :param im: 灰度图 m*n
    :param order: 阶数
    :return:
    """
    imT = np.transpose(im.copy())
    imud = np.flipud(im)
    im[:, 1::2] = imud[:, 1::2]
    imud = np.flipud(imT)
    imT[:, 1::2] = imud[:, 1::2]

    im = im.transpose().flatten()
    imt = imT.transpose().flatten()
    marple_data = np.append(im, imt)
    a, b, rho = arburg(marple_data, order)
    real = np.real(a)
    real = np.round(real, 4)
    # print(real)
    return real


if __name__ == '__main__':
    x = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    x = np.array(x)
    my_arburg(x, order=6)
