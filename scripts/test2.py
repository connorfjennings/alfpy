import numpy as np
from mpi4py.futures import MPIPoolExecutor


if __name__ == '__main__':
    executor = MPIPoolExecutor(max_workers=4)
    future = executor.submit(pow, 2, np.arange(4))
    print(future.result())
    #future.shutdone()
    assert future.done()
