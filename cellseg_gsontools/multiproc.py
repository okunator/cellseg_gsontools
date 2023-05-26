import os
import warnings
from typing import Any, Callable, Generator, List, Union

from pathos.pools import ProcessPool, SerialPool, ThreadPool
from tqdm import tqdm

__all__ = ["run_pool"]


def iter_pool_generator(
    it: Generator, res: List = None, pbar: bool = True, length: int = None
) -> Union[List[Any], None]:
    """Iterate over a pool generator object.

    Parameters
    ----------
        it : Generator
            A Generator object containing results from a concurrent run.
        res : List | None
            An empty list, where the results from the generator will be saved.
            If None, no results will be saved.
        pbar : bool, default=True
            Flag, whether to use tqdm or not.
        length : int, optional
            The length of the generator object. If None, will not use tqdm.

    Returns
    -------
        Union[List[Any], None]:
            A list of results or None.
    """
    it = tqdm(it, total=length) if pbar else it
    if res is not None:
        for x in it:
            res.append(x)
    else:
        for _ in it:
            pass

    return res


def run_pool(
    func: Callable,
    args: List[Any],
    ret: bool = True,
    pooltype: str = "thread",
    maptype: str = "amap",
    n_jobs: int = -1,
    pbar: bool = True,
) -> Union[List[Any], None]:
    """Run a pathos Thread, Process or Serial pool object.

    NOTE: if `ret` is set to True and `func` callable does not return anything. This
          will return a list of None values.

    Parameters
    ----------
        func : Callable
            The function that will be copied to existing cores and run in parallel.
        args : List[Any]
            A list of arguments for each of the parallelly executed functions.
        ret : bool, default=True
            Flag, whether to return a list of results from the pool object. Will be set
            to False e.g. when saving data to disk in parallel etc.
        pooltype : str, default="thread"
            The pathos pooltype. Allowed: ("process", "thread", "serial")
        maptype : str, default="amap"
            The map type of the pathos Pool object.
            Allowed: ("map", "amap", "imap", "uimap")
        n_jobs : int, default=-1
            Number of processes/threads to use. If -1, will use all available cores.
        pbar : bool, default=True
            Flag, whether to use tqdm or not. Does not work with `maptype` "amap".
            Makes sense to use only with `maptype` "uimap" and "imap".

    Raises
    ------
        ValueError: if illegal `pooltype` or `maptype` is given.

    Returns
    -------
        Union[List[Any], None]:
            A list of results or None.

    Example
    -------
        >>> f = myfunc # any function.
        >>> args = (1, 2, 3)
        >>> res_list = run_pool(f, args)
    """
    allowed = ("process", "thread", "serial")
    if pooltype not in allowed:
        raise ValueError(f"Illegal `pooltype`. Got {pooltype}. Allowed: {allowed}")

    allowed = ("map", "amap", "imap", "uimap")
    if maptype not in allowed:
        raise ValueError(f"Illegal `maptype`. Got {maptype}. Allowed: {allowed}")

    Pool = None
    if pooltype == "thread":
        Pool = ThreadPool
    elif pooltype == "process":
        Pool = ProcessPool
    else:
        if maptype in ("amap", "uimap"):
            raise ValueError(
                f"`SerialPool` has only `map` & `imap` implemented. Got: {maptype}."
            )
        Pool = SerialPool

    n_jobs = os.cpu_count() if n_jobs == -1 else n_jobs
    results = [] if ret else None
    if maptype == "map":
        with Pool(nodes=n_jobs) as pool:
            it = pool.map(func, args)
            results = iter_pool_generator(it, results, pbar=pbar, length=len(args))
    elif maptype == "amap":
        if pbar:
            warnings.warn("`amap` does not support progress bar.")
        with Pool(nodes=n_jobs) as pool:
            results = pool.amap(func, args).get()
    elif maptype == "imap":
        with Pool(nodes=n_jobs) as pool:
            it = pool.imap(func, args)
            results = iter_pool_generator(it, results, pbar=pbar, length=len(args))
    elif maptype == "uimap":
        with Pool(nodes=n_jobs) as pool:
            it = pool.uimap(func, args)
            results = iter_pool_generator(it, results, pbar=pbar, length=len(args))

    return results
