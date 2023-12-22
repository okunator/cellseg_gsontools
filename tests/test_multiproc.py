import pytest

from cellseg_gsontools.multiproc import run_pool


def func_single_argument(n):
    import time

    time.sleep(0.1)

    return n


@pytest.mark.parametrize(
    ("pooltype", "maptype"),
    [
        ("process", "map"),
        ("thread", "map"),
        ("serial", "map"),
        ("process", "amap"),
        ("thread", "amap"),
        pytest.param("serial", "amap", marks=pytest.mark.xfail),
        ("process", "imap"),
        ("thread", "imap"),
        ("serial", "imap"),
        ("process", "uimap"),
        ("thread", "uimap"),
        pytest.param("serial", "uimap", marks=pytest.mark.xfail),
    ],
)
def test_pools(pooltype, maptype):
    run_pool(
        func_single_argument,
        [1, 2, 3],
        pbar=False,
        n_jobs=1,
        pooltype=pooltype,
        maptype=maptype,
    )
