import unittest
from threading import Lock

import pytest
from docker.errors import APIError

from compose.parallel import CascadeLimiter
from compose.parallel import GlobalLimit
from compose.parallel import parallel_execute
from compose.parallel import parallel_execute_iter
from compose.parallel import ParallelLimitExceededByCascadeCalls
from compose.parallel import ParallelStreamWriter
from compose.parallel import UpstreamError


web = 'web'
db = 'db'
data_volume = 'data_volume'
cache = 'cache'

objects = [web, db, data_volume, cache]

deps = {
    web: [db, cache],
    db: [data_volume],
    data_volume: [],
    cache: [],
}


def get_deps(obj):
    return [(dep, None) for dep in deps[obj]]


class ParallelTest(unittest.TestCase):

    def test_parallel_execute(self):
        results, errors = parallel_execute(
            objects=[1, 2, 3, 4, 5],
            func=lambda x: x * 2,
            get_name=str,
            msg="Doubling",
        )

        assert sorted(results) == [2, 4, 6, 8, 10]
        assert errors == {}

    def test_parallel_execute_with_limit(self):
        limit = 1
        tasks = 20
        lock = Lock()

        def f(obj):
            locked = lock.acquire(False)
            # we should always get the lock because we're the only thread running
            assert locked
            lock.release()
            return None

        results, errors = parallel_execute(
            objects=list(range(tasks)),
            func=f,
            get_name=str,
            msg="Testing",
            limit=limit,
        )

        assert results == tasks * [None]
        assert errors == {}

    def test_parallel_execute_with_global_limit(self):
        GlobalLimit.set_global_limit(1)
        self.addCleanup(GlobalLimit.set_global_limit, None)
        tasks = 20
        lock = Lock()

        def f(obj):
            locked = lock.acquire(False)
            # we should always get the lock because we're the only thread running
            assert locked
            lock.release()
            return None

        results, errors = parallel_execute(
            objects=list(range(tasks)),
            func=f,
            get_name=str,
            msg="Testing",
        )

        assert results == tasks * [None]
        assert errors == {}

    @pytest.mark.timeout(10)
    def test_parallel_cascade_execute_with_global_limit(self):
        COMPOSE_PARALLEL_LIMIT = 2

        GlobalLimit.set_global_limit(COMPOSE_PARALLEL_LIMIT)
        CascadeLimiter.set_global_limit(COMPOSE_PARALLEL_LIMIT)
        self.addCleanup(GlobalLimit.set_global_limit, None)
        self.addCleanup(CascadeLimiter.set_global_limit, None)

        tasks = COMPOSE_PARALLEL_LIMIT
        inner_tasks = 2

        lock = Lock()

        def inner_f(obj):
            locked = lock.acquire(False)
            # we should always get the lock because we're the only thread running
            assert locked
            lock.release()
            return None

        def outer_f(obj, limit):
            return parallel_execute(
                objects=list(range(inner_tasks)),
                func=inner_f,
                get_name=str,
                msg="Testing cascade parallel level 1 func",
                limit=limit
            )

        results, errors = parallel_execute(
            objects=list(range(tasks)),
            func=outer_f,
            get_name=str,
            msg="Testing cascade parallel level 0 func",
            limit=CascadeLimiter()
        )

        assert results == tasks * [(inner_tasks * [None], {})]
        assert errors == {}

    @pytest.mark.timeout(10)
    def test_parallel_multiple_cascade_execute_with_global_limit(self):
        COMPOSE_PARALLEL_LIMIT = 3

        GlobalLimit.set_global_limit(COMPOSE_PARALLEL_LIMIT)
        CascadeLimiter.set_global_limit(COMPOSE_PARALLEL_LIMIT)
        self.addCleanup(GlobalLimit.set_global_limit, None)
        self.addCleanup(CascadeLimiter.set_global_limit, None)

        tasks = COMPOSE_PARALLEL_LIMIT
        middle_tasks = 2
        inner_tasks = 2

        lock = Lock()

        def inner_f(obj):
            locked = lock.acquire(False)
            # we should always get the lock because we're the only thread running
            assert locked
            lock.release()
            return None

        def middle_f(obj, limit):
            return parallel_execute(
                objects=list(range(inner_tasks)),
                func=inner_f,
                get_name=str,
                msg="Testing cascade parallel level 2 func",
                limit=limit
            )

        def outer_f(obj, limit):
            return parallel_execute(
                objects=list(range(middle_tasks)),
                func=middle_f,
                get_name=str,
                msg="Testing cascade parallel level 1 func",
                limit=CascadeLimiter(limit)
            )
        limiter = CascadeLimiter()
        results, errors = parallel_execute(
            objects=list(range(tasks)),
            func=outer_f,
            get_name=str,
            msg="Testing cascade parallel level 0 func",
            limit=limiter
        )
        assert results == tasks * [(inner_tasks * [(middle_tasks * [None], {})], {})]
        assert errors == {}

    @pytest.mark.timeout(10)
    def test_parallel_cascade_execute_with_global_limit_less_than_call_stack_height(self):
        COMPOSE_PARALLEL_LIMIT = 2

        GlobalLimit.set_global_limit(COMPOSE_PARALLEL_LIMIT)
        CascadeLimiter.set_global_limit(COMPOSE_PARALLEL_LIMIT)
        self.addCleanup(GlobalLimit.set_global_limit, None)
        self.addCleanup(CascadeLimiter.set_global_limit, None)

        tasks = COMPOSE_PARALLEL_LIMIT
        middle_tasks = 2
        inner_tasks = 2

        lock = Lock()

        def inner_f(obj):
            locked = lock.acquire(False)
            # we should always get the lock because we're the only thread running
            assert locked
            lock.release()
            return None

        def middle_f(obj, limit):
            print('middle limit:', limit)
            return parallel_execute(
                objects=list(range(inner_tasks)),
                func=inner_f,
                get_name=str,
                msg="Testing cascade parallel level 2 func",
                limit=limit
            )

        def outer_f(obj, limit):
            print('outer limit:', limit)
            return parallel_execute(
                objects=list(range(middle_tasks)),
                func=middle_f,
                get_name=str,
                msg="Testing cascade parallel level 1 func",
                limit=CascadeLimiter(limit)
            )

        with pytest.raises(ParallelLimitExceededByCascadeCalls):
            results, errors = parallel_execute(
                objects=list(range(tasks)),
                func=outer_f,
                get_name=str,
                msg="Testing cascade parallel level 0 func",
                limit=CascadeLimiter()
            )

            assert results == tasks * [(inner_tasks * [(middle_tasks * [None], {})], {})]
            assert errors == {}

    def test_parallel_execute_with_deps(self):
        log = []

        def process(x):
            log.append(x)

        parallel_execute(
            objects=objects,
            func=process,
            get_name=lambda obj: obj,
            msg="Processing",
            get_deps=get_deps,
        )

        assert sorted(log) == sorted(objects)

        assert log.index(data_volume) < log.index(db)
        assert log.index(db) < log.index(web)
        assert log.index(cache) < log.index(web)

    def test_parallel_execute_with_upstream_errors(self):
        log = []

        def process(x):
            if x is data_volume:
                raise APIError(None, None, "Something went wrong")
            log.append(x)

        parallel_execute(
            objects=objects,
            func=process,
            get_name=lambda obj: obj,
            msg="Processing",
            get_deps=get_deps,
        )

        assert log == [cache]

        events = [
            (obj, result, type(exception))
            for obj, result, exception
            in parallel_execute_iter(objects, process, get_deps, None)
        ]

        assert (cache, None, type(None)) in events
        assert (data_volume, None, APIError) in events
        assert (db, None, UpstreamError) in events
        assert (web, None, UpstreamError) in events


def test_parallel_execute_alignment(capsys):
    ParallelStreamWriter.instance = None
    results, errors = parallel_execute(
        objects=["short", "a very long name"],
        func=lambda x: x,
        get_name=str,
        msg="Aligning",
    )

    assert errors == {}

    _, err = capsys.readouterr()
    a, b = err.split('\n')[:2]
    assert a.index('...') == b.index('...')


def test_parallel_execute_ansi(capsys):
    ParallelStreamWriter.instance = None
    ParallelStreamWriter.set_noansi(value=False)
    results, errors = parallel_execute(
        objects=["something", "something more"],
        func=lambda x: x,
        get_name=str,
        msg="Control characters",
    )

    assert errors == {}

    _, err = capsys.readouterr()
    assert "\x1b" in err


def test_parallel_execute_noansi(capsys):
    ParallelStreamWriter.instance = None
    ParallelStreamWriter.set_noansi()
    results, errors = parallel_execute(
        objects=["something", "something more"],
        func=lambda x: x,
        get_name=str,
        msg="Control characters",
    )

    assert errors == {}

    _, err = capsys.readouterr()
    assert "\x1b" not in err
