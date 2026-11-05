import pytest
from spark_jobs.spark_session import get_spark


@pytest.fixture(scope="session")
def spark():
    s = get_spark("Tests")
    yield s
    s.stop()