import pytest
from environments.compat import HAVE_PETTINGZOO

@pytest.mark.skipif(not HAVE_PETTINGZOO, reason='optional PettingZoo/Gymnasium not installed locally')
def test_official_parallel_api():
    from pettingzoo.test import parallel_api_test
    from environments.pettingzoo_env import BallMARLEnv
    parallel_api_test(BallMARLEnv(),num_cycles=100)
