import json
from pathlib import Path

import pytest

from msynd.pos import pos

CASES = json.loads((Path(__file__).parent / "fixtures/pos_cases.json").read_text())


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["name"])
def test_pos_shared_fixture(case):
    assert pos(**case["input"]) == pytest.approx(case["expected"])
