import sys
import types

# recommender imports the HNSW pipeline (torch, hnswlib, Mongo); the boost is a pure function, so stub that import
_stub = types.ModuleType("vectorizer_hnsw_pipeline")
for _n in ("APPROVED_STATUSES", "LIVE_COLLECTION", "PLATFORM_COLLECTION", "get_db", "get_search_manager"):
    setattr(_stub, _n, None)
sys.modules.setdefault("vectorizer_hnsw_pipeline", _stub)

from recommender import CITY_BOOST, REMOTE_BOOST, STATE_BOOST, location_boost

PUNE = [{"city": "Pune", "state": "Maharashtra", "country": "India"}]


def test_no_location_no_boost():
    assert location_boost(None, True, PUNE) == 0
    assert location_boost({"city": "", "state": None}, True, PUNE) == 0


def test_city_state_remote():
    assert location_boost({"city": "pune "}, False, PUNE) == CITY_BOOST
    assert location_boost({"city": "Mumbai", "state": "Maharashtra"}, False, PUNE) == STATE_BOOST
    assert location_boost({"city": "Delhi"}, True, PUNE) == REMOTE_BOOST
    assert location_boost({"city": "Delhi"}, False, PUNE) == 0


def test_bad_address_entries_are_ignored():
    assert location_boost({"city": "Pune"}, False, [None, "Pune", {"city": "Pune"}]) == CITY_BOOST
