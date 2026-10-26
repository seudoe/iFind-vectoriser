"""
End-to-end API test against a RUNNING vectoriser and a THROWAWAY database (never the production one).

    # terminal 1: a second instance on its own port + scratch DB
    MONGODB_DB_NAME=ifind_vectest PORT=6100 python app.py
    # terminal 2
    VECTORISER_TEST_URL=http://localhost:6100 VECTORISER_TEST_DB=ifind_vectest pytest tests/test_api_integration.py -v

Skipped unless VECTORISER_TEST_URL is set. Refuses to run when the DB name is "ifind" (production).
Seeds its own internships and students, drops the scratch DB afterwards.
"""

import os
import time

import pytest
import requests

BASE = os.environ.get("VECTORISER_TEST_URL")
DB_NAME = os.environ.get("VECTORISER_TEST_DB", "ifind_vectest")

pytestmark = pytest.mark.skipif(not BASE, reason="set VECTORISER_TEST_URL to run the API integration tests")

APPROVED = {"status": "auto_approved", "scamScore": 0.05}

INTERNSHIPS = [
    ("Python Backend Intern", "Build REST APIs with Python, FastAPI and PostgreSQL. Docker, Git, unit testing."),
    ("Machine Learning Intern", "Train NLP models with PyTorch and scikit-learn; embeddings, transformers, data pipelines."),
    ("Frontend React Intern", "Build responsive UIs with React, TypeScript, Tailwind CSS and Next.js."),
    ("Data Analyst Intern", "SQL, Excel, Tableau dashboards, statistics and business reporting."),
    ("Digital Marketing Intern", "SEO, social media campaigns, content writing and Google Analytics."),
    ("Graphic Design Intern", "Photoshop, Illustrator, Figma, branding and poster design."),
]

PARSED_RESUME = {
    "name": "Test Student",
    "skills": ["Python", "FastAPI", "PostgreSQL", "Docker", "Git", "REST APIs"],
    "education": [{"institution": "Test University", "degree": "B.Tech", "field": "Computer Science"}],
    "experience": [{"title": "Backend Developer Intern", "company": "Acme", "description": "Built REST APIs in Python."}],
    "projects": [{"name": "URL shortener", "description": "FastAPI + PostgreSQL service deployed with Docker."}],
}


@pytest.fixture(scope="module")
def db():
    assert DB_NAME != "ifind", "refusing to run against the production database"
    import pymongo
    from dotenv import load_dotenv
    from pathlib import Path
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    client = pymongo.MongoClient(os.environ["MONGODB_URI"], serverSelectionTimeoutMS=15000)
    d = client[DB_NAME]
    for name in d.list_collection_names():
        d[name].drop()
    yield d
    client.drop_database(DB_NAME)


def post(path, body=None, **kw):
    return requests.post(f"{BASE}{path}", json=body if body is not None else {}, timeout=300, **kw)


def test_health():
    r = requests.get(f"{BASE}/health", timeout=10)
    assert r.status_code == 200 and r.json()["ready"] is True


def test_root():
    assert requests.get(f"{BASE}/", timeout=10).status_code == 200


def test_encode_internships_shape_and_mongo_doc():
    r = post("/encode-internships", {"internships": [
        {"id": "a1", "title": INTERNSHIPS[0][0], "description": INTERNSHIPS[0][1]},
        {"_id": {"$oid": "664a00000000000000000001"}, "name": INTERNSHIPS[1][0], "summary": INTERNSHIPS[1][1]},
    ]})
    assert r.status_code == 200
    body = r.json()
    assert body["encoded"] == 2
    assert [v["id"] for v in body["vectors"]] == ["a1", "664a00000000000000000001"]
    for v in body["vectors"]:
        assert len(v["tfidf"]) == 15000 and len(v["bert"]) == 768
        assert abs(sum(x * x for x in v["bert"]) - 1) < 1e-3          # L2-normalised


def test_encode_internships_is_deterministic_and_discriminates():
    def enc(t, d):
        return post("/encode-internships", {"internships": [{"id": "x", "title": t, "description": d}]}).json()["vectors"][0]["bert"]
    a, a2, b = enc(*INTERNSHIPS[0]), enc(*INTERNSHIPS[0]), enc(*INTERNSHIPS[4])
    dot = lambda u, v: sum(x * y for x, y in zip(u, v))
    assert dot(a, a2) > 0.999
    assert dot(a, b) < 0.9


def test_encode_internships_batch_limit():
    items = [{"id": str(i), "title": "t", "description": "d"} for i in range(71)]
    assert post("/encode-internships", {"internships": items}).status_code == 422


def test_encode_internships_validation():
    assert post("/encode-internships", {}).status_code == 422


def test_encode_resume_parsed_json_and_text():
    j = post("/encode-resume", {"resume": PARSED_RESUME})
    t = post("/encode-resume", {"resume": "Python developer with FastAPI, PostgreSQL and Docker experience"})
    assert j.status_code == 200 and t.status_code == 200
    for r in (j, t):
        body = r.json()
        assert len(body["tfidf"]) == 15000 and len(body["bert"]) == 768 and body["recommended"] is None


def test_encode_resume_empty_rejected():
    assert post("/encode-resume", {"resume": "   "}).status_code == 422
    assert post("/encode-resume", {}).status_code == 422


def test_pipeline_end_to_end(db):
    from bson import ObjectId
    # 1. staging: 6 approved + 1 not approved
    ids = []
    for title, desc in INTERNSHIPS:
        _id = ObjectId()
        ids.append(_id)
        db["internships.mod-unvectorised"].insert_one({
            "_id": _id, "title": title, "description": desc, "isActive": True,
            "applyLink": f"https://example.com/{_id}", "moderation": APPROVED,
        })
    held = ObjectId()
    db["internships.mod-unvectorised"].insert_one({
        "_id": held, "title": "Held", "description": "needs review", "isActive": True,
        "moderation": {"status": "pending_review", "scamScore": 0.9},
    })

    r = post("/vectorize-hnsw", {"recommend": False})
    assert r.status_code == 200, r.text
    stats = r.json()["stats"]
    assert stats["processed"] == 6 and stats["errors"] == 0

    # approved ones moved with vectors, unapproved stays
    assert db["internships.mod-unvectorised"].count_documents({}) == 1
    assert db["internships.mod-unvectorised"].find_one({"_id": held})
    live = db["internships"].find_one({"_id": ids[0]})
    assert len(live["tfidf_vector"]) == 15000 and len(live["bert_vector"]) == 768
    assert db["internships.graph"].count_documents({}) == 1

    # 2. employer post vectorised in place
    pid = ObjectId()
    db["internships.this-platform"].insert_one({
        "_id": pid, "name": "Python API Intern", "company": "Acme", "summary": "Backend developer: Python, FastAPI, PostgreSQL, Docker and REST APIs",
        "isActive": True, "status": "published", "moderation": APPROVED,
    })
    r = post("/vectorize-platform", {"ids": [str(pid)], "recommend": False})
    assert r.status_code == 200 and r.json()["stats"]["processed"] == 1
    assert len(db["internships.this-platform"].find_one({"_id": pid})["bert_vector"]) == 768

    # 3. graph search finds the Python backend internship for a Python resume
    vec = post("/encode-resume", {"resume": PARSED_RESUME}).json()
    res = post("/search-internships", {"vector": vec["bert"], "k": 3}).json()["results"]
    assert res[0]["id"] in {str(ids[0]), str(pid)}                            # nearest neighbour is a Python backend role

    # 4. encode-resume with user_id saves vectors and recommends
    uid = ObjectId()
    db["users"].insert_one({"_id": uid, "username": "vectest", "role": "student", "city": "Pune", "state": "Maharashtra"})
    r = post("/encode-resume", {"resume": PARSED_RESUME, "user_id": str(uid), "recommend": True})
    assert r.status_code == 200 and r.json()["recommended"] > 0
    user = db["users"].find_one({"_id": uid})
    assert len(user["resume"]["bert_vector"]) == 768
    recs = user["recommendedInternships"]
    scores = recs["recommendedScores"]
    assert str(recs["recommendedList"][0]) in {str(ids[0]), str(pid)}       # the Python backend roles lead
    assert scores[0]["score"] - scores[-1]["score"] > 0.2                    # and clearly beat the unrelated ones
    assert [x["score"] for x in scores] == sorted((x["score"] for x in scores), reverse=True)
    assert str(pid) in [str(x) for x in recs["recommendedList"]]             # employer posts are recommended too

    # 5. recommend-all (sync) for that student
    r = post("/recommend-all", {"user_ids": [str(uid)], "background": False})
    assert r.status_code == 200 and r.json()["status"] == "ok"

    # 6. rebuild graph from stored vectors
    r = post("/rebuild-graph")
    assert r.status_code == 200 and r.json()["indexed"] >= 6
    assert len(post("/search-internships", {"vector": vec["bert"], "k": 3}).json()["results"]) > 0


def test_vectorize_hnsw_unknown_ids_are_skipped(db):
    from bson import ObjectId
    r = post("/vectorize-hnsw", {"ids": [str(ObjectId())], "recommend": False})
    assert r.status_code == 200 and r.json()["stats"]["processed"] == 0


def test_search_validation():
    assert post("/search-internships", {"vector": [0.1], "k": 0}).status_code == 422
