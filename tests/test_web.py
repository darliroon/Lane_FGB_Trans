"""Web API (needs Flask)."""
import pytest

pytest.importorskip("flask")

from lane_trans.web.app import create_app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    return create_app().test_client()


@pytest.fixture(scope="module")
def example(client):
    return client.get("/api/example").get_json()


def test_config_lists_defaults_and_kernels(client):
    j = client.get("/api/config").get_json()
    assert j["defaults"] == {"max_loop_len": 200.0, "connector_margin": 0.1, "parallel_ratio": 1.5}
    assert "v0" in j["kernels"] and "v1" in j["kernels"] and j["kernel"] == "v0"


def test_static_page_is_served(client):
    assert client.get("/").status_code == 200
    assert client.get("/main.js").status_code == 200


def test_convert_example_removes_loops(client, example):
    r = client.post("/api/convert", json=example)
    j = r.get_json()
    assert r.status_code == 200
    assert j["stats"]["before"]["loops"] > 0 and j["stats"]["after"]["loops"] == 0
    assert len(j["log"]) > 0


def test_kernel_field_selects_the_algorithm(client, example):
    v0 = client.post("/api/convert", json=example).get_json()
    v1 = client.post("/api/convert", json={**example, "kernel": "v1"}).get_json()
    assert v1["stats"]["after"]["loops"] == 0
    assert any(l.startswith("connector") for l in v0["log"])
    assert not any(l.startswith("connector") for l in v1["log"])


def test_params_are_applied(client, example):
    j = client.post("/api/convert", json={**example, "params": {"max_loop_len": 1}}).get_json()
    assert j["stats"]["after"]["loops"] == j["stats"]["before"]["loops"]


def test_empty_graph(client):
    r = client.post("/api/convert", json={"nodes": [], "edges": []})
    assert r.status_code == 200 and r.get_json()["graph"] == {"nodes": [], "edges": []}


@pytest.mark.parametrize("body, msg", [
    ({"nodes": [{"id": 1, "x": 0, "y": 0}], "edges": [{"source": 1, "target": 9}]}, "unknown node"),
    ({"nodes": [], "edges": [], "params": {"parallel_ratio": 0.5}}, "parallel_ratio"),
    ({"nodes": [], "edges": [], "params": {"max_loop_len": "nan"}}, "finite"),
    ({"nodes": [], "edges": [], "kernel": "nope"}, "unknown kernel"),
    ({"nodes": [], "edges": [], "kernel": ["x"]}, "unknown kernel"),
    ({"nodes": [{"id": 1, "x": "a", "y": 0}], "edges": []}, "malformed"),
])
def test_bad_requests_get_400(client, body, msg):
    r = client.post("/api/convert", json=body)
    assert r.status_code == 400 and msg in r.get_json()["error"]


def test_non_object_body_gets_400(client):
    assert client.post("/api/convert", data="[]", content_type="application/json").status_code == 400
