import json
import threading
import urllib.error
import urllib.request

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper


def tiny_model(path, C=4):
    """[1, T, 356] -> every other frame -> linear -> log-softmax [1, ceil(T/2), C]."""
    w = np.random.default_rng(0).normal(size=(356, C)).astype(np.float32)
    ints = lambda name, v: helper.make_tensor(name, TensorProto.INT64, [1], [v])  # noqa: E731
    g = helper.make_graph(
        [helper.make_node("Slice", ["feats", "st", "en", "ax", "sp"], ["half"]),
         helper.make_node("MatMul", ["half", "W"], ["z"]),
         helper.make_node("LogSoftmax", ["z"], ["logprobs"], axis=-1)],
        "tiny",
        [helper.make_tensor_value_info("feats", TensorProto.FLOAT, [1, "T", 356])],
        [helper.make_tensor_value_info("logprobs", TensorProto.FLOAT, [1, "T2", C])],
        [helper.make_tensor("W", TensorProto.FLOAT, [356, C], w.ravel()),
         ints("st", 0), ints("en", 10**9), ints("ax", 1), ints("sp", 2)])
    # ir_version 10: onnx 1.23 defaults to 14, which onnxruntime 1.30 cannot load
    onnx.save(helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)], ir_version=10), path)


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    d = tmp_path_factory.mktemp("m")
    tiny_model(str(d / "m.onnx"))
    json.dump({"glosses": ["a", "b", "c"], "lookup": {"a b": "AB"}, "blank": 0}, open(d / "v.json", "w"))
    import app

    srv = app.make_server(str(d / "m.onnx"), str(d / "v.json"), port=0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def post(url, arr):
    req = urllib.request.Request(url + "/api/sentence", data=np.asarray(arr, "<f4").tobytes(), method="POST",
                                 headers={"Content-Type": "application/octet-stream"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def test_ok(server):
    code, body = post(server, np.random.default_rng(1).normal(size=(40, 356)))
    assert code == 200 and set(body) >= {"ids", "glosses", "text", "conf", "ms", "model"}
    assert len(body["glosses"]) == len(body["ids"]) == len(body["conf"])


def test_all_zero_frames_ok(server):  # nothing detected in any frame
    code, body = post(server, np.zeros((40, 356)))
    assert code == 200 and isinstance(body["text"], str)


def test_limits(server):
    assert post(server, np.zeros((451, 356)))[0] == 413  # over 30 s
    assert post(server, np.zeros((4, 356)))[0] == 400  # too short
    assert post(server, np.zeros(1001))[0] == 400  # not a whole number of frames
    bad = np.zeros((20, 356), np.float32)
    bad[3, 7] = np.nan
    assert post(server, bad)[0] == 400


def test_health(server):
    with urllib.request.urlopen(server + "/api/sentence/health") as r:
        assert json.loads(r.read()) == {"ok": True, "model": "large-us", "name": "SLI Sentences Large US"}
