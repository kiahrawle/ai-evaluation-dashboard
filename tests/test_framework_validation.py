import pytest

from src.research import framework_validation as fv


# --- pure stats --------------------------------------------------------------
def test_pearson_perfect_and_inverse():
    assert fv.pearson([1, 2, 3], [2, 4, 6]) == pytest.approx(1.0)
    assert fv.pearson([1, 2, 3], [6, 4, 2]) == pytest.approx(-1.0)


def test_pearson_zero_variance():
    assert fv.pearson([1, 1, 1], [1, 2, 3]) == 0.0
    assert fv.pearson([], []) == 0.0


def test_confusion_binary_counts():
    cm = fv.confusion_binary([1, 1, 0, 0], [1, 0, 0, 1])
    assert cm["tp"] == 1 and cm["fn"] == 1 and cm["tn"] == 1 and cm["fp"] == 1
    assert cm["accuracy"] == 0.5


def test_confusion_matrix_multiclass():
    cm = fv.confusion_matrix(
        ["LOW", "HIGH", "MEDIUM", "LOW"],
        ["LOW", "HIGH", "LOW", "LOW"],
        ["LOW", "MEDIUM", "HIGH"],
    )
    assert cm["n"] == 4
    assert cm["accuracy"] == pytest.approx(3 / 4)
    # row=MEDIUM(idx1), col=LOW(idx0) should be the single off-diagonal hit
    assert cm["matrix"][1][0] == 1


# --- end-to-end aggregation (engines stubbed, no embeddings) -----------------
def test_validate_framework_monkeypatched(tmp_path, monkeypatch):
    csv = tmp_path / "labels.csv"
    csv.write_text(
        "question,response,context,factual_groundedness,risk_level\n"
        "q1,grounded answer,ctx,1,LOW\n"
        "q2,wrong answer,ctx,0,HIGH\n"
        "q3,hedged maybe,ctx,1,MEDIUM\n",
        encoding="utf-8",
    )

    def fake_predict(response, context):
        if "grounded" in response:
            return {"pred_grounded": 1, "risk_score": 0.1, "pred_risk_level": "LOW"}
        if "wrong" in response:
            return {"pred_grounded": 0, "risk_score": 0.8, "pred_risk_level": "HIGH"}
        return {"pred_grounded": 1, "risk_score": 0.5, "pred_risk_level": "MEDIUM"}

    monkeypatch.setattr(fv, "predict_row", fake_predict)

    out = fv.validate_framework(csv)
    assert out["n"] == 3
    # All three predictions match the human labels in this stub.
    assert out["groundedness"]["accuracy"] == 1.0
    assert out["risk"]["accuracy"] == 1.0
    # Risk score rises monotonically with human ordinal -> strong correlation.
    assert out["risk_pearson"] > 0.9


def test_validate_framework_missing_columns(tmp_path):
    csv = tmp_path / "bad.csv"
    csv.write_text("response,context\nx,y\n", encoding="utf-8")
    with pytest.raises(ValueError):
        fv.validate_framework(csv)
