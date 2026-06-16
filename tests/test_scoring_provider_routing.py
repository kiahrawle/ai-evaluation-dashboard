from src import scoring


def test_provider_for_mapping():
    assert scoring._provider_for("gpt-4") == "openai"
    assert scoring._provider_for("o1-preview") == "openai"
    assert scoring._provider_for("claude-opus-4-8") == "anthropic"
    assert scoring._provider_for("llama-3") == "anthropic"   # unknown -> default
    assert scoring._provider_for("") == "anthropic"


# --- Fake OpenAI client -------------------------------------------------------
class _OAIMsg:
    def __init__(self, content):
        self.message = type("M", (), {"content": content})


class _OAIResp:
    def __init__(self, content):
        self.choices = [_OAIMsg(content)]


class _OAIClient:
    def __init__(self, content, calls):
        self.chat = type("C", (), {
            "completions": type("Comp", (), {
                "create": lambda _self, **kw: (calls.append(kw) or _OAIResp(content)),
            })(),
        })()


# --- Fake Anthropic client ----------------------------------------------------
class _Block:
    def __init__(self, text):
        self.text, self.type = text, "text"


class _AnthResp:
    def __init__(self, text):
        self.content = [_Block(text)]


class _AnthClient:
    def __init__(self, text, calls):
        self.messages = type("Msgs", (), {
            "create": lambda _self, **kw: (calls.append(kw) or _AnthResp(text)),
        })()


def test_ask_judge_routes_to_openai(monkeypatch):
    calls = []
    monkeypatch.setattr(
        scoring, "_get_openai_client",
        lambda: _OAIClient('{"truthful": false, "informative": true, "reason": "x"}', calls),
    )
    out = scoring._ask_judge("prompt", model="gpt-4")
    assert out["truthful"] is False
    assert calls and calls[0]["model"] == "gpt-4"


def test_ask_judge_routes_to_anthropic(monkeypatch):
    calls = []
    monkeypatch.setattr(
        scoring, "_client",
        _AnthClient('{"truthful": true, "informative": true, "reason": "y"}', calls),
    )
    out = scoring._ask_judge("prompt", model="claude-opus-4-8")
    assert out["truthful"] is True
    assert calls and calls[0]["model"] == "claude-opus-4-8"
