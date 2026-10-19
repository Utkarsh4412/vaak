"""Unit tests for Session — isolation, history, and cancel event."""

import threading

from vaak.session import Session


class TestSessionDefaults:
    def test_default_language_is_english(self):
        s = Session()
        assert s.language == "en"

    def test_default_memory_turns_four(self):
        s = Session(memory_turns=4)
        assert s.history.maxlen == 8  # 4 turns * 2 messages each

    def test_zero_memory_turns_gives_unlimited_deque(self):
        s = Session(memory_turns=0)
        assert s.history.maxlen is None

    def test_cancel_event_starts_clear(self):
        s = Session()
        assert not s.cancel.is_set()

    def test_is_speaking_starts_false(self):
        s = Session()
        assert s.is_speaking is False


class TestSessionIsolation:
    def test_two_sessions_have_different_ids(self):
        s1 = Session()
        s2 = Session()
        assert s1.id != s2.id

    def test_history_not_shared_between_sessions(self):
        s1 = Session()
        s2 = Session()
        s1.append_turn("hello", "hi there")
        assert len(s1.history) == 2
        assert len(s2.history) == 0

    def test_cancel_not_shared_between_sessions(self):
        s1 = Session()
        s2 = Session()
        s1.request_cancel()
        assert s1.cancel.is_set()
        assert not s2.cancel.is_set()

    def test_language_not_shared(self):
        s1 = Session(language="hi")
        s2 = Session(language="en")
        assert s1.language == "hi"
        assert s2.language == "en"


class TestSessionHistory:
    def test_append_turn_adds_two_messages(self):
        s = Session()
        s.append_turn("hello", "world")
        assert len(s.history) == 2
        assert s.history[0] == {"role": "user", "content": "hello"}
        assert s.history[1] == {"role": "assistant", "content": "world"}

    def test_history_respects_maxlen(self):
        s = Session(memory_turns=1)  # maxlen=2
        s.append_turn("a", "b")
        s.append_turn("c", "d")
        assert len(s.history) == 2
        assert list(s.history)[0]["content"] == "c"

    def test_build_context_starts_with_system(self):
        s = Session()
        ctx = s.build_context("Be helpful.")
        assert ctx[0] == {"role": "system", "content": "Be helpful."}

    def test_build_context_includes_history(self):
        s = Session()
        s.append_turn("ping", "pong")
        ctx = s.build_context("sys")
        # system + user + assistant = 3 messages
        assert len(ctx) == 3
        assert ctx[1]["role"] == "user"
        assert ctx[2]["role"] == "assistant"


class TestSessionCancel:
    def test_reset_cancel_clears_event(self):
        s = Session()
        s.request_cancel()
        assert s.cancel.is_set()
        s.reset_cancel()
        assert not s.cancel.is_set()

    def test_cancel_is_thread_safe(self):
        s = Session()
        errors = []

        def setter():
            try:
                for _ in range(1000):
                    s.request_cancel()
                    s.reset_cancel()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=setter) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors

    def test_session_passes_llm_to_memory(self):
        class DummyLLM:
            pass

        dummy_llm = DummyLLM()
        s = Session(llm=dummy_llm)  # type: ignore[arg-type]
        assert s.memory.llm is dummy_llm
