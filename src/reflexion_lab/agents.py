from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Literal

USE_MOCK = os.getenv("USE_MOCK", "0") == "1"

if USE_MOCK:
    from .mock_runtime import actor_answer as _actor_answer
    from .mock_runtime import evaluator as _evaluator
    from .mock_runtime import reflector as _reflector

    def actor_answer(example, attempt_id, agent_type, reflection_memory):
        return _actor_answer(example, attempt_id, agent_type, reflection_memory), 0

    def evaluator(example, answer):
        return _evaluator(example, answer), 0

    def reflector(example, attempt_id, judge):
        return _reflector(example, attempt_id, judge), 0
else:
    from .ollama_runtime import actor_answer, evaluator, reflector

from .schemas import AttemptTrace, QAExample, ReflectionEntry, RunRecord


def _classify_failure(question: str, predicted: str, gold: str, reflections: list) -> str:
    q = question.lower()
    p = predicted.lower()
    g = gold.lower()

    # Looping: phản chiếu nhiều lần nhưng vẫn sai
    if len(reflections) >= 2:
        strategies = [r.next_strategy for r in reflections]
        if len(set(strategies)) == 1:
            return "looping"

    # Reflection overfit: có reflection nhưng answer trở nên dài/lan man hơn gold
    if reflections and len(predicted) > len(gold) * 3:
        return "reflection_overfit"

    # Incomplete multi-hop: câu hỏi dạng "what X of the Y who/that Z"
    multi_hop_keywords = ["who wrote", "who directed", "who invented", "born in", "founded by", "what river", "what country", "what city", "what ocean", "what language"]
    if any(kw in q for kw in multi_hop_keywords):
        # Nếu answer là một phần của gold hoặc ngược lại → dừng sớm
        if p in g or g in p:
            return "incomplete_multi_hop"
        return "incomplete_multi_hop"

    # Entity drift: answer không liên quan gì đến gold
    gold_words = set(g.split())
    pred_words = set(p.split())
    if len(gold_words & pred_words) == 0 and len(gold_words) > 0:
        return "entity_drift"

    return "wrong_final_answer"

@dataclass
class BaseAgent:
    agent_type: Literal["react", "reflexion"]
    max_attempts: int = 1
    def run(self, example: QAExample) -> RunRecord:
        reflection_memory: list[str] = []
        reflections: list[ReflectionEntry] = []
        traces: list[AttemptTrace] = []
        final_answer = ""
        final_score = 0
        for attempt_id in range(1, self.max_attempts + 1):
            import time as _time
            t0 = _time.monotonic()
            answer, actor_tokens = actor_answer(example, attempt_id, self.agent_type, reflection_memory)
            judge, eval_tokens = evaluator(example, answer)
            latency_ms = int((_time.monotonic() - t0) * 1000)
            token_estimate = actor_tokens + eval_tokens
            trace = AttemptTrace(attempt_id=attempt_id, answer=answer, score=judge.score, reason=judge.reason, token_estimate=token_estimate, latency_ms=latency_ms)
            final_answer = answer
            final_score = judge.score
            if judge.score == 1:
                traces.append(trace)
                break

            if self.agent_type == "reflexion" and attempt_id < self.max_attempts:
                reflection, ref_tokens = reflector(example, attempt_id, judge)
                trace.token_estimate += ref_tokens
                trace.reflection = reflection
                reflections.append(reflection)
                reflection_memory.append(reflection.next_strategy)
            traces.append(trace)
        total_tokens = sum(t.token_estimate for t in traces)
        total_latency = sum(t.latency_ms for t in traces)
        if final_score == 1:
            failure_mode = "none"
        else:
            failure_mode = _classify_failure(example.question, final_answer, example.gold_answer, reflections)

        return RunRecord(qid=example.qid, question=example.question, gold_answer=example.gold_answer, agent_type=self.agent_type, predicted_answer=final_answer, is_correct=bool(final_score), attempts=len(traces), token_estimate=total_tokens, latency_ms=total_latency, failure_mode=failure_mode, reflections=reflections, traces=traces)

class ReActAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(agent_type="react", max_attempts=1)

class ReflexionAgent(BaseAgent):
    def __init__(self, max_attempts: int = 3) -> None:
        super().__init__(agent_type="reflexion", max_attempts=max_attempts)
