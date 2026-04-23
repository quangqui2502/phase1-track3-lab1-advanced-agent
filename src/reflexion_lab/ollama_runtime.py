from __future__ import annotations
import json
import time
import urllib.request
from .schemas import QAExample, JudgeResult, ReflectionEntry
from .prompts import ACTOR_SYSTEM, EVALUATOR_SYSTEM, REFLECTOR_SYSTEM

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "gemma3:4b"


def _chat(system: str, user: str) -> tuple[str, int, int]:
    """Returns (content, input_tokens, output_tokens)."""
    payload = json.dumps({
        "model": MODEL,
        "stream": False,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }).encode()
    req = urllib.request.Request(OLLAMA_URL, data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read())
    content = data["message"]["content"].strip()
    usage = data.get("usage", {})
    input_tokens = usage.get("prompt_tokens", 0)
    output_tokens = usage.get("completion_tokens", 0)
    # Ollama also exposes these fields at top level
    if input_tokens == 0:
        input_tokens = data.get("prompt_eval_count", 0)
        output_tokens = data.get("eval_count", 0)
    return content, input_tokens, output_tokens


def actor_answer(
    example: QAExample,
    attempt_id: int,
    agent_type: str,
    reflection_memory: list[str],
) -> tuple[str, int]:
    """Returns (answer, total_tokens)."""
    context_text = "\n\n".join(
        f"[{chunk.title}]\n{chunk.text}" for chunk in example.context
    )
    reflection_section = ""
    if reflection_memory:
        notes = "\n".join(f"- {r}" for r in reflection_memory)
        reflection_section = f"\n\nReflection notes from previous attempts:\n{notes}\nApply these strategies now."

    user_msg = (
        f"Context:\n{context_text}\n\n"
        f"Question: {example.question}{reflection_section}"
    )
    content, inp, out = _chat(ACTOR_SYSTEM, user_msg)
    return content, inp + out


def evaluator(example: QAExample, answer: str) -> tuple[JudgeResult, int]:
    """Returns (JudgeResult, total_tokens)."""
    user_msg = (
        f"Question: {example.question}\n"
        f"Gold answer: {example.gold_answer}\n"
        f"Predicted answer: {answer}"
    )
    raw, inp, out = _chat(EVALUATOR_SYSTEM, user_msg)
    try:
        # Strip markdown code fences if present
        clean = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        parsed = json.loads(clean)
        result = JudgeResult(
            score=int(parsed.get("score", 0)),
            reason=parsed.get("reason", ""),
            missing_evidence=parsed.get("missing_evidence", []),
            spurious_claims=parsed.get("spurious_claims", []),
        )
    except Exception:
        # Fallback: simple string match
        from .utils import normalize_answer
        score = 1 if normalize_answer(example.gold_answer) == normalize_answer(answer) else 0
        result = JudgeResult(score=score, reason=f"JSON parse failed, fallback match. Raw: {raw[:200]}")
    return result, inp + out


def reflector(
    example: QAExample,
    attempt_id: int,
    judge: JudgeResult,
) -> tuple[ReflectionEntry, int]:
    """Returns (ReflectionEntry, total_tokens)."""
    context_text = "\n\n".join(
        f"[{chunk.title}]\n{chunk.text}" for chunk in example.context
    )
    user_msg = (
        f"Question: {example.question}\n\n"
        f"Context:\n{context_text}\n\n"
        f"Wrong answer: {judge.reason}\n"
        f"Missing evidence: {judge.missing_evidence}"
    )
    raw, inp, out = _chat(REFLECTOR_SYSTEM, user_msg)
    try:
        clean = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        parsed = json.loads(clean)
        entry = ReflectionEntry(
            attempt_id=attempt_id,
            failure_reason=parsed.get("failure_reason", judge.reason),
            lesson=parsed.get("lesson", ""),
            next_strategy=parsed.get("next_strategy", ""),
        )
    except Exception:
        entry = ReflectionEntry(
            attempt_id=attempt_id,
            failure_reason=judge.reason,
            lesson="Could not parse reflection.",
            next_strategy="Re-read all context paragraphs carefully before answering.",
        )
    return entry, inp + out
