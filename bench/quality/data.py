"""Dataset loaders, normalised to one schema.

Both benchmarks reduce to the same shape: a question asked on a date, a
haystack of dated sessions made of turns, and labels saying which sessions and
which turns contain the evidence. Everything downstream (chunking, retrieval,
metrics) works on this schema and never sees dataset-specific quirks.

The quirks live here, and two are worth flagging because they fail silently:

  * LongMemEval's `has_answer` field is the *string* ``'True'``, not a boolean.
    ``if turn["has_answer"]`` is therefore true for ``'False'`` as well --
    every turn of every evidence session would count as evidence and
    turn-level recall would be quietly inflated.
  * LoCoMo category 5 is adversarial: the question has no evidence in the
    conversation. Those must be excluded from recall means (there is nothing
    to recall), the same way LongMemEval's ``_abs`` instances are.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class Turn:
    role: str
    content: str
    has_answer: bool = False
    dia_id: str | None = None  # LoCoMo turn label, e.g. "D3:12"


@dataclass
class Session:
    id: str
    dt: datetime | None
    turns: list[Turn]


@dataclass
class Instance:
    id: str
    qtype: str
    question: str
    q_dt: datetime | None
    answer: str
    sessions: list[Session]
    evidence_session_ids: set[str]
    evidence_turn_keys: set[tuple[str, int]]  # (session_id, turn index)
    abstention: bool
    # Instances sharing a haystack (all LoCoMo QAs over one conversation)
    # share chunks, embeddings and BM25 state under this key.
    haystack_key: str = ""


def _truthy(v) -> bool:
    """LongMemEval stores has_answer as the string 'True'/'False'."""
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() == "true"


_LME_DATE = re.compile(r"(\d{4})/(\d{1,2})/(\d{1,2}).*?(\d{1,2}):(\d{2})")


def _parse_lme_date(s: str | None) -> datetime | None:
    """'2023/04/10 (Mon) 17:50' -- the weekday token is decoration, skip it."""
    if not s:
        return None
    m = _LME_DATE.search(s)
    if not m:
        return None
    y, mo, d, h, mi = (int(g) for g in m.groups())
    try:
        return datetime(y, mo, d, h, mi)
    except ValueError:
        return None


def load_longmemeval(path: Path | str) -> list[Instance]:
    raw = json.loads(Path(path).read_text())
    out: list[Instance] = []
    for inst in raw:
        qid = str(inst["question_id"])
        evidence = set(inst.get("answer_session_ids") or [])
        sessions: list[Session] = []
        turn_keys: set[tuple[str, int]] = set()

        for sid, date_s, turns in zip(
            inst["haystack_session_ids"],
            inst["haystack_dates"],
            inst["haystack_sessions"],
        ):
            parsed = [
                Turn(
                    role=str(t.get("role", "user")),
                    content=str(t.get("content", "")),
                    has_answer=_truthy(t.get("has_answer", False)),
                )
                for t in turns
            ]
            sessions.append(Session(id=str(sid), dt=_parse_lme_date(date_s), turns=parsed))
            if sid in evidence:
                turn_keys.update(
                    (str(sid), i) for i, t in enumerate(parsed) if t.has_answer
                )

        out.append(
            Instance(
                id=qid,
                qtype=str(inst["question_type"]),
                question=str(inst["question"]),
                q_dt=_parse_lme_date(inst.get("question_date")),
                answer=str(inst.get("answer", "")),
                sessions=sessions,
                evidence_session_ids=evidence,
                evidence_turn_keys=turn_keys,
                # _abs questions are answerable-looking but the evidence was
                # deliberately withheld from the haystack.
                abstention=qid.endswith("_abs") or not evidence,
                haystack_key=qid,  # every LongMemEval question has its own haystack
            )
        )
    return out


def _parse_locomo_date(s: str | None) -> datetime | None:
    """'1:56 pm on 8 May, 2023'"""
    if not s:
        return None
    try:
        return datetime.strptime(s.strip(), "%I:%M %p on %d %B, %Y")
    except ValueError:
        return None


_SESS_KEY = re.compile(r"^session_(\d+)$")


def load_locomo(path: Path | str) -> list[Instance]:
    raw = json.loads(Path(path).read_text())
    out: list[Instance] = []
    for sample in raw:
        sample_id = str(sample["sample_id"])
        conv = sample["conversation"]

        sess_nums = sorted(
            int(m.group(1)) for k in conv if (m := _SESS_KEY.match(k))
        )
        sessions: list[Session] = []
        dia_index: dict[str, tuple[str, int]] = {}  # "D3:12" -> (session_id, turn idx)
        for n in sess_nums:
            sid = f"session_{n}"
            turns = []
            for i, t in enumerate(conv[sid]):
                dia = str(t.get("dia_id", ""))
                turns.append(
                    Turn(
                        # LoCoMo speakers are named people, not user/assistant.
                        # The name is kept as the role so chunk text reads
                        # "Caroline: ..." and lexical retrieval can use it.
                        role=str(t.get("speaker", "speaker")),
                        content=str(t.get("text", "")),
                        dia_id=dia or None,
                    )
                )
                if dia:
                    dia_index[dia] = (sid, i)
            sessions.append(
                Session(id=sid, dt=_parse_locomo_date(conv.get(f"{sid}_date_time")), turns=turns)
            )

        for j, qa in enumerate(sample.get("qa", [])):
            cat = qa.get("category")
            evidence_dias = [str(e) for e in (qa.get("evidence") or [])]
            turn_keys = {dia_index[d] for d in evidence_dias if d in dia_index}
            ev_sessions = {sk for sk, _ in turn_keys}
            # Category 5 is adversarial -- the evidence does not exist. Some
            # evidence labels also point at dia_ids missing from the transcript;
            # treat those as unanswerable rather than as recall failures.
            abstention = cat == 5 or not turn_keys

            # Evidence flags live on the QA, not the transcript, so has_answer
            # is stamped per-instance onto shared Session objects. Chunking
            # must therefore not rely on has_answer (it uses evidence_turn_keys).
            out.append(
                Instance(
                    id=f"{sample_id}_q{j}",
                    qtype=f"locomo-cat{cat}",
                    question=str(qa.get("question", "")),
                    q_dt=None,  # LoCoMo QAs carry no question date
                    answer=str(qa.get("answer", "")),
                    sessions=sessions,
                    evidence_session_ids=ev_sessions,
                    evidence_turn_keys=turn_keys,
                    abstention=abstention,
                    haystack_key=sample_id,  # all QAs of a sample share one conversation
                )
            )
    return out


def load(name: str, data_dir: Path | str = "data/quality") -> list[Instance]:
    data_dir = Path(data_dir)
    files = {
        "longmemeval_oracle": ("longmemeval_oracle.json", load_longmemeval),
        "longmemeval_s": ("longmemeval_s.json", load_longmemeval),
        "locomo": ("locomo10.json", load_locomo),
    }
    if name not in files:
        raise ValueError(f"unknown dataset {name!r}; options: {sorted(files)}")
    fname, loader = files[name]
    path = data_dir / fname
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- see scripts/fetch_quality_data.py"
        )
    return loader(path)
