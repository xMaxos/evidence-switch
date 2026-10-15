"""Experimental two-stage variant: ask the model for source IDs, not copies.

This module never computes relevance, ranks revisions, or reads expected answers.
The model chooses the IDs and the second model call still chooses the answer.
"""

import json

from evidenceswitch.io import parse_json
from evidenceswitch.workflow import prepare


EXTRACT_IDS = {
    "en": "Select ALL supplied records whose entity AND field exactly match the question. Keep both newer and older matching records. Do not answer the question or choose a winning revision yet. Return only JSON of the form {\"source_ids\":[...]} listing the source_id strings of ALL matching records, exactly as supplied. Do not repeat IDs or invent IDs. If none match, return {\"source_ids\":[]}. Do not include explanations or code fences.",
    "de": "Wähle ALLE vorgegebenen Datensätze aus, deren entity UND field exakt zur Frage passen. Behalte sowohl neuere als auch ältere passende Datensätze. Beantworte die Frage noch nicht und wähle noch keine maßgebliche Revision. Gib nur JSON der Form {\"source_ids\":[...]} mit den source_id-Strings ALLER passenden Datensätze aus, exakt wie vorgegeben. Wiederhole keine IDs und erfinde keine IDs. Falls keiner passt, gib {\"source_ids\":[]} aus. Keine Erklärungen oder Codeblöcke.",
    "es": "Selecciona TODOS los registros proporcionados cuyos entity Y field coincidan exactamente con la pregunta. Conserva tanto los registros coincidentes nuevos como los antiguos. Todavía no respondas la pregunta ni elijas la revisión ganadora. Devuelve solo JSON con la forma {\"source_ids\":[...]} que enumere las cadenas source_id de TODOS los registros coincidentes, exactamente como se proporcionaron. No repitas ni inventes IDs. Si no hay ninguno, devuelve {\"source_ids\":[]}. Sin explicaciones ni bloques de código.",
}


class PreparedIdsPrompt:
    def __init__(self, prepared):
        self._prepared = prepared
        # prepare() already validated the exact question/separator and records.
        # Parse only its known record suffix, never search for arbitrary brackets.
        records = parse_json(prepared._user[len(prepared._prefix):])
        self._by_id = {record["source_id"]: record for record in records}

    @property
    def extraction_messages(self):
        return [
            {"role": "system", "content": EXTRACT_IDS[self._prepared._language]},
            {"role": "user", "content": self._prepared._user},
        ]

    def finish(self, extraction_response):
        if not isinstance(extraction_response, str):
            raise ValueError("ID extraction response must be JSON text")
        extraction = parse_json(extraction_response)
        if not isinstance(extraction, dict) or set(extraction) != {"source_ids"}:
            raise ValueError("ID extraction must contain exactly source_ids")
        selected = extraction["source_ids"]
        if not isinstance(selected, list) or any(not isinstance(item, str) for item in selected):
            raise ValueError("source_ids must be an array of strings")
        if len(set(selected)) != len(selected):
            raise ValueError("Duplicate source_id in extraction")
        if any(source_id not in self._by_id for source_id in selected):
            raise ValueError("Extracted source_id does not exist in supplied records")
        # Preserve the model's order. Membership validation is not a relevance
        # check: unchanged, irrelevant records remain permissible.
        records = [self._by_id[source_id] for source_id in selected]
        return self._prepared.finish(json.dumps({"records": records}, ensure_ascii=False))


def prepare_ids(messages):
    """Validate an EvidenceSwitch prompt and prepare model-selected ID routing."""
    return PreparedIdsPrompt(prepare(messages))
