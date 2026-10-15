"""Two model passes, with validation of copied evidence between them."""

from dataclasses import dataclass
import json
import re

from .generate import _FIELDS, _LANGUAGES
from .io import parse_json


# These are the instructions frozen for the published 2026-10-15 experiment.
EXTRACT = {
    "en": "Select ALL supplied records whose entity AND field exactly match the question. Keep both newer and older matching records. Do not answer the question or choose a winning revision yet. Copy matching records exactly, without inventing or editing any field. Return only JSON of the form {\"records\":[...]} containing the complete matching record objects. If none match, return {\"records\":[]}. Do not include explanations or code fences.",
    "de": "Wähle ALLE vorgegebenen Datensätze aus, deren entity UND field exakt zur Frage passen. Behalte sowohl neuere als auch ältere passende Datensätze. Beantworte die Frage noch nicht und wähle noch keine maßgebliche Revision. Kopiere passende Datensätze exakt, ohne Felder zu erfinden oder zu verändern. Gib nur JSON der Form {\"records\":[...]} mit den vollständigen passenden Datensatzobjekten aus. Falls keiner passt, gib {\"records\":[]} aus. Keine Erklärungen oder Codeblöcke.",
    "es": "Selecciona TODOS los registros proporcionados cuyos entity Y field coincidan exactamente con la pregunta. Conserva tanto los registros coincidentes nuevos como los antiguos. Todavía no respondas la pregunta ni elijas la revisión ganadora. Copia los registros coincidentes exactamente, sin inventar ni modificar campos. Devuelve solo JSON con la forma {\"records\":[...]} que contenga los objetos completos de los registros coincidentes. Si no hay ninguno, devuelve {\"records\":[]}. Sin explicaciones ni bloques de código.",
}
_RECORD_FIELDS = {"source_id", "entity", "field", "revision", "value"}


def _validate_records(records):
    if not isinstance(records, list):
        raise ValueError("Records must be a JSON array")
    seen = set()
    for record in records:
        if not isinstance(record, dict) or set(record) != _RECORD_FIELDS:
            raise ValueError("Each record must have exactly source_id, entity, field, revision and value")
        if any(not isinstance(record[key], str) or not record[key]
               for key in _RECORD_FIELDS - {"revision"}):
            raise ValueError("Record text fields must be nonempty strings")
        if type(record["revision"]) is not int or record["revision"] < 1:
            raise ValueError("Record revision must be a positive integer")
        if record["source_id"] in seen:
            raise ValueError("Duplicate source_id in records")
        seen.add(record["source_id"])


@dataclass(frozen=True)
class PreparedPrompt:
    """A validated EvidenceSwitch prompt; no answer or relevance computation."""

    _language: str
    _system: str
    _user: str
    _prefix: str
    _records: list

    @property
    def extraction_messages(self):
        return [
            {"role": "system", "content": EXTRACT[self._language]},
            {"role": "user", "content": self._user},
        ]

    def finish(self, extraction_response):
        """Return the answering prompt, or reject altered/invented evidence.

        Irrelevant but unchanged records are allowed. Finding relevance and
        choosing the winning revision remain the model's responsibility.
        """
        if not isinstance(extraction_response, str):
            raise ValueError("Extraction response must be JSON text")
        extraction = parse_json(extraction_response)
        if not isinstance(extraction, dict) or set(extraction) != {"records"}:
            raise ValueError("Extraction must contain exactly records")
        selected = extraction["records"]
        _validate_records(selected)
        if any(record not in self._records for record in selected):
            raise ValueError("Extracted record does not exactly match a supplied record")
        return [
            {"role": "system", "content": self._system},
            {"role": "user", "content": self._prefix
             + json.dumps(selected, ensure_ascii=False, indent=2)},
        ]


def prepare(messages):
    """Validate the generated en/de/es format before making any model calls."""
    if (not isinstance(messages, list) or len(messages) != 2
            or any(not isinstance(message, dict) or set(message) != {"role", "content"}
                   or not isinstance(message["content"], str) for message in messages)
            or [message["role"] for message in messages] != ["system", "user"]):
        raise ValueError("Two-stage mode requires an EvidenceSwitch system and user message")
    system, content = (message["content"] for message in messages)
    language = next((code for code, text in _LANGUAGES.items()
                     if text["system"] == system), None)
    if language is None:
        raise ValueError("Unsupported EvidenceSwitch system template")
    text = _LANGUAGES[language]
    question, separator, raw_records = content.partition("\n\n" + text["records"] + ":\n")
    pattern = re.escape(text["question"])
    pattern = pattern.replace(re.escape("{field}"), "(?:" + "|".join(_FIELDS) + ")")
    pattern = pattern.replace(re.escape("{entity}"), r"Nuvora-[0-9a-f]{12}")
    if not separator or re.fullmatch(pattern, question) is None:
        raise ValueError("Unsupported EvidenceSwitch question template")
    records = parse_json(raw_records)
    _validate_records(records)
    return PreparedPrompt(language, system, content, question + separator, records)
