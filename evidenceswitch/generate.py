"""Deterministic, original synthetic evidence-switching fixtures."""

import copy
import hashlib
import json
import random


_LANGUAGES = {
    "en": {
        "system": (
            "Answer the question using only the supplied fictional records. "
            "Match both entity and field exactly. If matching records conflict, "
            "use the highest integer revision, comparing revisions numerically. "
            "Return only a JSON object with exactly answer and citations. "
            "Copy the winning value exactly as the answer string, and cite only "
            "its source_id in the citations array. If no matching record exists, "
            "return {\"answer\":null,\"citations\":[]}."
        ),
        "question": "What is the field {field} for entity {entity}?",
        "records": "Records",
    },
    "de": {
        "system": (
            "Beantworte die Frage ausschließlich mit den vorgegebenen fiktiven "
            "Datensätzen. Entity und field müssen beide exakt übereinstimmen. "
            "Bei widersprüchlichen passenden Datensätzen gilt die höchste "
            "ganzzahlige revision; vergleiche die Zahlen numerisch. Gib nur ein "
            "JSON-Objekt mit genau answer und citations zurück. Übernimm den "
            "maßgeblichen value unverändert als Antwortstring und nenne nur "
            "dessen source_id im Array citations. Fehlt ein passender Datensatz, "
            "gib {\"answer\":null,\"citations\":[]} zurück."
        ),
        "question": "Welchen Wert hat das Feld {field} für die Entität {entity}?",
        "records": "Datensätze",
    },
    "es": {
        "system": (
            "Responde usando únicamente los registros ficticios proporcionados. "
            "Tanto entity como field deben coincidir exactamente. Si hay "
            "registros coincidentes contradictorios, usa la mayor revision "
            "entera, comparando los números numéricamente. Devuelve solo un "
            "objeto JSON con exactamente answer y citations. Copia el value "
            "ganador sin cambios como cadena de respuesta y cita únicamente "
            "su source_id en el array citations. Si no hay ningún registro "
            "coincidente, devuelve {\"answer\":null,\"citations\":[]}."
        ),
        "question": "¿Qué valor tiene el campo {field} para la entidad {entity}?",
        "records": "Registros",
    },
}
_FIELDS = ("signal_code", "route_token", "archive_label", "beacon_key")


def generate_cases(seed=42, groups=10, languages=("en", "de", "es")):
    """Return two cases per task/language/group, in deterministic shuffled order.

    ``groups`` is a positive number of pairs for each task and language.
    Only a fresh local RNG is used; seeds must be integers and languages must
    be a nonempty iterable of distinct supported language codes.
    """
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if isinstance(groups, bool) or not isinstance(groups, int) or groups < 1:
        raise ValueError("groups must be a positive integer")
    if isinstance(languages, str):
        raise ValueError("languages must be an iterable of language codes")
    try:
        languages = tuple(languages)
    except TypeError as exc:
        raise ValueError("languages must be an iterable of language codes") from exc
    if not languages or any(not isinstance(code, str) or code not in _LANGUAGES
                            for code in languages):
        raise ValueError("languages must contain only en, de, or es")
    if len(set(languages)) != len(languages):
        raise ValueError("languages must not contain duplicates")

    rng = random.Random(seed)
    used_ids = set()

    def new_id(context=""):
        while True:
            identifier = f"{rng.getrandbits(48):012x}"
            if context:
                identifier = hashlib.sha256(
                    (context + "\0" + identifier).encode("utf-8")
                ).hexdigest()[:12]
            if identifier not in used_ids:
                used_ids.add(identifier)
                return identifier

    def revisions():
        # Every combination uses this same distribution, including distractors.
        if rng.randrange(2):
            boundary = rng.choice((10, 100, 1000, 10000))
            return rng.randint(boundary - 7, boundary - 1), rng.randint(boundary, boundary + 7)
        older = rng.randint(1, 10000)
        return older, older + rng.randint(1, 1000)

    def record(entity, field, revision, value):
        return {
            "source_id": new_id(), "entity": entity, "field": field,
            "revision": revision, "value": value,
        }

    def case(pair_id, task, language, variant, entity, field, records):
        # Source order is unrelated to revision, relevance, and the gold label.
        rng.shuffle(records)
        matches = [r for r in records if r["entity"] == entity and r["field"] == field]
        winner = max(matches, key=lambda r: r["revision"]) if matches else None
        language_text = _LANGUAGES[language]
        messages = [
            {"role": "system", "content": language_text["system"]},
            {"role": "user", "content": (
                language_text["question"].format(field=field, entity=entity)
                + "\n\n" + language_text["records"] + ":\n"
                + json.dumps(records, ensure_ascii=False, indent=2)
            )},
        ]
        identity = json.dumps({"pair_id": pair_id, "messages": messages},
                              ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return {
            "id": new_id(identity), "pair_id": pair_id, "task": task,
            "language": language, "variant": variant, "messages": messages,
            "expected": {
                "answer": winner["value"] if winner else None,
                "citations": [winner["source_id"]] if winner else [],
            },
        }

    cases = []
    for language in languages:
        for _ in range(groups):
            for task in ("update", "missing", "distractor"):
                pair_id = new_id("pair:" + language)
                # Random synthetic names and exact-copy values need no world knowledge.
                entities = ["Nuvora-" + new_id() for _ in range(2)]
                fields = rng.sample(_FIELDS, 2)
                # Balanced alternatives prevent repetition/frequency identifying the target.
                before = []
                for candidate_entity in entities:
                    for candidate_field in fields:
                        for revision in revisions():
                            before.append(record(candidate_entity, candidate_field,
                                                 revision, "V-" + new_id()))
                entity, field = rng.choice(entities), rng.choice(fields)
                after = copy.deepcopy(before)
                if task == "missing":
                    # Replace both target revisions; maintain count and paired alternatives.
                    replacement_entity = "Nuvora-" + new_id()
                    after = [record(replacement_entity, field, r["revision"], "V-" + new_id())
                             if r["entity"] == entity and r["field"] == field else r
                             for r in after]
                else:
                    changed_field = field if task == "update" else next(f for f in fields if f != field)
                    changed = max((r for r in after if r["entity"] == entity
                                   and r["field"] == changed_field), key=lambda r: r["revision"])
                    changed["value"] = "V-" + new_id()
                    changed["revision"] += rng.randint(1, 20)
                for variant, records in (("before", before), ("after", after)):
                    cases.append(case(pair_id, task, language, variant, entity, field, records))
    rng.shuffle(cases)
    return cases
