"""Experimental ID selection plus an explicit final-answer schema reminder."""

from candidate import prepare_ids


ANSWER_SCHEMA = {
    "en": 'Output format reminder: return exactly two keys, "answer" and "citations". "citations" must be an array of source_id strings, never an array of objects. Illustrative format only (replace both placeholder strings with the actual supplied value and source_id): {"answer":"ACTUAL_VALUE","citations":["ACTUAL_SOURCE_ID"]}. If no matching record exists, return {"answer":null,"citations":[]}. Return JSON only.',
    "de": 'Erinnerung zum Ausgabeformat: Gib genau zwei Schlüssel zurück, "answer" und "citations". "citations" muss ein Array von source_id-Strings sein, niemals ein Array von Objekten. Nur ein Formatbeispiel (ersetze beide Platzhalterstrings durch den tatsächlichen vorgegebenen value und die source_id): {"answer":"TATSAECHLICHER_VALUE","citations":["TATSAECHLICHE_SOURCE_ID"]}. Fehlt ein passender Datensatz, gib {"answer":null,"citations":[]} zurück. Gib nur JSON aus.',
    "es": 'Recordatorio del formato: devuelve exactamente dos claves, "answer" y "citations". "citations" debe ser un array de cadenas source_id, nunca un array de objetos. Solo es un ejemplo de formato (sustituye ambas cadenas de ejemplo por el value y el source_id reales proporcionados): {"answer":"VALUE_REAL","citations":["SOURCE_ID_REAL"]}. Si no existe ningún registro coincidente, devuelve {"answer":null,"citations":[]}. Devuelve solo JSON.',
}


class PreparedSchemaIdsPrompt:
    def __init__(self, prepared):
        self._prepared = prepared

    @property
    def extraction_messages(self):
        return self._prepared.extraction_messages

    def finish(self, extraction_response):
        messages = self._prepared.finish(extraction_response)
        language = self._prepared._prepared._language
        messages[0]["content"] += "\n\n" + ANSWER_SCHEMA[language]
        return messages


def prepare_schema_ids(messages):
    return PreparedSchemaIdsPrompt(prepare_ids(messages))
