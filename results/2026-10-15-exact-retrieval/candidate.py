"""Model-generated query, exact record retrieval, then model-generated answer."""
import json
from evidenceswitch.io import parse_json
from evidenceswitch.selection import PreparedSelection
from evidenceswitch.workflow import prepare

QUERY = {
    'en': 'Extract the entity and field explicitly requested in the question. Copy their complete strings exactly. Return only a JSON object with exactly two keys: {"entity":"ENTITY_FROM_QUESTION","field":"FIELD_FROM_QUESTION"}. Replace the placeholders with the actual strings. Do not answer the question or invent a value. No explanations or code fences.',
    'de': 'Extrahiere die in der Frage ausdrücklich gesuchte Entität und das Feld. Kopiere ihre vollständigen Strings exakt. Gib nur ein JSON-Objekt mit genau zwei Schlüsseln aus: {"entity":"ENTITAET_AUS_DER_FRAGE","field":"FELD_AUS_DER_FRAGE"}. Ersetze die Platzhalter durch die tatsächlichen Strings. Beantworte die Frage nicht und erfinde keinen Wert. Keine Erklärungen oder Codeblöcke.',
    'es': 'Extrae la entidad y el campo solicitados explícitamente en la pregunta. Copia sus cadenas completas exactamente. Devuelve solo un objeto JSON con exactamente dos claves: {"entity":"ENTIDAD_DE_LA_PREGUNTA","field":"CAMPO_DE_LA_PREGUNTA"}. Sustituye los ejemplos por las cadenas reales. No respondas la pregunta ni inventes un valor. Sin explicaciones ni bloques de código.',
}


class PreparedRetrieval:
    def __init__(self, prepared):
        self._prepared = prepared
        self._selection = PreparedSelection(prepared)

    @property
    def extraction_messages(self):
        return [{'role': 'system', 'content': QUERY[self._prepared._language]},
                {'role': 'user', 'content': self._prepared._prefix.split('\n\n', 1)[0]}]

    def finish(self, response):
        if not isinstance(response, str):
            raise ValueError('Retrieval query must be JSON text')
        query = parse_json(response)
        if (not isinstance(query, dict) or set(query) != {'entity', 'field'}
                or any(not isinstance(value, str) or not value for value in query.values())):
            raise ValueError('Retrieval query requires exactly nonempty entity and field strings')
        # Execute the model's query, without consulting the expected answer.
        # Retain source order and all revisions; the model must still answer.
        selected = [r['source_id'] for r in self._prepared._records
                    if r['entity'] == query['entity'] and r['field'] == query['field']]
        return self._selection.finish(json.dumps({'source_ids': selected}))


def prepare_retrieval(messages):
    return PreparedRetrieval(prepare(messages))
