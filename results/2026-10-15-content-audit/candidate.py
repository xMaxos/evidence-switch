"""Classify every source on two separate model-judged relevance conditions."""
import json
from evidenceswitch.io import parse_json
from evidenceswitch.selection import PreparedSelection
from evidenceswitch.workflow import prepare

AUDIT = {
    'en': 'Inspect EVERY supplied record against the question. For each record judge TWO independent conditions: (1) does its entire entity string exactly equal the entity asked for, character for character? (2) does its field exactly equal the field asked for? Sharing a field is not an entity match; sharing an entity is not a field match. A record with either condition false is irrelevant, even if it has the highest revision. Judge older and newer records separately; do not choose the winning revision or answer yet. Return only {"checks":[["SOURCE_ID",true,false],...]}: exactly one row per supplied source_id, in supplied order, each containing its unchanged ID, entity-match boolean, field-match boolean. Use JSON booleans, never strings. The example shows structure only; determine each boolean from the actual record and question. No explanations or code fences.',
    'de': 'Prüfe JEDEN vorgegebenen Datensatz anhand der Frage. Beurteile pro Datensatz ZWEI unabhängige Bedingungen: (1) Ist sein vollständiger entity-String Zeichen für Zeichen identisch mit der erfragten Entität? (2) Ist sein field identisch mit dem erfragten Feld? Ein gleiches Feld bedeutet keine gleiche Entität; eine gleiche Entität bedeutet kein gleiches Feld. Sobald eine Bedingung falsch ist, ist der Datensatz irrelevant, auch mit der höchsten Revision. Prüfe ältere und neuere Datensätze getrennt; wähle noch keine maßgebliche Revision und beantworte die Frage noch nicht. Gib nur {"checks":[["SOURCE_ID",true,false],...]} aus: genau eine Zeile pro vorgegebener source_id in vorgegebener Reihenfolge, mit unveränderter ID, Boolean für Entity-Übereinstimmung, Boolean für Field-Übereinstimmung. Verwende JSON-Booleans, keine Strings. Das Beispiel zeigt nur die Struktur; bestimme jeden Boolean anhand des tatsächlichen Datensatzes und der Frage. Keine Erklärungen oder Codeblöcke.',
    'es': 'Comprueba CADA registro proporcionado contra la pregunta. Evalúa DOS condiciones independientes por registro: (1) ¿su cadena entity completa es exactamente igual a la entidad preguntada, carácter por carácter? (2) ¿su field es exactamente igual al campo preguntado? Compartir field no implica compartir entity; compartir entity no implica compartir field. Si cualquiera de las condiciones es falsa, el registro es irrelevante, incluso con la mayor revision. Evalúa por separado registros antiguos y nuevos; todavía no elijas la revisión ganadora ni respondas la pregunta. Devuelve solo {"checks":[["SOURCE_ID",true,false],...]}: exactamente una fila por source_id proporcionada, en el orden proporcionado, con ID sin cambios, booleano de coincidencia de entity y booleano de coincidencia de field. Usa booleanos JSON, nunca cadenas. El ejemplo muestra solo la estructura; determina cada booleano a partir del registro y la pregunta reales. Sin explicaciones ni bloques de código.',
}


class PreparedAudit:
    def __init__(self, prepared):
        self._prepared = prepared
        self._selection = PreparedSelection(prepared)

    @property
    def extraction_messages(self):
        return [{'role': 'system', 'content': AUDIT[self._prepared._language]},
                {'role': 'user', 'content': self._prepared._user}]

    def finish(self, response):
        if not isinstance(response, str):
            raise ValueError('Audit response must be JSON text')
        value = parse_json(response)
        if not isinstance(value, dict) or set(value) != {'checks'} or not isinstance(value['checks'], list):
            raise ValueError('Audit must contain exactly a checks array')
        checks = value['checks']
        expected = [r['source_id'] for r in self._prepared._records]
        if len(checks) != len(expected):
            raise ValueError('Audit must cover every supplied record')
        for row, source_id in zip(checks, expected):
            if (not isinstance(row, list) or len(row) != 3 or row[0] != source_id
                    or type(row[1]) is not bool or type(row[2]) is not bool):
                raise ValueError('Audit rows require ordered source IDs and two booleans')
        # Both judgments come from the model. Do not compare entity/field in code.
        selected = [row[0] for row in checks if row[1] and row[2]]
        return self._selection.finish(json.dumps({'source_ids': selected}))


def prepare_audit(messages):
    return PreparedAudit(prepare(messages))
