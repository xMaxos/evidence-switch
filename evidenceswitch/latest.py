"""Select the unique highest numeric revision after the model's exact query."""
import json
from .io import parse_json
from .retrieval import PreparedRetrieval
from .workflow import prepare


class PreparedLatest(PreparedRetrieval):
    def finish(self, response):
        messages = super().finish(response)
        records = parse_json(messages[1]['content'][len(self._prepared._prefix):])
        if records:
            revision = max(record['revision'] for record in records)
            latest = [record for record in records if record['revision'] == revision]
            if len(latest) != 1:
                raise ValueError('Highest matching revision must be unique')
            messages[1]['content'] = self._prepared._prefix + json.dumps(latest, ensure_ascii=False, indent=2)
        return messages


def prepare_latest(messages):
    return PreparedLatest(prepare(messages))
