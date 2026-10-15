"""Show only relevance fields during selection; restore full records to answer."""
import json
from evidenceswitch.selection import PreparedSelection
from evidenceswitch.workflow import prepare


class PreparedFocus(PreparedSelection):
    @property
    def extraction_messages(self):
        messages = super().extraction_messages
        records = [{key: record[key] for key in ('source_id', 'entity', 'field')}
                   for record in self._prepared._records]
        messages[1]['content'] = self._prepared._prefix + json.dumps(records, ensure_ascii=False, indent=2)
        return messages


def prepare_focus(messages):
    return PreparedFocus(prepare(messages))
