import numbers
import logging
from pathlib import Path
import numpy as np

import json

logger = logging.getLogger(__name__)


class Exporter:
    CONFIG_JSON = 'config.json'

    def __init__(self, save_dir):
        self.save_dir = Path(save_dir)

    def process_result(self, under_test, result):
        metric = under_test.metric
        utterance, system = result

        def extract_fields(score, fields):
            if not fields:
                if isinstance(score, numbers.Number):
                    return score
                return score.__dict__
            if isinstance(fields, str):
                return getattr(score, fields)
            fields = tuple(fields)
            return {name: getattr(score, name) for name in fields}

        utterance = [extract_fields(score, metric.utterance_field) for score in utterance]
        if system is None:
            system = np.mean(utterance)
        else:
            system = extract_fields(system, metric.system_field)
        return dict(
            utterance=utterance,
            system=system,
            metric=under_test.metric_name,
            model=under_test.model_name,
            dataset=under_test.dataset_name,
        )

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool):
            return bool(obj)
        else:
            raise TypeError

    def export_json(self, result, under_test):
        result = self.process_result(under_test.metric, result)
        prefix = under_test.prefix
        output_path = self.save_dir.joinpath(prefix).with_suffix('.json')

        logger.info('Saving scores to %s', output_path)
        with output_path.open('w') as f:
            json.dump(result, f, default=self.default)

    def export_config(self, config):
        config_json = self.save_dir.joinpath(self.CONFIG_JSON)
        config_json.write_text(json.dumps(config))
