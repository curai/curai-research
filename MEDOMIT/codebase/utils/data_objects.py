
import pprint
from collections import defaultdict, OrderedDict
from types import NoneType

import yaml


def format_for_spreadsheet(value):
    if type(value) is str or type(value) is int or type(value) is float:
        return value
    else:
        return yaml.dump(value, allow_unicode=True, default_flow_style=False)


class PatientCase(object):
    def __init__(self, args, enc):
        self.eoc_title_to_encounter_id = defaultdict(list)
        self.encounter_id_to_eoc_title = defaultdict(list)
        self.encounter_info = {}
        self.args = args
        self.matched_facts = {}
        self.omissions_found = {}
        self.omissions_found_baseline = {}
        self.ddx = {}
        self.ddx_dialogue = {}

        self.ddx_scores = {}
        self.diagnosis_similarity = defaultdict(OrderedDict)
        self.facts = {}
        self.facts_relevance = {}
        self.facts_relevance_output = {}
        self.fact_clusters = {}
        self.fact_subclusters = {}
        self.fact_scores = {}
        self.fact_score_details = {}

        self.metadata = {}
        self.prompts = {}
        self.metrics = {}
        self.parsed_headers = {}
        self.parsed_header_sizes = {}
        self.subjective_altered_facts = {}
        self.notes_df = None
        self.subjective_corruption = {}
        self.diagnosis = None

    def get_ddx_string(self, ddx_name, from_dialogue=True):
        if from_dialogue:
            return "\n".join(f"{x[1]}: {x[2]}" for x in self.ddx_dialogue[ddx_name])
        else:
            return "\n".join(f"{x[-2]} ({x[1]}): {x[2]}" for x in self.ddx[ddx_name])

    def get_fact_list_str(self):
        output_string = ""
        for fact_type, fact_list in self.facts.items():
            output_string += "\n".join([f"{fact_id}: {fact}" for fact_id, fact in fact_list])
        return output_string

    def get_fact_list(self):
        output_list = []
        fact_set = set()
        for fact_type, fact_list in self.facts.items():
            output_list.extend([(fact_id, fact) for fact_id, fact in fact_list])
        return output_list

    def to_dict(self):
        pp = pprint.PrettyPrinter(depth=4)

        this_dict = OrderedDict()

        earlier_fields = ["dialogue", "subjective_output","user_id", "mother_encounter_id"]
        this_dict.update(
            {k: self.__dict__[k] for k in earlier_fields if k not in this_dict.keys() and k in self.__dict__})


        for metric_name, metric_value in self.metrics.items():
            this_dict[f'metric_{metric_name}'] = metric_value
        for key, value in self.facts_relevance_output.items():
            this_dict[key] = value
        for ddx, clusters in self.fact_clusters.items():
            this_dict[f"fact_clusters_{ddx}"] = ""
            for cluster_block, cluster_list in clusters.items():
                this_dict[f"fact_clusters_{ddx}"] += f"\n\nCluster:{cluster_block}\n"
                if type(cluster_list) is dict:
                    this_dict[f"fact_clusters_{ddx}"] += "\n".join(f"{x}:{v}" for x, v in cluster_list.items())
                elif len(cluster_list) > 0:
                    this_dict[f"fact_clusters_{ddx}"] += cluster_list

        for cluster_config, clusters in self.fact_subclusters.items():
            this_dict[f"fact_subclusters_{cluster_config}"] = ""
            for cluster_block, cluster_dict in clusters.items():
                this_dict[f"fact_subclusters_{cluster_config}"] += f"\nCluster:{cluster_block}\n{format_for_spreadsheet(cluster_dict)}\n\n"

        for ddx_name, ddx_list in self.ddx_scores.items():
            this_dict[f"ddx_scores_{ddx_name}"] = format_for_spreadsheet(ddx_list)

        for k, v in self.__dict__.items():
            if k not in this_dict.keys() and k not in  earlier_fields and (v is not None or type(v) is not NoneType):
                this_dict[k] = format_for_spreadsheet(v)

        return this_dict

    def to_prompt_dict(self):
        included_fields = ["request_uuid", "mother_encounter_id", "user_id"]
        prompt_dict = OrderedDict()

        for field in self.__dict__["metadata"].keys():
            prompt_dict[field] = self.__dict__["metadata"][field]
        for prompt_name, prompt_text in self.prompts.items():
            prompt_dict[prompt_name] = prompt_text


        return prompt_dict


class ACIPatientCase(PatientCase):
    _fields_to_include = {"src": "dialogue",
                          "encounter_id": "mother_encounter_id",
                          "tgt": "subjective_output",
                          "user_id" : "user_id"
                          }

    def __init__(self, enc, args, **entries):
        super().__init__(args, enc)
        self.metadata = {**entries}

        for input_name, output_name in self._fields_to_include.items():
            self.__dict__[output_name] = self.metadata[input_name]

