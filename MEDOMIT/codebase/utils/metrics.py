
import logging


def calculate_metrics(case_list, args, default_llm):
    log = logging.getLogger()
    def percentage_omissions_identified(case):
        for omission_and_subjective_names in case.omissions_found_baseline.keys():

            case.metrics[f"number of omissions"] = len(case.omissions_found_baseline[omission_and_subjective_names])

        return case

    def score_facts(case):
        category_to_score_mapping = {"Critical": 1,
                                     "Important": 0.5,
                                     "Other": 0.1}
        relevance_lambda = 0.5
        for fact_config, fact_list in case.omissions_found_baseline.items():
            case.fact_scores[fact_config] = {}
            case.fact_score_details[fact_config] = {}
            for fact_id, (fact, explanation) in fact_list.items():

                if fact_id in case.facts_relevance["three_groups"]:
                    relevance = case.facts_relevance["three_groups"][fact_id]
                    fact_relevance_score = category_to_score_mapping[relevance["category"]]
                else:
                    fact_relevance_score = 0
                    log.info(f"Missing fact {fact_id}")

                cluster_inverse_counts = {}
                for cluster_type in case.fact_clusters:
                    cluster_count = 0
                    for diagnosis, cluster_dict in case.fact_clusters[cluster_type].items():
                        if fact_id in cluster_dict:
                            cluster_count += 1
                    if cluster_count == 0:
                        cluster_inverse_counts[cluster_type] = 0
                    else:
                        cluster_inverse_counts[cluster_type] = 1 / cluster_count
                subcluster_inverse_counts = {}
                for subcluster_chain in case.fact_subclusters:
                    for diagnosis, subcluster_dict in case.fact_subclusters[subcluster_chain].items():
                        for type_cluster in subcluster_dict:
                            if type(subcluster_dict[type_cluster]) is dict:
                                for etiology, etiology_dict in subcluster_dict[type_cluster].items():
                                    if fact_id in etiology_dict:
                                        num_facts_included = len(etiology_dict)
                                        subcluster_inverse_counts[f"{diagnosis}_{etiology}"] = 1 / num_facts_included
                                        #log.info(f"{diagnosis} | {etiology} | {fact_id} | {subcluster_inverse_counts}")

                # also, arguably mean?
                fact_cluster_importance = max(
                    list(cluster_inverse_counts.values()) + list(subcluster_inverse_counts.values()))
                case.fact_score_details[fact_config][fact_id] = {"relevance": fact_relevance_score,
                                                                 "cluster": cluster_inverse_counts,
                                                                 "subcluster": subcluster_inverse_counts}
                case.fact_scores[fact_config][fact_id] = max(fact_relevance_score, fact_cluster_importance)
            case.metrics[f"combined_fact_score_{fact_config}"] = sum(case.fact_scores[fact_config].values())

        return case

    fn_list = [score_facts, percentage_omissions_identified]  # ddx_similarity_diff_fn
    for fn in fn_list:
        for i in range(len(case_list)):
            case_list[i] = fn(case_list[i])
    return case_list
