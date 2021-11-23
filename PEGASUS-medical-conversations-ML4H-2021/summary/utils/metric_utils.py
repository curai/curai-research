from metrics.metrics_report import MetricsReport
from summary.utils import read_json, subset_dict, write_json

def get_common_results_among_groups(group_names):
    all_datas, all_ids = [], []
    for group_name in group_names:
        data = read_json(f"results/pegasus/{group_name}.json")
        all_datas.append(data)

        curr_ids = []
        for snippet_id, data_pt in data.items():
            if "predicted_summary" in data_pt:
               curr_ids.append(snippet_id) 
        all_ids.append(set(curr_ids))

    common_ids = all_ids[0]
    for curr_ids in all_ids:
        common_ids = common_ids.intersection(curr_ids)
    common_ids = list(common_ids)
    print(f"{len(common_ids)} in evaluation set")

    res_datas = []
    for curr_data in all_datas:
        res_datas.append(subset_dict(curr_data, common_ids))
    return res_datas

def compute_metrics_for_all_groups(all_results, group_names, tag=""):
    auto_metrics = {}
    gold_metrics = {}
    for group, results in zip(group_names, all_results):
        gold_summaries, pred_summaries = [], []
        for _, data_pt in results.items():
            gold_summary = data_pt["summary"]
            pred_summary = data_pt["predicted_summary"]

            gold_summaries.append(gold_summary)
            pred_summaries.append(pred_summary)
        report = MetricsReport(gold_summaries, pred_summaries)
        gold_report = MetricsReport(gold_summaries, gold_summaries)
        auto_metrics[group] = report.to_json(tag=tag)
        gold_metrics[group] = gold_report.to_json(tag=f"{tag}.gold") if tag is not "" else gold_report.to_json(tag="gold")
    return auto_metrics, gold_metrics