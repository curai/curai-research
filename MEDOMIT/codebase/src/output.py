
import os
import pickle
import xlsxwriter
import pandas as pd
from scipy.stats import describe


def output_results(args, case_list, log):
    dictionary_list = sorted([h.to_dict() for h in case_list], key=lambda x: len(x), reverse=True)
    output_df = pd.DataFrame.from_dict(dictionary_list)

    metric_pd = print_metrics(case_list, log, output_df)

    prompt_dictionary_list = sorted([h.to_prompt_dict() for h in case_list], key=lambda x: len(x), reverse=True)

    prompt_df = pd.DataFrame.from_dict(prompt_dictionary_list)
    output_filename = os.path.join(args.output_path, f"{args.timestamp}.xlsx")
    with pd.ExcelWriter(output_filename, engine="xlsxwriter") as writer, \
            open(os.path.join(args.output_path, f"{args.timestamp}.pkl"), 'wb') as pickle_writer:
        pickle.dump(case_list, pickle_writer)
        output_df.to_excel(writer, index=False, sheet_name="results")
        worksheet = writer.sheets["results"]
        worksheet.freeze_panes(1, 0)
        worksheet.set_default_row(500)
        worksheet.set_row(row=0, height=20)
        worksheet.set_column(0, 6, 50)

        prompt_df.to_excel(writer, index=False, sheet_name="prompts")
        worksheet = writer.sheets["prompts"]
        worksheet.freeze_panes(1, 0)
        worksheet.set_default_row(20)

        metric_pd.to_excel(writer, index=False, sheet_name="metrics")

    log.info(f"Output written to {output_filename}")


def print_metrics(case_list, log, output_df):
    metric_rows = []
    for metric_name in case_list[0].metrics:
        metric_col = output_df[f"metric_{metric_name}"]
        mrow = {"metric_name": metric_name}
        mrow.update(describe(metric_col)._asdict())
        metric_rows.append(mrow)
    metric_pd = pd.DataFrame.from_dict(metric_rows)
    log.info(metric_pd.to_csv())
    return metric_pd