"""
Created by elliotschumacher, Curai.
12/14/23
"""
import pandas as pd
def binary_eval(label, prediction):
    if label == 'n' and prediction == 'n':
      return True
    # If both the label and prediction are strings representing numbers, it's a correct prediction
    elif label.strip().isdigit() and prediction.strip().isdigit():
      return True
    return False
def main():
    #expected format: list of (gold standard, prediction) files
    filepaths = [
        ("output/data/f_id_test.csv","output/data/f_id_test.csv"),
    ]

    for gold_standard_filepath, prediction_filename in filepaths:
        gold_pd = pd.read_csv(gold_standard_filepath)
        print(gold_standard_filepath)
        #TODO: change to file format
        prediction_pd = pd.read_csv(prediction_filename)
        output_row_list = []
        for ig, gold_row in gold_pd.iterrows():
            #NOTE: I'm assuming ith row in gold == ith row in prediction
            prediction_row = prediction_pd.iloc[ig].to_dict()

            #NOTE: I'm assuming the inference generation is output
            prediction_label = prediction_row["output"].strip().lower()

            #NOTE: I think completion is the right field here?
            gold_label = gold_row["completion"].strip().lower()

            output_row = gold_row.to_dict()
            output_row["prediction"] = prediction_label
            output_row["is_correct_binary"] = binary_eval(gold_label, prediction_label)
            output_row["is_correct_multi"] = gold_label.strip().lower() == prediction_label.strip().lower()

            output_row_list.append(output_row)
        output_pd = pd.DataFrame.from_dict(output_row_list)
        output_filename = gold_standard_filepath.replace(".csv", ".eval.csv")
        output_pd.to_csv(output_filename, index=False)
        for metric in ["is_correct_binary", "is_correct_multi"]:
            num_correct = len(output_pd[output_pd[metric] == True])
            accuracy = num_correct / len(output_pd)
            print(f"{metric}, acc: {accuracy}, n: {num_correct}")


if __name__ == "__main__":
    main()
