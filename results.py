from common import Common
import numpy as np


def trace_evaluation(output_file, correct_predictions, total_predictions, elapsed):
    accuracy_message = "Accuracy: {0}".format(str(correct_predictions / total_predictions))
    throughput_message = "Prediction throughput: %d" % int(total_predictions / (elapsed if elapsed > 0 else 1))
    output_file.write(accuracy_message + '\n')
    output_file.write(throughput_message)
    print(accuracy_message)
    print(throughput_message)


def calculate_results(true_positive, false_positive, false_negative):
    if true_positive + false_positive > 0:
        precision = true_positive / (true_positive + false_positive)
    else:
        precision = 0
    if true_positive + false_negative > 0:
        recall = true_positive / (true_positive + false_negative)
    else:
        recall = 0
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0
    return precision, recall, f1


def update_correct_predictions(beam_width, num_correct_predictions, output_file, results):
    for original_name, predicted in results:
        original_name_parts = original_name.split(Common.internal_delimiter)  # list
        filtered_original = Common.filter_impossible_names(original_name_parts)  # list
        predicted_first = predicted
        if beam_width > 0:
            predicted_first = predicted[0]
        filtered_predicted_first_parts = Common.filter_impossible_names(predicted_first)  # list

        if beam_width == 0:
            output_file.write('Original: ' + Common.internal_delimiter.join(original_name_parts) +
                              ' , predicted 1st: ' + Common.internal_delimiter.join(
                filtered_predicted_first_parts) + '\n')
            if filtered_original == filtered_predicted_first_parts or Common.unique(
                    filtered_original) == Common.unique(
                filtered_predicted_first_parts) or ''.join(filtered_original) == ''.join(
                filtered_predicted_first_parts):
                num_correct_predictions += 1
        else:
            filtered_predicted = [Common.internal_delimiter.join(Common.filter_impossible_names(p)) for p in
                                  predicted]

            true_ref = original_name
            output_file.write('Original: ' + ' '.join(original_name_parts) + '\n')
            for i, p in enumerate(filtered_predicted):
                output_file.write('\t@{}: {}'.format(i + 1, ' '.join(p.split(Common.internal_delimiter))) + '\n')
            if true_ref in filtered_predicted:
                index_of_correct = filtered_predicted.index(true_ref)
                update = np.concatenate(
                    [np.zeros(index_of_correct, dtype=np.int32),
                     np.ones(beam_width - index_of_correct, dtype=np.int32)])
                num_correct_predictions += update
    return num_correct_predictions


def update_per_subtoken_statistics(beam_width, results, true_positive, false_positive, false_negative):
    for original_name, predicted in results:
        if beam_width > 0:
            predicted = predicted[0]
        filtered_predicted_names = Common.filter_impossible_names(predicted)
        filtered_original_subtokens = Common.filter_impossible_names(original_name.split(Common.internal_delimiter))

        if ''.join(filtered_original_subtokens) == ''.join(filtered_predicted_names):
            true_positive += len(filtered_original_subtokens)
            continue

        for subtok in filtered_predicted_names:
            if subtok in filtered_original_subtokens:
                true_positive += 1
            else:
                false_positive += 1
        for subtok in filtered_original_subtokens:
            if not subtok in filtered_predicted_names:
                false_negative += 1
    return true_positive, false_positive, false_negative
