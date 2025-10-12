class MetricParser:
    '''
    Implementers should create specific benchmark parsing rules in subclasses. 
    The output of this should be a pair of lists, full of 1s and 0s indicating
    a positive and a negative. First element in the pair is the predictions,
    second element is the ground truth of the benchmark

    @todo use kwargs to not limit this interface
    '''
    def parse(self, prediction_file, ground_truth_file):
        pass

class calculator:
    def __init__(self, parser: MetricParser):
        self.parser = parser

    def parse(self, answer_file, label_file):
        self.answers, self.labels = self.parser.parse(answer_file, label_file)

    def calculate_results(self):
        pos = 1
        neg = 0
        yes_ratio = self.answers.count(1) / len(self.answers)
        true_pos, false_pos, true_neg, false_neg = 0, 0, 0, 0
        for pred, label in zip(self.answers, self.labels):
            if pred == pos and label == pos:
                true_pos += 1
            elif pred == pos and label == neg:
                false_pos += 1
            elif pred == neg and label == neg:
                true_neg += 1
            elif pred == neg and label == pos:
                false_neg += 1

        precision = float(true_pos) / float(true_pos + false_pos)
        recall = float(true_pos) / float(true_pos + false_neg)
        f1 = 2*precision*recall / (precision + recall)
        acc = (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg)
        return f"TP\tFP\tTN\tFN\t\n{true_pos}\t{false_pos}\t{true_neg}\t{false_neg}\nAccuracy: {acc}\nPrecision:\
        {precision}\nRecall: {recall}\nF1 score: {f1}\nYes ratio: {yes_ratio}"

