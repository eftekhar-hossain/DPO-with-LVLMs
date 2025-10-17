from .calculator import calculator
from .amber_metrics import AmberMetricParser

'''
This file is incredibly heavily based on AMBER/inference.py.
Most code from that file has been adapted here, to fit into
the framework established within this tool. This code is mostly
not original, except where it had to be moved around to fit into
our architecture.

Please see the LISCENCE file of the AMBER repo at
https://github.com/junyangwang0410/AMBER
for the original author and permissions to redistribute
and modify this code
'''
class AmberCalculator(calculator):
    def __init__(self, args):
        super().__init__(AmberMetricParser(args))

    def calculate_results(self):
        CHAIR = round(self.parse_results['chair_score'] / self.parse_results['chair_num'] * 100, 1)
        Cover = round(self.parse_results['safe_cover_score'] / self.parse_results['safe_cover_num'] * 100, 1)
        Ha = round(self.parse_results['hallu_cover_score'] / self.parse_results['hallu_cover_num'] * 100, 1)
        Ha_p = round(100 - self.parse_results['non_hallu_score'] / self.parse_results['non_hallu_num'] * 100, 1)
        print("Generative Task:")
        print("CHAIR:\t\t", CHAIR)
        print("Cover:\t\t", Cover)
        print("Hal:\t\t", Ha_p)
        print("Cog:\t\t", Ha, "\n")

        Accuracy = round(self.parse_results['qa_correct_score'] / self.parse_results['qa_correct_num'] * 100, 1)
        Precision = round(self.parse_results['qa_ans_no_score'] / self.parse_results['qa_ans_no_num'] * 100, 1)
        Recall = round(self.parse_results['qa_no_score'] / self.parse_results['qa_no_num'] * 100, 1)
        F1 = round(2 * (Precision/100) * (Recall/100) / ((Precision/100) + (Recall/100) + 0.0001) * 100, 1)
        print("Descriminative Task:")
        print("Accuracy:\t", Accuracy)
        print("Precision:\t", Precision)
        print("Recall:\t\t", Recall)
        print("F1:\t\t", F1, "\n")

        hallucination_Accuracy = round(self.parse_results['ha_qa_correct_score'] / self.parse_results['ha_qa_correct_num'] * 100, 1)
        hallucination_Precision = round(self.parse_results['ha_qa_ans_no_score'] / self.parse_results['ha_qa_ans_no_num'] * 100, 1)
        hallucination_Recall = round(self.parse_results['ha_qa_no_score'] / self.parse_results['ha_qa_no_num'] * 100, 1)
        hallucination_F1 = round(2 * (hallucination_Precision/100) * (hallucination_Recall/100) / ((hallucination_Precision/100) + (hallucination_Recall/100) + 0.001) * 100, 1)
        print("Exsitence:")
        print("Accuracy:\t", hallucination_Accuracy)
        print("Precision:\t", hallucination_Precision)
        print("Recall:\t\t", hallucination_Recall)
        print("F1:\t\t", hallucination_F1, "\n")

        attr_Accuracy = round((self.parse_results['as_qa_correct_score'] + self.parse_results['an_qa_correct_score'] + self.parse_results['aa_qa_correct_score']) / (self.parse_results['as_qa_correct_num'] + self.parse_results['an_qa_correct_num'] + self.parse_results['aa_qa_correct_num']) * 100, 1)
        attr_Precision = round((self.parse_results['as_qa_ans_no_score'] + self.parse_results['an_qa_ans_no_score'] + self.parse_results['aa_qa_ans_no_score']) / (self.parse_results['as_qa_ans_no_num'] + self.parse_results['an_qa_ans_no_num'] + self.parse_results['aa_qa_ans_no_num']) * 100, 1)
        attr_Recall = round((self.parse_results['as_qa_no_score'] + self.parse_results['an_qa_no_score'] + self.parse_results['aa_qa_no_score']) / (self.parse_results['as_qa_no_num'] + self.parse_results['an_qa_no_num'] + self.parse_results['aa_qa_no_num']) * 100, 1)
        attr_F1 = round(2 * (attr_Precision/100) * (attr_Recall/100) / ((attr_Precision/100) + (attr_Recall/100) + 0.0001) * 100, 1)
        state_Accuracy = round(self.parse_results['as_qa_correct_score'] / self.parse_results['as_qa_correct_num'] * 100, 1)
        state_Precision = round(self.parse_results['as_qa_ans_no_score'] / self.parse_results['as_qa_ans_no_num'] * 100, 1)
        state_Recall = round(self.parse_results['as_qa_no_score'] / self.parse_results['as_qa_no_num'] * 100, 1)
        state_F1 = round(2 * (state_Precision/100) * (state_Recall/100) / ((state_Precision/100) + (state_Recall/100) + 0.0001) * 100, 1)
        number_Accuracy = round(self.parse_results['an_qa_correct_score'] / self.parse_results['an_qa_correct_num'] * 100, 1)
        number_Precision = round(self.parse_results['an_qa_ans_no_score'] / self.parse_results['an_qa_ans_no_num'] * 100, 1)
        number_Recall = round(self.parse_results['an_qa_no_score'] / self.parse_results['an_qa_no_num'] * 100, 1)
        number_F1 = round(2 * (number_Precision/100) * (number_Recall/100) / ((number_Precision/100) + (number_Recall/100) + 0.0001) * 100, 1)
        action_Accuracy = round(self.parse_results['aa_qa_correct_score'] / self.parse_results['aa_qa_correct_num'] * 100, 1)
        action_Precision = round(self.parse_results['aa_qa_ans_no_score'] / self.parse_results['aa_qa_ans_no_num'] * 100, 1)
        action_Recall = round(self.parse_results['aa_qa_no_score'] / self.parse_results['aa_qa_no_num'] * 100, 1)
        action_F1 = round(2 * (action_Precision/100) * (action_Recall/100) / ((action_Precision/100) + (action_Recall/100) + 0.0001) * 100, 1)
        print("Attribute:")
        print("Accuracy:\t", attr_Accuracy)
        print("Precision:\t", attr_Precision)
        print("Recall:\t\t", attr_Recall)
        print("F1:\t\t", attr_F1, "\n")
        print("State:")
        print("Accuracy:\t", state_Accuracy)
        print("Precision:\t", state_Precision)
        print("Recall:\t\t", state_Recall)
        print("F1:\t\t", state_F1, "\n")
        print("Number:")
        print("Accuracy:\t", number_Accuracy)
        print("Precision:\t", number_Precision)
        print("Recall:\t\t", number_Recall)
        print("F1:\t\t", number_F1, "\n")
        print("Action:")
        print("Accuracy:\t", action_Accuracy)
        print("Precision:\t", action_Precision)
        print("Recall:\t\t", action_Recall)
        print("F1:\t\t", action_F1, "\n")

        relation_Accuracy = round(self.parse_results['asso_qa_correct_score'] / self.parse_results['asso_qa_correct_num'] * 100, 1)
        relation_Precision = round(self.parse_results['asso_qa_ans_no_score'] / self.parse_results['asso_qa_ans_no_num'] * 100, 1)
        relation_Recall = round(self.parse_results['asso_qa_no_score'] / self.parse_results['asso_qa_no_num'] * 100, 1)
        relation_F1 = round(2 * (relation_Precision/100) * (relation_Recall/100) / ((relation_Precision/100) + (relation_Recall/100) + 0.0001) * 100, 1)
        print("Relation:")
        print("Accuracy:\t", relation_Accuracy)
        print("Precision:\t", relation_Precision)
        print("Recall:\t\t", relation_Recall)
        print("F1:\t\t", relation_F1)