from .calculator import MetricParser
import json
import nltk
from nltk.stem import WordNetLemmatizer
import os
import json
import spacy
import warnings
from tqdm import tqdm
warnings.filterwarnings("ignore", category=UserWarning)
        
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

class AmberMetricParser(MetricParser):
    '''
    Function copied from source for ease of use - please see the AMBER code
    base for most up to date version.
    '''
    def check_synonyms_word(self, word1, word2):
        token1 = self.nlp(word1)
        token2 = self.nlp(word2)
        similarity = token1.similarity(token2)
        return similarity > self.sim_score

    '''
    Function copied from source for ease of use - please see the AMBER code
    base for most up to date version.
    '''
    def extract_nouns(self, text):
        lemmatizer = WordNetLemmatizer()
        tokens = nltk.word_tokenize(text)
        tagged = nltk.pos_tag(tokens)
        nouns = [lemmatizer.lemmatize(word) for word, pos in tagged if pos.startswith('NN')]
        return nouns

    def __init__(self, args):
        self.nlp = spacy.load("en_core_web_lg")
        self.data_path = os.path.join(args.amber_path, "data")
        association_file = os.path.join(self.data_path, "relation.json")
        safewords_file = os.path.join(self.data_path, "safe_words.txt")
        metrics_file = os.path.join(self.data_path, "metrics.txt")
        self.annotations_file = os.path.join(self.data_path, "annotations.json")
        self.sim_score = args.sim_score
        # parse word associations
        self.associations = json.load(open(association_file, 'r', encoding='utf-8'))
        self.hallucination_words = []

        for keyword in self.associations.keys():
            self.hallucination_words.append(keyword)
            for word in self.associations[keyword]:
                self.hallucination_words.append(word)
        
        # parse safe words
        self.global_safe_words = []
        with open(safewords_file, 'r', encoding='utf-8') as safe_file:
            for line in safe_file:
                line = line.split('\n')[0]
                self.global_safe_words.append(line)

        # parse metrics
        with open(metrics_file, "r") as file:
            metric_file_lines = file.readlines()
        self.metrics = {}
        for metric_line in metric_file_lines:
            kv = metric_line.strip().split('=')
            # metrics file is a list of key value pairs.
            # TODO: add error logging for this file
            if len(kv) == 2:
                key = kv[0].strip()
                val = eval(kv[1].strip())
                self.metrics[key] = val


    def parse(self, args):
        inference_data = [json.loads(q) for q in open(args[1], 'r', encoding='utf-8')]
        ground_truth = json.load(open(self.annotations_file, 'r', encoding='utf-8'))

        for i in tqdm(range(len(inference_data))):
            
            id = inference_data[i]['id']
            
            if ground_truth[id-1]['type'] == 'generative':
                nouns = self.extract_nouns(inference_data[i]['response'])
                after_process_nouns = []
                for noun in nouns:
                    if noun in self.hallucination_words:
                        after_process_nouns.append(noun)
                
                safe_words = []
                safe_list = []
                for idx, word in enumerate(ground_truth[id-1]['truth']):
                    safe_words += self.associations[word]
                    safe_list += [idx] * len(self.associations[word])
                    
                ha_words = []
                ha_list = []
                for idx, word in enumerate(ground_truth[id-1]['hallu']):
                    ha_words += self.associations[word]
                    ha_list += [idx] * len(self.associations[word])
                
                safe_words += ground_truth[id-1]['truth']
                safe_len = len(ground_truth[id-1]['truth'])
                safe_list += [0] * safe_len
                safe_flag_list = [0] * len(after_process_nouns)
                
                ha_words += ground_truth[id-1]['hallu']
                ha_len = len(ground_truth[id-1]['hallu'])
                ha_list += [0] * ha_len
                
                for idx, noun in enumerate(after_process_nouns):
                    if noun in self.global_safe_words:
                        continue
                    
                    if noun in safe_words:
                        for j in range(len(safe_words)):
                            if noun == safe_words[j]:
                                if j < (len(safe_list) - safe_len):
                                    safe_list[safe_list[j] + len(safe_list) - safe_len] = 1
                                else:
                                    safe_list[j] = 1
                                break
                        continue
                    
                    if noun in ha_words:
                        for j in range(len(ha_words)):
                            if noun == ha_words[j]:
                                if j < (len(ha_list) - ha_len):
                                    ha_list[ha_list[j] + len(ha_list) - ha_len] = 1
                                else:
                                    ha_list[j] = 1
                                break
                    
                    for j, check_word in enumerate(ha_words):
                        if self.check_synonyms_word(noun, check_word):
                            if j < (len(ha_list) - ha_len):
                                    ha_list[ha_list[j] + len(ha_list) - ha_len] = 1
                            else:
                                ha_list[j] = 1
                            break
                    
                    flag = False
                    for j, check_word in enumerate(safe_words):
                        if self.check_synonyms_word(noun, check_word):
                            flag = True
                            if j < (len(safe_list) - safe_len):
                                    safe_list[safe_list[j] + len(safe_list) - safe_len] = 1
                            else:
                                safe_list[j] = 1
                            break
                    if flag == True:
                        continue
                
                    safe_flag_list[idx] = 1

                self.metrics['chair_score'] += sum(safe_flag_list)
                self.metrics['chair_num'] += len(safe_flag_list)
                self.metrics['safe_cover_score'] += sum(safe_list[-safe_len:])
                self.metrics['safe_cover_num'] += len(safe_list[-safe_len:])
                self.metrics['hallu_cover_score'] += sum(ha_list[-ha_len:])
                self.metrics['hallu_cover_num'] += len(ha_list[-ha_len:])
                if sum(safe_flag_list) == 0:
                    self.metrics['non_hallu_score'] += 1
                self.metrics['non_hallu_num'] += 1
            
            else:
                self.metrics['qa_correct_num'] += 1
                if ground_truth[id-1]['type'] == 'discriminative-attribute-state':
                    self.metrics['as_qa_correct_num'] += 1
                elif ground_truth[id-1]['type'] == 'discriminative-attribute-number':
                    self.metrics['an_qa_correct_num'] += 1
                elif ground_truth[id-1]['type'] == 'discriminative-attribute-action':
                    self.metrics['aa_qa_correct_num'] += 1
                elif ground_truth[id-1]['type'] == 'discriminative-hallucination':
                    self.metrics['ha_qa_correct_num'] += 1
                else:
                    self.metrics['asso_qa_correct_num'] += 1
                
                truth = ground_truth[id-1]['truth']
                response = inference_data[i]['response']
                if truth == 'yes':
                    if response == 'Yes':
                        self.metrics['qa_correct_score'] += 1
                        if ground_truth[id-1]['type'] == 'discriminative-attribute-state':
                            self.metrics['as_qa_correct_score'] += 1
                        elif ground_truth[id-1]['type'] == 'discriminative-attribute-number':
                            self.metrics['an_qa_correct_score'] += 1
                        elif ground_truth[id-1]['type'] == 'discriminative-attribute-action':
                            self.metrics['aa_qa_correct_score'] += 1
                        elif ground_truth[id-1]['type'] == 'discriminative-hallucination':
                            self.metrics['ha_qa_correct_score'] += 1
                        else:
                            self.metrics['asso_qa_correct_score'] += 1
                else:
                    self.metrics['qa_no_num'] += 1
                    if ground_truth[id-1]['type'] == 'discriminative-attribute-state':
                        self.metrics['as_qa_no_num'] += 1
                    elif ground_truth[id-1]['type'] == 'discriminative-attribute-number':
                        self.metrics['an_qa_no_num'] += 1
                    elif ground_truth[id-1]['type'] == 'discriminative-attribute-action':
                        self.metrics['aa_qa_no_num'] += 1
                    elif ground_truth[id-1]['type'] == 'discriminative-hallucination':
                        self.metrics['ha_qa_no_num'] += 1
                    else:
                        self.metrics['asso_qa_no_num'] += 1
                    
                    if response == 'No':
                        self.metrics['qa_correct_score'] += 1
                        self.metrics['qa_no_score'] += 1
                        if ground_truth[id-1]['type'] == 'discriminative-attribute-state':
                            self.metrics['as_qa_correct_score'] += 1
                            self.metrics['as_qa_no_score'] += 1
                        elif ground_truth[id-1]['type'] == 'discriminative-attribute-number':
                            self.metrics['an_qa_correct_score'] += 1
                            self.metrics['an_qa_no_score'] += 1
                        elif ground_truth[id-1]['type'] == 'discriminative-attribute-action':
                            self.metrics['aa_qa_correct_score'] += 1
                            self.metrics['aa_qa_no_score'] += 1
                        elif ground_truth[id-1]['type'] == 'discriminative-hallucination':
                            self.metrics['ha_qa_correct_score'] += 1
                            self.metrics['ha_qa_no_score'] += 1
                        else:
                            self.metrics['asso_qa_correct_score'] += 1
                            self.metrics['asso_qa_no_score'] += 1
                
                if response == 'No':
                    self.metrics['qa_ans_no_num'] += 1
                    if ground_truth[id-1]['type'] == 'discriminative-attribute-state':
                        self.metrics['as_qa_ans_no_num'] += 1
                    elif ground_truth[id-1]['type'] == 'discriminative-attribute-number':
                        self.metrics['an_qa_ans_no_num'] += 1
                    elif ground_truth[id-1]['type'] == 'discriminative-attribute-action':
                        self.metrics['aa_qa_ans_no_num'] += 1
                    elif ground_truth[id-1]['type'] == 'discriminative-hallucination':
                        self.metrics['ha_qa_ans_no_num'] += 1
                    else:
                        self.metrics['asso_qa_ans_no_num'] += 1
                    if truth == 'no':
                        self.metrics['qa_ans_no_score'] += 1
                        if ground_truth[id-1]['type'] == 'discriminative-attribute-state':
                            self.metrics['as_qa_ans_no_score'] += 1
                        elif ground_truth[id-1]['type'] == 'discriminative-attribute-number':
                            self.metrics['an_qa_ans_no_score'] += 1
                        elif ground_truth[id-1]['type'] == 'discriminative-attribute-action':
                            self.metrics['aa_qa_ans_no_score'] += 1
                        elif ground_truth[id-1]['type'] == 'discriminative-hallucination':
                            self.metrics['ha_qa_ans_no_score'] += 1
                        else:
                            self.metrics['asso_qa_ans_no_score'] += 1
        return self.metrics