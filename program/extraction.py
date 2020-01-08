import json
import math
from preprocess import Preprocess
from eval import evaluation
import pandas as pd
import numpy as np


class HMM():
    def __init__(self, ep, sp):
        self.test_sen_argu = self.load_json('test_sen_argu.json')
        self.test_sen_trig = self.load_json('test_sen_trig.json')
        self.train_part_argu = self.load_json('train_part_argu.json')
        self.train_part_trig = self.load_json('train_part_trig.json')
        self.train_emissionProb_argu = self.load_json('train_emissionProb_argu.json')
        self.train_emissionProb_trig = self.load_json('train_emissionProb_trig.json')
        self.train_StateTransProb_argu = self.load_json('train_StateTransProb_argu.json')
        self.train_StateTransProb_trig = self.load_json('train_StateTransProb_trig.json')
        self.ep = ep
        self.sp = sp

    def load_json(self, path):
        with open(path) as f:
            dic = json.load(f)
        return dic

    def viterbi(self, sen, lst_part, statePr, emissPr):
        lenSen = len(sen)
        numPart = len(lst_part)
        D = dict()
        for i in range(0, lenSen):
            D[i] = dict()
            if i == 0:
                state_l = 'start'         # last state
                word_l = '*S*'            # last word
                sigma_l = 1               # last sigma
                ep = emissPr[state_l][word_l]
                D[i] = dict()
                for b in range(0, numPart):
                    state_c = lst_part[b]
                    sp = statePr[state_l][state_c]
                    if sp == 0.0:
                        sp = self.sp
                    sigma = math.log(sp) + math.log(ep) + math.log(sigma_l)
                    D[i][b] = [sigma, 0]
            else:
                word_l = sen[str(i-1)][0]
                for j in range(0, numPart):
                    D[i][j] = dict()
                    state_c = lst_part[j]         # current state
                    for k in range(0, numPart):   # calculate fi for the current state
                        state_l = lst_part[k]
                        sp = statePr[state_l][state_c]
                        if sp == 0.0:
                            sp = self.sp
                        try:
                            ep = emissPr[state_l][word_l]
                        except:
                            # ep = emissPr[state_l]['UNK']
                            ep = self.ep
                        sigma = math.log(sp) + math.log(ep) + D[i-1][k][0]
                        D[i][j][k] = sigma
                    D[i][j] = [max(D[i][j].values()), max(D[i][j], key=D[i][j].get)]

        # the end
        lst_end_sigma = list()
        word_l = sen[str(lenSen-1)][0]    # the last word of the sen
        for m in range(0, numPart):
            state_l = lst_part[m]
            sp = statePr[state_l]['end']
            if sp == 0.0:
                sp = self.sp
            try:
                ep = emissPr[state_l][word_l]
            except:
                # ep = emissPr[state_l]['UNK']
                ep = self.ep
            sigma = math.log(sp) + math.log(ep) + D[lenSen-1][m][0]
            lst_end_sigma.append(sigma)
        pathIdx = lst_end_sigma.index(max(lst_end_sigma))
        fi = lst_part[pathIdx]            # the part of the last word
        lst_path = list()
        lst_path.append(fi)

        # tracing the path
        for n in range(lenSen-1, 0, -1):
            pathIdx = D[n][pathIdx][1]
            fi = lst_part[pathIdx]
            lst_path.append(fi)
        lst_path.reverse()
        return lst_path

    def printResult(self, sentence, lst_part, statePr, emissPr, fn):   # self.sentence_argu/trig
        myfile = open(fn + '.txt', 'w')
        for sen in sentence.keys():
            s = sentence[sen]
            lenSen = len(s)
            if lenSen == 0:
                pass
            else:
                result = self.viterbi(s, lst_part, statePr, emissPr)
                for x in range(0, lenSen):
                    myfile.write(s[str(x)][0] + '\t'
                                 + s[str(x)][1] + '\t'
                                 + result[x]+ '\n')
                myfile.write('\n')
        myfile.close()

    def run(self):
        self.printResult(self.test_sen_argu, self.train_part_argu, self.train_StateTransProb_argu, self.train_emissionProb_argu, 'argument_result')
        self.printResult(self.test_sen_trig, self.train_part_trig, self.train_StateTransProb_trig, self.train_emissionProb_trig, 'trigger_result')


df_trig = pd.DataFrame(columns=['sp','ep', 'type_correct', 'accuracy', 'precision', 'recall', 'F1'])
df_argu = pd.DataFrame(columns=['sp','ep', 'type_correct', 'accuracy', 'precision', 'recall', 'F1'])

pre = Preprocess(0)
pre.run()
hmm = HMM(1e-6, 0.01)
hmm.run()
evaluation('trigger')
evaluation('argument')

# for m in np.logspace(-3, -9, 7, base=10):
#     for n in np.logspace(-2, -5, 4, base=10):
#         hmm = HMM(m, n)
#         hmm.run()
#         type_correct, accuracy, precision, recall, F1 = evaluation('trigger')
#         df_trig = df_trig.append({'ep': m,
#                                   'sp': n,
#                                   'type_correct': type_correct,
#                                   'accuracy': accuracy,
#                                   'precision': precision,
#                                   'recall': recall,
#                                   'F1': F1
#                                   }, ignore_index=True)
#         type_correct, accuracy, precision, recall, F1 = evaluation('argument')
#         df_argu = df_argu.append({'ep': m,
#                                   'sp': n,
#                                   'type_correct': type_correct,
#                                   'accuracy': accuracy,
#                                   'precision': precision,
#                                   'recall': recall,
#                                   'F1': F1
#                                   }, ignore_index=True)
#
# df_argu.to_csv('argu.csv', index=False)
# df_trig.to_csv('trig.csv', index=False)