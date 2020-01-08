import json
import math
from preprocess_tri import Preprocess_tri
from eval import evaluation
import pandas as pd
import numpy as np


class HMM_tri():
    def __init__(self, ep, topK_trig, topK_argu, lam1, lam2, lam3):
        self.test_sen_argu = self.load_json('test_sen_argu.json')
        self.test_sen_trig = self.load_json('test_sen_trig.json')
        self.train_part_argu = self.load_json('train_part_argu.json')
        self.train_part_trig = self.load_json('train_part_trig.json')
        self.train_emissionProb_argu = self.load_json('train_emissionProb_argu_tri.json')
        self.train_emissionProb_trig = self.load_json('train_emissionProb_trig_tri.json')
        self.train_StateTransProb_argu_tri = self.load_json('train_StateTransProb_argu_tri.json')
        self.train_StateTransProb_trig_tri = self.load_json('train_StateTransProb_trig_tri.json')
        self.train_StateTransProb_argu_uni = self.load_json('train_StateTransProb_argu_uni.json')
        self.train_StateTransProb_trig_uni = self.load_json('train_StateTransProb_trig_uni.json')
        self.train_StateTransProb_argu_bi = self.load_json('train_StateTransProb_argu.json')
        self.train_StateTransProb_trig_bi = self.load_json('train_StateTransProb_trig.json')
        self.ep = ep
        self.topK_argu = topK_argu
        self.topK_trig = topK_trig
        self.lam1 = lam1
        self.lam2 = lam2
        self.lam3 = lam3
        # self.sp = sp

    def load_json(self, path):
        with open(path) as f:
            dic = json.load(f)
        return dic

# beam inference + interpolate
    def viterbi(self, sen, lst_part, statePrUni, statePrBi, statePrTri,
                lam1, lam2, lam3, emissPr, topK):
        lenSen = len(sen)
        numPart = len(lst_part)
        D = dict()
        for i in range(0, lenSen):
            D[i] = dict()
            if i == 0:
                state_ll = 'bstart'       # last last state
                state_l = 'start'         # last state
                word_l = '*S*'            # last word
                sigma_l = 1               # last sigma
                ep = emissPr[state_l][word_l]
                D[i] = dict()
                for a in range(0, numPart):
                    state_c = lst_part[a]
                    spUni = statePrUni[state_c]
                    spBi = statePrBi[state_l][state_c]
                    spTri = statePrTri[state_ll][state_l][state_c]
                    sp = lam1 * spUni + lam2 * spBi + lam3 * spTri
                    sigma = math.log(sp) + math.log(ep) + math.log(sigma_l)
                    D[i][a] = [sigma, 0]
            elif i == 1:
                state_ll = 'start'
                word_l = sen[str(i-1)][0]
                for b in range(0, numPart):
                    D[i][b] = dict()
                    state_c = lst_part[b]
                    spUni = statePrUni[state_c]
                    mydict = D[i-1]
                    for c in [key for key, value in mydict.items()
                              if value[0] in [i[0] for i in (mydict.values())][:topK]]:
                        state_l = lst_part[c]
                        try:
                            ep = emissPr[state_l][word_l]
                        except:
                            # ep = emissPr[state_l]['UNK']
                            ep = self.ep
                        spBi = statePrBi[state_l][state_c]
                        spTri = statePrTri[state_ll][state_l][state_c]
                        sp = lam1 * spUni + lam2 * spBi + lam3 * spTri
                        sigma = math.log(sp) + math.log(ep) + D[i-1][c][0]
                        D[i][b][c] = sigma
                    D[i][b] = [max(D[i][b].values()), max(D[i][b], key=D[i][b].get)]
            else:
                word_l = sen[str(i-1)][0]
                for j in range(0, numPart):
                    D[i][j] = dict()
                    state_c = lst_part[j]
                    spUni = statePrUni[state_c]
                    mydict = D[i-1]
                    for k in [key for key, value in mydict.items()
                              if value[0] in [i[0] for i in (mydict.values())][:topK]]:
                        D[i][j][k] = dict()
                        state_l = lst_part[k]
                        try:
                            ep = emissPr[state_l][word_l]
                        except:
                            # ep = emissPr[state_l]['UNK']
                            ep = self.ep
                        mydict = D[i-2]
                        for l in [key for key, value in mydict.items()
                                  if value[0] in [i[0] for i in (mydict.values())][:topK]]:
                            state_ll = lst_part[l]
                            spBi = statePrBi[state_l][state_c]
                            spTri = statePrTri[state_ll][state_l][state_c]
                            sp = lam1 * spUni + lam2 * spBi + lam3 * spTri
                            sigma = math.log(sp) + math.log(ep) + D[i-1][k][0]
                            D[i][j][k][l] = sigma
                        D[i][j][k] = max(D[i][j][k].values())
                    D[i][j] = [max(D[i][j].values()), max(D[i][j], key=D[i][j].get)]

        # the end
        D_end = dict()
        spUni = statePrUni['end']
        word_l = sen[str(lenSen-1)][0]    # the last word of the sen
        mydict = D[lenSen-1]
        for m in [key for key, value in mydict.items()
                  if value[0] in [i[0] for i in (mydict.values())][:topK]]:
            D_end[m] = dict()
            state_l = lst_part[m]
            spBi = statePrBi[state_l]['end']
            try:
                ep = emissPr[state_l][word_l]
            except:
                self.ep = ep
            mydict = D[lenSen-2]
            for n in [key for key, value in mydict.items()
                      if value[0] in [i[0] for i in (mydict.values())][:topK]]:
                state_ll = lst_part[n]
                spTri = statePrTri[state_ll][state_l]['end']
                sp = lam1 * spUni + lam2 * spBi + lam3 * spTri
                sigma = math.log(sp) + math.log(ep) + D[lenSen-1][m][0]
                D_end[m][n] = sigma
            D_end[m] = [max(D_end[m].values()), max(D_end[m], key=D_end[m].get)]
        pathIdx = max(D_end, key=D_end.get)
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

    def printResult(self, sentence, lst_part, statePrUni, statePrBi, statePrTri, lam1, lam2, lam3, emissPr, topK, fn):
        myfile = open(fn + '.txt', 'w')
        for sen in sentence.keys():
            s = sentence[sen]
            lenSen = len(s)
            if lenSen == 0:
                pass
            else:
                result = self.viterbi(s, lst_part, statePrUni, statePrBi, statePrTri, lam1, lam2, lam3, emissPr, topK)
                for x in range(0, lenSen):
                    myfile.write(s[str(x)][0] + '\t'
                                 + s[str(x)][1] + '\t'
                                 + result[x]+ '\n')
                myfile.write('\n')
        myfile.close()

    def run(self):
        self.printResult(self.test_sen_argu, self.train_part_argu,
                         self.train_StateTransProb_argu_uni, self.train_StateTransProb_argu_bi,
                         self.train_StateTransProb_argu_tri,
                         self.lam1, self.lam2, self.lam3,
                         self.train_emissionProb_argu, self.topK_argu, 'argument_result')
        self.printResult(self.test_sen_trig, self.train_part_trig,
                         self.train_StateTransProb_trig_uni, self.train_StateTransProb_trig_bi,
                         self.train_StateTransProb_trig_tri,
                         self.lam1, self.lam2, self.lam3,
                         self.train_emissionProb_trig, self.topK_trig, 'trigger_result')


df_trig = pd.DataFrame(columns=['lam1', 'lam2', 'lam3', 'type_correct', 'accuracy', 'precision', 'recall', 'F1'])
df_argu = pd.DataFrame(columns=['lam1', 'lam2', 'lam3', 'type_correct', 'accuracy', 'precision', 'recall', 'F1'])


pre = Preprocess_tri(0)
pre.run()
hmm = HMM_tri(1e-5, 5,5, 0.6, 0.3, 0.2)
hmm.run()
evaluation('trigger')
evaluation('argument')




# lst = [[0.1,0.1,0.8], [0.1,0.2,0.7], [0.1,0.3,0.6], [0.2,0.2,0.6], [0.6, 0.3, 0.1]]
# for i in lst:
#     lam1 = i[0]
#     lam2 = i[1]
#     lam3 = i[2]
#     hmm = HMM_tri(1e-5, 1, lam1, lam2, lam3)
#     hmm.run()
#     type_correct, accuracy, precision, recall, F1 = evaluation('trigger')
#     df_trig = df_trig.append({'lam1': lam1,
#                               'lam2': lam2,
#                               'lam3': lam3,
#                               'type_correct': type_correct,
#                               'accuracy': accuracy,
#                               'precision': precision,
#                               'recall': recall,
#                               'F1': F1
#                               }, ignore_index=True)
#     type_correct, accuracy, precision, recall, F1 = evaluation('argument')
#     df_argu = df_argu.append({'lamb1': lam1,
#                               'lamb2': lam2,
#                               'lamb3': lam3,
#                               'type_correct': type_correct,
#                               'accuracy': accuracy,
#                               'precision': precision,
#                               'recall': recall,
#                               'F1': F1
#                               }, ignore_index=True)
#
# df_argu.to_csv('argu.csv', index=False)
# df_trig.to_csv('trig.csv', index=False)
