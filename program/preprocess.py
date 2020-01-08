import json
import codecs


class Preprocess():
    def __init__(self, lam):
        self.test_sen_argu = self.load('argument_test.txt')
        self.test_sen_trig = self.load('trigger_test.txt')
        self.train_sen_argu = self.load('argument_train.txt')
        self.train_sen_trig = self.load('trigger_train.txt')
        self.lam = lam
        self.train_emiss_argu, self.train_part_argu = self.pre(self.train_sen_argu, self.lam)
        self.train_emiss_trig, self.train_part_trig = self.pre(self.train_sen_trig, self.lam)

    def load(self, path):
        f = codecs.open(path, 'r', 'utf8')
        result = f.readlines()
        f.close()
        n = 0
        c = 0
        dic_sen = dict()
        dic_sen[n] = dict()
        for y in result:
            if len(y) == 1:      # if read an empty line
                c = 0
                n += 1
                dic_sen[n] = dict()  # start a new dict of sentence
                continue
            else:
                li = y.strip().split()
                word = li[0]
                part = li[1]
            dic_sen[n][c] = [word, part]
            c += 1
        return dic_sen

    def pre(self, dic_sen, lam):
        lst_part = []   # save the type of part
        emiss = dict()
        emiss['start'] = dict()
        emiss['start']['*S*'] = 0
        for i in dic_sen:
            emiss['start']['*S*'] += 1
            sen = dic_sen[i]
            for j in range(0, len(sen)):
                word = sen[j][0]
                part = sen[j][1]
                if part not in lst_part:
                    lst_part.append(part)
                else:
                    pass
                if part not in emiss.keys():
                    emiss[part] = dict()
                    emiss[part]['UNK'] = 0
                else:
                    pass
                if word not in emiss[part].keys():
                    emiss[part][word] = 1
                else:
                    emiss[part][word] += 1

        for a in emiss.keys():
            for b in emiss[a].keys():
                emiss[a][b] += lam      # smoothing
        return emiss, lst_part

    def StateTransProb(self, sen, lst_part, lam):   # using Bigram
        # initialize
        trans = dict()
        trans['start'] = dict()
        for j in lst_part:
            trans['start'][j] = 0
            trans[j] = dict()
            trans[j]['end'] = 0
            for k in lst_part:
                trans[j][k] = 0

        for i in sen.keys():
            dic_sen = sen[i]
            if len(dic_sen) == 0:
                continue
            elif len(dic_sen) == 1:
                s1 = 'start'
                s2 = dic_sen.values()
                trans[s1][s2] += 1
                s1 = s2
                s2 = 'end'
                trans[s1][s2] += 1
            else:
                for j in dic_sen.keys():
                    s1 = ''
                    s2 = ''    # state
                    if j == 0:
                        s1 = 'start'
                        s2 = dic_sen[j][1]
                    elif (j+1) % 2 == 0:
                        s1 = dic_sen[j-1][1]
                        s2 = dic_sen[j][1]
                    elif j == len(dic_sen)-1:
                        s1 = dic_sen[j-1][1]
                        s2 = 'end'
                    else:
                        continue
                    trans[s1][s2] += 1

        for m in trans.keys():
            for n in trans[m].keys():
                trans[m][n] += lam       # add-lambda-smoothing
            s = sum(trans[m].values())
            trans[m] = {k: v / s for k, v in trans[m].items()}
        return trans

    def emissionProb(self, emiss):
        for i in emiss.keys():
            s = sum(emiss[i].values())
            emiss[i] = {k: v / s for k, v in emiss[i].items()}
        return emiss

    def save(self, to_save, filename):
        s = json.dumps(to_save)
        f = open(filename + ".json", "w")
        f.write(s)
        f.close()

    def run(self):
        self.save(self.StateTransProb(self.train_sen_argu, self.train_part_argu, self.lam), 'train_StateTransProb_argu')
        self.save(self.StateTransProb(self.train_sen_trig, self.train_part_trig, self.lam), 'train_StateTransProb_trig')
        self.save(self.emissionProb(self.train_emiss_argu), 'train_emissionProb_argu')
        self.save(self.emissionProb(self.train_emiss_trig), 'train_emissionProb_trig')
        self.save(self.train_part_argu, 'train_part_argu')
        self.save(self.train_part_trig, 'train_part_trig')
        self.save(self.train_sen_argu, 'train_sen_argu')
        self.save(self.train_sen_trig, 'train_sen_trig')
        self.save(self.test_sen_argu, 'test_sen_argu')
        self.save(self.test_sen_trig, 'test_sen_trig')
