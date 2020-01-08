import json


class Preprocess_tri():
    def __init__(self, lam):
        self.train_sen_argu = self.load_json('train_sen_argu.json')
        self.train_sen_trig = self.load_json('train_sen_trig.json')
        self.train_part_argu = self.load_json('train_part_argu.json')
        self.train_part_trig = self.load_json('train_part_trig.json')
        self.lam = lam

    def load_json(self, path):
        with open(path) as f:
            dic = json.load(f)
        return dic

    def pre(self, lst_part, dic_sen, lam):
        # initialize
        emiss = dict()
        emiss['start'] = dict()
        emiss['start']['*S*'] = 0
        for k in range(0, len(lst_part)):
            k = lst_part[k]
            emiss[k] = dict()
            emiss[k]['UNK'] = 0

        for i in dic_sen:
            emiss['start']['*S*'] += 1
            sen = dic_sen[i]
            for j in range(0, len(sen)):
                word = sen[str(j)][0]
                part = sen[str(j)][1]
                if word not in emiss[part].keys():
                    emiss[part][word] = 1
                else:
                    emiss[part][word] += 1

        for a in emiss.keys():
            for b in emiss[a].keys():
                emiss[a][b] += lam      # smoothing
        return emiss

    def emissionProb(self, emiss):
        for i in emiss.keys():
            s = sum(emiss[i].values())
            emiss[i] = {k: v / s for k, v in emiss[i].items()}
        return emiss

    def StateTransProb(self, sen, lst_part, lam):   # using Trigram
        # initialize trigram
        trans = dict()
        trans['bstart'] = dict()
        trans['start'] = dict()
        trans['bstart']['start'] = dict()
        for j in lst_part:
            trans['bstart']['start'][j] = 0
            trans['start'][j] = dict()
            trans['start'][j]['end'] = 0
            trans[j] = dict()
            for i in lst_part:
                trans['start'][j][i] = 0
                trans[j][i] = dict()
                trans[j][i]['end'] = 0
                for k in lst_part:
                    trans[j][i][k] = 0

        # initialize unigram
        trans_uni = dict()
        trans_uni['end'] = 0

        for i in sen.keys():
            dic_sen = sen[i]
            if len(dic_sen) == 0:
                continue
            elif len(dic_sen) == 1:
                trans_uni['end'] += 1
                trans['bstart']['start'][dic_sen.values()] += 1
            elif len(dic_sen) == 2:
                trans_uni['end'] += 1
                w1 = dic_sen['0'][1]
                w2 = dic_sen['1'][1]
                trans['bstart']['start'][w1] += 1
                trans['start'][w1][w2] += 1
                trans[w1][w2]['end'] += 1
            else:
                trans_uni['end'] += 1
                for j in dic_sen.keys():
                    j = int(j)
                    u = dic_sen[str(j)][1]
                    if u not in trans_uni.keys():
                        trans_uni[u] = 1
                    else:
                        trans_uni[u] += 1
                    if j == 0:
                        s1 = 'bstart'
                        s2 = 'start'
                        s3 = dic_sen[str(j)][1]
                    elif (j+1) % 3 == 0:
                        s1 = dic_sen[str(j-2)][1]
                        s2 = dic_sen[str(j-1)][1]
                        s3 = dic_sen[str(j)][1]
                    elif j == len(dic_sen)-1:
                        s1 = dic_sen[str(j-2)][1]
                        s2 = dic_sen[str(j-1)][1]
                        s3 = 'end'
                    else:
                        continue
                    trans[s1][s2][s3] += 1

        for m in trans.keys():
            for n in trans[m].keys():
                for l in trans[m][n].keys():
                    trans[m][n][l] += 1
                s = sum(trans[m][n].values())
                trans[m][n] = {k: v / s for k, v in trans[m][n].items()}

        s_uni = sum(trans_uni.values())
        trans_uni = {k: v / s_uni for k, v in trans_uni.items()}
        return trans, trans_uni

    def save(self, to_save, filename):
        s = json.dumps(to_save)
        f = open(filename + ".json", "w")
        f.write(s)
        f.close()

    def run(self):
        argu = self.StateTransProb(self.train_sen_argu, self.train_part_argu, self.lam)
        trig = self.StateTransProb(self.train_sen_trig, self.train_part_trig, self.lam)
        self.save(argu[0], 'train_StateTransProb_argu_tri')
        self.save(trig[0], 'train_StateTransProb_trig_tri')
        self.save(argu[1], 'train_StateTransProb_argu_uni')
        self.save(trig[1], 'train_StateTransProb_trig_uni')
        self.save(self.emissionProb(self.pre(self.train_part_argu, self.train_sen_argu, self.lam)), 'train_emissionProb_argu_tri')
        self.save(self.emissionProb(self.pre(self.train_part_trig, self.train_sen_trig, self.lam)), 'train_emissionProb_trig_tri')