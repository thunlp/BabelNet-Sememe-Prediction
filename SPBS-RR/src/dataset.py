import os
import pandas as pd
import numpy as np
import random
import math

class KnowledgeGraph:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.entity_dict = {}
        self.entities = []
        self.relation_dict = {}
        self.entity2vec_dict = {}
        self.n_entity = 0
        self.n_relation = 0
        self.training_triples = []  # list of triples in the form of (h, t, r)

        self.training_synset_sememes = [] # list of triples in the form of (h, [s1, s2, s3])
        self.max_sememe_num = 30
        self.validation_triples = []
        self.test_triples = []
        self.n_training_triple = 0
        self.n_validation_triple = 0
        self.n_test_triple = 0
        self.embedding_array = {}
        '''load dicts and triples'''
        self.load_dicts()
        self.load_triples()
        #self.load_vec()
        '''construct pools after loading'''
        self.training_triple_pool = set(self.training_triples)
        self.golden_triple_pool = set(self.training_triples) | set(self.validation_triples) | set(self.test_triples)

    def load_dicts(self):
        entity_dict_file = 'entity2id.txt'
        relation_dict_file = 'relation2id.txt'
        print('-----Loading entity dict-----')
        entity_df = pd.read_table(os.path.join(self.data_dir, entity_dict_file), header=None)
        self.entity_dict = dict(zip(entity_df[0], entity_df[1]))
        self.n_entity = len(self.entity_dict)
        self.entities = list(self.entity_dict.values())
        print('#entity: {}'.format(self.n_entity))
        print('-----Loading relation dict-----')
        relation_df = pd.read_table(os.path.join(self.data_dir, relation_dict_file), header=None)
        self.relation_dict = dict(zip(relation_df[0], relation_df[1]))
        self.n_relation = len(self.relation_dict)
        print('#relation: {}'.format(self.n_relation))

    def load_triples(self):
        training_file = 'train.txt'
        validation_file = 'valid.txt'
        test_file = 'test.txt'
        print('-----Loading training triples-----')
        training_df = pd.read_table(os.path.join(self.data_dir, training_file), header=None)
        self.training_triples = list(zip([self.entity_dict[h] for h in training_df[0]],
                                         [self.entity_dict[t] for t in training_df[1]],
                                         [self.relation_dict[r] for r in training_df[2]]))
        self.n_training_triple = len(self.training_triples)
        
        ss_batch = []

        #group training triple by synset
        temp_synset = {}
        for item in self.training_triples:
            head, tail, relation = item
            if relation != 0:
                continue
            if head not in temp_synset.keys():
                temp_synset[head] = [tail]
            else:
                temp_synset[head].append(tail)

        for key, value in temp_synset.items():
            temp = [key] + value
            if len(temp) > self.max_sememe_num:
                print('len:', len(temp))
            self.max_sememe_num = max(len(temp), self.max_sememe_num)
            if len(temp) <= 2:
                continue
            ss_batch.append(temp)

        #print(len(temp_synset))
        for i, item in enumerate(ss_batch):
            if len(item) < self.max_sememe_num:
                ss_batch[i] = ss_batch[i] + [self.n_entity + 1] * (self.max_sememe_num - len(item))

        self.training_synset_sememes = ss_batch
        self.n_training_triple_ss = len(ss_batch)

        print('#training triple: {}'.format(self.n_training_triple))
        print('-----Loading validation triples-----')
        validation_df = pd.read_table(os.path.join(self.data_dir, validation_file), header=None)
        self.validation_triples = list(zip([self.entity_dict[h] for h in validation_df[0]],
                                           [self.entity_dict[t] for t in validation_df[1]],
                                           [self.relation_dict[r] for r in validation_df[2]]))
        self.n_validation_triple = len(self.validation_triples)
        print('#validation triple: {}'.format(self.n_validation_triple))
        print('-----Loading test triples------')
        test_df = pd.read_table(os.path.join(self.data_dir, test_file), header=None)
        self.test_triples = list(zip([self.entity_dict[h] for h in test_df[0]],
                                     [self.entity_dict[t] for t in test_df[1]],
                                     [self.relation_dict[r] for r in test_df[2]]))
        self.n_test_triple = len(self.test_triples)
        print('#test triple: {}'.format(self.n_test_triple))

    def load_vec(self):
        entity2vec_file = '../../SPWE/synset_vec.txt'
        print('-----Loading entity vector-----')
        file_path = entity2vec_file
        with open(file_path, encoding = 'utf-8') as file:
            for line in file:
                line = line.strip().split()
                if len(line) == 2:
                    continue
                else:
                    synset = line[0]
                    vec = [float(val) for val in line[1:]]
                    self.entity2vec_dict[synset] = np.array(vec)

        cnt = 0
        for synset, vec in self.entity2vec_dict.items():
            if synset in self.entity_dict.keys():
                self.embedding_array[self.entity_dict[synset]] = vec
                cnt += 1
        print('cnt:', cnt)

        bound = 6 / math.sqrt(300)
        for i in range(len(self.entity_dict.keys())):
            if i not in self.embedding_array.keys():
                self.embedding_array[i] = np.random.uniform(-bound,bound,size=300)

        self.embedding_array = np.array(list(self.embedding_array.values()), dtype=np.float32)

        print('#entity2vec: {}'.format(self.embedding_array.shape))
        return self.embedding_array
        


    def next_raw_batch(self, batch_size):
        rand_idx = np.random.permutation(self.n_training_triple)
        rand_idx_ss = np.random.permutation(self.n_training_triple_ss)
        start = 0
        start_ss = 0
        while start < self.n_training_triple:
            if start_ss == self.n_training_triple_ss:
                start_ss = 0
            end = min(start + batch_size, self.n_training_triple)
            end_ss = min(int(start_ss + batch_size / 15), self.n_training_triple_ss)
            batch = [self.training_triples[i] for i in rand_idx[start:end]]
            ss_batch = [self.training_synset_sememes[i] for i in rand_idx_ss[start_ss:end_ss]]
            


            yield (batch, ss_batch)

            start = end

    def generate_training_batch(self, in_queue, out_queue):
        while True:
            raw_batch = in_queue.get()
            if raw_batch is None:
                return
            else:
                batch_pos, ss_batch = raw_batch
                batch_neg = []
                #corrupt_head_prob = np.random.binomial(1, 0.5)
                # #只产生尾负例
                corrupt_head_prob = 0
                for head, tail, relation in batch_pos:
                    head_neg = head
                    tail_neg = tail
                    while True:
                        if corrupt_head_prob:
                            head_neg = random.choice(self.entities)
                        else:
                            tail_neg = random.choice(self.entities)
                        if (head_neg, tail_neg, relation) not in self.training_triple_pool:
                            break
                    batch_neg.append((head_neg, tail_neg, relation))
                out_queue.put((batch_pos, batch_neg, ss_batch))

    def init_norm_Vector(relinit, entinit, embedding_size):
        lstent = []
        lstrel = []
        with open(relinit) as f:
            for line in f:
                tmp = [float(val) for val in line.strip().split()]
                # if np.linalg.norm(tmp) > 1:
                #     tmp = tmp / np.linalg.norm(tmp)
                lstrel.append(tmp)
        with open(entinit) as f:
            for line in f:
                tmp = [float(val) for val in line.strip().split()]
                # if np.linalg.norm(tmp) > 1:
                #     tmp = tmp / np.linalg.norm(tmp)
                lstent.append(tmp)
        assert embedding_size % len(lstent[0]) == 0
        return np.array(lstent, dtype=np.float32), np.array(lstrel, dtype=np.float32)