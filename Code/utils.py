import torch
import os
import numpy as np

class Utils():
    def __init__(self):
        self.data_path = "../data"
        self.train_path = "train.conll"
        self.test_path = "test.conll"
        self.dev_path = "dev.conll"
        self.embedding_file = "en-cw.txt"
        
    def get_data(self, file_path):
        dataset = []
        with open(file_path, "r") as file:
            words, tags, head, labels = [], [], [], []
            for line in file.readlines():
                line = line.strip().split("\t")
                if len(line) == 10:
                    words.append(line[1].lower())
                    tags.append(line[4].lower())
                    head.append(int(line[6]))
                    labels.append(line[7].lower())
                elif len(words) > 0:
                    dataset.append({"words":words, "tags":tags, "head":head, "labels":labels})
                    words, tags, head, labels = [], [], [], []
            if len(words) > 0:
                dataset.append({"words":words, "tags":tags, "head":head, "labels":labels})
        return dataset 

    def vectorize(self, dataset):
        vectorized_dataset = []
        # For each token in the dataset substitute it with its unique id
        for sent in dataset:    
            words, tags, labels = [], [], []
            for word in sent['words']:
                words.append(self.word2id.get(word, self.word2id['<unk>']))
            for tag in sent['tags']:
                tags.append(self.tag2id.get(tag, self.tag2id['<unk>']))
            for label in sent['labels']:
                labels.append(self.rel2id.get(label, self.rel2id['<unk>']))
            vectorized_dataset.append({"words": words, "tags": tags, "head": sent['head'], "labels":labels})
        return vectorized_dataset
    
    def get_transition(self, stack, buffer, sent):
        if len(stack) < 2:
            return self.tran2id['S'], None
        one = stack[-1]
        two = stack[-2]
        head_one = sent['head'][one-1]
        head_two = sent['head'][two-1] if two != 0 else None
        label_one = sent['labels'][one-1]
        label_two = sent['labels'][two-1] if two != 0 else None
        # Left arc
        if head_two == one:
            return self.tran2id['L-' + self.id2rel[label_two]], 'L'
        
        # Right arc
        elif head_one == two:
            check = [id for id in buffer if sent['head'][id-1] == one]
            # If dependent is head to any of the words left in the buffer
            if len(check) == 0:
                return self.tran2id['R-' + self.id2rel[label_one]], 'R'
            else:
                return self.tran2id['S'], None
        else:
            return self.tran2id['S'], None

    def get_features(self, stack, buffer, arcs, sent):

        # Get the first leftmost and second leftmost child of a word in the dependency tree
        def get_lc(i):
            dependents = []
            for h, d in arcs:
                if h == i and d < h:
                    dependents.append(d)
            dependents = sorted(dependents)
            if len(dependents) == 0:
                return None, None
            elif len(dependents) == 1: 
                return dependents[0], None
            elif len(dependents) >= 2:
                return dependents[0], dependents[1]
            
        # Get the first rightmost and second rightmost child of a word in the dependency tree
        def get_rc(i):
            dependents = []
            for h, d in arcs:
                if h == i and d > h:
                    dependents.append(d)
            dependents = sorted(dependents)
            if len(dependents) == 0:
                return None, None
            elif len(dependents) == 1: 
                return dependents[-1], None
            elif len(dependents) >= 2:
                return dependents[-1], dependents[-2]
        
        # Taking the first three words in stack and buffer if there are not present then using null word
        word_features = [self.word2id['<null>']] * (3 - len(stack)) + [sent['words'][x-1] for x in stack[-3:]]
        word_features += [sent['words'][x-1] for x in buffer[:3]] + [self.word2id['<null>']] * (3 - len(buffer))
        # Taking the corresponding pos tags for the first three words in stack and buffer
        # Null pos tag for null word
        tag_features = [self.tag2id['<null>']] * (3 - len(stack)) + [sent['tags'][x-1] for x in stack[-3:]]
        tag_features += [sent['tags'][x-1] for x in buffer[:3]] + [self.tag2id['<null>']] * (3 - len(buffer))
        # In label features the transition arc is also included
        label_features = []
        for i in range(2):
            if i < len(stack):
                word = stack[-i-1]
                lc1, lc2 = get_lc(word)
                rc1, rc2 = get_rc(word)
                llc1 = None if lc1 is None else get_lc(lc1)[0]
                rrc1 = None if rc1 is None else get_rc(rc1)[0]
                completeList = [lc1, lc2, rc1, rc2, llc1, rrc1]
                for x in completeList:
                    if x is None:
                        word_features.append(self.word2id['<null>'])
                        tag_features.append(self.tag2id['<null>'])
                        label_features.append(self.rel2id['<null>'])
                    else:
                        word_features.append(sent['words'][x-1])
                        tag_features.append(sent['tags'][x-1])
                        label_features.append(sent['labels'][x-1])
            else:
                word_features += [self.word2id['<null>']] * 6
                tag_features += [self.tag2id['<null>']] * 6
                label_features += [self.rel2id['<null>']] * 6
        return (torch.tensor(word_features), torch.tensor(tag_features), torch.tensor(label_features))

    def create_instances(self, dataset):
        inputs, outputs = [], []
        for sent in dataset:
            num_words = len(sent['words'])
            stack = [0]
            buffer = [i+1 for i in range(num_words)]
            arcs = []
            for _ in range(2 * num_words):
                transition_id, dir = self.get_transition(stack, buffer, sent)
                if transition_id == self.tran2id['S'] and len(buffer) == 0:
                    break
                features = self.get_features(stack, buffer, arcs, sent)
                inputs.append(features)
                outputs.append(torch.tensor(transition_id, dtype=torch.long))

                if transition_id == self.tran2id['S'] and len(buffer) > 0:
                    stack.append(buffer[0])
                    buffer.pop(0)
                elif dir == 'L':
                    arcs.append((stack[-1], stack[-2]))
                    stack.pop(-2)
                elif dir == 'R':
                    arcs.append((stack[-2], stack[-1]))
                    stack.pop(-1)
        return inputs, outputs
    


    def preprocessing(self):
        # Reading Data 
        train_set = self.get_data(os.path.join(self.data_path, self.train_path))
        dev_set = self.get_data(os.path.join(self.data_path, self.dev_path))
        test_set = self.get_data(os.path.join(self.data_path, self.test_path))

        # Getting unique dependency relations
        dep_rel = sorted(list(set([label for sent in train_set for label in sent['labels']])))
        dep_rel.extend(['<unk>', '<null>']) 
        rel2id = {rel: i for i, rel in enumerate(dep_rel)}

        # Getting vocabulary
        # Adding unk tag for unknown words and null which will be used while extracting features
        # Giving a unique id for each word
        tot_words = sorted(list(set([word for sent in train_set for word in sent['words']])))
        tot_words.extend(['<unk>', '<null>'])
        word2id = {word: i for i, word in enumerate(tot_words)}

        # Extracting unique pos tags
        pos_tags = sorted(list(set([tag for sent in train_set for tag in sent['tags']])))
        pos_tags.extend(['<unk>', '<null>'])
        tag2id = {tag: i for i, tag in enumerate(pos_tags)}

        # For each relation there can two possible transition left and right, there is also a shift transition
        transitions = ['L-' + rel for rel in dep_rel] + ['R-' + rel for rel in dep_rel] + ['S']

        # Technically only R-root transition is possible
        transitions.remove('L-root')
        transitions.remove('L-<null>')
        transitions.remove('R-<null>')
        tran2id = {transition: i for i, transition in enumerate(transitions)}


        self.word2id = word2id
        self.tag2id = tag2id
        self.tran2id = tran2id
        self.rel2id = rel2id
        self.id2rel = {i: rel for rel, i in rel2id.items()}
        self.id2tran = {i: t for t, i in tran2id.items()}

        vectorized_train_data = self.vectorize(train_set)
        vectorized_test_data = self.vectorize(test_set)
        vectorized_dev_data = self.vectorize(dev_set)

        train_instances = self.create_instances(vectorized_train_data)
        
        # Getting word vectors for the words in the dataset
        word_vectors = {}
        for line in open(os.path.join(self.data_path, self.embedding_file)).readlines():
            sp = line.strip().split()
            word_vectors[sp[0]] = [float(x) for x in sp[1:]]
        word_embeddings = np.asarray(np.random.random((len(self.word2id), 50)), dtype='float32')
    
        for k, v in self.word2id.items():
            if k in word_vectors:
                word_embeddings[v] = word_vectors[k]
            elif k == '<null>':
                word_embeddings[v] = word_vectors['null']
            else:
                word_embeddings[v] = word_vectors['UNKNOWN']

        return train_instances, vectorized_test_data, vectorized_dev_data, word_embeddings
    
class Sentence_Parser():
    def __init__(self, sent):
        self.sent = sent
        self.num_words = len(sent['words'])
        self.stack = [0]
        self.buffer = [i+1 for i in range(self.num_words)]
        self.arcs = []
        self.dependencies = []
    
    def parse(self, util, model):
        for _ in range(2 * self.num_words):
            w, t, l = util.get_features(self.stack, self.buffer, self.arcs, self.sent)
            model_output = model(w.unsqueeze(0), t.unsqueeze(0), l.unsqueeze(0))
            transition_id = torch.argmax(model_output[0]).item()
            dir = None
            if transition_id != util.tran2id['S']:
                dir = util.id2tran[transition_id][0]
            if transition_id == util.tran2id['S'] and len(self.buffer) == 0:
                break
            elif (dir == 'L' or dir == 'R') and len(self.stack) < 2:
                break

            if transition_id == util.tran2id['S'] and len(self.buffer) > 0:
                self.stack.append(self.buffer[0])
                self.buffer.pop(0)
            elif dir == 'L':
                self.arcs.append((self.stack[-1], self.stack[-2]))
                self.dependencies.append((self.stack[-1], self.stack[-2], transition_id))
                self.stack.pop(-2)
            elif dir == 'R':
                self.arcs.append((self.stack[-2], self.stack[-1]))
                self.dependencies.append((self.stack[-2], self.stack[-1], transition_id))
                self.stack.pop(-1)
    
    def get_pred(self, util, UAS = True):
        correct = 0
        for h, d, t in self.dependencies:
            r = util.rel2id[util.id2tran[t][2:]]
            if self.sent['head'][d-1] == h and (UAS or self.sent['labels'][d-1] == r):
                correct += 1
        total = len(self.sent['head'])
        return correct, total