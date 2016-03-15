#!/usr/bin/env python2

import adadelta
import corpus
from indices import WORD_INDEX
from modules import *
import util

import apollocaffe
from apollocaffe import ApolloNet
from apollocaffe.layers import Concat
from collections import defaultdict
import itertools
import logging
import numpy as np
import shutil
import sys
import yaml

CONFIG = """
opt:
    epochs: 10
    batch_size: 100
    alternatives: 1

    rho: 0.95
    eps: 0.000001
    lr: 1
    clip: 10

model:
    prop_embedding_size: 50
    word_embedding_size: 50
    hidden_size: 100
"""

N_TEST_IMAGES = 100
N_TEST = N_TEST_IMAGES * 10

N_EXPERIMENT_PAIRS = 100

class Listener0Model(object):
    def __init__(self, apollo_net, config):
        self.scene_encoder = LinearSceneEncoder("Listener0", apollo_net, config)
        self.string_encoder = LinearStringEncoder("Listener0", apollo_net, config)
        self.scorer = MlpScorer("Listener0", apollo_net, config)
        self.apollo_net = apollo_net

    def forward(self, data, alt_data, dropout):
        self.apollo_net.clear_forward()
        l_true_scene_enc = self.scene_encoder.forward("true", data, dropout)
        ll_alt_scene_enc = \
                [self.scene_encoder.forward("alt%d" % i, alt, dropout)
                 for i, alt in enumerate(alt_data)]
        l_string_enc = self.string_encoder.forward("", data, dropout)

        ll_scenes = [l_true_scene_enc] + ll_alt_scene_enc
        labels = np.zeros((len(data),))
        logprobs, accs = self.scorer.forward("", l_string_enc, ll_scenes, labels)

        return logprobs, accs

class Speaker0Model(object):
    def __init__(self, apollo_net, config):
        self.scene_encoder = LinearSceneEncoder("Speaker0", apollo_net, config)
        self.string_decoder = MlpStringDecoder("Speaker0", apollo_net, config)

        self.apollo_net = apollo_net

    def forward(self, data, alt_data, dropout):
        self.apollo_net.clear_forward()
        l_scene_enc = self.scene_encoder.forward("", data, dropout)
        losses = self.string_decoder.forward("", l_scene_enc, data, dropout)

        return losses, np.asarray(0)

    def sample(self, data, alt_data, dropout, viterbi, quantile=None):
        self.apollo_net.clear_forward()
        l_scene_enc = self.scene_encoder.forward("", data, dropout)
        probs, sample = self.string_decoder.sample("", l_scene_enc, viterbi)
        return probs, np.zeros(probs.shape), sample

class CompiledSpeaker1Model(object):
    def __init__(self, apollo_net, config):
        self.sampler = SamplingSpeaker1Model(apollo_net, config)
        self.scene_encoder = LinearSceneEncoder("CompSpeaker1Model", apollo_net, config)
        self.string_decoder = MlpStringDecoder("CompSpeaker1Model", apollo_net, config)

        self.config = config
        self.apollo_net = apollo_net

    def forward(self, data, alt_data, dropout):
        self.apollo_net.clear_forward()
        _, _, samples = self.sampler.sample(data, alt_data, dropout, True)

        l_true_scene_enc = self.scene_encoder.forward("true", data, dropout)
        ll_alt_scene_enc = \
                [self.scene_encoder.forward("alt%d" % i, alt, dropout)
                 for i, alt in enumerate(alt_data)]
        l_cat = "CompSpeaker1Model_concat"
        l_ip = "CompSpeaker1Model_ip"
        l_relu = "CompSpeaker1Model_relu"
        self.apollo_net.f(Concat(
            l_cat, bottoms=[l_true_scene_enc] + ll_alt_scene_enc))
        self.apollo_net.f(InnerProduct(
            l_ip, self.config.hidden_size, bottoms=[l_cat]))
        self.apollo_net.f(ReLU(l_relu, bottoms=[l_ip]))

        fake_data = [d._replace(description=s) for d, s in zip(data, samples)]

        losses = self.string_decoder.forward("", l_relu, fake_data, dropout)
        return losses, np.asarray(0)

    def sample(self, data, alt_data, dropout, viterbi, quantile=None):
        self.apollo_net.clear_forward()
        l_true_scene_enc = self.scene_encoder.forward("true", data, dropout)
        ll_alt_scene_enc = \
                [self.scene_encoder.forward("alt%d" % i, alt, dropout)
                        for i, alt in enumerate(alt_data)]
        l_cat = "CompSpeakerModel1_concat"
        l_ip = "CompSpeaker1Model_ip"
        l_relu = "CompSpeaker1Model_relu"
        self.apollo_net.f(Concat(
            l_cat, bottoms=[l_true_scene_enc] + ll_alt_scene_enc))
        self.apollo_net.f(InnerProduct(
            l_ip, self.config.hidden_size, bottoms=[l_cat]))
        self.apollo_net.f(ReLU(l_relu, bottoms=[l_ip]))

        probs, sample = self.string_decoder.sample("", l_relu, viterbi)
        return probs, np.zeros(probs.shape), sample


class SamplingSpeaker1Model(object):
    def __init__(self, apollo_net, config):
        self.listener0 = Listener0Model(apollo_net, config)
        self.speaker0 = Speaker0Model(apollo_net, config)

        self.apollo_net = apollo_net

    def sample(self, data, alt_data, dropout, viterbi, quantile=None):
        LAMBDA = 0.02
        #LAMBDA = 1
        N_SAMPLES = 100
        self.apollo_net.clear_forward()
        if viterbi or quantile is not None:
            n_samples = N_SAMPLES
        else:
            n_samples = 1

        speaker_scores = np.zeros((len(data), n_samples))
        listener_scores = np.zeros((len(data), n_samples))

        all_fake_scenes = []
        for i_sample in range(n_samples):
            speaker_logprobs, _, sample = self.speaker0.sample(data, alt_data, dropout, viterbi=False)

            fake_scenes = []
            for i in range(len(data)):
                fake_scenes.append(data[i]._replace(description=sample[i]))
            all_fake_scenes.append(fake_scenes)

            listener_logprobs, accs = self.listener0.forward(fake_scenes, alt_data, dropout)
            speaker_scores[:,i_sample] = speaker_logprobs
            listener_scores[:,i_sample] = listener_logprobs

        #scores = listener_scores
        scores = LAMBDA * speaker_scores + (1 - LAMBDA) * listener_scores

        out_sentences = []
        out_speaker_scores = np.zeros(len(data))
        out_listener_scores = np.zeros(len(data))
        for i in range(len(data)):
            if viterbi:
                q = scores[i,:].argmax()
            elif quantile is not None:
                idx = int(n_samples * quantile)
                if idx == n_samples:
                    q = scores.argmax()
                else:
                    q = scores[i,:].argsort()[idx]
            else:
                q = 0
            out_sentences.append(all_fake_scenes[q][i].description)
            out_speaker_scores[i] = speaker_scores[i][q]
            out_listener_scores[i] = listener_scores[i][q]

        return out_speaker_scores, out_listener_scores, out_sentences

def props(datum):
    props = {(p.type_index, p.object_index) for p in datum.props}
    props = {(i,j) if i not in (2,3) else (i,0) for i,j in props}
    return props

def similar(sprops, alt_props):
    out = []
    while len(out) < 5:
        r = np.random.randint(len(alt_props))
        if len(sprops ^ alt_props[r]) <= 4:
            out.append(r)
    return out

def train(train_scenes, test_scenes, model, apollo_net, config):
    n_train = len(train_scenes)
    n_test = len(test_scenes)

    train_props = [props(d) for d in train_scenes]
    test_props = [props(d) for d in test_scenes]

    train_similar = [similar(p, train_props) for p in train_props]
    test_similar = [similar(p, test_props) for p in test_props]

    opt_state = adadelta.State()
    for i_epoch in range(config.epochs):

        with open("vis.html", "w") as vis_f:
            print >>vis_f, "<html><body><table>"

        np.random.shuffle(train_scenes)

        e_train_loss = 0
        e_train_acc = 0
        e_test_loss = 0
        e_test_acc = 0

        n_train_batches = n_train / config.batch_size
        for i_batch in range(n_train_batches):
            if i_batch % (n_train_batches / 10) == 0:
                print ".",
            batch_data = train_scenes[i_batch * config.batch_size : 
                                      (i_batch + 1) * config.batch_size]
            batch_similar = train_similar[i_batch * config.batch_size : 
                                          (i_batch + 1) * config.batch_size]
            alt_indices = \
                    [np.random.choice(n_train, size=config.batch_size)
                     for i_alt in range(config.alternatives)]
            #alt_indices = [[np.random.choice(s) for s in batch_similar] for i in range(config.alternatives)]
            alt_data = [[train_scenes[i] for i in alt] for alt in alt_indices]
            
            #apollo_net.clear_forward()
            lls, accs = model.forward(batch_data, alt_data, dropout=True)
            apollo_net.backward()
            adadelta.update(apollo_net, opt_state, config)

            e_train_loss -= lls.sum()
            e_train_acc += accs.sum()
        print

        n_test_batches = n_test / config.batch_size
        for i_batch in range(n_test_batches):
            batch_data = test_scenes[i_batch * config.batch_size :
                                     (i_batch + 1) * config.batch_size]
            batch_similar = test_similar[i_batch * config.batch_size : 
                                          (i_batch + 1) * config.batch_size]

            alt_indices = \
                    [np.random.choice(n_test, size=config.batch_size)
                     for i_alt in range(config.alternatives)]
            #alt_indices = [[np.random.choice(s) for s in batch_similar] for i in range(config.alternatives)]
            alt_data = [[test_scenes[i] for i in alt] for alt in alt_indices]
            
            lls, accs = model.forward(batch_data, alt_data, dropout=False)

            e_test_loss -= lls.sum()
            e_test_acc += accs.sum()

        with open("vis.html", "a") as vis_f:
            print >>vis_f, "</table></body></html>"

        shutil.copyfile("vis.html", "vis2.html")

        e_train_loss /= n_train_batches
        e_train_acc /= n_train_batches
        e_test_loss /= n_test_batches
        e_test_acc /= n_test_batches

        print "%5.3f  (%5.3f)  :  %5.3f  (%5.3f)" % (
                e_train_loss, e_train_acc, e_test_loss, e_test_acc)


def demo(scenes, model, apollo_net, config):
    data = scenes[:config.batch_size]
    alt_indices = \
            [np.random.choice(len(scenes), size=config.batch_size)
             for i_alt in range(config.alternatives)]
    alt_data = [[scenes[i] for i in alt] for alt in alt_indices]

    _, samples = model.sample(data, alt_data, dropout=False)
    for i in range(10):
        sample = samples[i]
        print data[i].image_id
        print " ".join([WORD_INDEX.get(i) for i in sample])
        print

def run_experiment(name, cname, rname, models, data):
    data_by_image = defaultdict(list)
    for datum in data:
        data_by_image[datum.image_id].append(datum)

    with open("experiments/%s/%s.ids.txt" % (name, cname)) as id_f, \
         open("experiments/%s/%s.results.%s.txt" % (name, cname, rname), "w") as results_f:
        print >>results_f, "id,target,distractor,similarity,model_name,speaker_score,listener_score,description"
        counter = 0
        for line in id_f:
            img1, img2, similarity = line.strip().split(",")
            assert img1 in data_by_image and img2 in data_by_image
            d1 = data_by_image[img1][0]
            d2 = data_by_image[img2][0]
            for model_name, model in models.items():
                #for i_sample in range(10):
                speaker_scores, listener_scores, samples = \
                        model.sample([d1], [[d2]], dropout=False, viterbi=True)
                parts = [
                    counter,
                    img1,
                    img2,
                    similarity,
                    model_name,
                    speaker_scores[0],
                    listener_scores[0],
                    " ".join([WORD_INDEX.get(i) for i in samples[0][1:-1]])
                ]
                print >>results_f, ",".join([str(s) for s in parts])
                counter += 1

def main():
    apollocaffe.set_device(0)
    #apollocaffe.set_cpp_loglevel(0)
    apollocaffe.set_random_seed(0)
    np.random.seed(0)

    job = sys.argv[1]
    corpus_name = sys.argv[2]

    config = util.Struct(**yaml.load(CONFIG))
    if corpus_name == "abstract":
        train_scenes, dev_scenes, test_scenes = corpus.load_abstract()
    else:
        assert corpus_name == "birds"
        train_scenes, dev_scenes, test_scenes = corpus.load_birds()
    apollo_net = ApolloNet()
    print "loaded data"
    print "%d training examples" % len(train_scenes)

    listener0_model = Listener0Model(apollo_net, config.model)
    speaker0_model = Speaker0Model(apollo_net, config.model)
    sampling_speaker1_model = SamplingSpeaker1Model(apollo_net, config.model)
    compiled_speaker1_model = CompiledSpeaker1Model(apollo_net, config.model)

    if job == "train.base":
        train(train_scenes, dev_scenes, listener0_model, apollo_net, config.opt)
        train(train_scenes, dev_scenes, speaker0_model, apollo_net, config.opt)
        apollo_net.save("models/%s.base.caffemodel" % corpus_name)
        exit()

    elif job == "train.compiled":
        apollo_net.load("models/%s.base.caffemodel" % corpus_name)
        print "loaded model"
        train(train_scenes, dev_scenes, compiled_speaker1_model, apollo_net,
                config.opt)
        apollo_net.save("models/%s.compiled.caffemodel" % corpus_name)
        exit()

    elif job in ("sample.reasoning", "sample.compiled"):
        if job == "sample.reasoning":
            apollo_net.load("models/%s.base.caffemodel" % corpus_name)
        else:
            apollo_net.load("models/%s.compiled.caffemodel" % corpus_name)
        print "loaded model"
        if job == "sample.reasoning":
            models = {
                "sampling_speaker1": sampling_speaker1_model,
            }
        elif job == "sample.compiled":
            models = {
                "compiled_speaker1": compiled_speaker1_model,
            }

        name = job.split(".")[1]
        #name = "literal_speaker"

        run_experiment("dev/one_different", corpus_name, name, models, dev_scenes)
        run_experiment("dev/by_similarity", corpus_name, name, models, dev_scenes)
        #run_experiment("all_same", corpus_name, name, models, dev_scenes)
        exit()

    assert False

if __name__ == "__main__":
    main()
