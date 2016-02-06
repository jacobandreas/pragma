#!/usr/bin/env python2

import adadelta
import corpus
from indices import WORD_INDEX
from modules import *
import util

import apollocaffe
from apollocaffe import ApolloNet
import logging
import numpy as np
import shutil
import yaml

CONFIG = """
opt:
    epochs: 20
    batch_size: 100
    alternatives: 9

    rho: 0.95
    eps: 0.000001
    lr: 1
    clip: 10

model:
    prop_embedding_size: 50
    word_embedding_size: 50
    hidden_size: 100
"""


class Listener0Model(object):
    def __init__(self, apollo_net, config):
        #self.string_encoder = LinearStringEncoder("Listener0", apollo_net, config)
        #self.scene_encoder = BowSceneEncoder("Listener0", apollo_net, config)

        self.scene_encoder = LinearSceneEncoder("Listener0", apollo_net, config)
        self.string_encoder = LinearStringEncoder("Listener0", apollo_net, config)
        self.scorer = EuclideanScorer("Listener0", apollo_net, config)

    def forward(self, data, alt_data, dropout):
        l_true_scene_enc = self.scene_encoder.forward("true", data, dropout)
        ll_alt_scene_enc = \
                [self.scene_encoder.forward("alt%d" % i, alt, dropout)
                 for i, alt in enumerate(alt_data)]
        l_string_enc = self.string_encoder.forward("", data, dropout)

        ll_scenes = [l_true_scene_enc] + ll_alt_scene_enc
        labels = np.zeros((len(data),))
        losses, accs = self.scorer.forward("", l_string_enc, ll_scenes, labels)

        return losses, accs

class Speaker0Model(object):
    def __init__(self, apollo_net, config):
        self.scene_encoder = LinearSceneEncoder("Speaker0", apollo_net, config)
        self.string_decoder = MlpStringDecoder("Speaker0", apollo_net, config)
        #self.scene_encoder = BowSceneEncoder("Speaker0", apollo_net, config)
        #self.string_decoder = LstmStringDecoder("Speaker0", apollo_net, config)

        self.apollo_net = apollo_net

    def forward(self, data, alt_data, dropout):
        l_scene_enc = self.scene_encoder.forward("", data, dropout)
        loss = self.string_decoder.forward("", l_scene_enc, data, dropout)

        return loss, 0

    def sample(self, data, alt_data, dropout):
        l_scene_enc = self.scene_encoder.forward("", data, dropout)
        #print self.apollo_net.blobs[l_scene_enc].data
        sample = self.string_decoder.sample("", l_scene_enc)
        return sample
        #return self.string_decoder.sample("", l_scene_enc)

class CompiledSpeaker1Model(object):
    def __init__(self, apollo_net, config):
        self.speaker1 = SamplingSpeaker1Model(apollo_net, config)

class SamplingSpeaker1Model(object):
    def __init__(self, apollo_net, config):
        self.listener0 = Listener0Model(apollo_net, config)
        self.speaker0 = Speaker0Model(apollo_net, config)

        self.apollo_net = apollo_net

    def sample(self, data, alt_data, dropout):
        N_SAMPLES = 50

        scores = np.zeros((len(data), N_SAMPLES))

        all_fake_scenes = []
        for i_sample in range(N_SAMPLES):
            self.apollo_net.clear_forward()
            sample = self.speaker0.sample(data, alt_data, dropout)

            fake_scenes = []
            for i in range(len(data)):
                fake_scenes.append(data[i]._replace(description=sample[i]))
            all_fake_scenes.append(fake_scenes)

            self.apollo_net.clear_forward()
            probs, accs = self.listener0.forward(fake_scenes, alt_data, dropout)
            scores[:,i_sample] = probs

        for i in range(len(data)):
            q_order = np.argsort(scores[i,:])
            print all_fake_scenes[0][i].image_id
            for q in q_order[-10:]:
                print "%0.3f" % scores[i, q], 
                print " ".join([WORD_INDEX.get(w) for w in
                    all_fake_scenes[q][i].description])
            print

def train(train_scenes, test_scenes, model, apollo_net, config):
    n_train = len(train_scenes)
    n_test = len(test_scenes)

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
            batch_data = train_scenes[i_batch * config.batch_size : 
                                      (i_batch + 1) * config.batch_size]
            alt_indices = \
                    [np.random.choice(n_train, size=config.batch_size)
                     for i_alt in range(config.alternatives)]
            alt_data = [[train_scenes[i] for i in alt] for alt in alt_indices]
            
            apollo_net.clear_forward()
            loss, acc = model.forward(batch_data, alt_data, dropout=True)
            apollo_net.backward()
            adadelta.update(apollo_net, opt_state, config)
            #apollo_net.update(lr=0.01, clip_gradients=10, momentum=0.9)

            #if i_batch == 0:
            #    print apollo_net.blobs["EuclideanScorer_Listener0__reduce"].data

            e_train_loss += loss
            e_train_acc += acc

        ##with open("vis.html", "a") as vis_f:
        ##    samples = []
        ##    for i in range(5):
        ##        apollo_net.clear_forward()
        ##        sample = model.sample(batch_data, alt_data, dropout=False)
        ##        samples.append(sample)
        ##    #for sample in samples[:10]:
        ##    #    print sample
        ##    for i in range(10):
        ##        print >>vis_f, "<tr>"
        ##        print >>vis_f, "<td><img src='data/abstract/RenderedScenes/Scene%s.png'></td>" % batch_data[i].image_id
        ##        print >>vis_f, "<td>"
        ##        for j in range(5):
        ##            print >>vis_f, "%s<br/>" % " ".join(samples[j][i][1:-1])
        ##        print >>vis_f, "</td>"
        ##        print >>vis_f, "</tr>"
        ##        #print [WORD_INDEX.get(w) for w in batch_data[i].description]
        ##        #print batch_data[i].image_id
        ##        #print
        ##    print >>vis_f, "<tr style='border-bottom: 3px solid #000'><td colspan='3'></td></tr>"

        n_test_batches = n_test / config.batch_size
        for i_batch in range(n_test_batches):
            batch_data = test_scenes[i_batch * config.batch_size :
                                     (i_batch + 1) * config.batch_size]
            alt_indices = \
                    [np.random.choice(n_test, size=config.batch_size)
                     for i_alt in range(config.alternatives)]
            alt_data = [[test_scenes[i] for i in alt] for alt in alt_indices]
            
            apollo_net.clear_forward()
            loss, acc = model.forward(batch_data, alt_data, dropout=False)

            #if i_batch == 0:
            #    print apollo_net.blobs["LinearStringEncoder_Listener0__data"].data

            e_test_loss += loss
            e_test_acc += acc

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

    model.forward(data, alt_data, dropout=False)


def main():
    apollocaffe.set_device(0)
    np.random.seed(0)

    config = util.Struct(**yaml.load(CONFIG))
    scenes = corpus.load()
    np.random.shuffle(scenes)
    train_scenes = scenes[:-1000]
    test_scenes = scenes[-1000:]
    #train_scenes = scenes[:100]
    #test_scenes = scenes[-100:]
    #train_scenes = scenes[:10]
    #test_scenes = scenes[-10:]
    apollo_net = ApolloNet()

    listener0_model = Listener0Model(apollo_net, config.model)
    speaker0_model = Speaker0Model(apollo_net, config.model)

    #train(train_scenes, test_scenes, listener0_model, apollo_net, config.opt)
    #apollo_net.save("listener0_model.caffemodel")

    #train(train_scenes, test_scenes, speaker0_model, apollo_net, config.opt)
    #apollo_net.save("speaker0_model.caffemodel")

    apollo_net.load("listener0_model.caffemodel")
    apollo_net.load("speaker0_model.caffemodel")

    sampling_speaker1_model = SamplingSpeaker1Model(apollo_net, config.model)
    demo(test_scenes, sampling_speaker1_model, apollo_net, config.opt)

if __name__ == "__main__":
    main()
