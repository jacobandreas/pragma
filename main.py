#!/usr/bin/env python2

import adadelta
import corpus
from indices import WORD_INDEX
import util

from apollocaffe import ApolloNet
from apollocaffe.layers import *
import numpy as np

EPOCHS = 1000
BATCH_SIZE = 100
ALTERNATIVES = 1

N_PROP_TYPES = 8
N_OBJECT_TYPES = 35

SCENE_EMBEDDING_SIZE = 50
WORD_EMBEDDING_SIZE = 100
HIDDEN_SIZE = 100

class SpeakerModel(object):
    def __init__(self, net):
        self.net = net

    def encode_scenes(self, scenes, prefix):
        net = self.net

        max_props = max(len(scene.props) for scene in scenes)
        type_data = np.zeros((len(scenes), max_props))
        object_data = np.zeros((len(scenes), max_props))
        feature_data = np.zeros((len(scenes), max_props, 4))
        for i_scene, scene in enumerate(scenes):
            offset = max_props - len(scene.props)
            for i_prop, prop in enumerate(scene.props):
                type_data[i_scene, offset+i_prop] = prop.type_index
                object_data[i_scene, offset+i_prop] = prop.object_index
                feature_data[i_scene, offset+i_prop, :] = \
                        [prop.x, prop.y, prop.z, prop.flip]

        l_type = "SpeakerScene_%s_type_%d"
        l_object = "SpeakerScene_%s_object_%d"
        l_feature = "SpeakerScene_%s_feature_%d"
        l_type_vec = "SpeakerScene_%s_type_vec_%d"
        l_object_vec = "SpeakerScene_%s_object_vec_%d"
        l_prop = "SpeakerScene_%s_prop_%d"
        l_all_props = "SpeakerScene_%s_all_props" % prefix
        l_ip1 = "SpeakerScene_%s_ip1" % prefix
        l_relu1 = "SpeakerScene_%s_relu1" % prefix
        l_ip2 = "SpeakerScene_%s_ip2" % prefix

        p_type_vec = ["SpeakerScene_type_vec"]
        p_object_vec = ["SpeakerScene_object_vec"]
        p_ip1 = ["SpeakerScene_ip1_weight", "SpeakerScene_ip1_bias"]
        p_ip2 = ["SpeakerScene_ip2_weight", "SpeakerScene_ip2_bias"]

        for i_step in range(max_props):
            l_type_i = l_type % (prefix, i_step)
            l_object_i = l_object % (prefix, i_step)
            l_feature_i = l_feature % (prefix, i_step)
            l_type_vec_i = l_type_vec % (prefix, i_step)
            l_object_vec_i = l_object_vec % (prefix, i_step)
            l_prop_i = l_prop % (prefix, i_step)

            net.f(NumpyData(l_type_i, type_data[:, i_step]))
            net.f(NumpyData(l_object_i, object_data[:, i_step]))
            net.f(NumpyData(l_feature_i, feature_data[:, i_step]))
            net.f(Wordvec(
                    l_type_vec_i, SCENE_EMBEDDING_SIZE, N_PROP_TYPES,
                    bottoms=[l_type_i], param_names=p_type_vec))
            net.f(Wordvec(
                    l_object_vec_i, SCENE_EMBEDDING_SIZE, N_OBJECT_TYPES,
                    bottoms=[l_object_i], param_names=p_object_vec))
            net.f(Concat(
                    l_prop_i, 
                    bottoms=[l_type_vec_i, l_object_vec_i, l_feature_i]))

        net.f(Eltwise(
                l_all_props, "MAX", bottoms=[l_prop % (prefix, i_step)
                                             for i_step in range(max_props)]))

        net.f(InnerProduct(
                l_ip1, HIDDEN_SIZE, bottoms=[l_all_props], param_names=p_ip1))
        #net.f(ReLU(l_relu1, bottoms=[l_ip1]))
        #net.f(InnerProduct(
        #        l_ip2, HIDDEN_SIZE, bottoms=[l_relu1], param_names=p_ip2))

        return l_ip1

    def decode_descriptions(self, scenes, scene_encodings):
        net = self.net

        max_words = max(len(scene.description) for scene in scenes)
        word_data = np.zeros((len(scenes), max_words))
        for i_scene, scene in enumerate(scenes):
            offset = max_words - len(scene.description)
            for i_word, word in enumerate(scene.description):
                word_data[i_scene, offset+i_word] = word

        l_seed = "SpeakerDesc_seed"
        l_prev_word = "SpeakerDesc_word_%d"
        l_word_vec = "SpeakerDesc_word_vec_%d"
        l_concat = "SpeakerDesc_concat_%d"
        l_lstm = "SpeakerDesc_lstm_%d"
        l_hidden = "SpakerDesc_hidden_%d"
        l_mem = "SpeakerDesc_mem_%d"
        l_output = "SpeakerDesc_output_%d"
        l_target = "SpeakerDesc_target_%d"
        l_loss = "SpeakerDesc_loss_%d"

        p_word_vec = ["SpeakerDesc_word_vec"]
        p_lstm = ["SpeakerDesc_lstm_iv", "SpeakerDesc_lstm_ig",
                  "SpeakerDesc_lstm_fg", "SpeakerDesc_lstm_og"]
        p_output = ["SpeakerDesc_output_weight", "SpeakerDesc_output_bias"]

        loss = 0
        net.f(NumpyData(l_seed, np.zeros((len(scenes), HIDDEN_SIZE))))
        for i_step in range(1, max_words):
            l_prev_word_i = l_prev_word % i_step
            l_word_vec_i = l_word_vec % i_step
            l_concat_i = l_concat % i_step
            l_lstm_i = l_lstm % i_step
            l_hidden_i = l_hidden % i_step
            l_mem_i = l_mem % i_step
            l_output_i = l_output % i_step
            l_target_i = l_target % i_step
            l_loss_i = l_loss % i_step

            if i_step == 1:
                prev_hidden = l_seed
                prev_mem = l_seed
            else:
                prev_hidden = l_hidden % (i_step - 1)
                prev_mem = l_mem % (i_step - 1)


            net.f(NumpyData(l_prev_word_i, word_data[:, i_step-1]))
            net.f(Wordvec(
                    l_word_vec_i, WORD_EMBEDDING_SIZE, len(WORD_INDEX),
                    bottoms=[l_prev_word_i], param_names=p_word_vec))
            net.f(Concat(l_concat_i, bottoms=[prev_hidden, l_word_vec_i, scene_encodings]))
            net.f(LstmUnit(
                    l_lstm_i, bottoms=[l_concat_i, prev_mem],
                    param_names=p_lstm, tops=[l_hidden_i, l_mem_i],
                    num_cells=HIDDEN_SIZE))
            net.f(NumpyData(l_target_i, word_data[:, i_step]))
            net.f(InnerProduct(
                    l_output_i, len(WORD_INDEX), bottoms=[l_hidden_i],
                    param_names=p_output))
            loss += net.f(SoftmaxWithLoss(
                    l_loss_i, ignore_label=0, bottoms=[l_output_i, l_target_i]))
        return loss

    def sample_descriptions(self, scene_encodings):
        net = self.net

        max_words = 20

        l_seed = "SpeakerDesc_seed"
        l_prev_word = "SpeakerDesc_word_%d"
        l_word_vec = "SpeakerDesc_word_vec_%d"
        l_concat = "SpeakerDesc_concat_%d"
        l_lstm = "SpeakerDesc_lstm_%d"
        l_hidden = "SpakerDesc_hidden_%d"
        l_mem = "SpeakerDesc_mem_%d"
        l_output = "SpeakerDesc_output_%d"

        p_word_vec = ["SpeakerDesc_word_vec"]
        p_lstm = ["SpeakerDesc_lstm_iv", "SpeakerDesc_lstm_ig",
                  "SpeakerDesc_lstm_fg", "SpeakerDesc_lstm_og"]
        p_output = ["SpeakerDesc_output_weight", "SpeakerDesc_output_bias"]

        batch_size = net.blobs[scene_encodings].shape[0]
        samples = np.zeros((batch_size, max_words))
        samples[:,0] = WORD_INDEX["<s>"]

        loss = 0
        net.f(NumpyData(l_seed, np.zeros((batch_size, HIDDEN_SIZE))))
        for i_step in range(1, max_words):
            l_prev_word_i = l_prev_word % i_step
            l_word_vec_i = l_word_vec % i_step
            l_concat_i = l_concat % i_step
            l_lstm_i = l_lstm % i_step
            l_hidden_i = l_hidden % i_step
            l_mem_i = l_mem % i_step
            l_output_i = l_output % i_step

            if i_step == 1:
                prev_hidden = l_seed
                prev_mem = l_seed
            else:
                prev_hidden = l_hidden % (i_step - 1)
                prev_mem = l_mem % (i_step - 1)

            net.f(NumpyData(l_prev_word_i, samples[:,i_step-1]))
            net.f(Wordvec(
                    l_word_vec_i, WORD_EMBEDDING_SIZE, len(WORD_INDEX),
                    bottoms=[l_prev_word_i], param_names=p_word_vec))
            net.f(Concat(l_concat_i, bottoms=[prev_hidden, l_word_vec_i, scene_encodings]))
            net.f(LstmUnit(
                    l_lstm_i, bottoms=[l_concat_i, prev_mem],
                    param_names=p_lstm, tops=[l_hidden_i, l_mem_i],
                    num_cells=HIDDEN_SIZE))
            net.f(InnerProduct(
                    l_output_i, len(WORD_INDEX), bottoms=[l_hidden_i],
                    param_names=p_output))
            choices = np.argmax(net.blobs[l_output_i].data, axis=1)
            samples[:, i_step] = choices

        nl_samples = []
        for i in range(samples.shape[0]):
            this_sample = []
            for j in range(samples.shape[1]):
                word = WORD_INDEX.get(samples[i,j])
                this_sample.append(word)
                if word == "</s>":
                    break
            nl_samples.append(this_sample)
        return nl_samples

    def forward(self, scenes, update=False):
        self.net.clear_forward()

        scene_encodings = self.encode_scenes(scenes, "")
        loss = self.decode_descriptions(scenes, scene_encodings)

        if update:
            self.net.backward()
            adadelta.update(self.net, self.opt_state, self.opt_config)

        return loss

    def sample(self, scenes):
        self.net.clear_forward()
        scene_encodings = self.encode_scenes(scenes, "")
        samples = self.sample_descriptions(scene_encodings)
        return samples

    def train(self, scenes):
        self.opt_state = adadelta.State()
        self.opt_config = util.Struct(rho=0.95, eps=0.00001, lr=1, clip=10)

        #scenes = scenes[:100]

        for i_epoch in range(EPOCHS):
            np.random.shuffle(scenes)
            loss = 0
            for i_batch in range(len(scenes) / BATCH_SIZE):
                batch_start = i_batch * BATCH_SIZE
                batch_scenes = scenes[batch_start:batch_start+BATCH_SIZE]
                loss += self.forward(batch_scenes, update=True)
            print loss

            samples = self.sample(scenes[:BATCH_SIZE])
            self.visualize(scenes, samples)

    def visualize(self, scenes, samples):
        with open("vis.html", "w") as vis_f:
            print >>vis_f, "<html><body><table>"
            for i in range(100):
                print >>vis_f, "<tr><td>"
                print >>vis_f, " ".join(samples[i][1:-1])
                print >>vis_f, "</td><td>"
                print >>vis_f, " ".join([WORD_INDEX.get(w) for w in scenes[i].description[1:-1]])
                print >>vis_f, "</td><td>"
                image_name = "data/abstract/RenderedScenes/Scene%s.png" % scenes[i].image_id
                print >>vis_f, "<img src='%s' />" % image_name
                print >>vis_f, "</td></tr>"
            print >>vis_f, "</table></body></html>"


class ListenerModel(object):
    def __init__(self, net):
        self.net = net

    def forward(self, true_scenes, alt_scenes, update=False):
        self.net.clear_forward()

        true_scene_encodings = self.encode_scenes(true_scenes, "true")
        alt_scene_encodings = [self.encode_scenes(alt_scenes[i], "alt_%d" % i)
                               for i in range(len(alt_scenes))]
        desc_encodings = self.encode_descriptions(true_scenes)
        batch_loss, batch_acc = self.score_similarity(desc_encodings, true_scene_encodings, alt_scene_encodings)

        if update:
            self.net.backward()
            adadelta.update(self.net, self.opt_state, self.opt_config)

        return batch_loss, batch_acc

    def train(self, scenes):
        #scenes = scenes[:3*BATCH_SIZE]

        train_scenes = scenes[:-1000]
        test_scenes = scenes[-1000:]

        self.opt_state = adadelta.State()
        self.opt_config = util.Struct(rho=0.95, eps=0.00001, lr=1, clip=10)

        for i_epoch in range(EPOCHS):
            loss = 0
            acc = 0
            test_acc = 0

            np.random.shuffle(train_scenes)
            for i_batch in range(len(train_scenes) / BATCH_SIZE):
                batch_start = i_batch * BATCH_SIZE
                true_scene_choices = range(batch_start, batch_start + BATCH_SIZE)
                alt_scene_choices = [np.random.choice(len(train_scenes), BATCH_SIZE) 
                                     for i_scene in range(ALTERNATIVES)]
                true_scenes = [train_scenes[i_scene] for i_scene in true_scene_choices]
                alt_scenes = [[train_scenes[i_scene] for i_scene in alt] for alt in alt_scene_choices]
                batch_loss, batch_acc = self.forward(true_scenes, alt_scenes, update=True)
                loss += batch_loss
                acc += batch_acc

            loss /= (len(train_scenes) / BATCH_SIZE)
            acc /= (len(train_scenes) / BATCH_SIZE)

            for i_batch in range(len(test_scenes) / BATCH_SIZE):
                batch_start = i_batch * BATCH_SIZE
                true_scene_choices = range(batch_start, batch_start + BATCH_SIZE)
                alt_scene_choices = [np.random.choice(len(test_scenes), BATCH_SIZE) 
                                     for i_scene in range(ALTERNATIVES)]
                true_scenes = [test_scenes[i_scene] for i_scene in true_scene_choices]
                alt_scenes = [[test_scenes[i_scene] for i_scene in alt] for alt in alt_scene_choices]
                _, batch_acc = self.forward(true_scenes, alt_scenes, update=True)
                test_acc += batch_acc

            test_acc /= (len(test_scenes) / BATCH_SIZE)
                
            if i_epoch % 1 == 0:
                print "%7.3f %7.3f %7.3f" % (loss, acc, test_acc)
    

    def encode_scenes(self, scenes, prefix):
        net = self.net

        max_props = max(len(scene.props) for scene in scenes)
        type_data = np.zeros((len(scenes), max_props))
        object_data = np.zeros((len(scenes), max_props))
        feature_data = np.zeros((len(scenes), max_props, 4))
        for i_scene, scene in enumerate(scenes):
            offset = max_props - len(scene.props)
            for i_prop, prop in enumerate(scene.props):
                type_data[i_scene, offset+i_prop] = prop.type_index
                object_data[i_scene, offset+i_prop] = prop.object_index
                feature_data[i_scene, offset+i_prop, :] = \
                        [prop.x, prop.y, prop.z, prop.flip]

        l_type = "Scene_%s_type_%d"
        l_object = "Scene_%s_object_%d"
        l_feature = "Scene_%s_feature_%d"
        l_type_vec = "Scene_%s_type_vec_%d"
        l_object_vec = "Scene_%s_object_vec_%d"
        l_prop = "Scene_%s_prop_%d"
        l_all_props = "Scene_%s_all_props" % prefix
        l_ip1 = "Scene_%s_ip1" % prefix
        l_relu1 = "Scene_%s_relu1" % prefix
        l_ip2 = "Scene_%s_ip2" % prefix

        p_type_vec = ["Scene_type_vec"]
        p_object_vec = ["Scene_object_vec"]
        p_ip1 = ["Scene_ip1_weight", "Scene_ip1_bias"]
        p_ip2 = ["Scene_ip2_weight", "Scene_ip2_bias"]

        for i_step in range(max_props):
            l_type_i = l_type % (prefix, i_step)
            l_object_i = l_object % (prefix, i_step)
            l_feature_i = l_feature % (prefix, i_step)
            l_type_vec_i = l_type_vec % (prefix, i_step)
            l_object_vec_i = l_object_vec % (prefix, i_step)
            l_prop_i = l_prop % (prefix, i_step)

            net.f(NumpyData(l_type_i, type_data[:, i_step]))
            net.f(NumpyData(l_object_i, object_data[:, i_step]))
            net.f(NumpyData(l_feature_i, feature_data[:, i_step]))
            net.f(Wordvec(
                    l_type_vec_i, SCENE_EMBEDDING_SIZE, N_PROP_TYPES,
                    bottoms=[l_type_i], param_names=p_type_vec))
            net.f(Wordvec(
                    l_object_vec_i, SCENE_EMBEDDING_SIZE, N_OBJECT_TYPES,
                    bottoms=[l_object_i], param_names=p_object_vec))
            net.f(Concat(
                    l_prop_i, 
                    bottoms=[l_type_vec_i, l_object_vec_i, l_feature_i]))

        net.f(Eltwise(
                l_all_props, "MAX", bottoms=[l_prop % (prefix, i_step)
                                             for i_step in range(max_props)]))


        net.f(InnerProduct(
                l_ip1, HIDDEN_SIZE, bottoms=[l_all_props], param_names=p_ip1))
        #net.f(ReLU(l_relu1, bottoms=[l_ip1]))
        #net.f(InnerProduct(
        #        l_ip2, HIDDEN_SIZE, bottoms=[l_relu1], param_names=p_ip2))

        return l_ip1

    def encode_descriptions(self, scenes):
        net = self.net
        
        max_words = max(len(scene.description) for scene in scenes)
        word_data = np.zeros((len(scenes), max_words))
        for i_scene, scene in enumerate(scenes):
            offset = max_words - len(scene.description)
            for i_word, word in enumerate(scene.description):
                word_data[i_scene, offset+i_word] = word

        l_word = "Desc_word_%d"
        l_word_vec = "Desc_word_vec_%d"
        l_all_words = "Desc_all_words"
        l_ip1 = "Desc_ip1"
        l_relu1 = "Desc_relu1"
        l_ip2 = "Desc_ip2"

        p_word_vec = ["Desc_word_vec"]
        p_ip1 = ["Desc_ip1_weight", "Desc_ip1_param"]
        p_ip2 = ["Desc_ip2_weight", "Desc_ip2_param"]

        # TODO LSTM
        for i_step in range(max_words):
            l_word_i = l_word % i_step
            l_word_vec_i = l_word_vec % i_step

            net.f(NumpyData(l_word_i, word_data[:, i_step]))
            net.f(Wordvec(
                    l_word_vec_i, WORD_EMBEDDING_SIZE, len(WORD_INDEX),
                    bottoms=[l_word_i], param_names=p_word_vec))

        net.f(Eltwise(
                l_all_words, "MAX", bottoms=[l_word_vec % i_step 
                                             for i_step in range(max_words)]))

        net.f(InnerProduct(
                l_ip1, HIDDEN_SIZE, bottoms=[l_all_words], param_names=p_ip1))
        #net.f(ReLU(l_relu1, bottoms=[l_ip1]))
        #net.f(InnerProduct(
        #        l_ip2, HIDDEN_SIZE, bottoms=[l_relu1], param_names=p_ip2))

        return l_ip1

    def score_similarity(self, desc_encodings, true_scene_encodings, alt_scene_encodings):
        net = self.net

        l_cat_scene = "Score_cat_scene"
        l_tile_desc = "Score_tile_desc"
        l_inv = "Score_inv"
        l_sum = "Score_sum"
        l_sq = "Score_sq"
        l_reduce = "Score_reduce"
        l_target = "Score_target"
        l_loss = "Score_loss"

        p_sum = ["Score_sum_weight", "Score_sum_bias"]

        batch_size = net.blobs[desc_encodings].shape[0]
        n_alts = len(alt_scene_encodings)
        net.blobs[desc_encodings].reshape((batch_size, HIDDEN_SIZE, 1, 1))
        net.blobs[true_scene_encodings].reshape((batch_size, HIDDEN_SIZE, 1, 1))
        for alt in alt_scene_encodings:
            net.blobs[alt].reshape((batch_size, HIDDEN_SIZE, 1, 1))

        target_data = np.zeros((batch_size,))

        net.f(Concat(
            l_cat_scene, axis=2, 
            bottoms=[true_scene_encodings]+alt_scene_encodings))
        net.f(Tile(
            l_tile_desc, axis=2, tiles=n_alts+1, bottoms=[desc_encodings]))
        
        net.f(Power(l_inv, scale=-1, bottoms=[l_tile_desc]))
        net.f(Eltwise(l_sum, "SUM", bottoms=[l_cat_scene, l_inv]))
        net.f(Power(l_sq, power=2, bottoms=[l_sum]))
        net.f(Convolution(
            l_reduce, (1,1), 1, bottoms=[l_sq], param_names=p_sum, 
            param_lr_mults=[0,0], weight_filler=Filler("constant", 1), 
            bias_filler=Filler("constant", 0)))
        net.blobs[l_reduce].reshape((batch_size, n_alts+1))
        net.f(NumpyData(l_target, target_data))
        loss = net.f(SoftmaxWithLoss(l_loss, bottoms=[l_reduce, l_target]))

        #print net.blobs[l_reduce].data
        #exit()
        #print np.argmax(net.blobs[l_reduce].data, axis=1)

        acc = (np.argmax(net.blobs[l_reduce].data, axis=1) == 0).mean()

        return loss, acc

def main():
    scenes = corpus.load()

    net = ApolloNet()
    #listener_model = ListenerModel(net)
    #listener_model.train(scenes)
    speaker_model = SpeakerModel(net)
    speaker_model.train(scenes)

if __name__ == "__main__":
    main()
