#!/usr/bin/env python2

from indices import WORD_INDEX

from collections import defaultdict, namedtuple
import numpy as np
import re

Prop = namedtuple("Prop", ["type_index", "object_index", "x", "y", "z", "flip"])
Scene = namedtuple("Scene", ["image_id", "props", "description", "features"])
Bird = namedtuple("Bird", ["image_id", "description", "features"])

N_IMAGES = 10020
N_DEV_IMAGES = 1000
N_TEST_IMAGES = 1000
MIN_WORD_COUNT = 5
DEV_RANGE = range(N_IMAGES - N_TEST_IMAGES - N_DEV_IMAGES, N_IMAGES - N_TEST_IMAGES)
TEST_RANGE = range(N_IMAGES - N_TEST_IMAGES, N_IMAGES)

def load_props():
    scene_props = []
    with open("data/abstract/Scenes_10020.txt") as scene_f:
        scene_f.readline()
        while True:
            line = scene_f.readline().strip()
            if not line:
                break
            length = line.split()[1]
            length = int(length)
            props = []
            for i_object in range(length):
                line = scene_f.readline().strip()
                parts = line.split()[1:]
                parts = [int(p) for p in parts]
                props.append(Prop(*parts))

            scene_props.append(props)

    return scene_props

def normalize_props(scene_props):
    feats = np.zeros(4)
    feats_sq = np.zeros(4)
    count = 0

    for props in scene_props:
        for prop in props:
            feats_here = np.asarray([prop.x, prop.y, prop.z, prop.flip])
            feats += feats_here
            feats_sq += feats_here ** 2
            count += 1

    mean = feats / count
    std = np.sqrt(feats_sq / count - mean ** 2)
    assert (std > 0).all()

    norm_scene_props = []
    for props in scene_props:
        new_props = []
        for prop in props:
            prop_feats = np.asarray([prop.x, prop.y, prop.z, prop.flip], dtype=float)
            prop_feats -= mean
            prop_feats /= std
            x, y, z, flip = prop_feats
            new_prop = Prop(prop.type_index, prop.object_index, x, y, z, flip)
            new_props.append(new_prop)
        norm_scene_props.append(new_props)

    return norm_scene_props

def load_scenes(scene_props):
    scenes = []

    word_counter = defaultdict(lambda: 0)
    for sent_file_id in range(1, 3):
        with open("data/abstract/SimpleSentences/SimpleSentences%d_10020.txt" %
                sent_file_id) as sent_f:
            for sent_line in sent_f:
                sent_parts = sent_line.strip().split("\t")
                sent = sent_parts[2]
                sent = sent.replace('"', ' " ')
                sent = sent.replace("'", " ' ")
                sent = re.sub(r"[.?!]", "", sent)
                words = sent.lower().split()
                words = ["<s>"] + words + ["</s>"]
                for word in words:
                    word_counter[word] += 1
    for word, count in word_counter.items():
        if count >= MIN_WORD_COUNT:
            WORD_INDEX.index(word)

    for sent_file_id in range(1, 3):
        with open("data/abstract/SimpleSentences/SimpleSentences%d_10020.txt" %
                sent_file_id) as sent_f:
            for sent_line in sent_f:
                sent_parts = sent_line.strip().split("\t")

                scene_id = int(sent_parts[0])
                props = scene_props[scene_id]

                sent_id = int(sent_parts[1])
                image_id = scene_id / 10
                image_subid = scene_id % 10
                image_strid = "%d_%d" % (image_id, image_subid)

                sent = sent_parts[2]
                sent = sent.replace('"', "")
                sent = re.sub(r"[.?!']", "", sent)
                words = sent.lower().split()
                words = ["<s>"] + words + ["</s>"]
                word_ids = [WORD_INDEX[w] or 0 for w in words]

                with np.load("data/abstract/EmbeddedScenes/Scene%s.png.npz" %
                        image_strid) as feature_f:
                    features = feature_f[feature_f.keys()[0]]
                scenes.append(Scene(image_strid, props, word_ids, features))

    return scenes

def load_abstract():
    props = load_props()
    norm_props = normalize_props(props)
    scenes = load_scenes(norm_props)
    train_scenes = []
    dev_scenes = []
    test_scenes = []
    for scene in scenes:
        raw_id = int(scene.image_id.replace("_", ""))
        if raw_id in DEV_RANGE:
            dev_scenes.append(scene)
        elif raw_id in TEST_RANGE:
            test_scenes.append(scene)
        else:
            train_scenes.append(scene)
    return train_scenes, dev_scenes, test_scenes

def load_birds():
    birds = []
    feats = np.zeros(4096)
    feats_sq = np.zeros(4096)


    word_counter = defaultdict(lambda: 0)
    with open("data/birds/cub_0917_5cap.tsv") as caption_f:
        for line in caption_f:
            parts = line.strip().split("\t")
            caption = parts[-1]
            caption = (caption.lower()
                              .replace(".", "")
                              .replace(",", " , "))
            words = ["<s>"] + caption.split() + ["</s>"]
            for word in words:
                word_counter[word] += 1
    for word, count in word_counter.items():
        if count >= MIN_WORD_COUNT:
            WORD_INDEX.index(word)

    with open("data/birds/cub_0917_5cap.tsv") as caption_f:
        caption_f.readline()
        for line in caption_f:
            parts = line.strip().split("\t")
            caption = parts[-1]
            image_path = parts[-2]
            image_id = image_path.split("/")[-1]

            caption = (caption.lower()
                              .replace(".", "")
                              .replace(",", " , "))
            words = ["<s>"] + caption.split() + ["</s>"]
            word_ids = [WORD_INDEX[w] for w in words]

            with np.load("data/birds/embeddings/%s.npz" % image_id) as feature_f:
                features = feature_f[feature_f.keys()[0]]

            birds.append(Bird(image_id, word_ids, features))

            feats += features
            feats_sq += features ** 2

    mean_feats = feats / len(birds)
    mean_feats_sq = feats_sq / len(birds)
    var_feats = mean_feats_sq - (mean_feats ** 2)
    std_feats = np.sqrt(var_feats)
    std_feats += 0.0001

    for bird in birds:
        bird.features[...] -= mean_feats
        bird.features[...] /= std_feats

    train_birds = birds[:-1100]
    val_birds = birds[-1100:-100]
    test_birds = birds[-100:]

    return train_birds, val_birds, test_birds

            #print image_id, caption
