#!/usr/bin/env python2

from indices import WORD_INDEX

from collections import namedtuple
import numpy as np
import re

Prop = namedtuple("Prop", ["type_index", "object_index", "x", "y", "z", "flip"])
Scene = namedtuple("Scene", ["image_id", "props", "description"])

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

            props = props[:2]
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
    with open("data/abstract/SimpleSentences/SimpleSentences1_10020.txt") as sent_f:
        for sent_line in sent_f:
            sent_parts = sent_line.strip().split("\t")

            scene_id = int(sent_parts[0])
            props = scene_props[scene_id]

            sent_id = int(sent_parts[1])
            image_id = scene_id / 10
            image_subid = scene_id % 10
            image_strid = "%d_%d" % (image_id, image_subid)

            #if image_subid > 1:
            #    continue

            #if sent_id > 0:
            #    continue

            sent = sent_parts[2]
            sent = sent.replace('"', "")
            sent = re.sub(r"[.?!']", "", sent)
            words = sent.lower().split()
            words = ["<s>"] + words + ["</s>"]
            word_ids = [WORD_INDEX.index(w) for w in words]

            scenes.append(Scene(image_strid, props, word_ids))

    return scenes

def load():
    props = load_props()
    norm_props = normalize_props(props)
    scenes = load_scenes(norm_props)
    return scenes
