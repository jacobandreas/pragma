#!/usr/bin/env python2

import corpus
from indices import WORD_INDEX

from collections import defaultdict
import numpy as np

N_PAIRS = 200
MAX_DIFFERENCES = 4
N_PAIRS_PER = N_PAIRS / MAX_DIFFERENCES

def make_abstract():
    train_data, val_data, test_data = corpus.load_abstract()

    by_prop = defaultdict(list)

    for datum in val_data:
        props = {(p.type_index, p.object_index) for p in datum.props}
        props = {(i,j) if i not in (2,3) else (i,0) for i,j in props}
        props = tuple(sorted(props))
        by_prop[props].append(datum)

    print len(by_prop)

    by_similarity = defaultdict(list)
    keys = list(by_prop.keys())
    keys2 = list(by_prop.keys())
    np.random.shuffle(keys)
    for key1 in by_prop:
        np.random.shuffle(keys2)
        for key2 in by_prop:
            if key2 == key1:
                continue
            similarity = len(set(key1) ^ set(key2))
            i_img1 = np.random.choice(len(by_prop[key1]))
            i_img2 = np.random.choice(len(by_prop[key2]))
            img1 = by_prop[key1][i_img1]
            img2 = by_prop[key2][i_img2]
            by_similarity[similarity].append((img1, img2))

    with open("experiments/dev/one_different/abstract.ids.txt", "w") as id_f:
        one_different = list(by_similarity[2])
        np.random.shuffle(one_different)
        for img1, img2 in one_different[:N_PAIRS]:
            print >>id_f, "%s,%s,2" % (img1.image_id, img2.image_id)

    with open("experiments/dev/by_similarity/abstract.ids.txt", "w") as id_f:
        for i in range(1, MAX_DIFFERENCES + 1):
            different = list(by_similarity[i])
            np.random.shuffle(different)
            for img1, img2 in different[:N_PAIRS_PER]:
                print >>id_f, "%s,%s,%d" % (img1.image_id, img2.image_id, i)

    #with open("experiments/all_same/abstract.ids.txt", "w") as id_f:
    #    counter = 0
    #    for key in by_prop:
    #        if counter >= N_PAIRS:
    #            break
    #        images = set(d.image_id for d in by_prop[key])
    #        if len(images) <= 3:
    #            continue
    #        images = list(images)
    #        np.random.shuffle(images)
    #        images = images[:2]
    #        print >>id_f, ",".join(images) + ",0"
    #        counter += 1

    #with open("experiments/one_different/abstract.ids.txt", "w") as id_f:
    #    counter = 0
    #    for key1 in by_prop:
    #        if counter >= N_PAIRS:
    #            break
    #        keys = list(by_prop.keys())
    #        np.random.shuffle(keys)
    #        for key2 in by_prop:
    #            if len(key1) != len(key2):
    #                continue
    #            if len(set(key1) ^ set(key2)) != 2:
    #                continue
    #            if key1 > key2:
    #                continue
    #            images = [by_prop[key1][0].image_id, by_prop[key2][0].image_id]
    #            print >>id_f, ",".join(images) + ",2"
    #            counter += 1
    #            break

    #with open("experiments/by_similarity/abstract.ids.txt", "w") as id_f:
    #    keys = list(by_prop.keys())
    #    counter = 0
    #    for key1 in by_prop:
    #        if counter >= N_PAIRS:
    #            break
    #        attempts = 0
    #        while attempts < 250:
    #            attempts += 1
    #            key2 = keys[np.random.randint(len(keys))]
    #            similarity = len(set(key1) ^ set(key2))
    #            if similarity > MAX_DIFFERENCES:
    #                continue
    #            if np.random.random() > 0.45 ** similarity:
    #                continue
    #            img1 = by_prop[key1][0].image_id
    #            img2 = by_prop[key2][0].image_id
    #            if img1 == img2:
    #                continue
    #            print >>id_f, "%s,%s,%s" % (img1, img2, similarity)
    #            counter += 1
    #            #print counter
    #            break

def make_birds():
    train_data, val_data, test_data = corpus.load_birds()

    by_bird = defaultdict(list)
    all_birds = []
    for datum in val_data:
        id_parts = datum.image_id.split("_")
        bird_name = "_".join(id_parts[:-2])
        by_bird[bird_name].append(datum)
        all_birds.append(datum)

    counter = 0
    with open("experiments/all_same/birds.ids.txt", "w") as id_f:
        for bird in by_bird:
            if counter == 100:
                break
            birds = by_bird[bird]
            np.random.shuffle(birds)
            bird1 = birds[0]
            for bird2 in birds:
                if bird2.image_id == bird1.image_id:
                    continue
                print >>id_f, "%s,%s,0" % (bird1.image_id, bird2.image_id)
                counter += 1

    with open("experiments/one_different/birds.ids.txt", "w") as id_f:
        for i in range(100):
            bird1 = all_birds[np.random.randint(len(all_birds))]
            bird2 = all_birds[np.random.randint(len(all_birds))]
            print >>id_f, "%s,%s,0" % (bird1.image_id, bird2.image_id)

    with open("experiments/by_similarity/birds.ids.txt", "w") as id_f:
        pass

if __name__ == "__main__":
    make_abstract()
    #make_birds()
