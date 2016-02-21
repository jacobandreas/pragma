#!/usr/bin/env python2

import sys
import numpy as np

results_file = sys.argv[1]

dest_file = ".".join(results_file.split(".")[:-1]) + ".turk.csv"

records = []

with open(results_file) as results_f:
    results_f.readline()
    for line in results_f:
        example_id, tgt, dis, _, _, _, _, caption = line.strip().split(",")

        if "abstract" in results_file:
            tgt = "http://fromage.banatao.berkeley.edu/pragma/data/abstract/RenderedScenes/Scene%s.png" % tgt
            dis = "http://fromage.banatao.berkeley.edu/pragma/data/abstract/RenderedScenes/Scene%s.png" % dis
        elif "birds" in results_file:
            assert False

        if np.random.random() < 0.5:
            img1 = tgt
            img2 = dis
            tgt_id = 1
        else:
            img1 = dis
            img2 = tgt
            tgt_id = 2

        caption = caption.replace(",", " ")
        parts = [example_id, caption, img1, img2, tgt_id]
        records.append(parts)

np.random.shuffle(records)
print len(records)

with open(dest_file, "w") as dest_f:
    print >>dest_f, "id,caption,img1,img2,tgt_id"
    for parts in records:
        print >>dest_f, ",".join([str(s) for s in parts])
