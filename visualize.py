#!/usr/bin/env python2

from collections import defaultdict

results = defaultdict(list)

corpus = "birds"
exp = "one_different"

with open("experiments/%s/%s.results.base.txt" % (exp, corpus)) as results_f:
    results_f.readline()
    for line in results_f:
        target, distractor, sim, model, speaker, listener, desc = line.strip().split(",", 6)
        listener = float(listener)
        speaker = float(speaker)
        score = listener
        #score = speaker
        results[target,distractor,model].append((score, desc))

opt_results = {}
for key, result in results.items():
    opt_results[key] = max(result)

keys = sorted(opt_results.keys())

if corpus == "birds":
    image_template = "data/birds/images/%s"
else:
    image_template = "data/abstract/RenderedScenes/Scene%s.png"

print "<html><body><table>"
for key in keys:
    score, desc = opt_results[key]
    tgt, dis, model = key
    print "  <tr>"
    print "    <td>%s</td>" % model
    print "    <td><img src='%s'></td>" % (image_template % tgt)
    print "    <td><img src='%s'></td>" % (image_template % dis)
    print "    <td>%s</td>" % desc
    print "  </tr>"
print "</table></body></html>"
