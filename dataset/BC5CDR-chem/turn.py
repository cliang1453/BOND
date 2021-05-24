import os
import json
import tqdm

with open('labels.txt') as f:
    tag_to_id={}
    i=0
    for l in f:
        if "B-category" in l:
            continue
        tag_to_id[l.strip()] = i
        i += 1

with open('tag_to_id.json', 'w') as f:
    json.dump(tag_to_id, f)


for filename in ['dev.txt', 'test.txt', 'train.txt', 'weak.txt']:
    data = []
    with open(filename, 'r') as f:
        words = []
        tags = []
        for l in tqdm.tqdm(f):
            l = l.strip()
            if len(words)>0 and l == '':
                data.append({
                    "str_words": words,
                    "tags": tags,
                })
                words = []
                tags = []
                continue
            w,t = l.split('\t')
            if "B-category" in t:
                continue
            words.append(w)
            assert t in tag_to_id
            tags.append(tag_to_id[t])
        if len(words)>0 and l == '':
            data.append({
                "str_words": words,
                "tags": tags,
            })

    with open(filename.replace('.txt', '.json'), 'w') as f:
        json.dump(data, f)
    # break



