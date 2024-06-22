import os
import json

labels = {}
with open('imagenet1k-words.txt', 'r') as file:
    data = file.read()
    labels = json.loads(data)


# print(labels)

# MUST MODIFY STABLE DIFFUSION SAMPLE PATH IN text2img.py

os.chdir('stable-diffusion')
for key in labels.keys():
    label = labels[key]
    print(key, label[0], label[1])
    if int(key) < 0:
        continue
    res = os.system('python scripts/txt2img.py --ckpt sd-v1-4.ckpt --prompt "%s" --plms --skip_grid --n_iter 1 --n_samples 5 --outdir "../imagenet1k-mini/train/% s"' % (label[1], label[0]))
    if res != 0:
        quit()
