import codecs
import os
import pickle

import emoji
import numpy as np
from sklearn.model_selection import train_test_split

num_data = 50000
num_emojis = 64


class EmojiTweetsPreprocessing:

    def __init__(self):
        self.path = "data/emoji-tweets.txt"

        self.emojis = emoji.UNICODE_EMOJI

        self.cur_emojis = {}

    def get_emojis(self, text):
        emojis = [c for c in text if c in self.emojis]
        for e in emojis:
            text = text.replace(e, "")
            self.cur_emojis[e] = self.cur_emojis.get(e, 0) + 1

        emojis= sorted(list(set(emojis)))

        return text, emojis

    def run(self):

        print(self.emojis)

        f = codecs.open(self.path, "r", encoding="utf-8")

        res = {
            "texts": [],
            "info": []
        }

        all_texts = []
        all_emojis = []
        i = 0
        for line in f:
            line = line.strip("\n")

            text, emojis = self.get_emojis(line)

            all_texts.append(text)
            all_emojis.append(emojis)
            i += 1
            if num_data > 0 and i > num_data:
                break

        top_emojis = list(sorted(self.cur_emojis.items(), key=lambda k: -k[1])[:num_emojis])
        top_emojis = [self.emojis[e] for e, v in top_emojis]

        emoji_indices = {e:i for i, e in enumerate(top_emojis)}

        for i, emojis in enumerate(all_emojis):
            emoji_lst = [False]*len(top_emojis)
            have_top_emojis = False
            for emoji in emojis:
                emoji_idx = self.emojis[emoji]
                if emoji_idx in top_emojis:
                    emoji_ind = emoji_indices[emoji_idx]
                    emoji_lst[emoji_ind] = True
                    have_top_emojis = True

            if have_top_emojis:
                res["info"].append({"label": emoji_lst})
                res["texts"].append(all_texts[i])

        all_inds = np.arange(len(res["texts"]))

        train_inds, test_inds = train_test_split(all_inds, test_size=0.15)
        train_inds, val_inds = train_test_split(train_inds, test_size=0.1)

        res["train_ind"] = train_inds
        res["val_ind"] = val_inds
        res["test_ind"] = test_inds

        if not os.path.exists("data/emoji-tweets"):
            os.makedirs("data/emoji-tweets")

        with open("data/emoji-tweets/raw.pickle", "wb+") as f:
            pickle.dump(res, f)
