import os
import pickle
import numpy as np
import emoji
import pandas as pd
import plotly.express as ex
from tqdm import tqdm


class EmojisZipfLaw:

    def run(self):
        pkl_path = "data/emojis_zipf.pkl"
        if not os.path.exists(pkl_path):
            with open("data/emojitweets-01-04-2018.txt", "r") as f:
                text = "\n".join(f.readlines())

            emojis_dict = {}

            for chr in tqdm(text):
                if chr in emoji.UNICODE_EMOJI:
                    emojis_dict[chr] = emojis_dict.get(chr, 0) + 1

            emojis_dict = {k: v for k, v in sorted(emojis_dict.items(), key=lambda item: -item[1])}

            pickle.dump(emojis_dict, open(pkl_path, "wb+"))

        emojis_dict = pickle.load(open(pkl_path, "rb"))
        print(emojis_dict)

        print(len(emojis_dict))

        num_emojis = 64 # len(emojis_dict)

        x = np.array(list(range(1, len(emojis_dict)+1))[:num_emojis])
        y =  np.array(list(emojis_dict.values())[:num_emojis])

        m = np.sum((x - np.mean(x))*(y - np.mean(y)))/np.sum((x - np.mean(x))**2)
        print("slope", m)
        m = np.round(np.rad2deg(np.arctan(m)), 2)
        print("in degrees", m)


        df = pd.DataFrame({
            "Rank": x,
            "Frequency": y,
            "Emoji": list(emojis_dict.keys())[:num_emojis],
        }, columns = ["Rank", "Frequency", "Emoji"])


        fig = ex.scatter(df, x="Rank", y="Frequency", text="Emoji", log_x=False, log_y=False, template="ggplot2")

        fig.update_traces(textposition='middle center', textfont_size=10)

        fig.update_layout(
            height=800,
            width=800
        )

        fig.show()
        fig.write_image("figures/emojis_zipf.png")

        x = np.log(x)
        y = np.log(y)
        df = pd.DataFrame({
            "Rank": x,
            "Frequency": y,
            "Emoji": list(emojis_dict.keys())[:num_emojis],
        }, columns = ["Rank", "Frequency", "Emoji"])
        # LogLog Plot
        fig = ex.scatter(df, x="Rank", y="Frequency", text="Emoji", log_x=False, log_y=False, template="ggplot2")

        fig.update_traces(textposition='middle center', textfont_size=10)
        fig.update_xaxes(title_text='Rank (Log)')
        fig.update_yaxes(title_text='Frequency (Log)')

        fig.update_layout(
            height=800,
            width=800
            #title_text="Zipf's Law Plot for Emojis"
        )

        fig.show()
        fig.write_image("figures/emojis_zipf_log.png")
