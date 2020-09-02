import os
from tqdm import tqdm
from googletrans import Translator
#from translate import Translator
from tqdm import tqdm

from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons


def fix_text(text):
    text = text.replace("&amp;", "&")
    text = text.replace('&gt', '>')
    text = text.replace('&lt', '<')
    
    return text

translator = Translator()
#translator = Translator(to_lang='tr')

text_processor = TextPreProcessor(
    normalize=['url', 'email', 'percent', 'money', 'phone', 'user',
        'time', 'url', 'date', 'number'],
    annotate={"hashtag", "allcaps", "elongated", "repeated",
        'emphasis', 'censored'},
    fix_html=True,  # fix HTML tokens
    #segmenter="twitter", 
    #corrector="twitter", 
    #unpack_hashtags=True,  # perform word segmentation on hashtags
    #unpack_contractions=True,  # Unpack contractions (can't -> can not)
    spell_correct_elong=False,  # spell correction for elongated words
    tokenizer=SocialTokenizer(lowercase=False).tokenize,
    mode='fast',
    dicts=[emoticons]
)

lang = "English"
short_lang = "En"

target_lang = 'Tran_Spanish'#"Turkish"
short_target_lang = 'tEs'#"Tr"

pts = ['So @Ryanair site crashes everytime I try to book - how do they help? Tell me there\'s nothing wrong & hang up #furious #helpless @SimonCalder', 
       '@RKsAllyCat82 No, but I\'m planning on it before September 😊',
       'I have the best girlfriend in the world 😍 #blessed',
       'Bruhhh🙄why is Ar\'mon so cute from @TheRealAandT . He just chill and ughhhhh😍😍😍😍. #attractive #whyamIsingle #Houston',
       'Your #smile is always yours, we make it as easy as possible for you to get the best orthodontic care no matter when you need it.',
       'Remember, for everything you have #lost, you have gained something else. Without the #dark, you would never see the #stars.',
       'I was thinking about Fergie\'s music M.I.L.F and seriously, if I\'m a mother someday, I\'ll be a M.I.L.F  #lmao #Empowerment #adorable',
       'Am \'ugly\' guy whos pleasant geuine  and happy &gt; an evil attractive guy whos bitter and coniving and superficial',
       '@157Gale @GatorDave_SEC I\'d be thrilled if he committed but nervous the whole time until NSD. Kid being from Cali makes me nervous.']

failed = 0
for split in tqdm(["dev", "train", "test-gold"]):
    split_dir = f"data/SemEval_{lang}"
    if not tqdm(os.path.exists(split_dir)):
        os.makedirs(split_dir)
    f = open(f"{split_dir}/2018-E-c-{short_lang}-{split}.txt", "r")

    res = ""
    for i, line in tqdm(enumerate(f)):
        line = line.rstrip("\n")
        if i == 0:
            res += line + "\n"
            continue
        
        text = line[14:-22]
        #text = " ".join(text_processor.pre_process_doc(text))
        text = fix_text(text)
        #text = pts[-1]
        #if i >= 400:
            #break

        translated = False
        patience = 0
        while not translated:
            if patience == 2:
                failed = failed + 1
                translated_text = text
                print(failed)
                break
            translated_text = translator.translate(text, src="en", dest="es").text
            if text == translated_text:
                translator = Translator()
                import time
                time.sleep(5)
                patience = patience + 1
            else:
                translated = True
            #translated_text = translator.translate(text)
            #print(text)
            #print(translated, translated_text)
            
        res += line[:14] + translated_text + line[-22:] + "\n"
    f.close()
    print(failed)

    target_dir = f"data/SemEval_{target_lang}"
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    f = open(f"{target_dir}/2018-E-c-{short_target_lang}-{split}.txt", "w+")
    f.write(res)
    f.close()

#Problem tweets:
#So @Ryanair site crashes everytime I try to book - how do they help? Tell me there's nothing wrong & hang up #furious #helpless @SimonCalder
#@RKsAllyCat82 No, but I'm planning on it before September 😊
# I have the best girlfriend in the world 😍 #blessed
# Bruhhh🙄why is Ar'mon so cute from @TheRealAandT . He just chill and ughhhhh😍😍😍😍. #attractive #whyamIsingle #Houston
# Your #smile is always yours, we make it as easy as possible for you to get the best orthodontic care no matter when you need it.
# Remember, for everything you have #lost, you have gained something else. Without the #dark, you would never see the #stars.
