import json
import os
import string
from collections import Counter
from empath import Empath
from scipy.constants import sigma
from textblob import TextBlob

from joblib import Parallel, delayed
from nltk import ngrams
from nltk.corpus import stopwords

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


lexicon = Empath()
cv = CountVectorizer(binary=True)

np.random.seed(7)
sw = set(stopwords.words('english'))
trantab = "".maketrans('', '', string.punctuation)
totalTweets=0
ownertweets=0
OriginaltweetMap={}

def mutualInformation(word1, word2, unigram_freq, bigram_freq):


  prob_word1 = unigram_freq[word1] / float(sum(unigram_freq.values()))
  prob_word2 = unigram_freq[word2] / float(sum(unigram_freq.values()))
  prob_word1_word2 = bigram_freq[(word1,word2)] / float(sum(bigram_freq.values()))
  return np.math.log((float(prob_word1_word2) / float(prob_word1 * prob_word2)), 2)


import re


def Find(string):
    # findall() has been used
    # with valid conditions for urls in string
    completestring =""
    url = re.findall('(?:(?:https?|ftp):\/\/)?[\w/\-?=%.]+\.[\w/\-?=%.]+', string)
    if len(url)>0:
        for u in url:
            completestring=string.replace(u,"")
            string=completestring
    else:
        completestring=string

    prefix =('@')
    stringfix=""
    mylist=completestring.split()
    for w in mylist :
        for s in w[:]:
            if s.startswith(prefix):
                mylist.remove(s)
    return " ".join(mylist)
def Read_ownerTweets(file):
    global x, y, totalTweets, ownertweets
    fileObject = open(file, encoding="utf8")
    Lines = fileObject.readlines()
    totalTweets = len(Lines)
    for line in Lines:
        try:
            parsed_json_tweets = json.loads(line)
            this_user_handle = parsed_json_tweets['user']['screen_name'].lstrip().strip()
            retweet_count = parsed_json_tweets['retweet_count']
            UserFollowerCount = parsed_json_tweets['user']['followers_count']
            tweet_text = parsed_json_tweets['text'].lstrip().strip()
            if 'retweeted_status' in parsed_json_tweets:
                ownertweets += 1
                ownerName = parsed_json_tweets['retweeted_status']['user'][
                    'screen_name'].lstrip().strip()
                ownerTweetTimeStamp = parsed_json_tweets['retweeted_status'][
                    'created_at'].lstrip().strip()
                ownerFollercount = parsed_json_tweets['retweeted_status']['user']['followers_count']
                ownerretweetcount = parsed_json_tweets['retweeted_status']['retweet_count']
                try:
                    Owner_tweet_text =parsed_json_tweets['retweeted_status']['extended_tweet'][
                        'full_text'].lstrip().strip()
                except:
                    Owner_tweet_text = parsed_json_tweets['retweeted_status'][
                        'text'].lstrip().strip()
                    # bigrams = [b for l in wordlist for b in zip(l[:-1], l[1:])]
                Owner_tweet_text=Find(Owner_tweet_text)
                if Owner_tweet_text!="":
                    bigrams = []
                    bigramcounter = {}
                    wordLength = len(clean(Owner_tweet_text))
                    wordlist = clean(Owner_tweet_text)
                    EachWordCount = Counter.__call__(clean(Owner_tweet_text))
                    if len(wordlist) > 3:
                        bigrams = list(ngrams(wordlist, 2))
                        bigramcounter = dict(Counter.__call__(bigrams))
                    values = []
                    V1MI = 0
                    adder = 0
                    for s in bigrams:
                        tt = mutualInformation(s[0], s[1], EachWordCount, bigramcounter)
                        adder += tt
                        values.append(tt)
                    if adder != 0:
                        V1MI = float(adder / len(values))

                    V2sentiObject = TextBlob(Owner_tweet_text).sentiment

                    # p = []
                    # for x, y in EachWordCount.items():
                    #     pi = EachWordCount[x] / wordLength
                    #     p.append(pi * float(math.log(pi, 2)))
                    V13 = lexicon.analyze(Owner_tweet_text, normalize=True)
                    try:
                        deception =(V13['deception']+( V13['money']+V13['hate']+V13['envy']+V13['crime']+V13['magic']+V13['fear']+V13['lust']+V13['power']/8))
                    except:
                        deception=V13['deception']

                    OTCnorm = [float(i) / max([V1MI, 1 - V2sentiObject.subjectivity, 1 - deception]) for i in
                               [V1MI, 1 - V2sentiObject.subjectivity, 1 - deception] if
                               max([V1MI, 1 - V2sentiObject.subjectivity, 1 - deception]) != 0]
                    recp= abs((sum(OTCnorm)/3))
                    # print(Owner_tweet_text.replace('\n', ''),recp)
                    OriginaltweetMap[ownerName + "," + ownerTweetTimeStamp] = [ownerName, ownerTweetTimeStamp,
                                                                               ownerFollercount, ownerretweetcount,
                                                                               Owner_tweet_text.replace('\n', ''),OTCnorm,recp
                                                                           ]
        except ValueError:
            continue
    return totalTweets, OriginaltweetMap


def clean(line,status=False):
    lowerlist = [x.lower().translate(trantab) for x in line.split() if x.lower() not in sw]
    lowerlist = [''.join(e for e in x if e.isalnum()) for x in lowerlist if
                 x is not "" and len(x.strip()) > 2 and not x.isdigit() and 'https' not in x.lower()]
    if(status):
        ss=' '.join(e for e in lowerlist)
        lowerlist=ss
    return lowerlist


folder="C:\\Users\\OSU user\\Desktop\\share\\stuff\\Data\\NRA\\"
outputFolder="C:\\Users\\OSU user\\Desktop\\share\\stuff\\OutData\\NRA\\"
onlyfiles = [val for sublist in [[os.path.join(i[0], j) for j in i[2]] for i in os.walk(folder.__str__())] for val
             in sublist if 'Flum' in val and val is not None]

import os
if not os.path.exists(outputFolder):
    os.makedirs(outputFolder)
TotalTweetCount=0
TweetMapper={}
TweetsOnly_str=[]
TweetsOnly=[]
labels=[]
temp = Parallel(n_jobs=-1,prefer="threads")(delayed(Read_ownerTweets)(file) for file in onlyfiles)
foutbias = open(outputFolder + "biased",
                                 encoding="utf8", mode='w')
foutneutral = open(outputFolder + "neutral",
                                 encoding="utf8", mode='w')
fouttruth = open(outputFolder + "truth",
                                 encoding="utf8", mode='w')

foutAll = open(outputFolder + "Democrats_tweets",
                                 encoding="utf8", mode='w')
for x in temp:
    TotalTweetCount += x[0]
    for k, v in x[1].items():
        TweetMapper[k] = v

listV=[]

for x, y in TweetMapper.items():
    if(len(clean(y[4]))>0):
        cleanedTweet=y[4]
        listV.append(float(y[6]))
        TweetsOnly_str.append(cleanedTweet)
        TweetsOnly.append(clean(y[4]))



s=np.random.normal(np.mean(listV),np.std(np.asarray(listV)),len(listV))

# from matplotlib import pyplot as plt
# count, bins, ignored = plt.hist(s, 30, normed=True)
# plt.plot(bins, 1/(np.std(np.asarray(listV)) * np.sqrt(2 * np.pi)) * np.exp( - (bins - np.mean(listV) )**2 / (2 * np.std(np.asarray(listV))**2) ),linewidth=2, color='r')
# plt.show()


for y,z in zip(listV,TweetsOnly_str):
    foutAll.write(z + "\n")
    if (float(y) < 0.36):
        foutbias.write(z + "\n")
    elif float(y) > 0.45 and float(y) < 0.55:
        foutneutral.write(z + "\n")
    elif float(y) > 0.58 and float(y) < 0.70:
        fouttruth.write(z + "\n")

foutbias.close()
foutneutral.close()
fouttruth.close()
foutAll.close()

# print(s)


