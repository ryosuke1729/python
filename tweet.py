import json
import re

with open("tweets.js", 'r', encoding='utf-8') as f:
    json2 = f.read()
    json3 = re.sub("window.YTD.tweets.part0 = ", "", json2)
    tweets = json.loads(json3)

num = []
twe = []
for status in tweets:
    tweet = status["tweet"]
    tw = tweet["full_text"]
    ide = tweet["id"]
    num.append(ide)
    twe.append(tw)

list1, list2 = zip(*sorted(zip(num, twe)))
for status in list2:
    print(status)
    print()
