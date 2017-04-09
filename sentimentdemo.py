import tweepy
from textblob import TextBlob

consumer_key = 'wv3Y6pp1mnn1wSVbbpRnXU7UG'
consumer_secret = 'yohJkaAdwUH4cdjoAchGNvMiXGPkhdpAOEiykT13CGVfqeGqro'

access_token = '2264238294-es7QaM1Lfvt19OdZi8y7JYczssTaaLixUIvqUck'
access_token_secret = 'hI0t7N7zVe8tcR4w0gS4pLVUUSg3oXlciJuoboGzmSMwA'

auth = tweepy.OAuthHandler(consumer_key,consumer_secret)
auth.set_access_token(access_token,access_token_secret)

api = tweepy.API(auth)
public_tweets = api.search('Trump')

for tweet in public_tweets:
	print(tweet.text)
	analysis = TextBlob(tweet.text)
	print(analysis.sentiment)
