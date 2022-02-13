import tweepy
import tweepy as tw

import sentiment_mod as s


def calculate_sentiment(text):
    return text + "\n" + str(s.sentiment(text)) + "\n"


# Connecting to your Twitter Developer APIs

api_key = '07LgMonW8TpwyKEXG6tWCKrr7'
api_secret = 'syoceNXkcFLWZrin2TVknIS8J3JjvIkqDyVB9AnvEFlQ9i8yfY'
baerer_token = 'AAAAAAAAAAAAAAAAAAAAAMiYZAEAAAAAHAjsXmVZ1J3dQt3Py014thyZYPo%3DIy2nQUmVjDMYhrsdH06FRYuVkKJfzXDDnLLp1lFeQgi0fH46FU'
access_token = '1060741249-r8UouANdyolMO8yu7pyWbnrz6htlAxBf8cxRHwV'
access_token_secret = 'HxXnBHamDLHtdbtftU7a21kDU9MCMvIiN3AvIqZQ3m3e4'

# Authenticating the APIs
auth = tw.OAuthHandler(api_key, api_secret)
auth.set_access_token(access_token, access_token_secret)
api = tw.API(auth, wait_on_rate_limit=True)

# Define the search term and the date_since date as variables
search_words = "#Happy"
date_since = "2018-01-16"

# Collect tweets
tweets = tweepy.Cursor(api.search_tweets, search_words, lang="en", since_id=date_since, tweet_mode='extended').items(1000)

print(tweets)

# Iterate and print tweets
for tweet in tweets:
    # print(tweet.full_text)
    sentiment_value, confidence = s.sentiment(tweet.full_text)
    print(tweet, sentiment_value, confidence)

    if sentiment_value == "pos":
        output = open("twitter-out.txt", "a")
        output.write(sentiment_value)
        output.write('\n')
        output.close()
    elif confidence * 100 >= 80:
        output = open("twitter-out.txt", "a")
        output.write(sentiment_value)
        output.write('\n')
        output.close()

# Collect a list of tweets
tweet_list = [tweet.text for tweet in tweets]





"""
# Define the search term and the date_since date as variables
search_words = "bruins"
date_since = "2021-02-20"

# Collect tweets
tweets = tw.Cursor(api.search,
                   q=search_words,
                   lang="en",
                   since=date_since).items(5)


# Function to get related tweets
def get_related_tweets(key_word):
    twitter_users = []
    tweet_time = []
    tweet_string = []
    for tweet in tw.Cursor(api.search, q=key_word, count=1000).items(1000):
        if (not tweet.retweeted) and ('RT @' not in tweet.text):
            if tweet.lang == "en":
                twitter_users.append(tweet.user.name)
                tweet_time.append(tweet.created_at)
                tweet_string.append(tweet.text)
                # print([tweet.user.name,tweet.created_at,tweet.text])
    df = pd.DataFrame({'name': twitter_users, 'time': tweet_time, 'tweet': tweet_string})
    df.to_csv(f"{key_word}.csv")
    return df


# Getting bruins related tweets and passing them to a dataframe
tweets_df = get_related_tweets("bruins")
tweets_df.head(5)
"""

"""
tweet = tw.Tweet(data)
print(tweet.text)
print(tweet.created_at)
print(tweet.user.location)
print(tweet.user.followers_count)
print(tweet.user.friends_count)
print(tweet.user.statuses_count)
print(tweet.user.screen_name)
print(tweet.user.name)
print(tweet.user.time_zone)
print(tweet.user.geo_enabled)
print(tweet.user.lang)
print(tweet.user.verified)
print(tweet.user.favourites_count)
print(tweet.user.created_at)
print(tweet.user.contributors_enabled)
print(tweet.user.default_profile_image)
print(tweet.user.default_profile)
print(tweet.user.description)
print(tweet.user.entities)
print(tweet.user.url)
print(tweet.user.utc_offset)
print(tweet.user.location)
print(tweet.user.profile_background_image_url)
print(tweet.user.profile_background_image_url_https)
print(tweet.user.profile_background_tile)
print(tweet.user.profile_image_url)
print(tweet.user.profile_image_url_https)
print(tweet.user.profile_banner_url)
print(tweet.user.profile_link_color)
print(tweet.user.profile_sidebar_border_color)
print(tweet.user.profile_sidebar_fill_color)
print(tweet.user.profile_text_color)
print(tweet.user.profile_use_background_image)
print(tweet.user.protected)
"""
