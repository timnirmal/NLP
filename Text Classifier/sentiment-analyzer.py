import sentiment_mod as s


# print("This movie was awesome! The acting was great, plot was wonderful, and there were pythons...so yea!")
# print(s.sentiment("This movie was awesome! The acting was great, plot was wonderful, and there were pythons...so yea!"))


def calculate_sentiment(text):
    return text + "\n" + str(s.sentiment(text)) + "\n"


print(calculate_sentiment(
    "This movie was awesome! The acting was great, plot was wonderful, and there were pythons...so yea!"))

print(calculate_sentiment(
    "This movie was utter junk. There were absolutely 0 pythons. I don't see what the point was at all. Horrible "
    "movie, 0/10"))