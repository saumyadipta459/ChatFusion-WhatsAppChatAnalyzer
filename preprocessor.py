import re
import pandas as pd
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def preprocess(data):
    pattern = r'\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s[APap][Mm]\s-\s'
    messages = re.split(pattern, data)[1:]
    dates = re.findall(pattern, data)

    # Create DataFrame
    df = pd.DataFrame({'user_message': messages, 'message_date': dates})

    # NEW: Ensure user_message is string type
    df['user_message'] = df['user_message'].astype(str)

    # Replace non-breaking space with a regular space and remove trailing characters
    df['message_date'] = df['message_date'].str.replace('\u202F', ' ', regex=False).str.rstrip(' - ')

    # Convert message_date type
    df['message_date'] = pd.to_datetime(df['message_date'], format='%d/%m/%y, %I:%M %p')

    # Rename the column
    df.rename(columns={'message_date': 'date'}, inplace=True)

    users = []
    messages = []
    for message in df['user_message']:
        entry = re.split('([\w\W]+?):\s', str(message))  # NEW: Added str() conversion here
        if entry[1:]:  # user name
            users.append(entry[1])
            messages.append(" ".join(entry[2:]))
        else:
            users.append('group_notification')
            messages.append(entry[0])

    df['user'] = users
    df['message'] = messages
    df.drop(columns=['user_message'], inplace=True)

    # NEW: Ensure critical columns are strings
    df['message'] = df['message'].astype(str)
    df['user'] = df['user'].astype(str)

    df['Specific_Date'] = df['date'].dt.date
    df['year'] = df['date'].dt.year
    df['month_num'] = df['date'].dt.month
    df['month'] = df['date'].dt.month_name()
    df['day'] = df['date'].dt.day
    df['day_name'] = df['date'].dt.day_name()
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute

    # NEW: Add try-catch for sentiment analysis
    try:
        df['sentiment'] = df['message'].apply(lambda msg: TextBlob(str(msg)).sentiment.polarity)
        analyzer = SentimentIntensityAnalyzer()
        df['vader_sentiment'] = df['message'].apply(lambda msg: analyzer.polarity_scores(str(msg))['compound'])
    except Exception as e:
        print(f"Sentiment analysis error: {str(e)}")
        df['sentiment'] = 0
        df['vader_sentiment'] = 0

    period = []
    for hour in df[['day_name', 'hour']]['hour']:
        if hour == 23:
            period.append(str(hour) + "-" + str('00'))
        elif hour == 0:
            period.append(str('00') + "-" + str(hour + 1))
        else:
            period.append(str(hour) + "-" + str(hour + 1))

    df['period'] = period

    return df
