from urlextract import URLExtract
from wordcloud import WordCloud
import pandas as pd
from collections import Counter
import emoji

extract = URLExtract()

def fetch_stats(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    num_messages = df.shape[0]
    words = []
    for message in df['message']:
        words.extend(message.split())

    # Fetching number of media messages
    num_media_messages = df[df['message'] == '<Media omitted>\n'].shape[0]

    # Fetch all the links shared
    links = []
    for message in df['message']:
        links.extend(extract.find_urls(message))

    return num_messages, len(words), num_media_messages, len(links)

def most_busy_users(df):
    x = df['user'].value_counts().head()
    df = round((df['user'].value_counts() / df.shape[0]) * 100, 2).reset_index().rename(
        columns={'index': 'Name', 'user': 'Percent'})
    return x, df

def create_wordcloud(selected_user, df):
    # Remove group notifications and media messages
    df = df[(df['user'] != 'group_notification') & (df['message'] != '<Media omitted>\n')]

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    # Read stop words
    with open('stop_hinglish.txt', 'r') as f:
        stop_words = set(f.read().split())

    # Function to remove stop words
    def remove_stop_words(message):
        return " ".join([word for word in message.lower().split() if word not in stop_words])

    # Apply the stop words removal
    df['message'] = df['message'].apply(remove_stop_words)

    # Combine all messages into a single string
    messages_text = df['message'].str.cat(sep=" ")

    # Check if the message text is non-empty
    if not messages_text.strip():
        print("No meaningful text found for WordCloud.")
        return None

    # Generate WordCloud
    wc = WordCloud(width=500, height=500, min_font_size=10, background_color='white')
    df_wc = wc.generate(messages_text)
    return df_wc

def most_common_words(selected_user, df):
    # Read stop words
    with open('stop_hinglish.txt', 'r') as f:
        stop_words = set(f.read().split())

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    # Remove group notifications and media messages
    df = df[(df['user'] != 'group_notification') & (df['message'] != '<Media omitted>\n')]

    words = []

    for message in df['message']:
        words.extend([word for word in message.lower().split() if word not in stop_words])

    most_common_df = pd.DataFrame(Counter(words).most_common(20))
    return most_common_df

def emoji_helper(selected_user, df):
    # Filter the DataFrame by selected user
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    emojis = []
    for message in df['message']:
        emojis.extend([c for c in message if emoji.is_emoji(c)])

    emoji_df = pd.DataFrame(Counter(emojis).most_common(len(Counter(emojis))))

    return emoji_df

def monthly_timeline(selected_user,df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    timeline = df.groupby(['year', 'month_num', 'month']).count()['message'].reset_index()

    time = []
    for i in range(timeline.shape[0]):
        time.append(timeline['month'][i] + "-" + str(timeline['year'][i]))
    timeline['time'] = time

    return timeline

def daily_timeline(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    daily_timeline = df.groupby('Specific_Date').count()['message'].reset_index()

    return daily_timeline

def week_activity_map(selected_user,df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    return df['day_name'].value_counts()

def month_activity_map(selected_user,df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df['month'].value_counts()


def activity_heatmap(selected_user,df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    user_heatmap = df.pivot_table(index='day_name', columns='period', values='message', aggfunc='count').fillna(0)
    return user_heatmap

def sentiment_analysis(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    # Calculate average sentiment score using TextBlob
    avg_sentiment = df['sentiment'].mean()

    # Calculate average sentiment score using VADER
    avg_vader_sentiment = df['vader_sentiment'].mean()

    return avg_sentiment, avg_vader_sentiment



