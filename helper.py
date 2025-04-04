from urlextract import URLExtract
from wordcloud import WordCloud
import pandas as pd
from collections import Counter
import emoji

extract = URLExtract()


def fetch_stats(selected_user, df):
    # Input validation
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")
    if not all(col in df.columns for col in ['user', 'message']):
        raise ValueError("DataFrame missing required columns 'user' or 'message'")

    # Ensure string type
    df = df.copy()
    df['message'] = df['message'].astype(str)
    df['user'] = df['user'].astype(str)

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
        try:
            links.extend(extract.find_urls(message))
        except:
            continue  # Skip if URL extraction fails

    return num_messages, len(words), num_media_messages, len(links)


def most_busy_users(df):
    # Input validation
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")
    if 'user' not in df.columns:
        raise ValueError("DataFrame missing required column 'user'")

    df['user'] = df['user'].astype(str)
    x = df['user'].value_counts().head()
    busy_df = round((df['user'].value_counts() / df.shape[0]) * 100, 2).reset_index().rename(
        columns={'index': 'Name', 'user': 'Percent'})
    return x, busy_df


def create_wordcloud(selected_user, df):
    # Input validation and type safety
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")
    if not all(col in df.columns for col in ['user', 'message']):
        raise ValueError("DataFrame missing required columns")

    df = df.copy()
    df['message'] = df['message'].astype(str)
    df['user'] = df['user'].astype(str)

    # Filter data
    df = df[(df['user'] != 'group_notification') & (df['message'] != '<Media omitted>\n')]

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    # Read stop words with error handling
    try:
        with open('stop_hinglish.txt', 'r', encoding='utf-8') as f:
            stop_words = set(f.read().split())
    except:
        stop_words = set()

    # Process messages
    df['message'] = df['message'].apply(
        lambda msg: " ".join([word for word in str(msg).lower().split() if word not in stop_words])
    )

    messages_text = df['message'].str.cat(sep=" ")

    if not messages_text.strip():
        return None

    # Generate WordCloud
    try:
        wc = WordCloud(width=500, height=500, min_font_size=10, background_color='white')
        return wc.generate(messages_text)
    except:
        return None


def most_common_words(selected_user, df):
    # Input validation
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")
    if not all(col in df.columns for col in ['user', 'message']):
        raise ValueError("DataFrame missing required columns")

    df = df.copy()
    df['message'] = df['message'].astype(str)
    df['user'] = df['user'].astype(str)

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    df = df[(df['user'] != 'group_notification') & (df['message'] != '<Media omitted>\n')]

    # Read stop words with error handling
    try:
        with open('stop_hinglish.txt', 'r', encoding='utf-8') as f:
            stop_words = set(f.read().split())
    except:
        stop_words = set()

    words = []
    for message in df['message']:
        words.extend([word for word in str(message).lower().split() if word not in stop_words])

    return pd.DataFrame(Counter(words).most_common(20))


def emoji_helper(selected_user, df):
    # Input validation
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")
    if 'message' not in df.columns:
        raise ValueError("DataFrame missing required column 'message'")

    df = df.copy()
    df['message'] = df['message'].astype(str)

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    emojis = []
    for message in df['message']:
        try:
            emojis.extend([c for c in str(message) if emoji.is_emoji(c)])
        except:
            continue

    if not emojis:
        return pd.DataFrame()

    return pd.DataFrame(Counter(emojis).most_common(len(Counter(emojis))))


def monthly_timeline(selected_user, df):
    # Input validation
    required_cols = ['user', 'message', 'year', 'month_num', 'month']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"DataFrame missing required columns: {required_cols}")

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    timeline = df.groupby(['year', 'month_num', 'month']).count()['message'].reset_index()
    timeline['time'] = timeline.apply(lambda x: f"{x['month']}-{x['year']}", axis=1)

    return timeline


def daily_timeline(selected_user, df):
    if 'Specific_Date' not in df.columns:
        raise ValueError("DataFrame missing required column 'Specific_Date'")

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df.groupby('Specific_Date').count()['message'].reset_index()


def week_activity_map(selected_user, df):
    if 'day_name' not in df.columns:
        raise ValueError("DataFrame missing required column 'day_name'")

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df['day_name'].value_counts()


def month_activity_map(selected_user, df):
    if 'month' not in df.columns:
        raise ValueError("DataFrame missing required column 'month'")

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df['month'].value_counts()


def activity_heatmap(selected_user, df):
    required_cols = ['day_name', 'period', 'message']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"DataFrame missing required columns: {required_cols}")

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df.pivot_table(index='day_name', columns='period', values='message', aggfunc='count').fillna(0)


def sentiment_analysis(selected_user, df):
    required_cols = ['user', 'sentiment', 'vader_sentiment']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"DataFrame missing required columns: {required_cols}")

    # Ensure numeric types
    df = df.copy()
    df['sentiment'] = pd.to_numeric(df['sentiment'], errors='coerce').fillna(0)
    df['vader_sentiment'] = pd.to_numeric(df['vader_sentiment'], errors='coerce').fillna(0)

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    avg_sentiment = df['sentiment'].mean()
    avg_vader_sentiment = df['vader_sentiment'].mean()

    return avg_sentiment, avg_vader_sentiment


def first_last_message_times(selected_user, df):
    # Input validation
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")
    if 'date' not in df.columns:
        raise ValueError("DataFrame missing required column 'date'")

    df = df.copy()

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    if df.empty:
        return None, None

    # Get first and last messages
    first_msg = df.iloc[0]
    last_msg = df.iloc[-1]

    return first_msg['date'], last_msg['date']

def get_conversation_duration(first_msg_time, last_msg_time):
    if not first_msg_time or not last_msg_time:
        return None, None
    duration = last_msg_time - first_msg_time
    return duration.days, duration.days * 24


def first_last_message_details(selected_user, df):
    """Returns (first_msg, first_time, last_msg, last_time)"""
    if not isinstance(df, pd.DataFrame) or 'date' not in df.columns or 'message' not in df.columns:
        raise ValueError("DataFrame missing required columns")

    df = df.copy()
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    if df.empty:
        return None, None, None, None

    first_msg = df.iloc[0]['message']
    first_time = df.iloc[0]['date']
    last_msg = df.iloc[-1]['message']
    last_time = df.iloc[-1]['date']

    return first_msg, first_time, last_msg, last_time
