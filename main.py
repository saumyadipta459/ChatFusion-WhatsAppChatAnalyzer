import streamlit as st
import preprocessor, helper
import matplotlib.pyplot as plt
import seaborn as sns


# Alternate version with icon and link
# ===== STYLISH DEVELOPER ATTRIBUTION =====
st.markdown("""
<style>
    @keyframes subtleGlow {
        0% { text-shadow: 0 0 5px rgba(0, 150, 255, 0.3); }
        50% { text-shadow: 0 0 10px rgba(0, 150, 255, 0.5); }
        100% { text-shadow: 0 0 5px rgba(0, 150, 255, 0.3); }
    }
    .dev-badge {
        text-align: right;
        margin: -15px 10px 15px 0;
        font-family: 'Segoe UI', Tahoma, sans-serif;
    }
    .dev-name {
        background: linear-gradient(90deg, #0066cc, #00ccff);
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
        font-weight: 600;
        font-size: 0.95rem;
        animation: subtleGlow 3s ease-in-out infinite;
        display: inline-block;
        padding: 2px 8px;
        border-radius: 12px;
        border: 1px solid rgba(0, 150, 255, 0.2);
    }
    .dev-prefix {
        color: #555;
        font-size: 0.8rem;
        margin-right: 5px;
    }
    .dev-icon {
        font-size: 0.9rem;
        margin-right: 3px;
        transform: translateY(1px);
        display: inline-block;
    }
</style>
<div class="dev-badge">
    <span class="dev-icon">👨‍💻</span>
    <span class="dev-prefix">Developed by</span>
    <a href="https://saumyadiptasaha.vercel.app/" target="_blank" style="text-decoration: none;">
        <span class="dev-name">Saumyadipta Saha</span>
    </a>
</div>
""", unsafe_allow_html=True)

# Mobile sidebar instruction
# ===== HIGH-VISIBILITY MOBILE INSTRUCTIONS =====
st.markdown("""
<style>
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
    }
    .pulse-arrow {
        animation: pulse 2s infinite;
        display: inline-block;
        font-size: 24px;
        transform: translateY(4px);
        color: white;
    }
    .privacy-box {
        background-color: #000000;
        color: white;
        padding: 12px;
        border-radius: 8px;
        margin: 10px 0;
        border: 1px solid #444;
    }
    .privacy-badge {
        background-color: #222;
        color: white;
        padding: 6px 10px;
        border-radius: 6px;
        font-weight: bold;
        display: inline-block;
        margin-top: 8px;
        border: 1px solid #444;
    }
</style>

<div class="privacy-box">
<p style="margin:0;font-size:16px;">
<span class="pulse-arrow">></span> <strong>Tap the arrow (top-left)</strong> to upload<br>
<code style="background:#333;color:#fff;padding:2px 6px;border-radius:4px;">_chat.txt</code> from WhatsApp
</p>
<div class="privacy-badge">🔐 WE NEVER SAVE YOUR CHATS</div>
</div>
""", unsafe_allow_html=True)

with st.expander("🔍 Analyze Any WhatsApp Chat", expanded=False):
    st.markdown("""
     **📱 All Chat Types Supported:**  
    ▸ **Groups**: Compare all members or focus on individuals  
    ▸ **Private Chats**: 1-on-1 conversation analytics  
    ▸ **Broadcasts**: Message statistics  

    **🔄 How To Export:**  
    1. Open chat → ⋮ → "Export chat"  
    2. Choose "Without Media"  
    3. Extract the ZIP file  

    **📊 Insights You'll Get:  
    1. Sentiment Trends (TextBlob + VADER)  
    2. Activity Heatmaps (days/times)  
    3. Top Emojis & Words 
    4. Media/Link Statistics

    <div style="background:#000;color:#fff;padding:12px;border-radius:8px;margin:10px 0;border:1px solid #444">
    <strong>🔒 PRIVACY PROTECTED</strong>  
    • All processing happens in your browser  
    • Your data is deleted when you close the tab  
    • We have no server to store your chats  
    </div>

    ⚠️ <strong>For Best Results:</strong>  
    • Use exported chats <3 months old  
    • Always select "Without Media"  
    """, unsafe_allow_html=True)

#plt.rcParams['font.family'] = 'DejaVu Sans'  # Fixes missing emoji/glyph warnings
# Set font early to prevent glyph warnings
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']

st.sidebar.title("Chat Fusion: Sentiment Analysis  and Behavioural Insights from WhatsApp Conversations")

uploaded_file = st.sidebar.file_uploader("Choose a file")
if uploaded_file is not None:
    try:
        bytes_data = uploaded_file.getvalue()
        try:
            data = bytes_data.decode("utf-8")
        except UnicodeDecodeError:
            try:
                data = bytes_data.decode("utf-16")
            except UnicodeDecodeError:
                data = bytes_data.decode("latin-1")

        # Validate it looks like a WhatsApp export
        if not any(x in data[:100] for x in [' - ', ']: ', '[']):
            st.error("This doesn't appear to be a WhatsApp chat export file")
            st.stop()

        df = preprocessor.preprocess(data)

        # Additional type safety
        df['message'] = df['message'].astype(str)
        df['user'] = df['user'].astype(str)

    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        st.stop()

    # Fetch the unique users
    user_list = df['user'].unique().tolist()
    user_list.remove('group_notification')
    user_list.sort()
    user_list.insert(0, "Overall")

    selected_user = st.sidebar.selectbox("Show analysis with respect to", user_list)

    if st.sidebar.button("Show Sentiment Analysis", key='sentiment_analysis'):
        # Display sentiment analysis
        avg_sentiment, avg_vader_sentiment = helper.sentiment_analysis(selected_user, df)
        st.title("Sentiment Analysis")

        # Sentiment Analysis Explanation
        st.write("ℹ️ **Click below to understand Sentiment Analysis and how to interpret the scores:**")
        with st.expander("What is Sentiment Analysis?"):
            st.write("""
            Sentiment analysis helps determine the emotional tone of a conversation. We use two models:

            1. **TextBlob**
               - **Range:** `-1.0` (Negative) to `1.0` (Positive)
               - Provides a general sentiment score based on word polarity.

            2. **VADER (Valence Aware Dictionary and sEntiment Reasoner)**
               - **Range:** `-1.0` (Negative) to `1.0` (Positive)
               - Specially designed for social media and short texts.
               - It considers punctuation and intensity (e.g., "great!!!" is more positive than "great").

            **How to interpret the scores?**
            - **Negative (-1.0 to -0.1):** Mostly negative emotions.
            - **Neutral (-0.1 to 0.1):** Balanced or neutral tone.
            - **Positive (0.1 to 1.0):** Mostly positive mood.

            The scores may vary slightly between models, so focus on the overall trend rather than exact numbers.
            """)

        col1, col2 = st.columns(2)
        with col1:
            st.header("Average Sentiment (TextBlob):")
            st.write(f"{avg_sentiment:.2f} (Range: -1.0 to 1.0)")

        with col2:
            st.header("Average Sentiment (VADER):")
            st.write(f"{avg_vader_sentiment:.2f} (Range: -1.0 to 1.0)")

    if st.sidebar.button("Show Sentiment Over Time", key='sentiment_over_time'):
        # Sentiment over time
        st.title("Sentiment Over Time")

        # Add description for better clarity
        st.write("""
        📈 **How does sentiment change over time?**  

        The graphs below compare **mood trends** using two different sentiment analysis models: **TextBlob** and **VADER**. Each model processes text differently:  

        - **TextBlob** provides a general sentiment score based on **polarity** (how positive or negative a text is).  
        - **VADER** is specifically designed for **social media and informal text**, making it more sensitive to context, emojis, and slang.  

        Both graphs use **color-coded scatter points** to show changes in sentiment over time:  
        🟢 **Green** → Positive sentiment  
        ⚪ **Gray** → Neutral sentiment  
        🔴 **Red** → Negative sentiment  

        By comparing these two models, you can see how each interprets mood shifts in conversations differently. 🚀
        """)

        # Improved TextBlob Sentiment Graph
        st.header("Sentiment Trend (TextBlob)")
        fig, ax = plt.subplots()

        colors = ['red' if x < -0.1 else 'green' if x > 0.1 else 'gray' for x in df['sentiment']]
        ax.scatter(df['date'], df['sentiment'], color=colors)
        ax.set_xlabel("Date")
        ax.set_ylabel("Sentiment Score (-1 to 1)")
        plt.xticks(rotation='vertical')

        # Legend
        import matplotlib.patches as mpatches

        negative_patch = mpatches.Patch(color='red', label='Negative')
        neutral_patch = mpatches.Patch(color='gray', label='Neutral')
        positive_patch = mpatches.Patch(color='green', label='Positive')
        ax.legend(handles=[negative_patch, neutral_patch, positive_patch])

        st.pyplot(fig)

        # Plot VADER sentiment over time (Keep this as it is)
        # Improved VADER Sentiment Graph
        st.header("Sentiment Trend (VADER)")
        fig, ax = plt.subplots()

        colors = ['red' if x < -0.1 else 'green' if x > 0.1 else 'gray' for x in df['vader_sentiment']]
        ax.scatter(df['date'], df['vader_sentiment'], color=colors)
        ax.set_xlabel("Date")
        ax.set_ylabel("Sentiment Score (-1 to 1)")
        plt.xticks(rotation='vertical')

        # Legend
        negative_patch = mpatches.Patch(color='red', label='Negative')
        neutral_patch = mpatches.Patch(color='gray', label='Neutral')
        positive_patch = mpatches.Patch(color='green', label='Positive')
        ax.legend(handles=[negative_patch, neutral_patch, positive_patch])

        st.pyplot(fig)

    if st.sidebar.button("Show Statistics", key='statistics'):
        num_messages, words, num_media_messages, num_links = helper.fetch_stats(selected_user, df)
        st.title("Statistics Of The Chats")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.header("Total Messages:")
            st.title(num_messages)

        with col2:
            st.header("Total Words:")
            st.title(words)

        with col3:
            st.header("Media Shared:")
            st.title(num_media_messages)

        with col4:
            st.header("Links Shared:")
            st.title(num_links)

        # Monthly basis timeline
        st.title("Monthly Timeline Data")
        timeline = helper.monthly_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(timeline['time'], timeline['message'], color='red')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        # Daily basis timeline
        st.title("Daily Timeline Data")
        daily_timeline = helper.daily_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(daily_timeline['Specific_Date'], daily_timeline['message'], color='brown')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        # Activity map
        st.title('Activity Map')
        col1, col2 = st.columns(2)

        with col1:
            st.header("Most Busy Day")
            busy_day = helper.week_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_day.index, busy_day.values)
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        with col2:
            st.header("Most Busy Month")
            busy_month = helper.month_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_month.index, busy_month.values, color='purple')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        st.title("Weekly Activity Map")
        user_heatmap = helper.activity_heatmap(selected_user, df)
        fig, ax = plt.subplots()
        ax = sns.heatmap(user_heatmap)
        st.pyplot(fig)

        # Finding the busiest users in the group (Group level)
        if selected_user == 'Overall':
            st.title("Most Busy User")
            x, new_df = helper.most_busy_users(df)
            fig, ax = plt.subplots()

            col1, col2 = st.columns(2)

            with col1:
                ax.bar(x.index, x.values, color='red')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            with col2:
                st.dataframe(new_df)

        # Wordcloud
        st.title("WordCloud")
        df_wc = helper.create_wordcloud(selected_user, df)
        if df_wc:
            fig, ax = plt.subplots()
            ax.imshow(df_wc, interpolation='bilinear')
            ax.axis("off")
            st.pyplot(fig)
        else:
            st.write("Not enough data to generate a wordcloud for the selected user.")

        # Most common words
        st.title('Most Common Words')
        most_common_df = helper.most_common_words(selected_user, df)

        if not most_common_df.empty and 0 in most_common_df.columns and 1 in most_common_df.columns:
            fig, ax = plt.subplots()
            ax.barh(most_common_df[0], most_common_df[1])
            plt.xticks(rotation='vertical')
            st.pyplot(fig)
        else:
            st.write("Not enough data to display the most common words for the selected user.")

        # Emoji analysis
        st.title('Emojis Analysis Data')
        emoji_df = helper.emoji_helper(selected_user, df)

        if not emoji_df.empty and 1 in emoji_df.columns:
            col1, col2 = st.columns(2)

            with col1:
                st.dataframe(emoji_df)
            with col2:
                fig, ax = plt.subplots()
                ax.pie(emoji_df[1].head(), labels=emoji_df[0].head(), autopct="%0.2f")
                st.pyplot(fig)
        else:
            st.write("No emojis found for the selected user.")
