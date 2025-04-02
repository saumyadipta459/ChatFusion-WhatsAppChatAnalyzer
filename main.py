import streamlit as st
import preprocessor, helper
import matplotlib.pyplot as plt
import seaborn as sns


# Mobile sidebar instruction
# ===== ENHANCED MOBILE INSTRUCTION WITH STRONG PRIVACY =====
st.markdown("""
<style>
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.6; }
        100% { opacity: 1; }
    }
    .pulse-arrow {
        animation: pulse 2s infinite;
        display: inline-block;
        font-size: 24px;
        transform: translateY(4px);
    }
    .privacy-highlight {
        background-color: #004d40;
        color: white;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
</style>

**<span class="pulse-arrow">></span> Tap this arrow (top-left)** to upload:  
`_chat.txt` from WhatsApp  
<span class="privacy-highlight">🔒 YOUR DATA NEVER LEAVES YOUR DEVICE</span>
""", unsafe_allow_html=True)

with st.expander("🔍 Analyze Any WhatsApp Chat", expanded=False):
    st.markdown("""
    **📱 All Chat Types Supported:**  
    ▸ **Groups**: Compare all members or focus on individuals  
    ▸ **Private Chats**: 1-on-1 conversation analytics  

    **🔄 Simple 3-Step Process:**  
    1. Export chat (⋮ → *Without Media*)  
    2. Extract the ZIP → find `_chat.txt`  
    3. Upload & analyze  

    **📊 Insights You'll Get:**  
    • Sentiment trends • Activity patterns • Top emojis  

    <div style="background-color: #e8f5e9; padding: 12px; border-radius: 8px; margin-top: 10px;">
    <strong>🛡️ Privacy Guarantee:</strong>  
    • <strong>No data storage</strong> - analysis happens in your browser  
    • <strong>No server uploads</strong> - we never see your chats  
    • <strong>Automatic deletion</strong> - when you close the tab  
    </div>

    ⚠️ <strong>Troubleshooting</strong>:  
    - Garbled text? Re-export as *"Without Media"*  
    - Errors? Ensure you upload the `.txt` file  
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
