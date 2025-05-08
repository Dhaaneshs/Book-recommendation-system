import streamlit as st
import pickle
import numpy as np
import requests

model = pickle.load(open('artifacts/model.pkl', 'rb'))
book_names = pickle.load(open('artifacts/book_names.pkl', 'rb'))
final_rating = pickle.load(open('artifacts/final_rating.pkl', 'rb'))

book_pivot = final_rating.pivot_table(columns='user_id', index='title', values='rating').fillna(0)
avg_rating = final_rating.groupby('title')['rating'].mean(numeric_only=True).to_dict()

def get_openlibrary_books(title):
    url = f"https://openlibrary.org/search.json?title={title}"
    try:
        res = requests.get(url)
        data = res.json()
        books = []
        for doc in data["docs"][:5]:
            link = f"https://openlibrary.org{doc.get('key')}"
            cover_id = doc.get("cover_i")
            cover_url = f"http://covers.openlibrary.org/b/id/{cover_id}-M.jpg" if cover_id else None
            books.append({
                "title": doc.get("title"),
                "author": ", ".join(doc.get("author_name", [])),
                "link": link,
                "cover_url": cover_url
            })
        return books
    except:
        return []

if 'search_history' not in st.session_state:
    st.session_state['search_history'] = []

st.set_page_config(page_title="Book Recommendation System", layout="centered")

theme = st.toggle("🌙 Toggle Dark Mode", value=False)

if theme:
    bg_color = "#0e1117"
    text_color = "#ffffff"
    card_color = "#161b22"
else:
    bg_color = "#ffffff"
    text_color = "#000000"
    card_color = "#f0f2f6"

st.markdown(
    f"""
    <style>
        body {{
            background-color: {bg_color};
            color: {text_color};
        }}
        .stApp {{
            background-color: {bg_color};
            color: {text_color};
        }}
        h1, h2, h3, h4, h5, h6, p, span {{
            color: {text_color};
        }}
        .stButton > button {{
            background-color: #007bff;
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.5em 1em;
            font-weight: bold;
            transition: 0.3s;
            outline: none !important;
            box-shadow: none !important;
        }}
        .stButton > button:hover {{
            background-color: #0056b3;
        }}
        .stButton > button:focus {{
            outline: none !important;
            box-shadow: none !important;
        }}
        .stSlider .css-1aumxhk {{
            color: {text_color};
        }}
    </style>
    """,
    unsafe_allow_html=True
)

st.title("📚 Book Recommendation System")
st.markdown("Type a book title or select from the list to get recommendations.")

input_book = st.text_input("🔍 Search for a book:", "")
selected_from_dropdown = st.selectbox("📖 Or pick from existing books:", [""] + list(book_pivot.index), index=0)

final_input = selected_from_dropdown if selected_from_dropdown else input_book.strip()
min_rating = st.slider("⭐ Minimum average rating for recommended books:", 0.0, 3.0, 0.5)

st.markdown("---")

def recommend_from_dataset(book_name):
    book_id = np.where(book_pivot.index == book_name)[0][0]
    distance, suggestion = model.kneighbors(book_pivot.iloc[book_id, :].values.reshape(1, -1), n_neighbors=10)

    recommended_books = []

    st.success(f"You searched for: **{book_name}**")
    st.markdown("### 📖 Recommended Books:")

    for i in range(1, len(suggestion[0])):
        recommended_title = book_pivot.index[suggestion[0][i]]
        avg = avg_rating.get(recommended_title, 0)

        if avg >= min_rating:
            image_url = final_rating[final_rating['title'] == recommended_title]['image_url'].values
            image_url = image_url[0] if len(image_url) > 0 else None

            col1, col2 = st.columns([1, 4])
            with col1:
                if image_url:
                    st.image(image_url, width=100)
            with col2:
                st.write(f"**{recommended_title}**")
                st.write(f"⭐ Average Rating: {avg:.2f}")

                link = get_openlibrary_books(recommended_title)
                if link and link[0].get("link"):
                    st.markdown(f"[🔗 Read on Open Library]({link[0]['link']})")

            recommended_books.append(recommended_title)
    
    st.session_state['search_history'].append({"searched": book_name, "recommendations": recommended_books})

def show_openlibrary_results(book_name):
    st.warning("This book is not in our local dataset. Showing results from Open Library...")
    results = get_openlibrary_books(book_name)
    if results:
        for book in results:
            col1, col2 = st.columns([1, 4])
            with col1:
                if book["cover_url"]:
                    st.image(book["cover_url"], width=100)
            with col2:
                st.write(f"**{book['title']}**")
                st.write(f"✍ Author: {book['author']}")
                st.markdown(f"[🔗 Read on Open Library]({book['link']})")
        
        recommended_titles = [book['title'] for book in results]
        st.session_state['search_history'].append({"searched": book_name, "recommendations": recommended_titles})
    else:
        st.error("No results found on Open Library.")

if st.button("Recommend"):
    if final_input == "":
        st.warning("Please enter or select a book title.")
    elif final_input in book_pivot.index:
        recommend_from_dataset(final_input)
    else:
        show_openlibrary_results(final_input)

if st.session_state['search_history']:
    st.markdown("---")
    st.markdown("## 📜 Search History")
    for item in st.session_state['search_history'][::-1]:
        with st.expander(f"🔍 {item['searched']}"):
            for rec in item['recommendations']:
                st.write(f"- {rec}")
