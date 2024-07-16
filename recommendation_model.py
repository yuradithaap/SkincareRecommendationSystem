import pandas as pd
import numpy as np
import pickle

def preprocess_data():
    data = pd.read_csv('skincares.csv')

    # Cleansing Data
    data_clean = data.dropna().drop_duplicates()
    data_clean = data_clean[~(data_clean['user'] == ' ')]
    id_count = pd.crosstab(index=data_clean.user, columns='count').sort_values(by='count', ascending=True)
    name_r = id_count[id_count['count'] > 1]
    name_u = name_r.index.to_list()
    data_clean = data_clean[data_clean.user.isin(name_u)]
    data_clean['rate'] = pd.to_numeric(data_clean['rate'], errors='coerce')
    return data_clean

def pivot_data(data_clean):
    data_pivot = pd.pivot_table(data_clean, values='rate', index='user', columns='id').fillna(0)
    with open('pivot.pkl', 'wb') as f:
        pickle.dump(data_pivot, f)
    return data_pivot

def calculate_similarity(data_pivot, data_clean):
    # Hitung Pearson Correlation
    corr = data_pivot.T.corr(method='pearson').round(2)
    with open('similarity.pkl', 'wb') as f:
        pickle.dump(corr, f)
    return corr

def calculate_positive_reviews(data_clean):
    reviews = data_clean[['id','user','rate','reviews']]

    reviews = reviews.dropna().drop_duplicates()

    def lowercase(review_text):
        low = review_text.lower()
        return low

    reviews['reviews'] = reviews['reviews'].apply(lambda low: lowercase(str(low)))

    import re
    def remove_emoji(review_text):
        emoji_pattern = re.compile("["
                                   u"\U0001F600-\U0001F64F"  # emoticons
                                   u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                   u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                   u"\U00002702-\U000027B0"
                                   u"\U000024C2-\U0001F251"
                                   u"\U0001f926-\U0001f937"
                                   u"\U00010000-\U0010ffff"
                                   "]+", flags=re.UNICODE)
        return emoji_pattern.sub(r'', review_text)

    reviews['reviews'] = reviews['reviews'].apply(lambda emoji: remove_emoji(emoji))

    def remove_hashtag(review_text, default_replace=""):
        hashtag = re.sub(r'#\w+', default_replace, review_text)
        return hashtag

    reviews['reviews'] = reviews['reviews'].apply(lambda hashtag: remove_hashtag(hashtag))

    def remove_number(review_text, default_replace=" "):
        num = re.sub(r'\d+', default_replace, review_text)
        return num

    reviews['reviews'] = reviews['reviews'].apply(lambda num: remove_number(num))

    import string
    def remove_punctuation(review_text, default_text=" "):
        list_punct = string.punctuation
        delete_punct = str.maketrans(list_punct, ' ' * len(list_punct))
        new_review = ' '.join(review_text.translate(delete_punct).split())

        return new_review

    reviews['reviews'] = reviews['reviews'].apply(lambda punct: remove_punctuation(punct))

    def remove_superscript(review_text):
        number = re.compile("["u"\U00002070"
                            u"\U000000B9"
                            u"\U000000B2-\U000000B3"
                            u"\U00002074-\U00002079"
                            u"\U0000207A-\U0000207E"
                            u"U0000200D"
                            "]+", flags=re.UNICODE)
        return number.sub(r'', review_text)

    reviews['reviews'] = reviews['reviews'].apply(lambda num: remove_superscript(num))

    def word_repetition(review_text):
        review = re.sub(r'(.)\1+', r'\1\1', review_text)
        return review

    reviews['reviews'] = reviews['reviews'].apply(lambda word: word_repetition(word))

    def repetition(review_text):
        repeat = re.sub(r'\b(\w+)(?:\W\1\b)+', r'\1', review_text, flags=re.IGNORECASE)
        return repeat

    reviews['reviews'] = reviews['reviews'].apply(lambda word: repetition(word))

    def remove_extra_whitespaces(review_text):
        review = re.sub(r'\s+', ' ', review_text)
        return review

    reviews['reviews'] = reviews['reviews'].apply(lambda extra_spaces: remove_extra_whitespaces(extra_spaces))

    reviews['rate'] = pd.to_numeric(reviews['rate'], errors='coerce')

    lexicon_positive = pd.read_csv('kata_positif.csv')
    lexicon_positive_dict = {}
    for index, row in lexicon_positive.iterrows():
        if row[0] not in lexicon_positive_dict:
            lexicon_positive_dict[row[0]] = row[1]

    lexicon_negative = pd.read_csv('kata_negatif.csv')
    lexicon_negative_dict = {}
    for index, row in lexicon_negative.iterrows():
        if row[0] not in lexicon_negative_dict:
            lexicon_negative_dict[row[0]] = row[1]

    def sentiment_analysis_lexicon_indonesia(ulasan, rating):
        if isinstance(ulasan, str):
            score = 0
            for word in ulasan.split():
                if word in lexicon_positive_dict:
                    score += lexicon_positive_dict[word]
                if word in lexicon_negative_dict:
                    score += lexicon_negative_dict[word]

            if rating >= 4:
                score *= 1.2
                score = np.abs(score)
            elif rating <= 3:
                score *= 0.8

            sentiment = 'positif' if score > 0 else ('negatif' if score < 0 else 'netral')
            return score, sentiment
        else:
            return None, 'tidak valid'

    results = reviews.apply(lambda row: sentiment_analysis_lexicon_indonesia(row['reviews'], row['rate']), axis=1)
    results = list(zip(*results))

    reviews['skor'] = results[0]
    reviews['label'] = results[1]
    new_reviews = reviews[['id','reviews','label']]

    produk_positif = new_reviews[new_reviews['label'] == 'positif'].groupby(['id']).size().reset_index(name='positif_reviews')

    total_reviews_per_product = reviews.groupby('id')['reviews'].count()
    threshold = total_reviews_per_product * 0.5
    top_produk_positif = []
    for product in total_reviews_per_product.index:
        product_threshold = threshold[product]
        produk = produk_positif[(produk_positif['id'] == product) & (produk_positif['positif_reviews'] >= product_threshold)]
        top_produk_positif.append(produk)
    top_produk_positif = pd.concat(top_produk_positif)
    top_produk_positif = top_produk_positif.set_index('id')['positif_reviews'].to_dict()

    positif = produk_positif[produk_positif['positif_reviews'] >= 0]
    positif.to_csv('positif.csv', index=False)

    with open('positif.pkl', 'wb') as f:
        pickle.dump(top_produk_positif, f)
    return top_produk_positif

def weighted_sum_similarity(similarity_matrix, user_ratings, user_i, product_p):
    similarities = similarity_matrix[user_i]
    other_user_ratings = user_ratings.loc[:, product_p]
    positive_similarities = similarities[similarities > 0.3]
    weighted_sum_similarity = np.sum(positive_similarities * other_user_ratings[similarities > 0.3]) / np.sum(
        np.abs(positive_similarities))
    return weighted_sum_similarity

def predict_ratings(user_id, corr, produk_positif, data_pivot):
    # Hitung Prediksi
    result_list = []
    predicted_ratings = {}
    if user_id not in data_pivot.index:
        return {"message": "User ID tidak terdaftar"}, 404

    for product_p in data_pivot.columns:
        prediction = weighted_sum_similarity(corr, data_pivot, user_id, product_p)
        if product_p in produk_positif:
            predicted_ratings[product_p] = prediction

    df = pd.DataFrame(list(predicted_ratings.items()), columns=['id', 'predicted_rate']).round(2)
    df1 = pd.DataFrame(list(produk_positif.items()), columns=['id', 'positif_reviews']).round(2)

    product = pd.merge(df, df1, on='id', how='left')

    product = product[product['predicted_rate'] > 0]
    top_predicted = product.sort_values(by=['predicted_rate', 'positif_reviews'], ascending=[False, False])
    # top_predicted_dict = top_predicted.set_index('name')[['predicted_rate', 'positif_reviews']].apply(
    #     lambda row: {"predicted rate": row['predicted_rate'], "positive reviews": row['positif_reviews']}, axis=1
    # ).to_dict()
    top_predicted = top_predicted.head(10)
    top_predicted['positif_reviews'] = pd.to_numeric(top_predicted['positif_reviews'], errors='coerce')
    for index, row in top_predicted.iterrows():
        # Membuat dictionary untuk setiap baris dengan format yang diinginkan
        product_dict = {
            "id": row['id'],
            "predicted rate": row['predicted_rate'],
            "positive reviews": row['positif_reviews']
        }
        # Menambahkan dictionary ke dalam list
        result_list.append(product_dict)
    return result_list

def panggil_jam_24():
    data_clean = preprocess_data()
    data_pivot = pivot_data(data_clean)
    corr = calculate_similarity(data_pivot, data_clean)
    produk_positif = calculate_positive_reviews(data_clean)
    new_rate_positif(data_clean, produk_positif)
    update()

def new_rate_positif(data_clean, positif):
    average = data_clean.groupby('id')['rate'].mean().round(1)
    average_df = pd.DataFrame(list(average.items()), columns=['id', 'rate'])
    produk_positif_df = pd.DataFrame(list(positif.items()), columns=['id', 'positif_reviews'])
    merged_data = pd.merge(average_df, produk_positif_df, on='id', how='left')
    merged_data.to_csv('positif.csv', index=False)
    return average_df

def update():
    product_df = pd.read_csv('product.csv')
    positif_df = pd.read_csv('positif.csv')
    merged_df = pd.merge(product_df, positif_df[['id', 'positif_reviews']], on='id', how='left')
    merged_df['positif'] = merged_df['positif_reviews'].fillna(merged_df['positif'])
    merged_df.drop(columns=['positif_reviews'], inplace=True)
    merged_df.rename(columns={'positif': 'positif'}, inplace=True)
    merged_df.to_csv('product.csv', index=False)

# panggil_jam_24()