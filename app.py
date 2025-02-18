from flask import Flask, render_template, request
import joblib  # Use joblib instead of pickle
import pandas as pd

app = Flask(__name__)

# Load the model using joblib
model_filename = 'knn_movie_recommendation_model.joblib'
model = joblib.load(model_filename)

file_path = "C:/Users/Administrator/Downloads/ML Data/movies_recommendation_system/movies.xlsx"
data = pd.read_excel(file_path, engine="openpyxl")
data.dropna(inplace=True)
data.drop_duplicates(subset=["Movies Titles"], inplace=True)
data.reset_index(drop=True, inplace=True)
data["Genres"] = data["Genres"].str.lower().str.strip()
data["Genres"] = data["Genres"].apply(lambda x: x.split(",") if "," in x else [x])

from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer()
genres_encoded = mlb.fit_transform(data["Genres"])
genres_df = pd.DataFrame(genres_encoded, columns=mlb.classes_)

data = pd.concat([data, genres_df], axis=1)
data.drop(columns=["Genres"], inplace=True)
X = data[genres_df.columns]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    name = request.form.get('name')
    age = request.form.get('age')
    selected_genres = request.form.getlist('genres')

    user_input_encoded = [1 if genre in selected_genres else 0 for genre in genres_df.columns]

    # Use the model to make predictions
    distances, indices = model.kneighbors([user_input_encoded])

    # Get the recommended movies
    recommended_movies = data.iloc[indices[0]]['Movies Titles'].tolist()

    return render_template('recommendations.html', name=name, age=age, recommended_movies=recommended_movies)

if __name__ == '__main__':
    app.run(debug=True)
