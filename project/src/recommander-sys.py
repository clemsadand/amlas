from import_lib import pd, pickle
from preprocessing import *
from training import *
from recommander import *

#Hyperparameters
K, num_epoch, tau, lambd = 10, 20, 0.4, 5

#Load model's dependencies and parameters
def loading(path):
    with open(path+"data_by_movie.txt", "rb") as f:
        data_by_movie = pickle.load(f)
    with open(path+"movie_to_id.txt", "rb") as f:
        movie_to_id = pickle.load(f)
    with open(path+"id_to_movie.txt", "rb") as f:
        id_to_movie = pickle.load(f)
    with open(path+"item_vec.txt", "rb") as f:
        item_vec = pickle.load(f)
    with open(path+"item_biases.txt", "rb") as f:
        item_biases = pickle.load(f)
    
    return data_by_movie, movie_to_id, id_to_movie, item_vec, item_biases

#Dependencies loading
movies =pd.read_csv("../dataset/ml-25m/movies.csv")
data_by_movie, movie_to_id, id_to_movie, item_vec, item_biases = loading("../dependencies/")
movie_rated_more_than_20 = movie_rated_more_than_20 = [n for n in range(N) if len(data_by_movie[n])>20]

#Recommandation
movie_id = movie_liked(movies)
if movie_id>0:
	print(f"You like this movie : {movies.iloc[np.where(movies['movieId'].values == movie_id)[0][0]].title}\n")
	rates = 5
	movie_idx = movie_to_id[movie_id]
	recommended_movies = predict(movie_idx, rates, movie_rated_more_than_20, item_vec, item_biases, K, num_epoch, tau, lambd)
	#Looking for the original indices of these movies
	recommended_movies = [np.where(movies["movieId"].values == id_to_movie[idx])[0][0] for idx in recommended_movies]
	print("\nYou may also like this: ")
	print(movies.iloc[recommended_movies]["title"])
