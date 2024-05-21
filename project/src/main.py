from import_lib import pd, plt
from preprocessing import *
from training import *
from recommander import *

path = "../dataset/ml-25m/"
ratings = pd.read_csv(path+"ratings.csv")
ratings = ratings.values

movies =pd.read_csv(path+"movies.csv")

M = len(np.unique(ratings[:,0]))#Number of users
N = len(np.unique(ratings[:,1]))#Number of movies rated

#
#Shuffle
np.random.shuffle(ratings)

#Data indexing
(
movie_to_id,
id_to_movie,
user_to_id,
id_to_user, 
data_by_user, 
data_by_movie,
number_of_ratings_per_users
) = data_indexing(ratings)


#Split data in training set and test set

(
	train_data_by_user,
	train_data_by_movie,
	test_data_by_user,
	test_data_by_movie
) = split_data(movie_to_id, id_to_movie, user_to_id, id_to_user, number_of_ratings_per_users, M, N, ratings)

(genres, dict_genres, features_per_movies) = feature_per_movies(N, movies,id_to_movie)

num_feat = len(genres)

movies_with_feature_i = movies_per_feature(num_feat, N, features_per_movies)

#Get all movie rated more than 20 times
movie_rated_more_than_20 = [n for n in range(N) if len(data_by_movie[n])>20]


#Hyperparamters
K = 10
lambd = 1
gamma = 0.01
tau = 0.5
tau_feat = 0.01
num_epoch = 20

# (
# 	user_vec,
# 	user_biases,
# 	item_vec,
# 	item_biases,
# 	train_loss_history,
# 	train_rmse_history,
# 	test_loss_history,
# 	test_rmse_history
# ) = training(train_data_by_user, train_data_by_movie, test_data_by_user, M, N, K=10, lambd=1, gamma=0.01, tau=1, num_epoch=20)


(
	user_vec,
	user_biases,
	item_vec,
	item_biases,
	feature_vec,
	train_loss_history,
	train_rmse_history,
	test_loss_history,
	test_rmse_history
) = training_with_features(
	train_data_by_user,
	train_data_by_movie,
	test_data_by_user,
	features_per_movies,
	movies_with_feature_i,
	M,
	N,
	K,
	lambd,
 	gamma,
	tau,
	tau_feat,
	num_epoch
)

# fig, ax = plt.subplots(figsize=(5,5))
# ax.plot(train_loss_history, label="Training")
# ax.plot(test_loss_history, label="Test")
# ax.set_xlabel("Number of iterations")
# ax.set_ylabel("Loss")
# ax.legend()
# # ax.title("Pos regularized log loss for biases+user and ")
# plt.savefig("reg_pos_log_loss_biases_with.pdf", format="pdf", bbox_inches="tight")
# plt.show()


#Prediction
# movie_id = movie_liked(movies)
# if movie_id>0:
# 	print(f"You like this movie : {movies.iloc[np.where(movies['movieId'].values == movie_id)[0][0]].title}\n")
# 	rates = 5
# 	movie_idx = movie_to_id[movie_id]

# 	recommended_movies = predict(movie_idx, rates, movie_rated_more_than_20, item_vec, item_biases, K, num_epoch, tau, lambd)
# 	#Looking for the original indices of these movies
# 	recommended_movies = [np.where(movies["movieId"].values == id_to_movie[idx])[0][0] for idx in recommended_movies]
# 	print("\nYou may also like this: ")
# 	print(movies.iloc[recommended_movies]["title"])
 


###
#@title Save my model
# %mkdir model_amls
# %mkdir model_amls/datasets
import pickle


path = "../dependencies/"

with open(path+"params.txt", "w") as file:
	file.write(f"K = {K}\nlambd={lambd}\ngamma={gamma}\ntau={tau}\ntau_feat={tau_feat}\nnum_epoch = {num_epoch}")

with open(path+"user_vec.txt", "wb") as f:
	pickle.dump(user_vec, f)

with open(path+"item_vec.txt", "wb") as f:
	pickle.dump(item_vec, f)

with open(path+"user_biases.txt", "wb") as f:
	pickle.dump(user_biases, f)

with open(path+"item_biases.txt", "wb") as f:
	pickle.dump(item_biases, f)

with open(path+"id_to_movie.txt", "wb") as f:
	pickle.dump(id_to_movie, f)

with open(path+"movie_to_id.txt", "wb") as f:
	pickle.dump(movie_to_id, f)

with open(path+"data_by_movie.txt", "wb") as f:
	pickle.dump(data_by_movie, f)

print("done...")
