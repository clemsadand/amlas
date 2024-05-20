from import_lib import pd, plt
from preprocessing import *
from training import *

path = ""#"ml-25m/"
ratings = pd.read_csv(path+"ratings.csv")
ratings = ratings.values#[:1000,:]
movies =pd.read_csv(path+"movies.csv")
#
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
	K = 10,
	lambd = 1,
	gamma = 0.01,
	tau = 0.5,
	tau_feat = 0.01,
	num_epoch = 10,
	)

fig, ax = plt.subplots(figsize=(5,5))
ax.plot(list(range(len(train_loss_history))), train_loss_history, label="Training")
ax.plot(list(range(len(train_loss_history))), test_loss_history, label="Test")
ax.set_xlabel("Number of iterations")
ax.set_ylabel("Loss")
ax.legend()
# ax.title("Pos regularized log loss for biases+user and ")
plt.savefig("reg_pos_log_loss_biases_with.pdf", format="pdf", bbox_inches="tight")
plt.show()

