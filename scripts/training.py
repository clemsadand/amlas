from import_lib import  np, tqdm

def loss_func(train_data_by_user, user_vec, item_vec, user_biases, item_biases, N, tau, gamma, lambd, feature_vec= None, tau_feat=None, features_per_movies=None):
	residual = 0
	count = 0
	for m in range(len(train_data_by_user)):
		movie_indices = train_data_by_user[m][0]
		r_m = np.array(train_data_by_user[m][1])
		count += len(movie_indices)
		residual += np.sum((r_m - (np.dot(user_vec[m].T, item_vec[movie_indices].T)+user_biases[m]+item_biases[movie_indices]))**2)
	if tau_feat !=None and feature_vec.any() !=None:
		diff_item_feat = 0
		feat_vec1 = np.array([np.sum(feature_vec[features_per_movies[n],:], axis=0)/np.sqrt(len(features_per_movies[n])) for n in range(N)])
		feat_vec2 = np.array([np.sum(feature_vec[features_per_movies[n],:], axis=0) for n in range(N)])
		loss = 0.5*lambd*residual + 0.5*tau*(np.sum(user_vec**2)+np.sum((item_vec - feat_vec1)**2)) + 0.5*tau_feat*np.sum(feat_vec2**2) + 0.5*gamma*(np.sum(user_biases**2)+np.sum(item_biases**2))
	else:
		loss = 0.5*lambd*residual + 0.5*tau*(np.sum(user_vec**2)+np.sum(item_vec**2))+ 0.5*gamma*(np.sum(user_biases**2)+np.sum(item_biases**2))
	rmse = np.sqrt(residual/count)
	return loss, rmse

def update_bias(movie_indices, r_m, u_vec, item_vec, item_biases, tau, gamma, lambd):
	numerator = np.sum(np.array(r_m) - np.dot(u_vec.T, item_vec[movie_indices,:].T)-item_biases[movie_indices])
	bias = lambd*numerator/(lambd*len(r_m)+gamma)
	return bias

def update_vector(movie_indices, r_m, u_bias, item_vec, item_biases, K, tau, gamma, lambd):
	term1 = lambd*np.dot(item_vec[movie_indices].T, item_vec[movie_indices])+tau*np.eye(K,K)
	term2 = np.array(r_m) - u_bias - item_biases[movie_indices]
	term2 = lambd*np.sum(item_vec[movie_indices]*term2.reshape(-1,1), axis=0)
	vector = np.linalg.solve(term1, term2)
	return vector

def update_item_vec_with_feature(user_indices, r_n, i_bias, user_vec, user_biases, feature_vec, movie_features, K, tau, gamma, lambd, tau_feat):
    # user_indices = train_data_by_movie[n][0]
    # r_n = np.array(train_data_by_movie[n][1])
    term1 = lambd*np.dot(user_vec[user_indices].T, user_vec[user_indices])+tau*np.eye(K,K)
    term2 = np.array(r_n) - user_biases[user_indices] - i_bias
    term2 = lambd*np.sum(user_vec[user_indices]*term2.reshape(-1,1), axis=0)+tau_feat*np.sum(feature_vec[movie_features,:], axis=0)/np.sqrt(len(movie_features))
    vector = np.linalg.solve(term1, term2)
    return vector

def update_feature(item_vec, feature_vec, movie_features, feature_movies, tau, tau_feat):
	denominator = tau_feat + tau* sum([1./np.sqrt(len(movie_features)) for n in feature_movies])
	feature = tau*np.sum(item_vec[feature_movies,:], axis=0)/denominator
	return feature

def training(
		train_data_by_user,
		train_data_by_movie,
		test_data_by_user,
		M,
		N,
		K = 10,
		lambd = 1,
		gamma = 0.01,
		tau = 0.5,
		num_epoch = 40
	):

	#Initialization
	user_vec = np.random.normal(0, 1/np.sqrt(K), size= (M, K))
	user_biases = np.zeros(M)
	item_vec = np.random.normal(0, 1/np.sqrt(K), size=(N,K))
	item_biases = np.zeros(N)
	##
	train_loss_history = []
	test_loss_history = []
	train_rmse_history = []
	test_rmse_history = []
	##
	for epoch in tqdm(range(num_epoch)):
		#user biases
		for m in range(M):
			movie_indices = train_data_by_user[m][0]
			r_m = train_data_by_user[m][1]
			user_biases[m] = update_bias(movie_indices, r_m, user_vec[m], item_vec, item_biases, tau, gamma, lambd)
		#item biases
		for n in range(N):
			user_indices = train_data_by_movie[n][0]
			r_n = train_data_by_movie[n][1]
			item_biases[n] = update_bias(user_indices, r_n, item_vec[n], user_vec, user_biases, tau, gamma, lambd)

		#user_vec
		for m in range(M):
			movie_indices = train_data_by_user[m][0]
			r_m = train_data_by_user[m][1]
			user_vec[m] = update_vector(movie_indices, r_m, user_biases[m], item_vec, item_biases, K, tau, gamma, lambd)
		#item_vec
		for n in range(N):
			user_indices = train_data_by_movie[n][0]
			r_n = train_data_by_movie[n][1]
			item_vec[n] = update_vector(user_indices, r_n, item_biases[n], user_vec, user_biases, K, tau, gamma, lambd)
		#Loss
		loss, rmse = loss_func(train_data_by_user, user_vec, item_vec, user_biases, item_biases, N, tau, gamma, lambd)
		train_loss_history.append(loss.copy())
		train_rmse_history.append(rmse.copy())
		test_loss, test_rmse = loss_func(test_data_by_user, user_vec, item_vec, user_biases, item_biases, N, tau, gamma, lambd)
		test_loss_history.append(test_loss.copy())
		test_rmse_history.append(test_rmse.copy())
	return (
		user_vec,
		user_biases,
		item_vec,
		item_biases,
		train_loss_history,
		train_rmse_history,
		test_loss_history,
		test_rmse_history
	)
 
 
def training_with_features(
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
	):
    #Initialization
	user_vec = np.random.normal(0, 1/np.sqrt(K), size= (M, K))
	user_biases = np.zeros(M)
	item_vec = np.random.normal(0, 1/np.sqrt(K), size=(N,K))
	item_biases = np.zeros(N)
	feature_vec = np.random.normal(0, 1/np.sqrt(K), size=(20,K))
	##
	train_loss_history = []
	test_loss_history = []
	train_rmse_history = []
	test_rmse_history = []
	##
	for epoch in tqdm(range(num_epoch)):
		#user biases
		for m in range(M):
			movie_indices = train_data_by_user[m][0]
			r_m = train_data_by_user[m][1]
			user_biases[m] = update_bias(movie_indices, r_m, user_vec[m], item_vec, item_biases, tau, gamma, lambd)
		#item biases
		for n in range(N):
			user_indices = train_data_by_movie[n][0]
			r_n = train_data_by_movie[n][1]
			item_biases[n] = update_bias(user_indices, r_n, item_vec[n], user_vec, user_biases, tau, gamma, lambd)

		#user_vec
		for m in range(M):
			movie_indices = train_data_by_user[m][0]
			r_m = train_data_by_user[m][1]
			user_vec[m] = update_vector(movie_indices, r_m, user_biases[m], item_vec, item_biases, K, tau, gamma, lambd)
		#item_vec
		for n in range(N):
			user_indices = train_data_by_movie[n][0]
			r_n = train_data_by_movie[n][1]
			i_bias = item_biases[n]
			movie_features = features_per_movies[n]
			item_vec[n] = update_item_vec_with_feature(user_indices, r_n, i_bias, user_vec, user_biases, feature_vec, movie_features, K, tau, gamma, lambd, tau_feat)
		#feature_vec
		for i in range(20):
			movie_features = features_per_movies[n]
			feature_movies = movies_with_feature_i[i]
			feature_vec[i,:] = update_feature(item_vec, feature_vec, movie_features, feature_movies, tau, tau_feat)
			
		#Loss
		loss, rmse = loss_func(train_data_by_user, user_vec, item_vec, user_biases, item_biases, N, tau, gamma, lambd, feature_vec, tau_feat, features_per_movies)
		train_loss_history.append(loss.copy())
		train_rmse_history.append(rmse.copy())
		test_loss, test_rmse = loss_func(test_data_by_user, user_vec, item_vec, user_biases, item_biases, N, tau, gamma, lambd, feature_vec, tau_feat, features_per_movies)
		test_loss_history.append(test_loss.copy())
		test_rmse_history.append(test_rmse.copy())
	return (
		user_vec,
		user_biases,
		item_vec,
		item_biases,
		feature_vec,
		train_loss_history,
		train_rmse_history,
		test_loss_history,
		test_rmse_history
	)

# def predict(movie_indices, rates, K, item_vec, lambd, tau, gamma):
# 	#get a movie id
# 	#4993#2116#
# 	movie_id = 122892#
# 	rate = 5
# 	K = 10
# 	# print(f"You rate this movie : {movies.iloc[np.where(movies['movieId'].values == movie_id)[0][0]].title}")
# 	#train a dummpy user
# 	new_user = np.zeros(K)
# 	# r_m = np.array([rate])
# 	# movie_indices = [movie_to_id[movie_id]]
# 	for epoch in range(num_epoch):
# 		term1 = lambd*np.dot(item_vec[movie_indices].T, item_vec[movie_indices])+tau*np.eye(K,K)
# 		term2 = np.array(rates) - item_biases[movie_indices]
# 		term2 = lambd*np.sum(item_vec[movie_indices]*term2.reshape(-1,1), axis=0)
# 		new_user = np.linalg.solve(term1, term2)

# 	#Compute the rates
# 	score_for_item = np.array([np.dot(new_user, item_vec[n])+item_biases[n] for n in movie_rated_more_than_20])
# 	# #Sort the score_for_item and their indices
# 	recommendations = sort_with_indices(score_for_item)[:20]#[np.where(movies["movieId"].values == id_to_movie[idx])[0][0] for idx in recommendation_idx]
# 	# #  and get the indices of 20 best rates
# 	recommended_movies = [movie_rated_more_than_20[n] for n in recommendations]
# 	# #Looking for the original indices of these movies
# 	recommended_movies = [np.where(movies["movieId"].values == id_to_movie[idx])[0][0] for idx in recommended_movies]

# 	movies.iloc[recommended_movies]