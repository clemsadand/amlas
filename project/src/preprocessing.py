from tqdm import tqdm

def data_indexing(ratings):
	"""
	returns: movie_to_id, id_to_movie, user_to_id, id_to_user, data_by_user, data_by_movie
	"""
	#@title Indexing the whole dataset
	movie_to_id = {}#dictionary
	id_to_movie = []
	user_to_id = {}#dictionary
	id_to_user = []
	data_by_user = []
	data_by_movie = []

	for i in tqdm(range(ratings.shape[0])):
		u_id = ratings[i,0]
		#check if u_id is already in user_to_id
		if not u_id in user_to_id.keys():
			user_to_id[u_id] = len(user_to_id)
			id_to_user.append(u_id)
		#get the movieId watched by user
		m_id = ratings[i,1]
		#check if movieId (watched by user) is already in movie_to_id
		if not m_id in movie_to_id.keys():
			movie_to_id[m_id] = len(movie_to_id)
			id_to_movie.append(m_id)
		###########################################
		#add (movieId_new_id, rating) in the dict
		try:
			data_by_user[user_to_id[u_id]].append((movie_to_id[m_id], ratings[i,2]))
		except:
			data_by_user.append([(movie_to_id[m_id], ratings[i,2])])
		##########################################
		#add (user_new_id, rating) in the dict
		try:
			data_by_movie[movie_to_id[m_id]].append((user_to_id[u_id], ratings[i,2]))
		except:
			data_by_movie.append([(user_to_id[u_id], ratings[i,2])])

	#Number of ratings per users for the training set
	number_of_ratings_per_users = [int(0.8*len(l)) for l in data_by_user]
	#
	return (
		movie_to_id,
		id_to_movie,
		user_to_id,
		id_to_user, 
		data_by_user, 
		data_by_movie,
		number_of_ratings_per_users
	)


def split_data(movie_to_id, id_to_movie, user_to_id, id_to_user, number_of_ratings_per_users, M, N, ratings):
	"""
	returns: train_data_by_user, train_data_by_movie, test_data_by_user, test_data_by_movie
	"""
	#
	
	#Get number of ratings per user

	# movie_to_id = {}
	# id_to_movie = []
	# user_to_id = {}
	# id_to_user = []
	train_data_by_user = [[[], []] for _ in range(M)]
	train_data_by_movie = [[[], [], [0]*20] for _ in range(N)]#three components, the last reserved to feature embedding vector
	#
	test_data_by_user = [[[], []] for _ in range(M)]
	test_data_by_movie = [[[], [], [0]*20] for _ in range(N)]#three components, the last reserved to feature embedding vector

	#
	for i in tqdm(range(ratings.shape[0])):
		u_id = ratings[i, 0]
		if u_id not in user_to_id:
		  	user_to_id[u_id] = len(user_to_id)
		  	id_to_user.append(u_id)

		m_id = ratings[i, 1]
		if m_id not in movie_to_id:
		  	movie_to_id[m_id] = len(movie_to_id)
		  	id_to_movie.append(m_id)

		rating = ratings[i, 2]
		user_index = user_to_id[u_id]
		movie_index = movie_to_id[m_id]

		if len(train_data_by_user[user_to_id[u_id]][0]) < number_of_ratings_per_users[user_to_id[u_id]]:
		  	train_data_by_user[user_index][0].append(movie_index)
		  	train_data_by_user[user_index][1].append(rating)
		  	train_data_by_movie[movie_index][0].append(user_index)
		  	train_data_by_movie[movie_index][1].append(rating)
		else:
		  	test_data_by_user[user_index][0].append(movie_index)
		  	test_data_by_user[user_index][1].append(rating)
		  	test_data_by_movie[movie_index][0].append(user_index)
		  	test_data_by_movie[movie_index][1].append(rating)
	return (
		train_data_by_user,
		train_data_by_movie,
		test_data_by_user,
		test_data_by_movie
	)

def feature_per_movies(N, movies,id_to_movie):
	"""
	Returns the list of genres per movies
	
	args: 
	"""
	#Feature per movies
	genres = sorted(list(set('|'.join(list(movies["genres"])).split("|"))))#Get every genre in a list
	#Attribute a number to each movie
	dict_genres = dict(zip(genres, list(range(len(genres)))))#{'(no genres listed)': 0, 'Action': 1, 'Adventure': 2, 'Ani
	#Encode genres per movie
	features_per_movies = [[] for _ in range(N)]#a list of lists
	for n in tqdm(range(N)):
		#get the features of the movie whose new id is i
		genres_per_movie = list(movies[movies["movieId"] == id_to_movie[n]]["genres"])[0].split("|")
		for g in genres_per_movie:
			# test_data_by_movie[movie_index][2][dict_genres[g]] = 1
			features_per_movies[n].append(dict_genres[g])#[dict_genres[g]] = 1

	return (genres, dict_genres, features_per_movies) 

def movies_per_feature(num_feat, N, features_per_movies):
	#List of movies per feature
	movies_with_feature_i = [[] for _ in range(num_feat)]
	#loop over feature
	for i in range(num_feat):
		#loop over movies
		for n in range(N):
			if i in features_per_movies[n]:
				movies_with_feature_i[i].append(n)
	return movies_with_feature_i