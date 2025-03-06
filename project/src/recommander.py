from import_lib import np, pd


def predict(movie_idx, rates, movie_rated_more_than_20, item_vec, item_biases, K, num_epoch, tau, lambd):
	
	#train a dummpy user
	new_user = np.zeros(K)
	r_m = np.array([rates])
	movie_indices = [movie_idx]
	for epoch in range(num_epoch):
		term1 = lambd*np.dot(item_vec[movie_indices].T, item_vec[movie_indices])+tau*np.eye(K,K)
		term2 = r_m - item_biases[movie_indices]
		term2 = lambd*np.sum(item_vec[movie_indices]*term2.reshape(-1,1), axis=0)
		new_user = np.linalg.solve(term1, term2)
	#Compute the rates
	score_for_item = np.array([np.dot(new_user, item_vec[n])+item_biases[n] for n in movie_rated_more_than_20])
	# #Sort the score_for_item and their indices
	recommendations = -np.argsort(score_for_item)[:20]
	# #  and get the indices of 20 best rates
	recommended_movies = [movie_rated_more_than_20[n] for n in recommendations]
	return recommended_movies

def search_related_words(word, text_list):
    #Returns the indices of each sentences related to word in text_list
    indices = []
    for i in range(len(text_list)):
        words = text_list[i].lower().split()
        if word.lower() in words:
            indices.append(i)
    return indices

def movie_liked(movies):
    texts = list(movies["title"])
    # Get user input
    search_word = input("\nEnter a word to search for related movies: ")

    # Search for related words
    results = search_related_words(search_word, texts)
    movie_id = -1
    if results:
        print("\nRelated movies found:")
        #Display the related movies
        for idx in results:
            print(f"{idx}: " + " " + texts[idx])

        idx = input("Enter the number of the movie: ")
        movie_id = movies.iloc[int(idx)]["movieId"]
    else:
        print("No related movies found.")
    return movie_id

if __name__ == "__main__":
    # texts
    movies = pd.read_csv("../dataset/ml-25m/movies.csv")
    
    
    
    movie_id = movie_liked(movies)
    
