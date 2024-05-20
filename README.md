# Movie Recommender System

This repository contains the implementation of a movie recommender system using collaborative filtering with a latent factor model. The system is built and trained using the MovieLens dataset and evaluates its performance using the Root Mean Square Error (RMSE) metric.

## Overview

The goal of this project is to build a recommender system that suggests movies to users based on their past ratings. The system uses a latent factor model, which represents users and items (movies) in a shared latent feature space. The model learns user and item vectors, as well as biases, to predict ratings and make recommendations. It is trained on a [MovieLens dataset](https://files.grouplens.org/datasets/movielens/ml-25m.zip).

The attached notebooks 3 and 4 provide an overview of how the recommender system works. 

## Model Description

### Latent Factor Model

In the latent factor model, each user $u_m$ and each item $v_{n}$ are associated with K-dimensional vectors. The predicted rating $\hat r_{mn}$ for a user $u_m$ and an item $v_n$ is given by:

$$
\hat r_{mn} = u_m^T * v_n + b^u_m + b^i_n
$$


where $b^u_m$ and $b^i_n$ are the biases for the user and item, respectively.

### Objective Function

The parameters are learned by minimizing the regularized negative log-likelihood:

$$
\begin{aligned}
\mathcal L &= \sum_{m} \sum_{n\in \Omega(m)} \frac{\lambda}{2} (r_{mn} -(u_m^T v_n+ b_m^{(m)} + b_n^{(i)}))^2\\
&+\frac{\tau}{2}\left(\sum_{m} u_m^Tu_m + \sum_{n} v_n^Tv_n\right)\\
&+\frac{\gamma}{2}\left(\sum_{m} (b_{m}^{(u)})^2 + \sum_{n} (b_{n}^{(i)})^2\right)
\end{aligned}
$$


Regularization terms with mean-zero Gaussian priors are added to the user and item vectors and biases.

### Optimization

The optimization is performed using the Alternative Least Squares (ALS) algorithm. The ALS algorithm iteratively updates the user vectors, item vectors, and biases by solving a series of least squares problems.

## Implementation

The implementation includes the following steps:

1. **Data Preprocessing**: Load and preprocess the MovieLens dataset.
2. **Model Training**: Train the model using the ALS algorithm.
3. **Evaluation**: Evaluate the model's performance using RMSE.
4. **Prediction**: Generate movie recommendations for users.

## Evaluation

The model's performance is evaluated using the Root Mean Square Error (RMSE) on both the training and test datasets. The RMSE decreases steadily over iterations, indicating the model's learning process. The plot below illustrates the change the RMSE over iterations.

[reg_rmse_for_biases_with (4).pdf](https://github.com/clemsadand/amlas/files/15381390/reg_rmse_for_biases_with.4.pdf)

## Example Recommendations

The system provides movie recommendations based on user preferences. For example, if a user likes "The Lord of the Rings," the system recommends other similar movies like "The Hobbit" and "Star Wars" series.

When a user likes (gives starts) the movie: **Avengers: Age of Ultron (2015)**, these the recommended movie by the system.

| movieId | title                                              | genres                                       |
|---------|----------------------------------------------------|----------------------------------------------|
| 25068   | 122914 | Avengers: Infinity War - Part II (2019)              | Action|Adventure|Sci-Fi                      |
| 25067   | 122912 | Avengers: Infinity War - Part I (2018)               | Action|Adventure|Sci-Fi                      |
| 25071   | 122920 | Captain America: Civil War (2016)                   | Action|Sci-Fi|Thriller                       |
| 17067   | 89745  | Avengers, The (2012)                                | Action|Adventure|Sci-Fi|IMAX                 |
| 21348   | 110102 | Captain America: The Winter Soldier (2014)          | Action|Adventure|Sci-Fi|IMAX                 |
| 25069   | 122916 | Thor: Ragnarok (2017)                               | Action|Adventure|Sci-Fi                      |
| 25058   | 122892 | Avengers: Age of Ultron (2015)                      | Action|Adventure|Sci-Fi                      |
| 25074   | 122926 | Untitled Spider-Man Reboot (2017)                   | Action|Adventure|Fantasy                     |
| 49883   | 179819 | Star Wars: The Last Jedi (2017)                     | Action|Adventure|Fantasy|Sci-Fi              |
| 25064   | 122906 | Black Panther (2017)                                | Action|Adventure|Sci-Fi                      |
| 25066   | 122910 | Captain Marvel (2018)                               | Action|Adventure|Sci-Fi                      |
| 20602   | 106487 | The Hunger Games: Catching Fire (2013)              | Action|Adventure|Sci-Fi|IMAX                 |
| 16725   | 88140  | Captain America: The First Avenger (2011)           | Action|Adventure|Sci-Fi|Thriller|War         |
| 29958   | 135133 | The Hunger Games: Mockingjay - Part 2 (2015)        | Adventure|Sci-Fi                            |
| 19678   | 102125 | Iron Man 3 (2013)                                   | Action|Sci-Fi|Thriller|IMAX                  |
| 23024   | 116823 | The Hunger Games: Mockingjay - Part 1 (2014)        | Adventure|Sci-Fi|Thriller                   |
| 25055   | 122886 | Star Wars: Episode VII - The Force Awakens (2015)   | Action|Adventure|Fantasy|Sci-Fi|IMAX         |
| 33522   | 143355 | Wonder Woman (2017)                                 | Action|Adventure|Fantasy                     |
| 53867   | 188301 | Ant-Man and the Wasp (2018)                         | Action|Adventure|Comedy|Fantasy|Sci-Fi       |
| 59844   | 201773 | Spider-Man: Far from Home (2019)                    | Action|Adventure|Sci-Fi                      |


## Usage

To use this repository, follow these steps:

1. Clone the repository:
    ```
    git clone https://github.com/clemsadand/amlas.git
    cd project
    ```

2. Install the required dependencies:
    ```
    pip install -r requirements.txt
    ```

3. Run the training script:
    ```
    python src/main.py
    ```

<!---4. Generate recommendations for a new user:
    ```
    python recommend.py --user_id <USER_ID>
    ```
--->

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open an issue.

<!---## License

This project is licensed under the MIT License.
--->

