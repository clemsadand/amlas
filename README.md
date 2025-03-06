# Movie Recommender System

<img src="https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue" /> <img src="https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white" /> <img src ="https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white" />

This repository contains the implementation of a movie recommender system using collaborative filtering with a latent factor model. The system is built and trained using the MovieLens dataset and evaluates its performance using the Root Mean Square Error (RMSE) metric.

## 1. Overview

The goal of this project is to build a recommender system that suggests movies to users based on their past ratings. The system uses a latent factor model, which represents users and items (movies) in a shared latent feature space. The model learns user and item vectors, as well as biases, to predict ratings and make recommendations. It is trained on a [MovieLens dataset](https://files.grouplens.org/datasets/movielens/ml-25m.zip).

The attached notebooks [3](https://github.com/clemsadand/amlas/blob/main/2_4_amls_practice_3.ipynb) and [4](https://github.com/clemsadand/amlas/blob/main/2_0_AMLS_practice_4.ipynb) provide an overview of how the recommender system works. The notebook 3 implements a recommender system based only on the interaction between users and movies. In addition to this interaction, the in the notebook 4 implements a recommender system taking the genres of each movies into accounts. In the following, we describe the first recommender system.

## 2. Model Description

### 2.1 Latent Factor Model

In the latent factor model, each user $u_m$ and each item $v_{n}$ are associated with K-dimensional vectors. The predicted rating $\hat r_{mn}$ for a user $u_m$ and an item $v_n$ is given by:

$$
\hat r_{mn} = u_m^T \cdot v_n + b^u_m + b^i_n
$$


where $b^u_m$ and $b^i_n$ are the biases for the user and item, respectively.

### 2.2 Objective Function

The parameters are learned by minimizing the regularized negative log-likelihood:

$$
\begin{aligned}
\mathcal L &= \sum_{m} \sum_{n\in \Omega(m)} \frac{\lambda}{2} (r_{mn} -(u_m^T v_n+ b_m^{(m)} + b_n^{(i)}))^2\\
&+\frac{\tau}{2}\left(\sum_{m} u_m^Tu_m + \sum_{n} v_n^Tv_n\right)\\
&+\frac{\gamma}{2}\left(\sum_{m} (b_{m}^{(u)})^2 + \sum_{n} (b_{n}^{(i)})^2\right)
\end{aligned}
$$


Regularization terms with mean-zero Gaussian priors are added to the user and item vectors and biases.

### 2.3 Optimization

The optimization is performed using the Alternative Least Squares (ALS) algorithm. The ALS algorithm iteratively updates the user vectors, item vectors, and biases by solving a series of least squares problems.

## 3. Implementation

The implementation includes the following steps:

1. **Data Preprocessing**: Load and preprocess the MovieLens dataset.
2. **Model Training**: Train the model using the ALS algorithm.
3. **Evaluation**: Evaluate the model's performance using RMSE.
4. **Prediction**: Generate movie recommendations for users.

## 4. Evaluation

The model's performance is evaluated using the Root Mean Square Error (RMSE) on both the training and test datasets. The plot below illustrates the change the RMSE over iterations. The RMSE decreases steadily over iterations, indicating the model's learning process. 

![rmse_progress](https://github.com/clemsadand/amlas/assets/132694770/c055457f-92dd-4283-8cb0-9c24c4bcd2ce)

## 4.1 Example Recommendations

The system provides movie recommendations based on user preferences. 

When a user likes (gives starts) the movie: **Avengers: Age of Ultron (2015)**, the recommended movies by the system are listed asin following the table. 

You rate this movie: **Avengers: Age of Ultron (2015)**

| Title                                              |
|----------------------------------------------------|
| Avengers: Infinity War - Part II (2019)            |
| Avengers: Infinity War - Part I (2018)             |
| Captain America: Civil War (2016)                  |
| Avengers, The (2012)                               |
| Captain America: The Winter Soldier (2014)         |
| Thor: Ragnarok (2017)                              |
| Avengers: Age of Ultron (2015)                     |
| Untitled Spider-Man Reboot (2017)                  |
| Star Wars: The Last Jedi (2017)                    |
| Black Panther (2017)                               |
| Captain Marvel (2018)                              |
| The Hunger Games: Catching Fire (2013)             |
| Captain America: The First Avenger (2011)          |
| The Hunger Games: Mockingjay - Part 2 (2015)       |
| Iron Man 3 (2013)                                  |
| The Hunger Games: Mockingjay - Part 1 (2014)       |
| Star Wars: Episode VII - The Force Awakens (2015)  |
| Wonder Woman (2017)                                |
| Ant-Man and the Wasp (2018)                        |
| Spider-Man: Far from Home (2019)                   |


## 4.2 Usage

To use this repository, follow these steps:

1. Clone the repository:
    ```
    git clone https://github.com/clemsadand/amlas.git
    cd amlas/project
    ```
<!--
2. Install the required dependencies:
    ```
    pip install -r requirements.txt
    ```-->

2. Run the training script:
    ```
    python src/main.py
    ```

<!---4. Generate recommendations for a new user:
    ```
    python recommend.py --user_id <USER_ID>
    ```
--->

## 4.3 Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open an issue.

## License

This project is licensed under the MIT License.


