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

The model's performance is evaluated using the Root Mean Square Error (RMSE) on both the training and test datasets. The RMSE decreases steadily over iterations, indicating the model's learning process.

## Example Recommendations

The system provides movie recommendations based on user preferences. For example, if a user likes "The Lord of the Rings," the system recommends other similar movies like "The Hobbit" and "Star Wars" series.

## Usage

To use this repository, follow these steps:

1. Clone the repository:
    ```
    git clone https://github.com/clemsadand/amlas.git
    cd recommender-system
    ```

2. Install the required dependencies:
    ```
    pip install -r requirements.txt
    ```

3. Run the training script:
    ```
    python train.py
    ```

4. Generate recommendations for a new user:
    ```
    python recommend.py --user_id <USER_ID>
    ```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open an issue.

<!---## License

This project is licensed under the MIT License.
--->

