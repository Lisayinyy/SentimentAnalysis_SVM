"""
EECS 445 - Winter 2024

Project 1 main file.
"""


import itertools
import string
import warnings
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from gensim.models import Word2Vec
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.metrics import roc_curve
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

from helper import *


warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=ConvergenceWarning)

np.random.seed(445)



def extract_word(input_string: str) -> list[str]:
    """Preprocess review text into list of tokens.

    Convert input string to lowercase, replace punctuation with spaces, and split along
    whitespace. Return the resulting array.

    Example:
        > extract_word("I love EECS 445. It's my favorite course!")
        > ["i", "love", "eecs", "445", "it", "s", "my", "favorite", "course"]

    Args:
        input_string: text for a single review

    Returns:
        a list of words, extracted and preprocessed according to the directions
        above.
    """
    # TODO: Implement this function
    lower_case_string = input_string.lower()

    for punctuation in string.punctuation:
        lower_case_string = lower_case_string.replace(punctuation, ' ')
    
    return lower_case_string.split()




def extract_dictionary(df: pd.DataFrame) -> dict[str, int]:
    """
    Map words to index.

    Reads a pandas dataframe, and returns a dictionary of distinct words mapping from each
    distinct word to its index (ordered by when it was found).

    Example:
        Input df:

        | reviewText                    | label | ... |
        | It was the best of times.     |  1    | ... |
        | It was the blurst of times.   | -1    | ... |

        The output should be a dictionary of indices ordered by first occurence in
        the entire dataset. The index should be autoincrementing, starting at 0:

        {
            it: 0,
            was: 1,
            the: 2,
            best: 3,
            of: 4,
            times: 5,
            blurst: 6,
        }

    Args:
        df: dataframe/output of load_data()

    Returns:
        a dictionary mapping words to an index
    """
    word_dict = {}
    idx = 0

    # Iterate over the array of text.
    for text in df['reviewText']:
        # Extract words using the 'extract_word' function.
        words = extract_word(text)
        for word in words:
            if word not in word_dict:
                word_dict[word] = idx
                idx += 1  # Increment the index for the next unique word.

    return word_dict


def generate_feature_matrix(
    df: pd.DataFrame, word_dict: dict[str, int]
) -> npt.NDArray[np.float64]:
    """
    Create matrix of feature vectors for dataset.

    Reads a dataframe and the dictionary of unique words to generate a matrix
    of {1, 0} feature vectors for each review. For each review, extract a token
    list and use word_dict to find the index for each token in the token list.
    If the token is in the dictionary, set the corresponding index in the review's
    feature vector to 1. The resulting feature matrix should be of dimension
    (# of reviews, # of words in dictionary).

    Args:
        df: dataframe that has the text and labels
        word_dict: dictionary of words mapping to indices

    Returns:
        a numpy matrix of dimension (# of reviews, # of words in dictionary)
    """
    number_of_reviews = df.shape[0]
    number_of_words = len(word_dict)
    feature_matrix = np.zeros((number_of_reviews, number_of_words))
    # TODO: Implement this function
    for i, review in enumerate(df['reviewText']):  
        tokens = extract_word(review)
        for token in tokens:
            if token in word_dict: 
                feature_matrix[i, word_dict[token]] = 1  

    return feature_matrix




def performance(
    y_true: npt.NDArray[np.float64],
    y_pred: npt.NDArray[np.int64],
    metric: str = "accuracy",
) -> np.float64:
    """
    Calculate performance metrics.

    Performance metrics are evaluated on the true labels y_true versus the
    predicted labels y_pred.

    Args:
        y_true: (n,) array containing known labels
        y_pred: (n,) array containing predicted scores
        metric: string specifying the performance metric (default='accuracy'
                 other options: 'f1-score', 'auroc', 'precision', 'sensitivity',
                 and 'specificity')

    Returns:
        the performance as an np.float64
    """
    # TODO: Implement this function
    # This is an optional but very useful function to implement.
    # See the sklearn.metrics documentation for pointers on how to implement
    # the requested metrics.
    if metric == 'accuracy':
        return metrics.accuracy_score(y_true, y_pred)
    elif metric == 'f1-score':
        return metrics.f1_score(y_true, y_pred)
    elif metric == 'auroc':
        return metrics.roc_auc_score(y_true, y_pred)
    elif metric == 'precision':
        return metrics.precision_score(y_true, y_pred)
    elif metric == 'sensitivity':
        return metrics.recall_score(y_true, y_pred)
    elif metric == 'specificity':
        tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
        return tn / (tn + fp)
    else:
        raise ValueError("Invalid metric")

def cv_performance(
    clf: LinearSVC | SVC,
    X: npt.NDArray[np.float64],
    y: npt.NDArray[np.int64],
    k: int = 5,
    metric: str = "accuracy",
) -> float:
    """
    Split data into k folds and run cross-validation.

    Splits the data X and the labels y into k-folds and runs k-fold
    cross-validation: for each fold i in 1...k, trains a classifier on
    all the data except the ith fold, and tests on the ith fold.
    Calculates and returns the k-fold cross-validation performance metric for
    classifier clf by averaging the performance across folds.

    Args:
        clf: an instance of LinearSVC() or SVC()
        X: (n,d) array of feature vectors, where n is the number of examples
           and d is the number of features
        y: (n,) array of binary labels {1, -1}
        k: an int specifying the number of folds (default=5)
        metric: string specifying the performance metric (default='accuracy'
             other options: 'f1-score', 'auroc', 'precision', 'sensitivity',
             and 'specificity')

    Returns:
        average 'test' performance across the k folds as np.float64
    """
    # TODO: Implement this function
    # HINT: You may find the StratifiedKFold from sklearn.model_selection
    # to be useful

    scores = []
    skf = StratifiedKFold(n_splits=k)
    
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        clf.fit(X_train, y_train)

        if metric == 'auroc':
            y_pred = clf.decision_function(X_test)
        else:
            y_pred = clf.predict(X_test)

        score = performance(y_test, y_pred, metric)
        scores.append(score)

    return np.array(scores).mean()

def select_param_linear(
    X: npt.NDArray[np.float64],
    y: npt.NDArray[np.int64],
    k: int = 5,
    metric: str = "accuracy",
    C_range: list[float] = [],
    loss: str = "hinge",
    penalty: str = "l2",
    dual: bool = True,
) -> float:
    """
    Search for hyperparameters from the given candidates of linear SVM with
    best k-fold CV performance.

    Sweeps different settings for the hyperparameter of a linear-kernel SVM,
    calculating the k-fold CV performance for each setting on X, y.

    Args:
        X: (n,d) array of feature vectors, where n is the number of examples
        and d is the number of features
        y: (n,) array of binary labels {1,-1}
        k: int specifying the number of folds (default=5)
        metric: string specifying the performance metric (default='accuracy',
             other options: 'f1-score', 'auroc', 'precision', 'sensitivity',
             and 'specificity')
        C_range: an array with C values to be searched over
        loss: string specifying the loss function used (default="hinge",
             other option of "squared_hinge")
        penalty: string specifying the penalty type used (default="l2",
             other option of "l1")
        dual: boolean specifying whether to use the dual formulation of the
             linear SVM (set True for penalty "l2" and False for penalty "l1")

    Returns:
        the parameter value for a linear-kernel SVM that maximizes the
        average 5-fold CV performance.
    """
    # TODO: Implement this function
    # HINT: You should be using your cv_performance function here
    # to evaluate the performance of each SVM
    best_score = -np.inf
    best_C = None

    for C in C_range:
        # Initialize a LinearSVC with the current C value and other fixed hyperparameters
        clf = LinearSVC(C=C, loss=loss, penalty=penalty, dual=dual, random_state=445)
        
        # Calculate the k-fold CV performance
        score = cv_performance(clf, X, y, k=k, metric=metric)
        
        # Update best_score and best_C if the current model is better
        if score > best_score:
            best_score = score
            best_C = C

    return best_C


def plot_weight(
    X: npt.NDArray[np.float64],
    y: npt.NDArray[np.int64],
    penalty: str,
    C_range: list[float],
    loss: str,
    dual: bool,
) -> None:
    """
    Create a plot of the L0 norm learned by a classifier for each C in C_range.

    Args:
        X: (n,d) array of feature vectors, where n is the number of examples
        and d is the number of features
        y: (n,) array of binary labels {1,-1}
        penalty: string for penalty type to be forwarded to the LinearSVC constructor
        C_range: list of C values to train a classifier on
        loss: string for loss function to be forwarded to the LinearSVC constructor
        dual: whether to solve the dual or primal optimization problem, to be
            forwarded to the LinearSVC constructor

    Returns: None
        Saves a plot of the L0 norms to the filesystem.
    """
    norm0 = []
    # TODO: Implement this part of the function
    # Here, for each value of c in C_range, you should
    # append to norm0 the L0-norm of the theta vector that is learned
    # when fitting an L2- or L1-penalty, degree=1 SVM to the data (X, y)

    for C in C_range:
        clf = LinearSVC(C=C, loss=loss, penalty=penalty, dual=dual)
        clf.fit(X, y)
        norm0.append(np.sum(clf.coef_ != 0))

    # Plot the L0-norm as a function of C
    plt.plot(C_range, norm0)
    plt.xscale("log")
    plt.legend(["L0-norm"])
    plt.xlabel("Value of C")
    plt.ylabel("Norm of theta")
    plt.title("Norm-" + penalty + "_penalty.png")
    plt.savefig("Norm-" + penalty + "_penalty.png")
    plt.close()


def select_param_quadratic(
    X: npt.NDArray[np.float64],
    y: npt.NDArray[np.int64],
    k: int = 5,
    metric: str = "accuracy",
    param_range: npt.NDArray[np.float64] = [],
) -> tuple[float, float]:
    """
    Search for hyperparameters from the given candidates of quadratic SVM
    with best k-fold CV performance.

    Sweeps different settings for the hyperparameters of a quadratic-kernel SVM,
    calculating the k-fold CV performance for each setting on X, y.

    Args:
        X: (n,d) array of feature vectors, where n is the number of examples
           and d is the number of features
        y: (n,) array of binary labels {1,-1}
        k: an int specifying the number of folds (default=5)
        metric: string specifying the performance metric (default='accuracy'
                 other options: 'f1-score', 'auroc', 'precision', 'sensitivity',
                 and 'specificity')
        param_range: a (num_param, 2)-sized array containing the
            parameter values to search over. The first column should
            represent the values for C, and the second column should
            represent the values for r. Each row of this array thus
            represents a pair of parameters to be tried together.

    Returns:
        The parameter values for a quadratic-kernel SVM that maximize
        the average 5-fold CV performance as a pair (C,r)
    """
    # TODO: Implement this function
    # Hint: This will be very similar to select_param_linear, except
    # the type of SVM model you are using will be different...
    best_C_val, best_r_val = 0.0, 0.0
    best_score = -np.inf
    perf_record = []

    for C, r in param_range:
        # Initialize a SVC with the current C and r values and other fixed hyperparameters
        clf = SVC(C=C, kernel="poly", degree=2, coef0=r, gamma = 'auto', random_state=445)
        
        # Calculate the k-fold CV performance
        score = cv_performance(clf, X, y, k=k, metric=metric)
        
        # Update best_score, best_C_val, and best_r_val if the current model is better
        if score > best_score:
            best_score = score
            best_C_val = C
            best_r_val = r
        
        perf_record.append((C, r, score))

    return best_C_val, best_r_val, perf_record


def train_word2vec(filename: str) -> Word2Vec:
    """
    Train a Word2Vec model using the Gensim library.

    First, iterate through all reviews in the dataframe, run your extract_word() function
    on each review, and append the result to the sentences list. Next, instantiate an
    instance of the Word2Vec class, using your sentences list as a parameter and using workers=1.

    Args:
        filename: name of the dataset csv

    Returns:
        created Word2Vec model
    """
    df = load_data(filename)
    sentences = []
    # TODO: Complete this function
    # Extract words from each review and add to sentences list
    for review in df['reviewText']:
        sentences.append(extract_word(review))
    # Train a Word2Vec model using the sentences list
    model = Word2Vec(sentences, workers=1)
    return model


def compute_association(filename: str, w: str, A: list[str], B: list[str]) -> float:
    """
    Args:
        filename: name of the dataset csv
        w: a word represented as a string
        A: set of English words
        B: set of English words

    Returns:
        association between w, A, and B as defined in the spec
    """
    model = train_word2vec(filename)

    # First, we need to find a numerical representation for the English language words in A and B
    # TODO: Complete words_to_array()
    def words_to_array(s: list[str]) -> npt.NDArray[np.float64]:
        """Convert a list of string words into a 2D numpy array of word embeddings,
        where the ith row is the embedding vector for the ith word in the input set (0-indexed).

            Args:
                s (list[str]): List of words to convert to word embeddings

            Returns:
                npt.NDArray[np.float64]: Numpy array of word embeddings
        """
        return np.array([model.wv[word] for word in s])
    
    # TODO: Complete cosine_similarity()
    def cosine_similarity(
        array: npt.NDArray[np.float64], w: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Calculate the cosine similarities between w and the input set.

        Args:
            array: array representation of the input set
            w: word embedding for w

        Returns:
            1D Numpy Array where the ith element is the cosine similarity between the word
            embedding for w and the ith embedding in input set
        """
        return np.array([np.dot(w, a) / (np.linalg.norm(w) * np.linalg.norm(a)) for a in array])
    

    # Although there may be some randomness in the word embeddings, we have provided the
    # following test case to help you debug your cosine_similarity() function:
    # This is not an exhaustive test case, so add more of your own!
    test_arr = np.array([[4, 5, 6], [9, 8, 7]])
    test_w = np.array([1, 2, 3])
    test_sol = np.array([0.97463185, 0.88265899])
    assert np.allclose(
        cosine_similarity(test_arr, test_w), test_sol, atol=0.00000001
    ), "Cosine similarity test 1 failed"

    # TODO: Return the association between w, A, and B.
    #      Compute this by finding the difference between the mean cosine similarity between w and the words in A,
    #      and the mean cosine similarity between w and the words in B
    A_array = words_to_array(A)
    B_array = words_to_array(B)
    w_array = model.wv[w]
    return np.mean(cosine_similarity(A_array, w_array)) - np.mean(cosine_similarity(B_array, w_array))




def main() -> None:
    # Read binary data
    # NOTE: Use the X_train, Y_train, X_test, and Y_test provided below as the training set and test set
    #       for the reviews in the file you read in.
    #
    #       Your implementations of extract_dictionary() and generate_feature_matrix() will be called
    #       to produce these training and test sets (for more information, see get_split_binary_data() in helper.py).
    #       DO NOT reimplement or edit the code we provided in get_split_binary_data().
    #
    #       Please note that dictionary_binary will not be correct until you have correctly implemented extract_dictionary(),
    #       and X_train, Y_train, X_test, and Y_test will not be correct until you have correctly
    #       implemented extract_dictionary() and generate_feature_matrix().
    filename = "data/dataset.csv"


    X_train, Y_train, X_test, Y_test, dictionary_binary = get_split_binary_data(
        filename=filename
    )
    

    # TODO: Questions 2, 3, 4, 5
    
    # Questions 2
    # (a)
    test_sentence = "It's a test sentence! Does it look CORRECT?"
    extracted_words = extract_word(test_sentence)
    print(extracted_words)

    # (b)
    print("d: ", len(dictionary_binary))

    # (c)
    # The average number of non-zero features per review in the training data.
    print("The average number of non-zero features per review in the training data: ", np.mean(np.sum(X_train, axis=1)))

    word_idx = np.argmax(np.sum(X_train, axis=0))
    word = list(dictionary_binary.keys())[list(dictionary_binary.values()).index(word_idx)]
    print("The word appearing in the greatest number of reviews: ", word)

    
    # Question 3
    # 3.1
    # (b)
    # Define the range of C values to test between 10^−3 and 10^3
    C_range = [10**i for i in range(-3, 4)]

    # Define the performance metrics to test
    performance_metrics = ["accuracy", "f1-score", "auroc", "precision", "sensitivity", "specificity"]
    
    # Initialize the results table
    results_table = []

    # Sweep over the performance metrics
    for metric in performance_metrics:
        # Find the best C value for the current performance metric
        best_C = select_param_linear(X_train, Y_train, k=5, metric=metric, C_range=C_range)
        
        # Evaluate the cross-validation performance with the best C
        clf = LinearSVC(C=best_C, loss="hinge", penalty="l2", dual=True, random_state=445)
        cv_perf = cv_performance(clf, X_train, Y_train, k=5, metric=metric)
        
        # Append the results to the results table
        results_table.append({
            "Performance Measure": metric,
            "C": best_C,
            "CV Performance": cv_perf
        })

    # Print out the results table
    print("Performance Measures | C | CV Performance")
    for result in results_table:
        print(f"{result['Performance Measure']} | {result['C']} | {result['CV Performance']}")

    
    # (c)
    # Use the best C value (0.1) to train SVM model
    # Define the performance metrics to test
    performance_metrics = ["accuracy", "f1-score", "auroc", "precision", "sensitivity", "specificity"]
    
    # Initialize the results table
    results_table = []

    # Sweep over the performance metrics
    for metric in performance_metrics:
        # Train the SVM model with the best C value
        clf = LinearSVC(C=0.1, loss="hinge", penalty="l2", dual=True, random_state=445)
        clf.fit(X_train, Y_train)
        
        # Evaluate the test performance
        if metric == "auroc":
            y_pred = clf.decision_function(X_test)
        else:
            y_pred = clf.predict(X_test)
        test_perf = performance(Y_test, y_pred, metric=metric)
        
        # Append the results to the results table
        results_table.append({
            "Performance Measure": metric,
            "Test Performance": test_perf
        })

    # Print out the results table
    print("Performance Measures | Test Performance")
    for result in results_table:
        print(f"{result['Performance Measure']} | {result['Test Performance']}")

    # (d)
    # Plot the L0-norm as a function of C
    C_values = [10 ** -3, 10 ** -2, 10 ** -1, 10 ** 0]
    plot_weight(X_train, Y_train, penalty="l2", loss="hinge", C_range=C_values, dual=True)
    
    # (e)
    # Use C = 0.1 to train the SVM model, kept other parameters the same as in (c)
    clf = LinearSVC(C=0.1, loss="hinge", penalty="l2", dual=True)
    clf.fit(X_train, Y_train)
    Y_pred = clf.predict(X_test)

    coef = clf.coef_[0]
    # sort the coefficient
    sorted_coef = np.argsort(coef)
    # get the five most positive and five most negative coefficients
    most_positive = sorted_coef[-5:]
    most_negative = sorted_coef[:5][::-1]
    word_list = np.array(list(dictionary_binary.keys()))
    positive_words = word_list[most_positive]
    negative_words = word_list[most_negative]
    positive_coef = coef[most_positive]
    negative_coef = coef[most_negative]
    word_list = np.array(list(dictionary_binary.keys()))
    positive_words = word_list[most_positive]
    negative_words = word_list[most_negative]
    all_words = np.concatenate((positive_words, negative_words))
    all_coef = np.concatenate((positive_coef, negative_coef))
    plt.barh(all_words, all_coef)
    plt.xlabel('Coefficient')
    plt.ylabel('Word')
    plt.title('Coefficient vs Word')
    plt.show()


    # 3.2
    # (a)
    # Define the range of C values to test {10^−3, 10^−2, 10^−1, 10^0}
    C_range = [10**i for i in range(-3, 1)]
    # find the setting for C that maximizes the mean AUROC
    best_C = select_param_linear(X_train, Y_train, k=5, metric="auroc", C_range=C_range, loss="squared_hinge", penalty="l1", dual=False)
    print(best_C)
    # Report the C value with the best CV performance
    clf = LinearSVC(C=best_C, loss="squared_hinge", penalty="l1", dual=False, random_state=445)
    cv_perf = cv_performance(clf, X_train, Y_train, k=5, metric="auroc")
    clf.fit(X_train, Y_train)
    Y_pred = clf.decision_function(X_test)
    print("CV AUROC: ", cv_perf)
    print("Test AUROC: ", metrics.roc_auc_score(Y_test, Y_pred))

    # (b)
    plot_weight(X_train, Y_train, "l1", C_range, "squared_hinge", False)

    # (c)
    print("part c \n")
    print("The L0-norm range for the L1 penalty graph starts slightly above 0 and increases to just under 600 as the regularization strength parameter C increases from 10^-3 to 10^0. "
          "In contrast, for the L2 penalty graph, the L0-norm starts just below 6000 and decreases to slightly above 4000 over the same range of C. "
          "The gradient of the L1 penalty indicates an increase in the number of non-zero coefficients in the model as C increases, due to less stringent penalization, leading to less sparsity. "
          "On the other hand, the L2 penalty does not promote sparsity but instead shrinks the coefficients in magnitude, resulting in a more gradual decrease in L0-norm with increasing C. "
          "These differences highlight how L1 regularization encourages sparsity with increasing C, while L2 regularization maintains most coefficients but reduces their sizes.")


    # 3.3
    # (a)
    # Method 1: Grid Search
    param_range = np.array([[10**i, 10**j] for i in range(-2, 4) for j in range(-2, 4)])
    best_C, best_r, grid_record = select_param_quadratic(X_train, Y_train, k=5, metric="auroc", param_range=param_range)
    print("Grid Search: " + str(best_C) + " " + str(best_r))
    # Report the C and r values with the best CV performance
    clf = SVC(C=best_C, kernel="poly", degree=2, coef0=best_r, gamma = 'auto', random_state=445)
    cv_perf = cv_performance(clf, X_train, Y_train, k=5, metric="auroc")
    clf.fit(X_train, Y_train)
    Y_pred = clf.decision_function(X_test)
    print("CV AUROC: ", cv_perf)
    print("Test AUROC: ", metrics.roc_auc_score(Y_test, Y_pred))
    print("Grid Search Record: ", grid_record)

    # Method 2: Random Search
    C_range = [10**i for i in range(-2, 3)]
    r_range = [10**i for i in range(-2, 3)]
    # Assuming np.random.choice is used correctly with a random seed for reproducibility
    np.random.seed(445)  # Set the seed for reproducibility

    # Correct way to generate 25 unique (C, r) pairs
    random_param_range = np.array([
        [float(10)**np.random.uniform(-2, 3), float(10)**np.random.uniform(-2, 3)]
        for _ in range(25)
    ])
    best_C, best_r, random_record = select_param_quadratic(X_train, Y_train, k=5, metric="auroc", param_range=random_param_range)
    print("Random Search: " + str(best_C) + " " + str(best_r))
    # Report the C and r values with the best CV performance
    clf = SVC(C=best_C, kernel="poly", degree=2, coef0=best_r, gamma = 'auto', random_state=445)
    cv_perf = cv_performance(clf, X_train, Y_train, k=5, metric="auroc")
    clf.fit(X_train, Y_train)
    Y_pred = clf.decision_function(X_test)
    print("CV AUROC: ", cv_perf)
    print("Test AUROC: ", metrics.roc_auc_score(Y_test, Y_pred))
    print("Random Search Record: ", random_record)



    # Question 4
    # 4.1 (c)
    # Define the performance metrics to test
    performance_metrics = ["accuracy", "f1-score", "auroc", "precision", "sensitivity", "specificity"]
    
    # Initialize the results table
    results_table = []

    # Sweep over the performance metrics
    for metric in performance_metrics:
        # Train the SVM model with the best C value
        clf = LinearSVC(C=0.01, loss="hinge", penalty="l2", class_weight={-1: 1, 1: 10}, random_state=445)
        clf.fit(X_train, Y_train)
        
        # Evaluate the test performance
        if metric == "auroc":
            y_pred = clf.decision_function(X_test)
        else:
            y_pred = clf.predict(X_test)
        test_perf = performance(Y_test, y_pred, metric=metric)
        
        # Append the results to the results table
        results_table.append({
            "Performance Measure": metric,
            "Test Performance": test_perf
        })

    # Print out the results table
    print("Performance Measures | Test Performance")
    for result in results_table:
        print(f"{result['Performance Measure']} | {result['Test Performance']}")
    

    # 4.2 (a)
    IMB_features, IMB_labels, IMB_test_features, IMB_test_labels = get_imbalanced_data(
        dictionary_binary, filename=filename
    )

    # Define the performance metrics to test
    performance_metrics = ["accuracy", "f1-score", "auroc", "precision", "sensitivity", "specificity"]
    
    # Initialize the results table
    results_table = []

    # Sweep over the performance metrics
    for metric in performance_metrics:
        # Train the SVM model with the best C value
        clf = LinearSVC(loss="hinge", penalty="l2", C=0.01, class_weight={-1: 1, 1: 1}, random_state=445)
        clf.fit(IMB_features, IMB_labels)
        
        # Evaluate the test performance
        if metric == "auroc":
            y_pred = clf.decision_function(IMB_test_features)
        else:
            y_pred = clf.predict(IMB_test_features)
        test_perf = performance(IMB_test_labels, y_pred, metric=metric)
        
        # Append the results to the results table
        results_table.append({
            "Performance Measure": metric,
            "Test Performance": test_perf
        })

    # Print out the results table
    print("Performance Measures | Test Performance")
    for result in results_table:
        print(f"{result['Performance Measure']} | {result['Test Performance']}")


    # 4.3 (a)
    class_weights = []
    for i in range(10):
        for j in range(10):
            class_weights.append({-1: i, 1: j})

    best_score = float('-inf')
    best_weight = None
    for weight in class_weights:
        clf = LinearSVC(loss="hinge", penalty="l2", C=0.01, class_weight=weight, random_state=445)
        score = cv_performance(clf, IMB_features, IMB_labels, k=5, metric="auroc")
        if score > best_score:
            best_score = score
            best_weight = weight
    print("Best class weight: " + str(best_weight))
    print("Best class weight score: " + str(best_score))

    # (b)
    performance_metrics = ["accuracy", "f1-score", "auroc", "precision", "sensitivity", "specificity"]
    results_table = []

    clf = LinearSVC(loss="hinge", penalty="l2", C=0.01, class_weight=best_weight)
    clf.fit(IMB_features, IMB_labels)
    for metric in performance_metrics:
        if metric == "auroc":
            y_pred = clf.decision_function(IMB_test_features)
        else:
            y_pred = clf.predict(IMB_test_features)
        test_perf = performance(IMB_test_labels, y_pred, metric=metric)
        results_table.append({
            "Performance Measure": metric,
            "Test Performance": test_perf
        })

    print("Performance Measures | Test Performance")
    for result in results_table:
        print(f"{result['Performance Measure']} | {result['Test Performance']}")


    # 4.4
    clf_equal = LinearSVC(loss="hinge", penalty="l2", C=0.01, class_weight={-1: 1, 1: 1})
    clf_equal.fit(IMB_features, IMB_labels)
    y_pred_equal = clf_equal.decision_function(IMB_test_features)
    fpr_equal, tpr_equal, threshold_equal = roc_curve(IMB_test_labels, y_pred_equal)

    clf_custom = LinearSVC(loss="hinge", penalty="l2", C=0.01, class_weight={-1: 9, 1: 6})
    clf_custom.fit(IMB_features, IMB_labels)
    y_pred_custom = clf_custom.decision_function(IMB_test_features)
    fpr_custom, tpr_custom, threshold_custom = roc_curve(IMB_test_labels, y_pred_custom)

    plt.plot(fpr_equal, tpr_equal, label="Equal Class Weight")
    plt.plot(fpr_custom, tpr_custom, label="Custom Class Weight")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()

    # Question 5
    # 5.1 
    # (a)
    actor, actresses = count_actors_and_actresses(filename)
    print("the number of actors: ", actor)  
    print("the number of actresses: ", actresses)

    # (b)
    plot_actors_and_actresses(filename, x_label="label")
    print("The majority of reviews that mention 'actor' are positive (label +1), with a proportion of 0.76. "
          "Reviews that mention 'actor' also have a smaller but significant proportion of negative (label -1) and neutral (label 0) reviews, at 0.15 and 0.09 respectively. \n"
          "The reviews that mention 'actress' show a lower proportion of positive sentiment (label +1) at 0.65. "
          "There is a notably higher proportion of neutral reviews (label 0) for 'actress' at 0.32, which is much higher than that for 'actor.'"
          "The proportion of negative reviews (label -1) for 'actress' is very low at 0.03, which is interestingly lower than the negative reviews for 'actor.' \n")
    print("There is a stronger positive sentiment associated with the term 'actor' compared to 'actress,' suggesting that reviewers may be implicitly more favorable towards male actors or the performances typically associated with male roles. "
          "The higher proportion of neutral reviews for 'actress' could indicate a tendency for reviewers to be less decisive or less passionate in their praise or criticism of female actors. " 
          "This may reflect an implicit bias where performances by actresses are less likely to be seen as standout or are more often categorized as average.")
    
    # (c)
    plot_actors_and_actresses(filename, x_label="rating")
    print("In light of the mislabeling error in the dataset where 5-star and 1-star ratings for reviews mentioning 'actor(s)' were swapped, "
          "the perception of bias towards actors based on review sentiment must be reassessed. The corrected data indicates that what was previously thought to be a large proportion of positive reviews for 'actor(s)' was actually negative. "
          "This suggests that the original positive bias for actors is significantly less pronounced, if not entirely absent. Conversely, the distribution for 'actress(es)' remains relatively unchanged, with a balanced spread across the ratings, and comparatively, it could now be perceived as more favorable. "
          "The corrected distribution, therefore, implies that any initial conclusions about a positive bias towards actors over actresses may not hold true, underscoring the impact of data accuracy on the analysis of potential biases in sentiment.")
    
    # (d)
    clf = LinearSVC(loss="hinge", penalty="l2", C=0.1, dual=True)
    clf.fit(X_train, Y_train)
    # report the theta vector’s coefficients for the words ‘actor’ and ‘actress’
    actor_idx = dictionary_binary['actor']
    actress_idx = dictionary_binary['actress']
    print("The coefficient for the word 'actor' is: ", clf.coef_[0][actor_idx])
    print("The coefficient for the word 'actress' is: ", clf.coef_[0][actress_idx])

    print("These coefficients suggest that the presence of the word 'actor' in a review is strongly associated with a positive outcome, "
          "while the presence of the word 'actress' is slightly associated with a negative outcome. This could imply that the model has learned a bias present in the training data, "
          "where reviews mentioning 'actor' tend to be more positive, and those mentioning 'actress' tend to be less positive or even negative. \n")
    print("The importance of interpretability in machine learning is demonstrated here as it allows us to understand the decisions made by the model. By examining the coefficients, "
          "we can detect potential biases that the model may have learned from the training data. This insight is crucial for taking corrective measures, such as adjusting the model or the training process to mitigate these biases, "
          "ensuring fairness and reliability in the model's predictions. It also reinforces the need for careful consideration of the training data used in machine learning to prevent the perpetuation of existing biases.")
    

    # 5.2
    # (a)
    # Using the returned model, print out the word embedding for the word ‘actor’ and report the dimensionality of this word embedding
    model = train_word2vec(filename)
    print("The word embedding for the word 'actor' is: \n", model.wv['actor'])
    print("The dimensionality of this word embedding is: ", len(model.wv['actor']))

    # (b)
    # Using the Word2Vec model trained in question 5.2(a), report the five most similar words to the word ‘plot’. Recalling that the Amazon dataset contains movie reviews, how has the context of the influenced the word embeddings? How might the most similar words have been different if the 
    # had been about geometry instead of movie reviews?
    similar_words = model.wv.most_similar('plot', topn=5)
    for word, similarity in similar_words:
        print("The word: ", word, " has a similarity of: ", similarity)
    
    print("The five words most similar to 'plot' according to the Word2Vec model trained on the Amazon movie reviews dataset are 'acting,' 'effects,' 'poor,' 'weak,' and 'script,' with their respective similarity scores. "
          "These words reflect the context of movie reviews, where the 'plot' is often discussed alongside acting quality, special effects, and script strength, indicating a focus on storytelling and production elements. "
          "For instance, 'poor' and 'weak' are evaluative terms that may be used to describe a plot, while 'script' is closely related as it's the written blueprint of the plot. \n")
    
    print("If the dataset had been about geometry instead, the most similar words to 'plot' would likely have reflected that domain, potentially including terms like 'graph,' 'coordinate,' 'axis,' 'point,' or 'line.' "
          "These words are related to the geometric concept of plotting points or lines on a graph, which is a different meaning of the word 'plot' compared to its narrative sense in movie reviews. "
          "This illustrates how word embeddings are sensitive to the context of the training corpus and can capture different meanings of a word based on how it is used in that specific domain.")


    # 5.3
    # (a)
    A = {"her", "woman", "women"}
    B = {"him", "man", "men"}
    w = {"smart"}
    association = compute_association(filename, w, A, B)
    print("Association between 'smart' and the sets {her, woman, women}, and {him, man, men}: ", association)
    

    # Read multiclass data
    # TODO: Question 6: Apply a classifier to heldout features, and then use
    #       generate_challenge_labels to print the predicted labels

    (
        multiclass_features,
        multiclass_labels,
        multiclass_dictionary,
    ) = get_multiclass_training_data()

    heldout_features = get_heldout_reviews(multiclass_dictionary)

    # Feature Engineering: scale the data
    scaler = StandardScaler()
    transformed = scaler.fit_transform(multiclass_features)

    # Hyperparameter selection
    C_range = [10**i for i in range(-3, 4)]
    best_C = select_param_linear(transformed, multiclass_labels, k=5, metric="accuracy", C_range=C_range)
    print("Best C value: ", best_C)

    # Algorithm selection
    # Multiclass method
    clf = LinearSVC(loss="squared_hinge", penalty="l2", C=best_C, multi_class="ovr", dual=False)
    clf.fit(transformed, multiclass_labels)
    y_pred = clf.predict(heldout_features)
    print("len(y_pred)" , len(y_pred))
    generate_challenge_labels(y_pred, "lisayin")


    print("Feature Engineering: In approaching feature engineering, I started by considering the raw data's richness. Since the bag-of-words model provided a simple but often effective baseline, "
          "we opted to scale the data to improve the performance of our SVM classifier. "
          "This standardization normalizes the feature vectors and ensures that each feature contributes equally to the distance calculations in the SVM. \n")
    print("Hyperparameter Selection: For hyperparameter tuning, I specifically targeted the regularization parameter 'C' of the SVM classifier. "
          "I experimented with a range of 'C' values, utilizing a logarithmic scale from 10^-3 to 10^3."
          "I also used k-fold cross-validation to identify the best 'C' value of 0.001., aiming for the one that maximized accuracy across the validation sets. \n")
    print("Algorithm Selection: I chose the Linear Support Vector Classifier (LinearSVC) as the algorithm with a linear kernel for its efficiency and scalability on large datasets. "
          "While a quadratic kernel could potentially model more complex relationships, it comes with a higher computational cost. Given the size of our dataset and the feature matrix, a linear kernel was deemed more appropriate and computationally feasible. \n")
    print("Multiclass Method: I implemented the 'one-vs-rest' (OvR) multiclass strategy with the 'LinearSVC' model. This method involves training a single classifier per class, with the samples of that class as positive samples and all other samples as negatives. "
          "This approach is preferred for its simplicity and scalability compared to 'one-vs-one,' which would require training a classifier for every pair of classes. \n")
    print("Advanced Techniques: Beyond the standard course material, I explored additional techniques to enhance the model. I employed feature scaling to normalize the data and improve the optimization process during training.")





if __name__ == "__main__":
    main()
    
    
