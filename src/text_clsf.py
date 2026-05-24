import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from sklearn.preprocessing import LabelEncoder





def load_data(file_path):
    """
    Load data from a CSV file.

    Parameters:
    file_path (str): The path to the CSV file.

    Returns:
    pd.DataFrame: A DataFrame containing the loaded data.
    """
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
def lowercase_text(text):
    """
    Convert text to lowercase.

    Parameters:
    text (str): The input text.

    Returns:
    str: The lowercase version of the input text.
    """
    return text.lower()

def punctuational_removal(text):
    """
    Remove punctuation from the text.

    Parameters:
    text (str): The input text.

    Returns:
    str: The text with punctuation removed.
    """
    import string
    return text.translate(str.maketrans('', '', string.punctuation))
def remove_stopwords(text):
    """
    Remove stop words from the text.

    Parameters:
    text (str): The input text.

    Returns:
    str: The text with stop words removed.
    """
    stop_words = set(stopwords.words('english'))
    return ' '.join([word for word in text.split() if word not in stop_words])
def stemming(text):
    """
    Apply stemming to the text.

    Parameters:
    text (str): The input text.

    Returns:
    str: The stemmed version of the input text.
    """
    stemmer = PorterStemmer()
    return ' '.join([stemmer.stem(word) for word in text.split()])
def tokenization(text):
    """
    Tokenize the text into words.

    Parameters:
    text (str): The input text.

    Returns:
    list: A list of tokens (words) from the input text.
    """
    return word_tokenize(text)
def preprocess_text(text):
    """
    Preprocess the input text by applying a series of transformations.

    Parameters:
    text (str): The input text.

    Returns:
    str: The preprocessed version of the input text.
    """
    text = lowercase_text(text)
    text = punctuational_removal(text)
    text = remove_stopwords(text)
    text = stemming(text)
    return text
def create_dictionary(messages):
    """
    Create a dictionary of unique words from the input text.

    Parameters:
    messages (list): A list of text messages.

    Returns:
    dict: A dictionary where keys are unique words and values are their frequencies.
    """
    # getting unique words of messages
    dictionary=[]
    for text in messages:
         for word in text:
            if word not in dictionary:
                dictionary.append(word)
    return dictionary
def vectorization_using_bow(messages, dictionary):
    """
    Vectorize the input messages using the Bag of Words (BoW) model.

    Parameters:
    messages (list): A list of text messages.
    dictionary (dict): A dictionary of unique words.

    Returns:
    list: A list of vectors representing the BoW representation of each message.
    """
    vectors = []
    for text in messages:
        vector = np.zeros(len(dictionary))
        for word in text:
            if word in dictionary:
                index = dictionary.index(word)
                vector[index] += 1
        vectors.append(vector)
    return np.array(vectors)
def split_data(X, y, test_size=0.125, val_size=0.2, random_state=42):
    """
    Split the data into training, validation, and test sets.

    Parameters:
    X (list): The input features (messages).
    y (list): The target labels.
    test_size (float): The proportion of the dataset to include in the test split.
    val_size (float): The proportion of the training set to include in the validation split.
    random_state (int): Controls the randomness of the splits.

    Returns:
    tuple: A tuple containing the training, validation, and test sets (X_train, X_val, X_test, y_train, y_val, y_test).
    """
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=test_size, shuffle=True, random_state=random_state)
    val_size_adjusted = val_size / (1 - test_size)  # Adjust validation size based on remaining data
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=val_size_adjusted, random_state=random_state)
    
    return X_train, X_val, X_test, y_train, y_val, y_test
if __name__ == "__main__":
    file_path = "data\\2cls_spam_text_cls.csv"  # Update this path to your CSV file
    df = load_data(file_path)
    messages=df['Message'].values.tolist()
    labels=df['Category'].values.tolist()

    texts = [preprocess_text(message) for message in messages]
    texts=[tokenization(t) for t in texts]
    dict=create_dictionary(texts)
    print(len(dict))
    vectors=vectorization_using_bow(texts, dict)
    print(f"First message vector: {vectors[0]}")
    le = LabelEncoder()
    y=le.fit_transform(labels)
    
    # model=GaussianNB()
    # model=model.fit(vectors,y)

    X_train, X_val, X_test, y_train, y_val, y_test= split_data(vectors, y)
    print(X_train.shape, y_train.shape)
    print(X_val.shape, y_val.shape)
    print(X_test.shape, y_test.shape)



    model=MultinomialNB()
    model=model.fit(X_train,y_train)

    y_val_pred=model.predict(X_val)
    y_test_pred=model.predict(X_test)
    print("Validation Set Evaluation:")
    val_accuracy_score=accuracy_score(y_val, y_val_pred)
    print(f"Accuracy: {val_accuracy_score:.4f}")
    test_accuracy_score=accuracy_score(y_test, y_test_pred)
    print(f"Test Accuracy: {test_accuracy_score:.4f}")


    cm=confusion_matrix(y_test, y_test_pred)
    print("Confusion Matrix:")
    print(cm)

    print("Classification Report:")
    print(classification_report(y_test, y_test_pred,target_names=le.classes_))