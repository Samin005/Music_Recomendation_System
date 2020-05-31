import pandas
# from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib


music_datafile_name = 'data/Music Data.csv'
music_recommendation_model_name = 'Decision_Tree_Model'

data_frame = pandas.read_csv(music_datafile_name)

# features = music_data_frame.drop(['genre'])
# all but last column as the features
features = data_frame.iloc[:, 0:-1]

# label = music_data_frame['genre']
# last column as the label
labels = data_frame.iloc[:, -1]


def train_model(model):
    model.fit(features, labels)


def save_model(model, model_name):
    joblib.dump(model, f'models/{model_name}.joblib')


def load_model(model_name):
    return joblib.load(f'models/{model_name}.joblib')


def check_model_accuracy(model):
    # splitting the data, 80% used for training and 20% used for testing
    features_for_training, features_for_testing, labels_for_training, labels_for_testing = train_test_split(features,
                                                                                                            labels,
                                                                                                            test_size=0.2)

    # using the 80% to train model
    model.fit(features_for_training, labels_for_training)

    # using the 20% for testing/predictions
    model_predictions = model.predict(features_for_testing)
    model_accuracy = accuracy_score(labels_for_testing, model_predictions) * 100
    return model_accuracy


# # To train and save initial data model
# decision_tree_model = DecisionTreeClassifier()
# train_model(decision_tree_model)
# save_model(decision_tree_model, music_recommendation_model_name)

# Load the trained model
decision_tree_model = load_model(music_recommendation_model_name)

# # To check accuracy of the model
# print(f'Model Accuracy: {check_model_accuracy(decision_tree_model)}%')

# take input from the user and check model predictions
play_again = 'yes'
while 'y' in play_again:
    user_age = input('Please enter age: ')
    user_gender = input('Are you a male or female? ')
    if 'f' in user_gender.lower():
        user_gender = 0
    else:
        user_gender = 1
    user_input_set = [user_age, user_gender]
    print(user_input_set)
    print(f"You probably like {decision_tree_model.predict([user_input_set])[0]} music!")
    play_again = input('Would you like to play again? ').lower()
print('Music Recommendation Program Terminated.')
