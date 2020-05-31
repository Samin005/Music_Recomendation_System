import pandas
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def check_model_accuracy():
    # splitting the data, 80% used for training and 20% used for testing
    features_for_training, features_for_testing, labels_for_training, labels_for_testing = train_test_split(features,
                                                                                                            label,
                                                                                                            test_size=0.2)

    # using the 80% to train model
    decision_tree_model.fit(features_for_training, labels_for_training)

    # using the 20% for testing/predictions
    model_predictions = decision_tree_model.predict(features_for_testing)
    model_accuracy = accuracy_score(labels_for_testing, model_predictions) * 100
    return model_accuracy


music_data_frame = pandas.read_csv('data/Music Data.csv')

# features = music_data_frame.drop(['genre'])
# label = music_data_frame['genre']

# all but last column as the features
features = music_data_frame.iloc[:, 0:-1]

# last column as the label
label = music_data_frame.iloc[:, -1]

decision_tree_model = DecisionTreeClassifier()

print(f'Model Accuracy = {check_model_accuracy()}%')

# Training the model with the entire dataset
decision_tree_model.fit(features, label)

# Test with sample inputs

# input_set_1 = [21, 1]
# input_set_2 = [22, 0]
#
# print(decision_tree_model.predict([input_set_1, input_set_2]))

# take input from the user
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
