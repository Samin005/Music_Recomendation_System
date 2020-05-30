import pandas
from sklearn.tree import DecisionTreeClassifier

music_data_frame = pandas.read_csv('data/Music Data.csv')

# features = music_data_frame.drop(['genre'])
# label = music_data_frame['genre']

# all but last column as the features
features = music_data_frame.iloc[:, 0:-1]

# last column as the label
label = music_data_frame.iloc[:, -1]

decision_tree_model = DecisionTreeClassifier()
decision_tree_model.fit(features, label)

input_set_1 = [21, 1]
input_set_2 = [22, 0]

print(decision_tree_model.predict([input_set_1, input_set_2]))

# take input from the user
user_age = input('Please enter age: ')
user_gender = input('Are you a male(1) or female(0)?')
user_input_set = [user_age, user_gender]
print(decision_tree_model.predict([user_input_set]))
