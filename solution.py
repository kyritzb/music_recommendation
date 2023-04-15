import numpy
from surprise import SVD, Dataset, Reader, KNNBasic, NMF
from surprise.model_selection import train_test_split
from surprise import accuracy
import os
import pandas as pd
from sklearn.metrics import roc_auc_score
from surprise.model_selection import GridSearchCV


file_name_test = "data/" + 'testTrack_hierarchy.txt'
file_name_train = "data/" + 'trainIdx2_matrix.txt'
output_file = 'output1.txt'
fTest = open(file_name_test, 'r')
fTrain = open(file_name_train, 'r')
Trainline = fTrain.readline()
fOut = open(output_file, 'w')
trackID_vec = [0] * 6
albumID_vec = [0] * 6
artistID_vec = [0] * 6
lastUserID = -1
user_rating_inTrain = numpy.zeros(shape=(6, 3))

# Create a hashmap for user ratings in the train file
user_ratings_map = {}

#the data
data_list = []

count=0;
total = 0;

for line in fTrain:
    arr_train = line.strip().split('|')
    trainUserID = arr_train[0]
    trainItemID = arr_train[1]
    trainRating = arr_train[2]
    Trainline = fTrain.readline()

    if trainUserID not in user_ratings_map:
        user_ratings_map[trainUserID] = {}

    user_ratings_map[trainUserID][trainItemID] = trainRating

fTrain.close()

for line in fTest:
    print(f"Processing test line: {line.strip()}")
    arr_test = line.strip().split('|')
    userID = arr_test[0]
    trackID = arr_test[1]
    albumID = arr_test[2]
    artistID = arr_test[3]
    if userID != lastUserID:
        ii = 0
        user_rating_inTrain = numpy.zeros(shape=(6, 3))
        print(f"New user: {userID}")

    trackID_vec[ii] = trackID
    albumID_vec[ii] = albumID
    artistID_vec[ii] = artistID
    ii = ii + 1
    lastUserID = userID

    if ii == 6:
        print("Processing ratings...")
        if userID in user_ratings_map:
            for nn in range(0, 6):
                if albumID_vec[nn] in user_ratings_map[userID]:
                    user_rating_inTrain[nn, 0] = user_ratings_map[userID][albumID_vec[nn]]
                if artistID_vec[nn] in user_ratings_map[userID]:
                    user_rating_inTrain[nn, 1] = user_ratings_map[userID][artistID_vec[nn]]

            for nn in range(0, 6):
                outStr = str(userID) + '|' + str(trackID_vec[nn]) + '|' + str(user_rating_inTrain[nn, 0]) + '|' + str(user_rating_inTrain[nn, 1])
                fOut.write(outStr + '\n')
                data_list.append([userID, trackID_vec[nn], user_rating_inTrain[nn, 0], user_rating_inTrain[nn, 1]])

fTest.close()
fOut.close()


print("using matrix factorization")


# Create a new DataFrame with the average rating
data_df = pd.DataFrame(data_list, columns=["userID", "itemID", "album_rating", "artist_rating"])
data_df['average_rating'] = data_df[['album_rating', 'artist_rating']].mean(axis=1)

print(data_df.head())
# Prepare the dataset for matrix factorization
reader = Reader(rating_scale=(0, 1))
data = Dataset.load_from_df(data_df[["userID", "itemID", "average_rating"]], reader)

trainset, testset = train_test_split(data, test_size=0.1)

param_grid = {
    "n_epochs": [5, 10, 20],
    "lr_all": [0.002, 0.005, 0.01],
    "reg_all": [0.02, 0.1, 0.5]
}

print("using grid search to find the best hyper paramaters")

gs = GridSearchCV(SVD, param_grid, measures=["rmse"], cv=3)
gs.fit(data)

print("Best RMSE score:", gs.best_score["rmse"])
print("Best parameters:", gs.best_params["rmse"])

algorithm = gs.best_estimator["rmse"]

# Apply SVD
algorithm.fit(trainset)

# Make predictions
predictions = []
for data_item in data_list:
    user_id, item_id, album_rating, artist_rating = data_item
    prediction = algorithm.predict(user_id, item_id)
    # Store predicted rating
    predicted_rating = prediction.est
    print(user_id +"_"+item_id, predicted_rating)
    # Apply threshold for binary output (1 if the user would like it, 0 otherwise)
    like_or_not = 1 if predicted_rating >= 0.5 else 0
    predictions.append([user_id +"_"+item_id, like_or_not])

# Save predictions to a CSV file
predictions_df = pd.DataFrame(predictions, columns=["TrackID", "Predictor"])
predictions_df.to_csv("predictions.csv", index=False)


