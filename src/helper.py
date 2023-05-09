import pandas as pd


def parseTestHierarchy():
    # Read the track data line by line
    with open('data/testTrack_hierarchy.txt', 'r') as file:
        lines = file.readlines()

    track_data = []
    for line in lines:
        fields = line.strip().split('|')
        # Use the second number as the trackId
        track_data.append([fields[1]] + fields[2:])

    # Create a DataFrame from the track data
    track_data_df = pd.DataFrame(track_data)

    # Fill NaN values with an empty string
    track_data_df.fillna("", inplace=True)

    # Determine the maximum number of genres
    max_genres = track_data_df.shape[1] - 4

    # Rename columns
    track_data_df.columns = ['userId', 'trackId', 'albumId', 'artistId'] + \
        [f'genreId_{i}' for i in range(1, max_genres + 1)]

    track_data_df['trackId'] = pd.to_numeric(
        track_data_df['trackId'], errors='coerce')
    track_data_df.dropna(subset=['trackId'], inplace=True)

    track_data_df['trackId'] = track_data_df['trackId'].astype(int)

    merged_df = track_data_df

    # Convert genres to a list of genres for each track
    merged_df['genres'] = merged_df[[
        f'genreId_{i}' for i in range(1, max_genres + 1)]].values.tolist()

    # Remove empty strings from the genre lists
    merged_df['genres'] = merged_df['genres'].apply(
        lambda x: [genre for genre in x if genre != ""])

    # Drop individual genre columns and itemId column
    merged_df.drop(columns=[f'genreId_{i}' for i in range(
        1, max_genres + 1)], inplace=True)

    return merged_df
