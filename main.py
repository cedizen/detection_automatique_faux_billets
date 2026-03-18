# retrieve the model
import joblib

# retrieve args in the command line terminal
import argparse
from pathlib import Path

import pandas as pd

from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline

def load_data(file_to_load):
  """
  Load data from a csv file.

  Parameters:
    file_to_load (file): CSV file 

    Returns:
      df: dataframe resulting from the file
  """
  return pd.read_csv(file_to_load)
  
def save_data_csv(df, path):
  """
  Save the dataframe to a csv file 

  Parameters:
      df (dataframe)
      path: path name of the new output file

  Returns:
      file CSV
  """
  return df.to_csv(path, index=False)

def main():

  parser = argparse.ArgumentParser()
  parser.add_argument("--file", required=True, help="Path to CSV file")

  args = parser.parse_args()
  csv_path_file = Path(args.file)

  # Dataframe given
  df = load_data(csv_path_file)

  # retrive the model
  model = joblib.load("model.pkl")

  # Check if the model is in the pipeline or not, and extract from it
  if isinstance(model, Pipeline):
    final_estimator = model.steps[-1][1]
  else:
    final_estimator = model


  # if there is a column id
  column_id = "id"
  if column_id in df.columns:
    X_df = df.drop(column_id, axis=1)
  else:
    X_df = df.copy()

  # Check if the model is unsupervised like KMeans or others
  if isinstance(final_estimator, KMeans):
    clusters = model.predict(df)
    X_df["clusters"] = clusters
    save_data_csv(df, "predictions.csv")

  # Or supervised
  else:
    predictions = model.predict(X_df)
    X_df["predictions"] = predictions

  # Attach again the id to the dataset after predictions
  X_df[column_id] = df[column_id]

  # Reorder the id at the first position
  new_order = [column_id] + [col for col in X_df.columns if col != column_id]
  output_df = X_df[new_order]

  save_data_csv(output_df, "predictions.csv")
  print("Prédictions générées et stockées dans le fichier 'predictions.csv'")
  

if __name__ == "__main__":
  main()