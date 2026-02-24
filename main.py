{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyMMVv0FsBRhylweqH2/L2T5",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/cedizen/detection_automatique_faux_billets/blob/target_model/main.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "ExY0xkOTi-7s"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from google.colab import drive\n",
        "drive.mount('/content/drive')"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "collapsed": true,
        "id": "gzDILzwhaAJG",
        "outputId": "476ae159-a0f8-4a72-e4f4-d7648371f00e"
      },
      "execution_count": 1,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount(\"/content/drive\", force_remount=True).\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import joblib\n",
        "import pandas as pd\n",
        "\n",
        "from sklearn.cluster import KMeans\n",
        "from sklearn.pipeline import Pipeline"
      ],
      "metadata": {
        "id": "gjyQCnWoUos5"
      },
      "execution_count": 2,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "root = \"/content/drive/MyDrive/projet_faux_monnayage/\"\n",
        "path_csv_file = \"data/billets_df_cleaned.csv\"\n",
        "field_target = \"is_genuine\""
      ],
      "metadata": {
        "id": "-hNHTjeiaah5"
      },
      "execution_count": 3,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "def set_target(column_target):\n",
        "  return column_target"
      ],
      "metadata": {
        "id": "3V3vKNVrxk68"
      },
      "execution_count": 4,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "def set_csv_file(file):\n",
        "  return file"
      ],
      "metadata": {
        "id": "w2gY1ZPmxZaO"
      },
      "execution_count": 5,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "def load_data(file_to_load):\n",
        "  file = set_csv_file(file_to_load)\n",
        "  df = pd.read_csv(file)\n",
        "  return df"
      ],
      "metadata": {
        "id": "WoMcB8Omz0pS"
      },
      "execution_count": 8,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "def save_data_csv(df, path):\n",
        "  return df.to_csv(path, index=False)"
      ],
      "metadata": {
        "id": "vK-rFHRA0cnC"
      },
      "execution_count": 15,
      "outputs": []
    },
    {
      "cell_type": "code",
      "execution_count": 18,
      "metadata": {
        "id": "urtzfU50JFDm"
      },
      "outputs": [],
      "source": [
        "def main():\n",
        "\n",
        "  file = set_csv_file(root + path_csv_file)\n",
        "  target = set_target(field_target)\n",
        "\n",
        "  # Dataframe given\n",
        "  df = load_data(file)\n",
        "\n",
        "  model = joblib.load(f\"{root}model.pkl\")\n",
        "\n",
        "  # Check if the model is in the pipeline or not, and extract from it\n",
        "  if isinstance(model, Pipeline):\n",
        "    final_estimator = model.steps[-1][1]\n",
        "  else:\n",
        "    final_estimator = model\n",
        "\n",
        "  # Check if the model is unsupervised like KMeans or others\n",
        "  if isinstance(final_estimator, KMeans):\n",
        "    clusters = model.predict(df)\n",
        "    df[\"clusters\"] = clusters\n",
        "    print(df)\n",
        "    save_data_csv(df, root + \"data/output.csv\")\n",
        "\n",
        "  # Or supervised and then need to split the target from the dataset\n",
        "  else:\n",
        "    # Check if the target is part of the columns\n",
        "    if target in df.columns:\n",
        "      X_df = df.drop(target, axis=1)\n",
        "      y_true = df[target]\n",
        "    else:\n",
        "      X_df = df.copy()\n",
        "      y_true = None\n",
        "\n",
        "    predictions = model.predict(X_df)\n",
        "    X_df[target] = y_true\n",
        "    X_df[\"predictions\"] = predictions\n",
        "\n",
        "    # calculate the ratio prediction\n",
        "    if y_true is not None:\n",
        "      ratio_false_predictions = (y_true != predictions).sum() / len(X_df)\n",
        "      print(f\"{ratio_false_predictions*100:.2f}% of false predictions\")\n",
        "\n",
        "    print(X_df)\n",
        "    save_data_csv(X_df, root + \"data/output.csv\")"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "if __name__ == \"__main__\":\n",
        "  main()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "DC4P-BH0lo3Q",
        "outputId": "22ecea09-bd5a-4c94-9038-e56b2916e760"
      },
      "execution_count": 19,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "0.87% of false predictions\n",
            "      diagonal  height_left  height_right  margin_low  margin_up  length  \\\n",
            "0       171.81       104.86        104.95        4.52       2.89  112.83   \n",
            "1       171.46       103.36        103.66        3.77       2.99  113.09   \n",
            "2       172.69       104.48        103.50        4.40       2.94  113.16   \n",
            "3       171.36       103.91        103.94        3.62       3.01  113.51   \n",
            "4       171.73       104.28        103.46        4.04       3.48  112.54   \n",
            "...        ...          ...           ...         ...        ...     ...   \n",
            "1495    171.75       104.38        104.17        4.42       3.09  111.28   \n",
            "1496    172.19       104.63        104.44        5.27       3.37  110.97   \n",
            "1497    171.80       104.01        104.12        5.51       3.36  111.95   \n",
            "1498    172.06       104.28        104.06        5.17       3.46  112.25   \n",
            "1499    171.47       104.15        103.82        4.63       3.37  112.07   \n",
            "\n",
            "      is_genuine  predictions  \n",
            "0           True         True  \n",
            "1           True         True  \n",
            "2           True         True  \n",
            "3           True         True  \n",
            "4           True         True  \n",
            "...          ...          ...  \n",
            "1495       False        False  \n",
            "1496       False        False  \n",
            "1497       False        False  \n",
            "1498       False        False  \n",
            "1499       False        False  \n",
            "\n",
            "[1500 rows x 8 columns]\n"
          ]
        }
      ]
    }
  ]
}