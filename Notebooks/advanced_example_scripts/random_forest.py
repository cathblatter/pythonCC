# Python Crash Course, 18.02.21
# Author: Maxim Samarin
#
# Random Forest Classifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import pandas as pd
import matplotlib.pyplot as plt

wine_data = pd.read_csv('../data/winequality-red.csv', sep=';')
wine_data['premium'] = wine_data['quality'] > 5

quality_counts = wine_data['quality'].value_counts()
plt.bar(quality_counts.index, quality_counts, color='grrgrg')
plt.xlabel('Quality assessment')
plt.ylabel('Amount of different wines')
plt.show()

target = wine_data['premium'].astype(int)
wine_features = wine_data.drop(['quality','premium'], axis=1)

print("Shape of wine_features:\t{}\nShape of target:\t{}\n"
      .format(wine_features.shape, target.shape)
      )

feat_train, feat_test, target_train, target_test = train_test_split(
    wine_features, target, test_size = 0.3)

print("After splitting into train and test sets:\n\n"
      "Shape of feat_train:\t{}\nShape of target_train:\t{}\n"
      "Shape of feat_test:\t{}\nShape of target_test:\t{}"
      .format(feat_train.shape, target_train.shape, feat_test.shape, 
              target_test.shape)
     )

random_forest = RandomForestClassifier(n_estimators=100, max_depth=8)
random_forest.fit(feat_train, target_train)

feature_scores = random_forest.feature_importances_
feature_names = list(wine_features.columns)

important_features = pd.Series(feature_scores, index=feature_names).sort_values()

plt.barh(important_features.index, important_features, color='Orange')
plt.xlabel('Feature importance')
plt.show()
# plt.savefig("RF_Feature_importance.png", bbox_inches='tight')