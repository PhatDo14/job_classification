import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
import re
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest, chi2, SelectPercentile
from imblearn.over_sampling import RandomOverSampler, SMOTEN

data = pd.read_csv("data/job_dataset.csv", dtype = str)

def filter_location(location):
    result1 = re.findall("\,\s[A-Z]{2}$", location)
    if len(result1) !=0 :
        return result1[0][2:]
    else:
        return location

data = data.dropna(axis=0)

data["location"] = data["location"].apply(filter_location)
print(data["career_level"].value_counts())

target = "career_level"
x = data.drop(target, axis= 1)
y = data[target]

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42 ,stratify=y)

ros = SMOTEN(random_state=0, k_neighbors=2, sampling_strategy={
    "director_business_unit_leader" : 500,
    "specialist" : 500,
    "managing_director_small_medium_company":500,
    "bereichsleiter":1000
})
# print(y_train.value_counts())
# x_train, y_train = ros.fit_resample(x_train, y_train)
# print("------------")
# print(y_train.value_counts() )



preprocessor = ColumnTransformer(transformers=[
    ("title_ft", TfidfVectorizer(stop_words="english", ngram_range=(1,1) ), "title"),
    ("location", OneHotEncoder(handle_unknown="ignore"), ["location"]),
    ("description_ft", TfidfVectorizer(stop_words="english", ngram_range=(1,2), min_df=0.01, max_df=0.95), "description"),
    ("function_ft", OneHotEncoder(handle_unknown="ignore"), ["function"]),
    ("industry_ft", TfidfVectorizer(stop_words="english", ngram_range=(1,1)), "industry")
])

cls = Pipeline(steps=[
    ("preprocessing", preprocessor),
    # ("feature_selector", SelectKBest(chi2, k=800)),
    ("feature_selector", SelectPercentile(chi2, percentile=10)),
    ("model", RandomForestClassifier())
])

params = {
    # "model__n_estimators" : [50, 100, 200],
    "model__criterion" : ["gini", "entropy", "log_loss"],
    # "model__max_depth": [None, 2 ,5 ,10],
    "feature_selector__percentile" : [1, 5, 10]
}

model = GridSearchCV(
    estimator = cls,
    param_grid= params,
    scoring= "recall_weighted",
    cv= 6,
    verbose=2,
    n_jobs=4
)

model.fit(x_train, y_train)
y_predict = model.predict(x_test)
print(classification_report(y_test,y_predict))

