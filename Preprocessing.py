import numpy as np
from custom.DataProcessor import *
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from custom.Normalization import *
from ast import literal_eval
import re
import langdetect
from textblob import TextBlob

import matplotlib.pyplot as plt

import nltk
nltk.download('stopwords')

USEFUL_FEATURES_LIST = [
    "id",
    # "host_response_time",
    #"host_response_rate",
    "host_acceptance_rate",
    "host_is_superhost",
    "host_total_listings_count",
    # "property_type",
    "room_type",
    "accommodates",
    # "bathrooms_text",
    "beds",
    "price",
    "maximum_nights",
    "number_of_reviews",
    "review_scores_rating",
    "review_scores_accuracy",
    "review_scores_cleanliness",
    "review_scores_checkin",
    "review_scores_communication",
    "review_scores_location",
    "review_scores_value",
    "instant_bookable",
    "reviews_per_month",
    "description",
    "num_amenities",
    "total_verification_types",
    #"neighbourhood_cleansed"
]

TARGET_LIST = [
    "review_scores_rating",
    "review_scores_accuracy",
    "review_scores_cleanliness",
    "review_scores_checkin",
    "review_scores_communication",
    "review_scores_location",
    "review_scores_value"
]

ENCODING_LIST = [
    "host_is_superhost",
    # "host_response_time",
    # "bathrooms_text",
    # "property_type",
    "room_type",
    "instant_bookable",

]

ONE_HOT_ENCODING_LIST= [
    #"neighbourhood_cleansed"
]

ACTUAL_FEATURES = []


LANGUAGE_COUNT = {}

LANGUAGES = ['hungarian',
 'swedish',
 'kazakh',
 'norwegian',
 'finnish',
 'arabic',
 'indonesian',
 'portuguese',
 'turkish',
 'azerbaijani',
 'slovene',
 'spanish',
 'danish',
 'nepali',
 'romanian',
 'greek',
 'dutch',
 'tajik',
 'german',
 'english',
 'russian',
 'french',
 'italian']

class Preprocessing:
    @staticmethod
    def Review_Text_PreprocessingFunc(review):
        try:
            if len(review.split(" ")) > 5:
                langdetect.DetectorFactory.seed = 0
                #print(tb.detect_language())
                #print(f"Processing review: {review}")
                lang = langdetect.detect(review)
                if lang not in LANGUAGE_COUNT:
                    LANGUAGE_COUNT[lang] = 0
                LANGUAGE_COUNT[lang] += 1
                # print(LANGUAGE_COUNT)
            return lang
        except:
            return "None"
        return "None"
    @staticmethod
    def tfidf_mean(list_data):
        # print(list_data)
        return list_data

    @staticmethod
    def Reviews_Preprocessing_Function(data):
        result = data
        result = result.dropna()
        result = result.reset_index(drop=True)
        result["comments"] = (result["comments"].str.encode("ascii","ignore")).str.decode("ascii")
        result["comments"] = (result["comments"].str.replace("<br/>", "")).str.replace("<br>", "")
        new_arr = result["comments"]
        #
        stop_words = nltk.corpus.stopwords.words(LANGUAGES[0])
        for i in range(1, len(LANGUAGES)):
            stop_words = stop_words + nltk.corpus.stopwords.words(LANGUAGES[i])
        vectorizer = TfidfVectorizer(stop_words=stop_words,max_features=500)
        X = vectorizer.fit_transform(new_arr)
        # feature_array = np.array([f"tfidf_{i}" for i in vectorizer.get_feature_names()])
        # tfidf_sorting = np.argsort(X.toarray()).flatten()[::-1]
        # n = 150
        # top_n = feature_array[tfidf_sorting][:n]
        # print("Top Features:")
        # print(top_n)
        new_df = pd.DataFrame(data=X.toarray(),columns=[f"tfidf_{i}" for i in vectorizer.get_feature_names()])
        merged = pd.concat((result, new_df), axis=1)
        merged = merged.drop(["id", "date", "reviewer_id", "reviewer_name"], axis=1)
        merged = merged.groupby("listing_id").mean()
        return [merged]
    @staticmethod
    def Description_PreProcessing_Function(data):
        if type(data) == type("string"):
            return re.sub(r"[^A-z]", " ", data)
        else:
            return None
    @staticmethod
    def Listing_Preprocessing_Function(data):
        # for target in TARGET_LIST:
            # fig, ax = plt.subplot(111)
            # ax.title(f"Bar plot for {target}")
            # print(data.loc[data[target] > 0 and data[target] < 1].value_counts())
            # plt.close(fig)
        """
        """
        column_na_counts = np.array([data[col].isna().sum() for col in data.columns])
        
        na_column_gt_zero = data.columns[np.where(column_na_counts > 0)]
        na_column_gt_zero2 = na_column_gt_zero
        na_column_count_gt_zero = column_na_counts[np.where(column_na_counts > 0)]
        na_column_count_gt_zero2 = na_column_count_gt_zero 
        na_column_gt_zero = na_column_gt_zero[np.argsort(na_column_count_gt_zero)][::-1]
        na_column_count_gt_zero = na_column_count_gt_zero[np.argsort(na_column_count_gt_zero)][::-1]
        export_csv = {"NaN Column Name" : na_column_gt_zero , "Count" : na_column_count_gt_zero }
        pd.DataFrame(data=export_csv).to_csv("./nan_values.csv")
        # fig = plt.figure(dpi=200)
        # plt.title("NaN counts")
        # plt.xticks(rotation=90)
        # plt.bar(na_column_gt_zero2, na_column_count_gt_zero2, color='red', width=0.75)
        # plt.show()
        # plt.close(fig)
        
        for col in USEFUL_FEATURES_LIST:
            if col not in TARGET_LIST and col != "description" and col != "id" and col not in ONE_HOT_ENCODING_LIST:
                ACTUAL_FEATURES.append(col)
        data["beds"] = data["beds"].fillna(data["beds"].dropna().mean())
        data["num_amenities"] = (data["amenities"].apply(literal_eval)).map(len)
        data["total_verification_types"] = data["host_verifications"].apply(literal_eval).map(len)
        print(data["num_amenities"].value_counts())
        print(data["total_verification_types"].value_counts())
        
        result = data[USEFUL_FEATURES_LIST]
        result["price"] = DataProcessor.replace_df_column(
            DataProcessor.replace_df_column(
                result["price"], '$', ''
            ), ',', ''
        ).apply(pd.to_numeric)

        
        # result["host_response_rate"] = DataProcessor.replace_df_column(
        #     result["host_response_rate"], '%', ''
        # ).apply(pd.to_numeric)
        #result["host_response_rate"] = result["host_response_rate"].fillna(result["host_response_rate"].dropna().mean())
        result["host_acceptance_rate"] = DataProcessor.replace_df_column(
            result["host_acceptance_rate"], '%', ''
        ).apply(pd.to_numeric)
        """
            Perform label encoding
        """
        LEncoder = LabelEncoder()
        for col in ENCODING_LIST:
            if col in USEFUL_FEATURES_LIST:
                result[col] = LEncoder.fit_transform(result[col])
                #result.drop(col)
        # OEncoder = OneHotEncoder()
        # for col in ONE_HOT_ENCODING_LIST:
        #     if col in USEFUL_FEATURES_LIST:
        #         result = result.join(pd.get_dummies(data["neighbourhood_cleansed"]))
        # result = result.drop("neighbourhood_cleansed", axis=1)
        """
            Perform quantization
        """
        target_qcut_map = {
            "review_scores_rating": {
                "total_quantiles": 3,
                "class_list": [0, 1, 2]
            },
            "review_scores_accuracy": {
                "total_quantiles": 3, 
                "class_list": [0, 1, 2]
            },
            "review_scores_cleanliness": {
                "total_quantiles": 3, 
                "class_list": [0, 1, 2]
            },
            "review_scores_checkin": {
                "total_quantiles": 2, 
                "class_list": [0, 1]
            },
            "review_scores_communication": {
                "total_quantiles": 2, 
                "class_list": [0, 1]
            },
            "review_scores_location": {
                "total_quantiles": 3, 
                "class_list": [0, 1, 2]
            },
            "review_scores_value": {
                "total_quantiles": 3, 
                "class_list": [0, 1, 2]
            }
        }
        # for target in TARGET_LIST:
        #     fig = plt.figure()
        #     plt.plot()
        #     plt.scatter(result["instant_bookable"], result[target])
        #     plt.xlabel("price")
        #     plt.ylabel(target)
        #     plt.show()
        #     plt.close(fig)
        target_df = result[TARGET_LIST]
        for target_label in TARGET_LIST:
            result[target_label] = pd.qcut(
                result[target_label],
                target_qcut_map[target_label]["total_quantiles"],
                duplicates='drop',
                labels=target_qcut_map[target_label]["class_list"]
            )
        
        result["description"] = result["description"]\
            .str.replace("<br />", "")\
            .str.replace("<b>", "")\
            .str.replace("</b>", "")
        result["description"] = result["description"].apply(Preprocessing.Description_PreProcessing_Function)
        result = result.dropna()
        result = result.reset_index(drop=True)
        #result["description"] = re.sub(r"^A-z", "", result["description"].str)
        new_arr = result["description"]
        vectorizer = TfidfVectorizer(stop_words='english', max_df=0.2, max_features=20)
        X = vectorizer.fit_transform(new_arr)
        new_df = pd.DataFrame(data=X.toarray(),columns=[
            f"tfidf_{i}" for i in vectorizer.get_feature_names()
        ])
        new_df["id"] = result["id"]
        feature_array = np.array([f"tfidf_{i}" for i in vectorizer.get_feature_names()])
        tfidf_sorting = np.argsort(X.toarray()).flatten()[::-1]
        n = 10
        #result = pd.merge(left=result, right=new_df, left_on="id", right_on="id")
        result.drop(["description"], axis=1, inplace=True)

        
        result = result.apply(pd.to_numeric)
        for key, value in enumerate(result):
            if value not in ENCODING_LIST and value not in TARGET_LIST and value != "id" and value not in new_df.columns and value not in ONE_HOT_ENCODING_LIST:
                result[value] = Normalization.z_score(result[value].astype('float'))
        # for column in new_df.columns:
        #     ACTUAL_FEATURES.append(column)
        #     print(new_df[column])
        #     result[column] = new_df[column]
        return [result, new_df, target_df]
