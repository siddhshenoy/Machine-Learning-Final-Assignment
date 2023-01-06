from custom.DataProcessor import *
from Preprocessing import *
from custom.Model import *
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.dummy import DummyClassifier
import warnings
warnings.filterwarnings("ignore")

dp = DataProcessor()
dp.load_from_csv(path="./listings.csv", alias="Listings")
dp.attach_preproc_func(alias="Listings", preproc_func=Preprocessing.Listing_Preprocessing_Function)
dp.preprocess_data()

dp_reviews = DataProcessor()
dp_reviews.load_from_csv("./reviews.csv", alias="Reviews")
dp_reviews.attach_preproc_func(alias="Reviews", preproc_func=Preprocessing.Reviews_Preprocessing_Function)
dp_reviews.preprocess_data()
listings = dp.get_data(alias="Listings")
review_data = dp_reviews.get_data(alias="Reviews")
review_data.to_csv("reviews_language.csv")
listings = pd.merge(left=listings,right=review_data, left_on="id", right_on="listing_id")
listings = listings.drop("id", axis=1)
for col in listings.columns:
    if col not in ACTUAL_FEATURES and col not in TARGET_LIST:
        ACTUAL_FEATURES.append(col)

model_data = {
    "LogisticRegression": [],
    "RidgeClassifier": [],
    "KNN": [],
    "SVC": [],
    "DummyClassifier": []
    }
model_col_data = {
    "LogisticRegression": [],
    "RidgeClassifier": [],
    "KNN": [],
    "SVC": [],
    "DummyClassifier": []
    }
C_list = [0.1, 0.5, 1.0, 5.0, 10.0, 15.0, 20.0, 25.0]
C_list.sort()
K_list = [1, 3, 5, 7, 9, 11, 13, 15]
        



for target in TARGET_LIST:
    mean_data = []
    std_data = []
    for C in C_list:
        print("-" * 70)
        print("LOGISTIC REGRESSION = {}".format(C))
        print(f"Target: {target}")
        features = listings[ACTUAL_FEATURES]
        targets = listings[target]
        model = Model()
        model.create(model=LogisticRegression,C_value = C)
        model.load_data(feature_data=features, target_data=targets)
        model.use_cross_validation(total_cross_validations=5)
        model.split_data()
        model.fit_model()        
        accuracy, mse = model.print_model_stats()
        for i, coefficient_list in enumerate(model.get_model().coef_):
            coeff_array = coefficient_list
            coeff_array = np.abs(coeff_array)           # Calculate the absolute value
            feature_array = np.array(ACTUAL_FEATURES)
            sorted_args = np.argsort(coeff_array)
            sorted_args = sorted_args[::-1]
            #print(sorted_args)
            coeff_array = coeff_array[sorted_args]
            feature_array = feature_array[sorted_args]
            fig = plt.figure()
            plt.title(f"Bar plot for coefficient vector {i}")
            plt.xticks(rotation=90)
            plt.bar(feature_array[:10], coeff_array[:10])
            plt.show()
            plt.close(fig)
            
        
        model.plot_ROC_AUC_curve(outpath=f"./plots/LogisticRegression/LogiReg_rocauc_{target}_c_{C}.png")
        model_data["LogisticRegression"].append([
                f"LogisticRegression(C={C})",
                f"Accuracy = {accuracy}",
                f"MSE = {mse}",
                f"Target Name = {target}"
            ])
        model_col_data["LogisticRegression"].append([
            model.get_sorted_coef_list(),
            [np.array(ACTUAL_FEATURES)[k] for k in model.get_sorted_coef_list_idx()]
            ])
        print("-" * 70)
        mean_data.append(model.get_cross_val_scores().mean())
        std_data.append(model.get_cross_val_scores().std())
        
    fig = plt.figure()
    plt.title(f"Errorbar - Logistic Regression\nTarget - {target}")
    plt.errorbar(C_list, mean_data,yerr=std_data)
    plt.savefig(f"./plots/LogisticRegression/LogiReg_Errorbar_Target_{target}.png")
    plt.show()
    
    plt.close(fig)

    
    """
        K-NearestNeighbors
    """
    
    mean_data = []
    std_data = []
    for k in K_list:
        print("-" * 70)
        features = listings[ACTUAL_FEATURES]
        targets = listings[target]
        model = Model()
        model.create(model=KNeighborsClassifier,K_value=k)
        model.load_data(feature_data=features, target_data=targets)
        model.use_cross_validation(total_cross_validations=5)
        model.split_data()
        model.fit_model()
        accuracy,mse = model.print_model_stats()
        model_data["KNN"].append([
            f"kNN(K={k})",
            f"Accuracy = {accuracy}",
            f"MSE = {mse}",
            f"Target Name = {target}"
        ])
        model.plot_ROC_AUC_curve(outpath=f"./plots/kNN/kNN_rocauc_{target}_k_{k}.png")
        mean_data.append(model.get_cross_val_scores().mean())
        std_data.append(model.get_cross_val_scores().std())
        print("-" * 70)
    fig = plt.figure()
    plt.title(f"Errorbar - k-NearestNeighbors\nTarget - {target}")
    plt.errorbar(K_list, mean_data,yerr=std_data)
    plt.savefig(f"./plots/kNN/kNN_Errorbar_Target_{target}.png")
    plt.show()
    plt.close(fig)
        
    # """
    #     Dummy Classifier
    # """
        
    features = listings[ACTUAL_FEATURES]
    targets = listings[target]
    model = Model()
    model.create(model=DummyClassifier, dummy_classifier_strategy="most_frequent")
    model.load_data(feature_data=features,target_data=targets)
    model.use_hold_out_strategy()
    model.split_data()
    model.fit_model()
    accuracy,mse = model.print_model_stats()
    model.plot_ROC_AUC_curve(outpath=f"./plots/DummyClassifier/DummyClassifier_rocauc_{target}")
    model_data["DummyClassifier"].append([
        f"DummyClassifier",
        f"Accuracy = {accuracy}",
        f"MSE = {mse}",
        f"Target Name = {target}"
    ])
    
        

for key in model_data:
    print(">" * 50)
    for results in model_data[key]:
        print(results)
    print("<" * 50)


    

