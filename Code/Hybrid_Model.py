from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import f1_score, precision_score, recall_score
import time
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns

def hybrid_model_formation(clf1, clf2, clf3, clf4, clf5, X, Y, columns):
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=1)
    
    print("Phase 1 Started")
    print("Training Models and Generating Report:")
    print("------------------------------------------------------------------------------------------------------")
    try:
        t1 = time.time()
        clf1.fit(x_train, y_train)
        clf1_acc = clf1.score(x_test, y_test)
        pred1 = clf1.predict(x_test)
        clf1_f1 = f1_score(pred1, y_test)
        clf1_pre = precision_score(pred1, y_test)
        clf1_rec = recall_score(pred1, y_test)
        fpr, tpr, thresholds = metrics.roc_curve(y_test, clf1.predict_proba(x_test)[:, 1])
        clf1_auc = metrics.auc(fpr, tpr)
        feature_imp1 = pd.Series([abs(i) for i in clf1.coef_[0]], index = columns[:26])
        t1_ = time.time()
        final_t1 = t1_ - t1
        print("Model 1 Trained and Tested: ")
        print("Accuracy: ", clf1_acc)
        print("F1 Score:", clf1_f1)
        print("AUC Score: ", clf1_auc)
        print("Precision Score: ", clf1_pre)
        print("Recall Score: ", clf1_rec)
        print("Time :", final_t1)
        
        print("------------------------------------------------------------------------------------------------------")
        t2 = time.time()
        clf2.fit(x_train, y_train)
        clf2_acc = clf2.score(x_test, y_test)
        pred2 = clf2.predict(x_test)
        clf2_pre = precision_score(pred2, y_test)
        clf2_rec = recall_score(pred2, y_test)
        clf2_f1 = f1_score(pred2, y_test)
        fpr, tpr, thresholds = metrics.roc_curve(y_test, clf2.predict_proba(x_test)[:, 1])
        clf2_auc = metrics.auc(fpr, tpr)
        feature_imp2 = pd.Series([abs(i) for i in clf2.coef_[0]], index = columns[:26])
        t2_ = time.time()
        final_t2 = t2_ - t2
        print("Model 2 Trained and Tested: ")
        print("Accuracy: ", clf2_acc)
        print("F1 Score:", clf2_f1)
        print("AUC Score: ", clf2_auc)
        print("Precision Score: ", clf2_pre)
        print("Recall Score: ", clf2_rec)
        print("Time :", final_t2)
        
        print("------------------------------------------------------------------------------------------------------")
        t3 = time.time()
        clf3.fit(x_train, y_train)
        clf3_acc = clf3.score(x_test, y_test)
        pred3 = clf3.predict(x_test)
        clf3_f1 = f1_score(pred3, y_test)
        clf3_pre = precision_score(pred3, y_test)
        clf3_rec = recall_score(pred3, y_test)
        fpr, tpr, thresholds = metrics.roc_curve(y_test, clf3.predict_proba(x_test)[:, 1])
        clf3_auc = metrics.auc(fpr, tpr)
        feature_imp3 = pd.Series([abs(i) for i in permutation_importance(clf3, x_test, y_test).importances_mean], index = columns[:26])
        t3_ = time.time()
        final_t3 = t3_ - t3
        print("Model 3 Trained and Tested: ")
        print("Accuracy: ", clf3_acc)
        print("F1 Score:", clf3_f1)
        print("AUC Score: ", clf3_auc)
        print("Precision Score: ", clf3_pre)
        print("Recall Score: ", clf3_rec)
        print("Time :", final_t3)
        
        print("------------------------------------------------------------------------------------------------------")
        t4 = time.time()
        clf4.fit(x_train, y_train)
        clf4_acc = clf4.score(x_test, y_test)
        pred4 = clf4.predict(x_test)
        clf4_f1 = f1_score(pred4, y_test)
        clf4_pre = precision_score(pred4, y_test)
        clf4_rec = recall_score(pred4, y_test)
        fpr, tpr, thresholds = metrics.roc_curve(y_test, clf4.predict_proba(x_test)[:, 1])
        clf4_auc = metrics.auc(fpr, tpr)
        feature_imp4 = pd.Series(clf4.feature_importances_, index = columns[:26])
        t4_ = time.time()
        final_t4 = t4_ - t4
        print("Model 4 Trained and Tested: ")
        print("Accuracy: ", clf4_acc)
        print("F1 Score:", clf4_f1)
        print("AUC Score: ", clf4_auc)
        print("Precision Score: ", clf4_pre)
        print("Recall Score: ", clf4_rec)
        print("Time :", final_t4)
        
        print("------------------------------------------------------------------------------------------------------")
        t5 = time.time()
        clf5.fit(x_train, y_train)
        clf5_acc = clf5.score(x_test, y_test)
        pred5 = clf5.predict(x_test)
        clf5_f1 = f1_score(pred5, y_test)
        clf5_pre = precision_score(pred5, y_test)
        clf5_rec = recall_score(pred5, y_test)
        fpr, tpr, thresholds = metrics.roc_curve(y_test, clf5.predict_proba(x_test)[:, 1])
        clf5_auc = metrics.auc(fpr, tpr)
        feature_imp5 = pd.Series(clf5.feature_importances_, index = columns[:26])
        t5_ = time.time()
        final_t5 = t5_ - t5
        print("Model 5 Trained and Tested: ")
        print("Accuracy: ", clf5_acc)
        print("F1 Score:", clf5_f1)
        print("AUC Score: ", clf5_auc)
        print("Precision Score: ", clf5_pre)
        print("Recall Score: ", clf5_rec)
        print("Time :", final_t5)
        print("------------------------------------------------------------------------------------------------------")
        print("Process Successfully completed without any errors....")
    except:
        print("Error Occured during model training and evaluation....")
        return "Error!!"
    
    print()
    print("Phase 2 Started")
    print("The slelection process has started....")
    all_classifiers = [clf1, clf2, clf3, clf4, clf5]
    all_classifiers_names = ["clf1", "clf2", "clf3", "clf4", "clf5"]
    selected_classifiers = [0 for i in range(5)]
    classifiers = {"clf1":clf1, "clf2":clf2, "clf3":clf3, "clf4":clf4, "clf5":clf5}
    accuracys = [clf1_acc, clf2_acc, clf3_acc, clf4_acc, clf5_acc]
    f1_scores = [clf1_f1, clf2_f1, clf3_f1, clf4_f1, clf5_f1]
    auc_scores = [clf1_auc, clf2_auc, clf3_auc, clf4_auc, clf5_auc]
    precision_scores = [clf1_pre, clf2_pre, clf3_pre, clf4_pre, clf5_pre]
    recall_scores = [clf1_rec, clf2_rec, clf3_rec, clf4_rec, clf5_rec]
    train_time = [final_t1, final_t2, final_t3, final_t4, final_t5]
    f_imp = [feature_imp1, feature_imp2, feature_imp3, feature_imp4, feature_imp5]
    
    fig = plt.figure(figsize=(15, 10))
    for o in range(5):
        plt.subplot(3,2,o+1)
        f_imp[o].nlargest(10).plot(kind = 'barh')
        plt.title(f"Feature Importance of {all_classifiers_names[o]}")
    
    
    fig = plt.figure(figsize=(15, 8))
    plt.plot([i*100 for i in accuracys], 'go-', color='lightblue', label='Accuracy')
    plt.plot([i*100 for i in f1_scores], 'go-', color='red', label='F1-Score')
    plt.plot([i*100 for i in auc_scores], 'go-', color='green', label='AUC-Score')
    plt.plot(train_time, 'go-', color='darkblue', label='Train Time(sec)')
    plt.title('Model Performance')
    plt.legend()
    plt.xticks([i for i in range(5)], all_classifiers_names)
    plt.xlabel('Model Names')
    plt.show()
    
    for i in range(5):
        if accuracys[i] >= .90 and f1_scores[i] >= .90 and auc_scores[i] >= .90 and precision_scores[i] >= 0.9 and recall_scores[i] >= 0.9:
            selected_classifiers[i] = 1
            
    for j in range(5):
        if not selected_classifiers[j]:
            all_classifiers[j] = None
            all_classifiers_names[j] = None
            f1_scores[j] = None
            
    all_classifiers = [i for i in all_classifiers if i != None]
    all_classifiers_names = [i for i in all_classifiers_names if i != None]
    f1_scores = [i for i in f1_scores if i != None]
    
    print("The selected classifiers are: ")
    for i in range(len(all_classifiers)):
        print(f"{i+1}. {all_classifiers_names[i]}")
    
    print()
    print("Phase 3 Started")
    # Final testing with formula:
    numerator = []
    denominator = sum([i for i in f1_scores])
    for i in range(len(all_classifiers)):
        proba = all_classifiers[i].predict_proba(x_test)[:, 1]
        # print(proba)
        final_num = np.array([x*f1_scores[i] for x in proba])
        numerator.append(final_num)
        
    numerator_val = np.zeros(shape=y_test.shape)
    for j in range(len(numerator)):
        numerator_val += numerator[j]
        
    numerator_val /= denominator
    
    for i in range(len(numerator_val)):
        if numerator_val[i] > 0.5:
            numerator_val[i] = 1
        else:
            numerator_val[i] = 0
    
    print("The final model is ready and has been evaluated the results are as following:")
    print("Accuracy :", metrics.accuracy_score(numerator_val, y_test))
    print("F1 Score :", f1_score(numerator_val, y_test))
    fpr, tpr, thresholds = metrics.roc_curve(y_test, numerator_val)
    print("AUC Score: ", metrics.auc(fpr, tpr))
    print("Average Precision :", precision_score(numerator_val, y_test))
    print("Average Recall :", recall_score(numerator_val, y_test))
    mat = confusion_matrix(numerator_val, y_test)
    x = [mat[0][0], mat[1][0]]
    y = [mat[0][1], mat[1][1]]
    df = pd.DataFrame({"Fraud": x, "Not Fraud": y})
    df.set_index(pd.Index(["Fraud", "Not Fraud"]), inplace = True)
    sns.heatmap(df, annot=True)
    print("Compilation successful!")


# hybrid_model_formation(LogisticRegression(max_iter=1000), SGDClassifier(loss="modified_huber", penalty = "l2"), 
#                       GaussianNB(), RandomForestClassifier(n_estimators=20, max_depth = 26), 
#                        XGBClassifier(use_label_encoder=False, disable_default_eval_metric=1), X, Y, list(df.columns))