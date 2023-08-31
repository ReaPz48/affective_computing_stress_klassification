# imports
import numpy as np
import pandas as pd
import seaborn as sns

import plotly.io
plotly.io.templates.default = "plotly_white"
import plotly
plotly.offline.init_notebook_mode()

from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc

from matplotlib import pyplot as plt
import scikitplot as skplt


def plot_model_results(model, X_test, y_test):
    print('best score: ' + str(model.best_score_) + '\n')
    print('best parameters: ' + str(model.best_params_) + '\n')
    print('score: ' + str(model.best_estimator_.score(X_test, y_test)) + '\n')
    
    report = classification_report(y_test, model.best_estimator_.predict(X_test))
    print(report)


def plot_feature_importance_rfc(model, title, save_path):
    # Creating importances_df dataframe
    importances_df = pd.DataFrame({"Features" : model.best_estimator_.feature_names_in_, 
                                "Importance" : model.best_estimator_.feature_importances_})
    
    importances_df = importances_df.sort_values(by='Importance', ascending=False)
                                
    # Plotting bar chart, g is from graph
    g = sns.barplot(x=importances_df["Features"], 
                    y=importances_df["Importance"], palette = "Blues_d")
    g.set_title("Feature Importance", fontsize=8);
    g.set_xticklabels(g.get_xticklabels(), rotation=90, horizontalalignment="right", fontsize=6)
    plt.axhline(y=0, color='black', linestyle='-') 
    plt.show() 

    importances_df = importances_df.loc[importances_df['Importance'] >= 0.015] 
                             
    # Plotting bar chart, g is from graph
    gn = sns.barplot(x=importances_df["Features"], 
                y=importances_df["Importance"], palette = "Blues_d")
    gn.set_title("Feature Importances of " + title, fontsize=10);
    gn.set_xticklabels(gn.get_xticklabels(), rotation=90, horizontalalignment="right")
    plt.axhline(y=0, color='black', linestyle='-')
    plt.savefig(save_path + '_feature_importance.png', bbox_inches='tight') 
    plt.show()


def plot_feature_importance(model, title, save_path, upper_boundry, lower_boundry):
    # Creating importances_df dataframe
    importances_df = pd.DataFrame({"Features" : model.best_estimator_.feature_names_in_, 
                                "Importance" : model.best_estimator_.coef_[0]})
    
    importances_df = importances_df.sort_values(by='Importance', ascending=False)
                                
    # Plotting bar chart, g is from graph
    g = sns.barplot(x=importances_df["Features"], 
                    y=importances_df["Importance"], palette = "Blues_d")
    g.set_title("Feature Importance", fontsize=8);
    g.set_xticklabels(g.get_xticklabels(), rotation=90, horizontalalignment="right", fontsize=6)
    plt.axhline(y=0, color='black', linestyle='-')
    plt.show() 

    importances_df_upper = importances_df.loc[importances_df['Importance'] >= upper_boundry]
    importances_df_lower = importances_df.loc[importances_df['Importance'] <= lower_boundry]  

    importances_df = pd.concat([importances_df_upper, importances_df_lower], ignore_index=True)
                             
    # Plotting bar chart, g is from graph
    gn = sns.barplot(x=importances_df["Features"], 
                y=importances_df["Importance"], palette = "Blues_d")
    gn.set_title("Feature Importances of " + title, fontsize=10)
    gn.set_xticklabels(gn.get_xticklabels(), rotation=90, horizontalalignment="right")
    plt.axhline(y=0, color='black', linestyle='-')

    plt.savefig(save_path + '_feature_importance.png', bbox_inches='tight') 
    plt.show()


def AUROC_draw_single(model, model_name, save_path, X_test, y_test):
    y_pred = model.best_estimator_.predict_proba(X_test)
    y_pred = y_pred[:, 1]

    # generate a no skill prediction (majority class)
    ns_pred = [0 for _ in range(len(y_test))]

    # calculate scores
    ns_auc = roc_auc_score(y_test, ns_pred)
    lr_auc = roc_auc_score(y_test, y_pred)

    # summarize scores
    print('No Skill: ROC AUC=%.3f' % (ns_auc))
    print('Logistic: ROC AUC=%.3f' % (lr_auc))

    # calculate roc curves
    ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_pred)
    lr_fpr, lr_tpr, _ = roc_curve(y_test, y_pred)

    label = '(AUC = %.3f)' % (lr_auc)
    # plot the roc curve for the model
    plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
    plt.plot(lr_fpr, lr_tpr, marker=',', label=model_name + ' ' + label)
    # axis labels and title
    plt.title(model_name + " - Receiver Operating Characteristic")
    plt.xlabel('False Positive Rate (1-Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    # show the legend
    plt.legend()
    # show the plot
    plt.savefig(save_path + '_ROC.png', bbox_inches='tight')
    plt.show()


def PRC_draw_single(model, model_name, save_path, X_test, y_test):
    y_pred = model.best_estimator_.predict_proba(X_test)
    y_pred = y_pred[:, 1]

    # calculate scores
    lr_precision, lr_recall, _ = precision_recall_curve(y_test, y_pred)
    lr_auc = auc(lr_recall, lr_precision)
    ns_auc = len(y_test[y_test==1]) / len(y_test)

    # summarize scores
    print('No Skill: PRC AUC=%.3f' % (ns_auc))
    print('Logistic: PRC AUC=%.3f' % (lr_auc))

    # create label
    label = '(AUC = %.3f)' % (lr_auc)
    label_no = '(AUC = %.3f)' % (ns_auc)
    # plot the precision-recall curves
    plt.plot([0, 1], [ns_auc, ns_auc], linestyle='--', label='No Skill' + label_no)
    plt.plot(lr_recall, lr_precision, marker=',', label=model_name + ' ' + label)
    # axis labels and title
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(model_name + " - Precision Recall Curve")
    # show the legend
    plt.legend()
    # show the plot
    plt.savefig(save_path + '_PRC.png', bbox_inches='tight')
    plt.show()


def AUROC_draw_all(rfc, svm, lr, knn, lda, save_path, X_test, y_test):
    y_pred_rfc = rfc.predict_proba(X_test)
    y_pred_rfc = y_pred_rfc[:, 1]

    y_pred_svm = svm.predict_proba(X_test)
    y_pred_svm = y_pred_svm[:, 1]

    y_pred_lr = lr.predict_proba(X_test)
    y_pred_lr = y_pred_lr[:, 1]

    y_pred_knn = knn.predict_proba(X_test)
    y_pred_knn = y_pred_knn[:, 1]

    y_pred_lda = lda.predict_proba(X_test)
    y_pred_lda = y_pred_lda[:, 1]

    # generate a no skill prediction (majority class)
    ns_pred = [0 for _ in range(len(y_test))]   

    # calculate scores
    ns_auc = roc_auc_score(y_test, ns_pred)

    rfc_auc = roc_auc_score(y_test, y_pred_rfc)
    svm_auc = roc_auc_score(y_test, y_pred_svm)
    lr_auc = roc_auc_score(y_test, y_pred_lr)
    knn_auc = roc_auc_score(y_test, y_pred_knn)
    lda_auc = roc_auc_score(y_test, y_pred_lda)

    # summarize scores
    print('No Skill: ROC AUC=%.3f' % (ns_auc))
    print('RFC: ROC AUC=%.3f' % (rfc_auc))
    print('SVM: ROC AUC=%.3f' % (svm_auc))
    print('LR: ROC AUC=%.3f' % (lr_auc))
    print('KNN: ROC AUC=%.3f' % (knn_auc))
    print('LDA: ROC AUC=%.3f' % (lda_auc))

        # calculate roc curves
    ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_pred)

    rfc_fpr, rfc_tpr, _ = roc_curve(y_test, y_pred_rfc)
    svm_fpr, svm_tpr, _ = roc_curve(y_test, y_pred_svm)
    lr_fpr, lr_tpr, _ = roc_curve(y_test, y_pred_lr)
    knn_fpr, knn_tpr, _ = roc_curve(y_test, y_pred_knn)
    lda_fpr, lda_tpr, _ = roc_curve(y_test, y_pred_lda)

    # create label
    rfc_label = '(AUC = %.3f)' % (rfc_auc)
    svm_label = '(AUC = %.3f)' % (svm_auc)
    lr_label = '(AUC = %.3f)' % (lr_auc)
    knn_label = '(AUC = %.3f)' % (knn_auc)
    lda_label = '(AUC = %.3f)' % (lr_auc)

    # plot the roc curve for the model
    plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
    plt.plot(rfc_fpr, rfc_tpr, marker=',', label='RFC' + ' ' + rfc_label)
    plt.plot(svm_fpr, svm_tpr, marker=',', label='SVM' + ' ' + svm_label)
    plt.plot(lr_fpr, lr_tpr, marker=',', label='LR' + ' ' + lr_label)
    plt.plot(knn_fpr, knn_tpr, marker=',', label='KNN' + ' ' + knn_label)
    plt.plot(lda_fpr, lda_tpr, marker=',', label='LDA' + ' ' + lda_label)

    # axis labels and title
    plt.xlabel('False Positive Rate (1-Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title("All Models - Receiver Operating Characteristic")
    # show the legend
    plt.legend()
    # show the plot
    plt.savefig(save_path + '_ROC_all.png', bbox_inches='tight')
    plt.show()


def PRC_draw_all(rfc, svm, lr, knn, lda, save_path, X_test, y_test):
    y_pred_rfc = rfc.predict_proba(X_test)
    y_pred_rfc = y_pred_rfc[:, 1]

    y_pred_svm = svm.predict_proba(X_test)
    y_pred_svm = y_pred_svm[:, 1]

    y_pred_lr = lr.predict_proba(X_test)
    y_pred_lr = y_pred_lr[:, 1]

    y_pred_knn = knn.predict_proba(X_test)
    y_pred_knn = y_pred_knn[:, 1]

    y_pred_lda = lda.predict_proba(X_test)
    y_pred_lda = y_pred_lda[:, 1]

    # calculate scores
    rfc_precision, rfc_recall, _ = precision_recall_curve(y_test, y_pred_rfc)
    svm_precision, svm_recall, _ = precision_recall_curve(y_test, y_pred_svm)
    lr_precision, lr_recall, _ = precision_recall_curve(y_test, y_pred_lr)
    knn_precision, knn_recall, _ = precision_recall_curve(y_test, y_pred_knn)
    lda_precision, lda_recall, _ = precision_recall_curve(y_test, y_pred_lda)

    rfc_auc = auc(rfc_recall, rfc_precision)
    svm_auc = auc(svm_recall, svm_precision)
    lr_auc = auc(lr_recall, lr_precision)
    knn_auc = auc(knn_recall, knn_precision)
    lda_auc = auc(lda_recall, lda_precision)

    ns_auc = len(y_test[y_test==1]) / len(y_test)

    # summarize scores
    print('No Skill: PRC AUC=%.3f' % (ns_auc))
    print('RFC: PRC AUC=%.3f' % (rfc_auc))
    print('SVM: PRC AUC=%.3f' % (svm_auc))
    print('LR: PRC AUC=%.3f' % (lr_auc))
    print('KNN: PRC AUC=%.3f' % (knn_auc))
    print('LDA: PRC AUC=%.3f' % (lda_auc))

    # create label
    label_no = '(AUC = %.3f)' % (ns_auc)
    rfc_label = '(AUC = %.3f)' % (rfc_auc)
    svm_label = '(AUC = %.3f)' % (svm_auc)
    lr_label = '(AUC = %.3f)' % (lr_auc)
    knn_label = '(AUC = %.3f)' % (knn_auc)
    lda_label = '(AUC = %.3f)' % (lda_auc)

    # plot the precision-recall curves
    plt.plot([0, 1], [ns_auc, ns_auc], linestyle='--', label='No Skill' + label_no)
    plt.plot(rfc_recall, rfc_precision, marker=',', label='RFC' + ' ' + rfc_label)
    plt.plot(svm_recall, svm_precision, marker=',', label='SVM' + ' ' + svm_label)
    plt.plot(lr_recall, lr_precision, marker=',', label='LR' + ' ' + lr_label)
    plt.plot(knn_recall, knn_precision, marker=',', label='KNN' + ' ' + knn_label)
    plt.plot(lda_recall, lda_precision, marker=',', label='LDA' + ' ' + lda_label)

    # axis labels and title
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title("All Models - Precision Recall Curve")
    # show the legend
    plt.legend()
    # show the plot
    plt.savefig(save_path + '_PRC_all.png', bbox_inches='tight')
    plt.show()


def plot_confusion_matrix(model, title, save_path, X_test, y_test):
    y_proba = model.best_estimator_.predict_proba(X_test)
    y_pred = np.where(y_proba[:,1] > 0.5, 1, 0)
    
    skplt.metrics.plot_confusion_matrix(y_test, y_pred, normalize=False, title = 'Confusion Matrix For ' + title)
    plt.savefig(save_path + '_confusion_matrix.png', bbox_inches='tight')
    plt.show()


def AUROC_draw_two(model, model_name, save_path, X_test, y_test, X_polar, y_polar):
    y_pred_wesad = model.predict_proba(X_test)
    y_pred_wesad = y_pred_wesad[:, 1]

    y_pred_polar = model.predict_proba(X_polar)
    y_pred_polar = y_pred_polar[:, 1]


    # generate a no skill prediction (majority class)
    ns_pred = [0 for _ in range(len(y_test))]   

    # calculate scores
    ns_auc = roc_auc_score(y_test, ns_pred)

    wesad_auc = roc_auc_score(y_test, y_pred_wesad)
    polar_auc = roc_auc_score(y_polar, y_pred_polar)

    # summarize scores
    print('No Skill: ROC AUC=%.3f' % (ns_auc))
    print('WESAD: ROC AUC=%.3f' % (wesad_auc))
    print('Polar: ROC AUC=%.3f' % (polar_auc))

    # calculate roc curves
    ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_pred)

    wesad_fpr, wesad_tpr, _ = roc_curve(y_test, y_pred_wesad)
    polar_fpr, polar_tpr, _ = roc_curve(y_polar, y_pred_polar)


    # create label
    wesad_label = '(AUC = %.3f)' % (wesad_auc)
    polar_label = '(AUC = %.3f)' % (polar_auc)

    # plot the roc curve for the model
    plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
    plt.plot(wesad_fpr, wesad_tpr, marker=',', label='WESAD' + ' ' + wesad_label)
    plt.plot(polar_fpr, polar_tpr, marker=',', label='Polar' + ' ' + polar_label)

    # axis labels and title
    plt.xlabel('False Positive Rate (1-Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title(model_name + " - Receiver Operating Characteristic")
    # show the legend
    plt.legend()
    # show the plot
    plt.savefig(save_path + '_ROC_comparison.png', bbox_inches='tight')
    plt.show()


def PRC_draw_two(model, model_name, save_path, X_test, y_test, X_polar, y_polar):
    y_pred_wesad = model.predict_proba(X_test)
    y_pred_wesad = y_pred_wesad[:, 1]

    y_pred_polar = model.predict_proba(X_polar)
    y_pred_polar = y_pred_polar[:, 1]


    # calculate scores
    wesad_precision, wesad_recall, _ = precision_recall_curve(y_test, y_pred_wesad)
    polar_precision, polar_recall, _ = precision_recall_curve(y_polar, y_pred_polar)

    wesad_auc = auc(wesad_recall, wesad_precision)
    polar_auc = auc(polar_recall, polar_precision)
    ns_auc = len(y_test[y_test==1]) / len(y_test)

    # summarize scores
    print('No Skill: PRC AUC=%.3f' % (ns_auc))
    print('RFC: PRC AUC=%.3f' % (wesad_auc))
    print('SVM: PRC AUC=%.3f' % (polar_auc))

    # create label
    label_no = '(AUC = %.3f)' % (ns_auc)
    wesad_label = '(AUC = %.3f)' % (wesad_auc)
    polar_label = '(AUC = %.3f)' % (polar_auc)

    # plot the precision-recall curves
    plt.plot([0, 1], [ns_auc, ns_auc], linestyle='--', label='No Skill' + label_no)
    plt.plot(wesad_recall, wesad_precision, marker=',', label='WESAD' + ' ' + wesad_label)
    plt.plot(polar_recall, polar_precision, marker=',', label='Polar' + ' ' + polar_label)

    # axis labels and title
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(model_name + " - Precision Recall Curve")
    # show the legend
    plt.legend()
    # show the plot
    plt.savefig(save_path + '_PRC_comparison.png', bbox_inches='tight')
    plt.show()


def plot_stresslevel(y, save_path, title, window_size, start_performance):
    window = window_size
    average_y = []

    for ind in range(len(y) - window + 1):
        average_y.append(np.mean(y[ind:ind+window]))

    for ind in range(window - 1):
        average_y.insert(0,0)

    x = np.linspace(1, y.shape, len(average_y))

    plt.figure(figsize=(20, 8))
    #plt.plot(x, y, 'k.-', label='Prediction Stress')
    plt.scatter(x, y, label='Prediction Stress', c="black", marker="x")
    plt.plot(x, average_y, 'r', label='Running Average with Windowsize = ' + str(window), solid_capstyle='round')
    plt.axvline(x=start_performance, color='g', ls='--', label='Start of Public Performance')
    plt.xlim(1,len(y)+1)
    plt.title("Result Of Stress Prediction For " + title)
    plt.ylabel("Stresslevel")
    plt.xlabel("Datapoints; Time+30 seconds each")
    plt.grid(linestyle=':')
    plt.legend()
    plt.savefig(save_path + '_stress_moving_average.png', bbox_inches='tight')
    plt.show()




























