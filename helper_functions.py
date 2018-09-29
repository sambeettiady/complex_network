def generate_and_save_training_history_plot(history,filename='Training History'):
    training_history = pd.DataFrame(history.history)
    training_history.columns = ['Training Accuracy','Training Loss','Testing Accuracy', 'Testing Loss']
    training_history['Epoch'] = training_history.index + 1
    training_history = training_history.melt('Epoch', var_name='type', value_name='Accuracy and Loss')
    g = sns.factorplot(x='Epoch', y='Accuracy and Loss', hue='type', data=training_history,size=7)
    ax = plt.gca()
    ax.set_title(filename)
    plt.show()
    g.savefig(filename + '.png')

def classfication_report(testing_data,model):
    y_probs = model.predict(testing_data)
    y_labels = [round(prob) for prob in y_probs]

#Plot ROC Curve
    fpr, tpr, thresholds = roc_curve(y_score=y_probs,y_true=y_test)
    roc_df = pd.DataFrame(columns=['FPR','TPR'])
    roc_df['FPR'] = pd.Series(fpr)
    roc_df['TPR'] = pd.Series(tpr)
    roc_plot = sns.lmplot(x = 'FPR', y = 'TPR', data = roc_df, fit_reg = False, size = 7)
    plt.plot([0,1],[0,1])
    ax = plt.gca()
    ax.set_title('ROC Curve')
    plt.show()

#Show metrics
    print 'AUC: ', roc_auc_score(y_score=y_probs,y_true=y_test)
    print 'Accuracy: ', accuracy_score(y_pred=y_labels,y_true=y_test)
    print 'Precision: ', precision_score(y_pred=y_labels,y_true=y_test)
    print 'Recall: ', recall_score(y_pred=y_labels,y_true=y_test)
    print 'F-1 Score (Micro): ', f1_score(y_pred=y_labels,y_true=y_test,average='micro')
