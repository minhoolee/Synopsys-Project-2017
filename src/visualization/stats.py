import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from tqdm import tqdm

def gen_roc_curves(inds=[]):
    name_width = 0
    for k in inds:
        if name_width < len(model_names[k]):
            name_width = len(model_names[k])
    print('Generating curves for ' + ', '.join(model_names[i] for i in inds))
    # Calculate AUC for each model
    for k in inds:
        # Calculate ROC AUC for each class
        print('\nUsing model {:s}'.format(model_names[k]))
        for c in tqdm(range(nb_samples)):
        #for c in range(10):
            fpr[k][c], tpr[k][c], _ = roc_curve(y_test[:,c], y_score[k][:,c])
            roc_auc[k].append(auc(fpr[k][c], tpr[k][c]))
            #print('AUC ( {:^{width}s} )[{:3d}]: {:f}'.format(model_names[k], c, roc_auc[k][c], width=name_width))

def plot_roc_curves(inds=[]):
    for k in inds:
        # ROC AUC for the model has not been created
        if roc_auc[k] is [0]:
            print('Generating ROC curves for {:s}'.format(model_names[k]))
            gen_roc_curves([k])

        #for c in range(nb_samples):
        for c in range(10):
            plt.plot(fpr[k][c], tpr[k][c], label='ROC curve (area = %0.2f)' % roc_auc[k][c])

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])

    #plt.legend(model_names[i] for i in inds, loc='lower right')
    plt.legend(loc='lower right')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic curve')

    # Plot ROC curves
    #print('Plotting ROC curves...')
    plt.show()
    plt.savefig('models/run_logs/auc.png', bbox_inches='tight') # Temporary

def plot_roc_aucs(a, b):
    for k in [a, b]:
        for c in range(nb_samples):
            plt.scatter(roc_auc[a][c], roc_auc[b][c], s=10, c='b', alpha=0.5)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel(model_names[a])
    plt.ylabel(model_names[b])
    #print('Generating curves for ' + ', '.join(model_names[i] for i in indscorr))
    plt.title('Area under the receiver operating characteristic curves')
    # Plot ROC curves
    #print('Plotting ROC AUCs...')
    plt.show()
    plt.savefig('models/run_logs/aucs.png', bbox_inches='tight') # Temporary

def plot_roc_aucs_vs_other(ind, roc_auc_arr, other_name="Competitor"):
    """
    # Arguments
        ind: 
            index of inds list to use
        roc_auc_arr:
            one dimensional numpy array of ROC AUC scores
        other_name:
            name of other model
    """
    for c in range(nb_samples):
        plt.scatter(roc_auc[ind][c], roc_auc_arr[c], s=10, c='b', alpha=0.5)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel(other_name)
    plt.ylabel(model_names[ind])
    plt.title('Area under the receiver operating characteristic curves')
    # Plot ROC curves
    #print('Plotting ROC AUCs...')
    plt.show()
    plt.savefig('models/run_logs/aucs.png', bbox_inches='tight') # Temporary

def plot_roc_aucs_others(arr_1, arr_2, arr_1_name="Competitor 1", arr_2_name="Competitor 2"):
    for c in range(nb_samples):
        plt.scatter(arr_1[c], arr_2[c], s=10, c='b', alpha=0.5)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel(arr_1_name)
    plt.ylabel(arr_2_name)
    plt.title('Area under the receiver operating characteristic curves')
    # Plot ROC curves
    #print('Plotting ROC AUCs...')
    plt.show()
    plt.savefig('models/run_logs/aucs.png', bbox_inches='tight') # Temporary

def plot_roc_aucs_diffs(a, b):
    for k in [a, b]:
        max_diff = 0
        min_diff = 1 # Dummy value
        for c in range(nb_samples):
            diff = roc_auc[a][c]-roc_auc[b][c]
            if diff > max_diff:
                max_diff = diff
            if diff < min_diff:
                min_diff = diff
            plt.scatter(c, diff, s=10, c='b', alpha=0.5)

    max_diff = abs(max_diff)
    min_diff = abs(min_diff)
    scale = 1.1 * (max_diff if max_diff > min_diff else min_diff) # Scale with greatest magnitude
    plt.plot([0, nb_samples], [0, 0], 'k--')
    plt.xlim([0.0, nb_samples])
    plt.ylim([-scale, scale])
    plt.xlabel('Epigenetic Class')
    plt.ylabel(model_names[a] + ' minus ' + model_names[b])
    #print('Generating curves for ' + ', '.join(model_names[i] for i in inds))
    plt.title('Difference between area under the receiver operating characteristic curves')
    # Plot ROC curves
    #print('Plotting ROC AUCs...')
    plt.show()
    plt.savefig('models/run_logs/aucs.png', bbox_inches='tight') # Temporary

def plot_interpolated_pr_curves(inds):
    """
    Do not run this yet, I have not tested it

    http://stackoverflow.com/questions/39836953/how-to-draw-a-precision-recall-curve-with-interpolation-in-python
    """
    precision["micro"], recall["micro"], _ = precision_recall_curve(y_test.ravel(),scores.ravel())
    pr = copy.deepcopy(precision[0])
    rec = copy.deepcopy(recall[0])
    prInv = np.fliplr([pr])[0]
    recInv = np.fliplr([rec])[0]
    j = rec.shape[0]-2
    while j>=0:
        if prInv[j+1]>prInv[j]:
            prInv[j]=prInv[j+1]
        j=j-1
    decreasing_max_precision = np.maximum.accumulate(prInv[::-1])[::-1]
    plt.plot(recInv, decreasing_max_precision, marker= markers[mcounter], label=methodNames[countOfMethods]+': AUC={0:0.2f}'.format(average_precision[0]))

def plot_pr_curves(inds):
    """
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html#sphx-glr-auto-examples-model-selection-plot-precision-recall-py
    """
    # setup plot details
    colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])
    lw = 2

    for k in inds:
        # Compute Precision-Recall and plot curve
        precision = [dict()*nb_of_models]
        recall = [dict()*nb_of_models]
        average_precision = [dict()*nb_of_models]
        for i in range(nb_classes):
            precision[k][i], recall[k][i], _ = precision_recall_curve(y_test[:, i],
                                                                y_score[k][:, i])
            average_precision[k][i] = average_precision_score(y_test[:, i], y_score[k][:, i])

        # Compute micro-average ROC curve and ROC area
        precision[k]["micro"], recall[k]["micro"], _ = precision_recall_curve(y_test.ravel(),
            y_score.ravel())
        average_precision[k]["micro"] = average_precision_score(y_test, y_score,
                                                             average="micro")

        # Plot Precision-Recall curve
        plt.clf()
        plt.plot(recall[k][0], precision[k][0], lw=lw, color='navy',
                 label='Precision-Recall curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.xlim([0.0, 1.05])
        plt.ylim([0.0, 1.05])
        plt.title('Precision-Recall example: AUC={0:0.2f}'.format(average_precision[k][0]))
        plt.legend(loc="lower left")
        plt.show()

        # Plot Precision-Recall curve for each class
        plt.clf()
        plt.plot(recall[k]["micro"], precision[k]["micro"], color='gold', lw=lw,
                 label='micro-average Precision-recall curve (area = {0:0.2f})'
                       ''.format(average_precision[k]["micro"]))
        for i, color in zip(range(nb_classes), colors):
            plt.plot(recall[k][i], precision[k][i], color=color, lw=lw,
                     label='Precision-recall curve of class {0} (area = {1:0.2f})'
                           ''.format(i, average_precision[k][i]))

        plt.xlim([0.0, 1.05])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Extension of Precision-Recall curve to multi-class')
        plt.legend(loc="lower right")
        plt.show()
