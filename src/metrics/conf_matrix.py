import numpy as np
import matplotlib.pyplot as plt

def confusion_matrix(y_pred, y, title):
    classes = np.unique(y)
    conf_mat = np.zeros((len(classes), len(classes)), dtype=np.uint32)
    y_pred -= np.min(classes)
    y -= np.min(classes)
    for true_val, pred_val in zip(y, y_pred):
        conf_mat[true_val][pred_val]+=1
    
    fig, ax = plt.subplots()
    im = ax.imshow(conf_mat)

    ax.set_yticks(classes, labels = classes)
    ax.set_xticks(classes, labels = classes)

    for i in range(classes.shape[0]):
        for j in range(classes.shape[0]):
            text = ax.text(j, i, conf_mat[i, j], ha='center', va='center', color='r')

    title_text = "{}_Confusion Matrix".format(title)
    ax.set_title(title_text)
    fig.tight_layout()
    plt.savefig(title_text)
    plt.show()
    return conf_mat
            