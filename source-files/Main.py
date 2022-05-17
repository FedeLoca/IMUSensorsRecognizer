import os

import numpy
import time
import sys

from DataRetriever import DataRetriever
from Classifier import Classifier
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class Main:

    # basin forward, hands up = BF-HU
    # basin right, left hand up = BR-LHU
    # basin left, right hand up = BL-RHU
    # basin back, hands on hips = BB-HH
    # left wrist rotation clockwise = LWR_C
    # right wrist rotation counter-clockwise = RWR_CC
    # wrist rotation hands up = WR_HU
    # wrist rotation hands down = WR_HD

    num_actions_dict1 = {0: 'BF_HU', 1: 'BR_LHU',
                        2: 'BL_RHU', 3: 'BB_HH'}
    actions_num_dict1 = {'BF_HU': 0, 'BR_LHU': 1,
                        'BL_RHU': 2, 'BB_HH': 3}
    num_actions_dict2 = {0: 'IDLE', 1: 'OTHER', 2: 'BF_HU', 3: 'BR_LHU',
                        4: 'BL_RHU', 5: 'BB_HH'}
    actions_num_dict2 = {'IDLE': 0, 'OTHER': 1, 'BF_HU': 2, 'BR_LHU': 3,
                        'BL_RHU': 4, 'BB_HH': 5}

    num_actions_dict3 = {0: 'LWR_C', 1: 'RWR_CC',
                        2: 'WR_HU', 3: 'WR_HD'}
    actions_num_dict3 = {'LWR_C': 0, 'RWR_CC': 1,
                        'WR_HU': 2, 'WR_HD': 3}
    num_actions_dict4 = {0: 'IDLE', 1: 'OTHER', 2: 'LWR_C', 3: 'RWR_CC',
                        4: 'WR_HU', 5: 'WR_HD'}
    actions_num_dict4 = {'IDLE': 0, 'OTHER': 1, 'LWR_C': 2, 'RWR_CC': 3,
                        'WR_HU': 4, 'WR_HD': 5}

    model_type = 'k-nn'
    # model_type = 'random forest'
    window_dim = 500  # in milliseconds
    actions = 1

    if __name__ == "__main__":

        print(sys.argv[1:])
        if len(sys.argv) > 1:
            _, model_type, window_dim, actions = sys.argv
            window_dim = int(window_dim)
            actions = int(actions)

        num_actions_dict = num_actions_dict1
        actions_num_dict = actions_num_dict1
        if actions == 1:
            num_actions_dict = num_actions_dict1
            actions_num_dict = actions_num_dict1
        elif actions == 2:
            num_actions_dict = num_actions_dict2
            actions_num_dict = actions_num_dict2
        elif actions == 3:
            num_actions_dict = num_actions_dict3
            actions_num_dict = actions_num_dict3
        elif actions == 4:
            num_actions_dict = num_actions_dict4
            actions_num_dict = actions_num_dict4

        start_time = time.time()
        # "..\\training-data-discrete"
        # "..\\training-data-with-other-idle-discrete"
        # "..\\training-data-continued"
        # "..\\training-data-with-other-idle-continued"
        training_path = ".." + os.sep + "training-data-discrete"
        training_data = DataRetriever.retrieve_training_data(training_path)
        # test_path = ".." + os.sep + "test-data"
        # test_data = DataRetriever.retrieve_test_data(test_path)
        print("\n\n--- %s retrieve data seconds ---\n\n" % (time.time() - start_time))

        classifier = Classifier(training_data, None, num_actions_dict, actions_num_dict)
        # classifier = Classifier(training_data, test_data, num_actions_dict, actions_num_dict)

        start_time = time.time()
        if window_dim > 0:
            classifier.compute_features_on_windows(window_dim)
        else:
            classifier.compute_features()
        print("\n\n--- %s compute features seconds ---\n\n" % (time.time() - start_time))

        scores = dict()
        confusion_matrices = dict()
        params = dict()
        tries = 5
        for t in range(tries):
            i = 0
            for test_size in range(1, 10):
                params[i] = "(tries: " + str(tries) + ", model-type: " + model_type + ", train-size: "\
                            + str(10 - test_size) + ")"
                if i in scores.keys():
                    start_time = time.time()
                    new_score, new_cf = classifier.classify(model_type, test_size/10)
                    print("\n\n--- %s classify seconds ---\n\n" % (time.time() - start_time))
                    scores[i] += new_score
                    confusion_matrices[i] += numpy.array(new_cf)
                else:
                    scores[i], confusion_matrices[i] = classifier.classify(model_type, test_size/10)
                i += 1
        for i in range(0, len(scores)):
            scores[i] = scores[i]/tries
            confusion_matrices[i] = confusion_matrices[i]/tries
        best = max(scores.values())
        best_string = "Best for "
        print("\n\nScores...")
        for i in range(0, len(scores)):
            print("Score for " + params[i] + ": " + str(scores[i]))

            '''
            # plot confusion matrix
            print(confusion_matrices[i])
            ax = sns.heatmap(confusion_matrices[i], annot=True, cmap='Blues')
            ax.set_title("Confusion Matrix for " + params[i])
            ax.set_xlabel("Predicted Values")
            ax.set_ylabel("Actual Values")
            ax.xaxis.set_ticklabels(list(num_actions_dict.values()))
            ax.yaxis.set_ticklabels(list(num_actions_dict.values()))
            # plt.savefig('confusion_matrix.png')
            plt.show()
            '''

            if scores[i] == best:
                best_string += params[i] + ", "
        print(best_string + ": " + str(best))

        # plot scores
        training_sizes = [x for x in range(9, 0, -1)]
        plt.plot(training_sizes, list(scores.values())[::-1], color='b', marker='o', linestyle='-', linewidth=2, markersize=5)
        plt.ylabel('accuracy')
        plt.xlabel('training size')
        plt.title("Average accuracy plot for " + model_type + " with " + str(tries)+ " tries")
        # plt.savefig('scores.png')
        plt.show()
