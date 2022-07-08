import os

import numpy
import time
import sys

from DataRetriever import DataRetriever
from Classifier import Classifier
import matplotlib.pyplot as plt
import seaborn as sns


class Main:
    test_path = "test-data"

    # left = L
    # right = R
    # up = U
    # down = D
    # forward = F
    # back = B
    # bit up = BU
    # bit left = BL
    # very up = VU
    # very left = VL
    # turn = T

    num_actions_dict1 = {0: 'left', 1: 'right',
                         2: 'forward', 3: 'turn'}
    actions_num_dict1 = {'left': 0, 'right': 1,
                         'forward': 2, 'turn': 3}
    training_path1 = "P1-data"

    num_actions_dict2 = {0: 'left', 1: 'right',
                         2: 'forward', 3: 'turn', 4: 'other'}
    actions_num_dict2 = {'left': 0, 'right': 1,
                         'forward': 2, 'turn': 3, 'other': 4}
    training_path2 = "P1-data-O"

    num_actions_dict3 = {0: 'bitleft', 1: 'veryleft',
                         2: 'bitup', 3: 'veryup'}
    actions_num_dict3 = {'bitleft': 0, 'veryleft': 1,
                         'bitup': 2, 'veryup': 3}
    training_path3 = "P2-data"

    num_actions_dict4 = {0: 'bitleft', 1: 'veryleft',
                         2: 'bitup', 3: 'veryup', 4: 'other'}
    actions_num_dict4 = {'bitleft': 0, 'veryleft': 1,
                         'bitup': 2, 'veryup': 3, 'other': 4}
    training_path4 = "P2-data-O"

    num_actions_dict5 = {0: 'left', 1: 'right',
                         2: 'up', 3: 'forward'}
    actions_num_dict5 = {'left': 0, 'right': 1,
                         'up': 2, 'forward': 3}
    training_path5 = "P3-data"

    num_actions_dict6 = {0: 'left', 1: 'right',
                         2: 'up', 3: 'forward', 4: 'other'}
    actions_num_dict6 = {'left': 0, 'right': 1,
                         'up': 2, 'forward': 3, 'other': 4}
    training_path6 = "P3-data-O"

    num_actions_dict7 = {0: 'left', 1: 'right',
                         2: 'up', 3: 'down'}
    actions_num_dict7 = {'left': 0, 'right': 1,
                         'up': 2, 'down': 3}
    training_path7 = "P4-data"

    num_actions_dict8 = {0: 'left', 1: 'right',
                         2: 'up', 3: 'down', 4: 'other'}
    actions_num_dict8 = {'left': 0, 'right': 1,
                         'up': 2, 'down': 3, 'other': 4}
    training_path8 = "P4-data-O"

    num_actions_dict9 = {0: 'left', 1: 'right',
                         2: 'up', 3: 'forward'}
    actions_num_dict9 = {'left': 0, 'right': 1,
                         'up': 2, 'forward': 3}
    training_path9 = "P5-data"

    num_actions_dict10 = {0: 'left', 1: 'right',
                         2: 'up', 3: 'forward', 4: 'other'}
    actions_num_dict10 = {'left': 0, 'right': 1,
                         'up': 2, 'forward': 3, 'other': 4}
    training_path10 = "P5-data-O"

    num_actions_dict11 = {0: 'down', 1: 'right',
                         2: 'up', 3: 'forward'}
    actions_num_dict11 = {'down': 0, 'right': 1,
                         'up': 2, 'forward': 3}
    training_path11 = "P6-data"

    num_actions_dict12 = {0: 'down', 1: 'right',
                         2: 'up', 3: 'forward', 4: 'other'}
    actions_num_dict12 = {'down': 0, 'right': 1,
                         'up': 2, 'forward': 3, 'other': 4}
    training_path12 = "P6-data-O"

    num_actions_dict13 = {0: 'back', 1: 'right',
                          2: 'up', 3: 'forward'}
    actions_num_dict13 = {'back': 0, 'right': 1,
                          'up': 2, 'forward': 3}
    training_path13 = "P7-data"

    num_actions_dict14 = {0: 'back', 1: 'right',
                          2: 'up', 3: 'forward', 4: 'other'}
    actions_num_dict14 = {'back': 0, 'right': 1,
                          'up': 2, 'forward': 3, 'other': 4}
    training_path14 = "P7-data-O"

    num_actions_dict15 = {0: 'left', 1: 'right',
                          2: 'up', 3: 'forward'}
    actions_num_dict15 = {'left': 0, 'right': 1,
                          'up': 2, 'forward': 3}
    training_path15 = "P8-data"

    num_actions_dict16 = {0: 'left', 1: 'right',
                          2: 'up', 3: 'forward', 4: 'other'}
    actions_num_dict16 = {'left': 0, 'right': 1,
                          'up': 2, 'forward': 3, 'other': 4}
    training_path16 = "P8-data-O"

    num_actions_dict17 = {0: 'left', 1: 'right',
                          2: 'up', 3: 'back'}
    actions_num_dict17 = {'left': 0, 'right': 1,
                          'up': 2, 'back': 3}
    training_path17 = "P9-data"

    num_actions_dict18 = {0: 'left', 1: 'right',
                          2: 'up', 3: 'back', 4: 'other'}
    actions_num_dict18 = {'left': 0, 'right': 1,
                          'up': 2, 'back': 3, 'other': 4}
    training_path18 = "P9-data-O"

    num_actions_dicts = [num_actions_dict1, num_actions_dict2, num_actions_dict3, num_actions_dict4,
                         num_actions_dict5, num_actions_dict6, num_actions_dict7, num_actions_dict8,
                         num_actions_dict9, num_actions_dict10, num_actions_dict11, num_actions_dict12,
                         num_actions_dict13, num_actions_dict14, num_actions_dict15, num_actions_dict16,
                         num_actions_dict17, num_actions_dict18]
    actions_num_dicts = [actions_num_dict1, actions_num_dict2, actions_num_dict3, actions_num_dict4,
                         actions_num_dict5, actions_num_dict6, actions_num_dict7, actions_num_dict8,
                         actions_num_dict9, actions_num_dict10, actions_num_dict11, actions_num_dict12,
                         actions_num_dict13, actions_num_dict14, actions_num_dict15, actions_num_dict16,
                         actions_num_dict17, actions_num_dict18]
    training_paths = [training_path1, training_path2, training_path3, training_path4, training_path5,
                      training_path6, training_path7, training_path8, training_path9, training_path10,
                      training_path11, training_path12, training_path13, training_path14, training_path15,
                      training_path16, training_path17, training_path18]

    model_type = 'k-nn'
    # model_type = 'random-forest'
    window_dim = 0  # in milliseconds
    actions = 1
    save = True
    tries = 10
    mobile_average_window_dim = 4
    max_test_size = 24

    if __name__ == "__main__":

        print(sys.argv[1:])
        if len(sys.argv) > 1:
            _, model_type, window_dim, actions, tries, mobile_average_window_dim, max_test_size, save = sys.argv
            window_dim = int(window_dim)
            actions = int(actions)
            tries = int(tries)
            mobile_average_window_dim = int(mobile_average_window_dim)
            max_test_size = int(max_test_size)

        if save == 't':
            save = True

        num_actions_dict = num_actions_dicts[actions]
        actions_num_dict = actions_num_dicts[actions]
        training_path = training_paths[actions]

        start_time = time.time()
        training_path_name = training_path
        training_path = ".." + os.sep + training_path
        training_data = DataRetriever.retrieve_training_data(training_path)
        # test_path = ".." + os.sep + test_path
        # test_data = DataRetriever.retrieve_test_data(test_path)
        print("\n\n--- %s retrieve data seconds ---\n\n" % (time.time() - start_time))

        classifier = Classifier(training_data, None, num_actions_dict, actions_num_dict, mobile_average_window_dim)
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
        for t in range(tries):
            i = 0
            for test_size in range(1, max_test_size + 1):
                params[i] = "(tries: " + str(tries) + ", model: " + model_type + \
                            ", mavg: " + str(mobile_average_window_dim) + \
                            ", train size: " + str(max_test_size + 1 - test_size) + "/" + str(max_test_size + 1) + ")"
                if i in scores.keys():
                    start_time = time.time()
                    new_score, new_cf = classifier.classify(model_type, test_size / (max_test_size + 1))
                    print("\n\n--- %s classify seconds ---\n\n" % (time.time() - start_time))
                    scores[i] += new_score
                    confusion_matrices[i] += numpy.array(new_cf)
                else:
                    scores[i], confusion_matrices[i] = classifier.classify(model_type, test_size / (max_test_size + 1))
                i += 1
        for i in range(0, len(scores)):
            scores[i] = scores[i] / tries
            confusion_matrices[i] = confusion_matrices[i] / tries
        best = max(scores.values())
        best_string = "Best for "
        print("\n\nScores...")
        images_folder = training_path_name + "-tries_" + str(tries) + "-model_" + model_type + \
                        "-mavg_" + str(mobile_average_window_dim)
        for i in range(0, len(scores)):
            print("Score for " + params[i] + ": " + str(scores[i]))

            # plot confusion matrix
            print(confusion_matrices[i])
            ax = sns.heatmap(confusion_matrices[i], annot=True, cmap='Blues')
            ax.set_title("Confusion Matrix for " + params[i])
            ax.set_xlabel("Predicted Values")
            ax.set_ylabel("Actual Values")
            ax.xaxis.set_ticklabels(list(num_actions_dict.values()))
            ax.yaxis.set_ticklabels(list(num_actions_dict.values()))
            if save:
                if not os.path.exists(".." + os.sep + "Images" + os.sep + images_folder):
                    os.makedirs(".." + os.sep + "Images" + os.sep + images_folder)
                plt.savefig(".." + os.sep + "Images" + os.sep + images_folder + os.sep +
                            "confusion-matrix" + str(i) + ".png")
            plt.show()

            if scores[i] == best:
                best_string += params[i] + ", "
        print(best_string + ": " + str(best))

        # plot scores
        training_sizes = [x for x in range(1, max_test_size + 1)]
        fig, ax = plt.subplots()
        plt.plot(training_sizes, list(scores.values())[::-1], color='b', marker='o', linestyle='-',
                 linewidth=2, markersize=5)

        rounded_scores = [round(x, 4) for x in list(scores.values())]
        print(rounded_scores)
        for i in range(scores.values().__len__()):
            ax.annotate(rounded_scores[::-1][i],
                        xy=(training_sizes[i], list(scores.values())[::-1][i]))
        plt.ylabel('accuracy')
        plt.xlabel('training size')
        plt.title("Average accuracy plot for " + model_type + " with " + str(tries) + " tries")
        if save:
            if not os.path.exists(".." + os.sep + "Images" + os.sep + images_folder):
                os.makedirs(".." + os.sep + "Images" + os.sep + images_folder)
            plt.savefig(".." + os.sep + "Images" + os.sep + images_folder + os.sep + "scores.png")
        plt.show()
