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
    # model_type = 'rf'
    # model_type = 'k-nn-dtw'
    # model_type = 'svm'
    # model_type = 'lstm'
    # model_type = 'nb'
    # model_type = 'hmm'
    # model_type = 'lr'
    # model_type = 'dt'
    window_dim = 0  # in milliseconds
    actions = 0
    save = False
    tuning = False
    tries = 10
    mobile_average_window_dim = 4
    max_test_size = 24
    overlap = 0

    if __name__ == "__main__":

        print(sys.argv[1:])
        if len(sys.argv) > 1:
            _, model_type, window_dim, overlap, actions, tries, mobile_average_window_dim, max_test_size, save, tuning = sys.argv
            window_dim = int(window_dim) * 1000000
            actions = int(actions)
            tries = int(tries)
            mobile_average_window_dim = int(mobile_average_window_dim)
            max_test_size = int(max_test_size)
            overlap = float(overlap)

        if save == 't':
            save = True
        else:
            save = False
        if tuning == 't':
            tuning = True
        else:
            tuning = False

        num_actions_dict = num_actions_dicts[actions]
        actions_num_dict = actions_num_dicts[actions]
        training_path = training_paths[actions]

        start_time = time.time()
        training_path_name = training_path
        training_path = ".." + os.sep + training_path
        # training_data = DataRetriever.retrieve_training_data(training_path)
        training_data = DataRetriever.retrieve_train_session_data(training_path.replace("-O", "") + "-sessions",
                                                                  mobile_average_window_dim)
        # test_path = ".." + os.sep + test_path
        # test_data = DataRetriever.retrieve_test_data(test_path)
        print("\n\n--- %s retrieve data seconds ---\n\n" % (time.time() - start_time))

        classifier = Classifier(training_data, None, num_actions_dict, actions_num_dict, mobile_average_window_dim, tuning)
        # classifier = Classifier(training_data, test_data, num_actions_dict, actions_num_dict)

        start_time = time.time()
        if model_type == 'lstm':
            classifier.compute_lstm_data(window_dim, overlap)
        else:
            if window_dim > 0:
                classifier.compute_features_on_windows(window_dim, overlap)
            else:
                classifier.compute_features()
        print("\n\n--- %s compute features seconds ---\n\n" % (time.time() - start_time))

        scores = dict()
        confusion_matrices = dict()
        train_times = dict()
        predict_times = dict()
        params = dict()
        # test_sizes = [round(x, 2) for x in numpy.arange(0.05, 0.95, 0.05)]
        # test_sizes = range(1, max_test_size + 1)
        test_sizes = [x for x in range(5, 21, 5)]
        for t in range(tries):
            i = 0
            new_try = True
            for test_size in test_sizes:
                params[i] = "(tries: " + str(tries) + ", model: " + model_type + \
                            ", mavg: " + str(mobile_average_window_dim) + \
                            ", train size: " + str(max_test_size + 1 - test_size) + "/" + str(max_test_size + 1) + \
                            ", win dim: " + str(window_dim/1000000) + "s, overlap: " + str(overlap)
                if i in scores.keys():
                    new_score, new_cf, new_train_t, new_predict_t = \
                        classifier.classify(model_type, test_size, max_test_size, new_try)
                    train_times[i] += new_train_t
                    predict_times[i] += new_predict_t
                    scores[i] += new_score
                    confusion_matrices[i] += numpy.array(new_cf)
                else:
                    scores[i], confusion_matrices[i], train_times[i], predict_times[i] = \
                        classifier.classify(model_type, test_size, max_test_size, new_try)
                i += 1
                new_try = False
            # for test_size in test_sizes:
            #     params[i] = "(tries: " + str(tries) + ", model: " + model_type + \
            #                 ", mavg: " + str(mobile_average_window_dim) + \
            #                 ", train size: " + str(test_size) + ", wdim: " + str(window_dim/1000000) + \
            #                 "ms, overlap: " + str(overlap) + ")"
            #     if i in scores.keys():
            #         new_score, new_cf, new_train_t, new_predict_t = \
            #             classifier.classify(model_type, test_size)
            #         train_times[i] += new_train_t
            #         predict_times[i] += new_predict_t
            #         scores[i] += new_score
            #         confusion_matrices[i] += numpy.array(new_cf)
            #     else:
            #         scores[i], confusion_matrices[i], train_times[i], predict_times[i] = \
            #             classifier.classify(model_type, test_size)
            #     i += 1
        for i in range(0, len(scores)):
            scores[i] = scores[i] / tries
            # confusion_matrices[i] = confusion_matrices[i] / tries
            train_times[i] = train_times[i] / tries
            predict_times[i] = predict_times[i] / tries
        best = max(scores.values())
        best_string = "Best for "
        print("\n\nScores...")
        images_folder = training_path_name + "-tries_" + str(tries) + "-model_" + model_type + \
                        "-mavg_" + str(mobile_average_window_dim) + "-wdim_" + str(window_dim/1000000) + \
                        "ms-overlap_" + str(overlap)
        for i in range(0, len(scores)):
            print("Score for " + params[i] + ": " + str(scores[i]))

            # plot confusion matrix
            print(confusion_matrices[i])
            ax = sns.heatmap(confusion_matrices[i], annot=True, cmap='Blues')
            ax.set_title("CM for " + params[i])
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
        fig, ax = plt.subplots()
        plt.plot(test_sizes, list(scores.values())[::-1], color='b', marker='o', linestyle='-',
                 linewidth=2, markersize=5)

        rounded_scores = [round(x, 4) for x in list(scores.values())]
        print(rounded_scores)
        for i in range(scores.values().__len__()):
            ax.annotate(rounded_scores[::-1][i],
                        xy=(test_sizes[i], list(scores.values())[::-1][i]))
        plt.ylabel('accuracy')
        plt.xlabel('training size')
        plt.title("Average accuracy plot for " + model_type + " with " + str(tries) + " tries")
        if save:
            if not os.path.exists(".." + os.sep + "Images" + os.sep + images_folder):
                os.makedirs(".." + os.sep + "Images" + os.sep + images_folder)
            plt.savefig(".." + os.sep + "Images" + os.sep + images_folder + os.sep + "scores.png")
        plt.show()

        sum = 0
        c = 0
        for n in train_times.values():
            sum = sum + n
            c = c + 1
        avg = sum / c
        print("Average train time per sample: " + str(avg) + " s")

        sum = 0
        c = 0
        for n in predict_times.values():
            sum = sum + n
            c = c + 1
        avg = sum / c
        print("Average predict time per sample: " + str(avg) + " s")
