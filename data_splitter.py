import random
import os

default_path = "TrainingSet/"
output_path = "out/"
train_split = 0.6
videos = {}
truth = {}

for line in open('truth_by_video.txt', 'r').read().strip().splitlines():
    plate, file = line.split(' - ')
    if file not in truth:
        truth[file] = []
    truth[file].append(plate)

for directory in os.listdir(default_path):
    if "Categorie" not in directory:
        continue
    videos[directory] = []
    for video in os.listdir(default_path + directory):
        videos[directory].append(video)

for category in videos.keys():
    if "Categorie" not in category:
        continue
    training_file = open(output_path + "train_" + category.replace(" ", "_").lower() + ".txt", "w")
    training_truth = open(output_path + "train_" + category.replace(" ", "_").lower() + "_plates.txt", "w")
    testing_file = open(output_path + "test_" + category.replace(" ", "_").lower() + ".txt", "w")
    testing_truth = open(output_path + "test_" + category.replace(" ", "_").lower() + "_plates.txt", "w")
    for video in videos.get(category):
        if random.randint(0, 1) < train_split:
            training_file.write(default_path + category + '/' + video + "\n")
            for plate in truth[default_path + category + '/' + video]:
                training_truth.write(plate + '\n')
        else:
            testing_file.write(default_path + category + '/' + video + "\n")
            for plate in truth[default_path + category + '/' + video]:
                testing_truth.write(plate + '\n')
    training_file.close()
    testing_file.close()
    training_truth.close()
    testing_truth.close()
