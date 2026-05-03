#!/bin/bash
VALIDATION_DIRECTORY="../validation/audio";
TRAINING_DIRECTORY="../training/audio";
TESTING_DIRECTORY="../testing/audio";

if [ ! -d "$VALIDATION_DIRECTORY" ]; then
    mkdir "$VALIDATION_DIRECTORY"
    echo "Validation Directory created."
fi

if [ ! -d "$TRAINING_DIRECTORY" ]; then
    mkdir "$TRAINING_DIRECTORY"
    echo "Training Directory created."
fi

if [ ! -d "$TESTING_DIRECTORY" ]; then
    mkdir "$TESTING_DIRECTORY"
    echo "Testing Directory created."
fi

for file in ../validation/video/*.mp4; do
    filename=$(basename "$file" .mp4)
    ffmpeg -n -i "$file" -vn -ac 1 -ar 16000 -acodec pcm_s16le "../validation/audio/${filename}.mp4.wav"
done

for file in ../training/video/*.mp4; do
    filename=$(basename "$file" .mp4)
    ffmpeg -n -i "$file" -vn -ac 1 -ar 16000 -acodec pcm_s16le "../training/audio/${filename}.mp4.wav"
done

for file in ../testing/video/*.mp4; do
    filename=$(basename "$file" .mp4)
    ffmpeg -n -i "$file" -vn -ac 1 -ar 16000 -acodec pcm_s16le "../testing/audio/${filename}.mp4.wav"
done