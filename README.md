# keras_unet_segmentation

This repo contains an implementation for solving a binary image segmentation problem. The repo contains training and predict code. The repo also contains code for data augmentation. Dataset must have images/labels split into two different directories.

For preproccesing and data augmentation two pixel values must be supplied pn labelled images:

- 0 for background
- 255 for class to segment. (Each value greater than 200 will be used as well for the class to segment)

You must run
python preprocess.py --source_dir dir_where_you_have_the_original_images --label_dir dir_where_you_have_the_original_labelled_images --output_dir dir_to_write_augmented_dataset --samples number_of_total_samples_to_generate

For training you must run
python train_keras_unet.py --images_dir images_dir --labels_dir labels_dir --epochs number_of_epochs_for_training --model_file file_model_name

After training you get a model file in H5 keras format (file_model_name.h5). You can use to get predictions like that:
python predict_keras.py --model file_model_name.h5 --image image_to_predict

The script obtain a segmented image "pred.png" where each pixel is labelled: black for background and white for segmented class.
