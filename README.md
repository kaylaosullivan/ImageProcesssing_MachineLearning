# ImageProcesssing_MachineLearning

Preprocessing.ipynb
=================================

Necessary packages:
- numpy
- matplotlib.pyplot
- cv2

NOTE: the steps listed below are all completed by executing each cell in the jupyter notebook

Load train and test images.

NOTE: This is the only file that requires the original train/test images. 
      They must be in the same directory as the Preprocessing.ipnyb file.


Process each image by calling:\
filter_image(image)
- returns filtered image as 1D array
- image input can be a 1D or 2D array of pixels
- use option plot=True to plot preprocessing stages


Once images are preprocessed, save image files as .npy files by executing:

np.save(open("ResizedTrain.npy","wb"), resized_train_imgs)



LinearSVM.ipynb
========================

Necessary packages:
- numpy
- matplotlib.pyplot
- csv
- sklearn

NOTE: the steps listed below are all completed by executing each cell in the jupyter notebook

Load resized testing and training images, as well as training classes.

To test cros-validation scores of various hyperparameter C-values, call:
linearSVM(Cs, features, classes)
      - returns a 2D array of elements [C-value, Validation Score]
      - Cs = an array of C values
      - features = training images
      - classes = training classes


Plot the scores obtained from the cross test validation (on logscale):
plt.semilogx(np.array(scores)[:,0],np.array(scores)[:,1])

Train for the correct hyperparameter (here, 6e-13 is the best C-value):
- clf = LinearSVC(C=6e-13) 
- clf.fit(train_images, train_classes) 
- SVM_prediction = clf.predict(test_images) 

Save the predictions in the format required by kaggle: 
  create_output(SVM_prediction,"finalSVM") 
NOTE: This assumes a folder called "Predictions" exists.

