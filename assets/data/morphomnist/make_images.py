import idx2numpy
import numpy
import matplotlib.pyplot as plt

file = 'perturbed/train-images-idx3-ubyte'
arr = idx2numpy.convert_from_file(file)
# arr is now a np.ndarray type of object of shape 60000, 28, 28

for i in range(10):
    plt.imshow(arr[i], cmap=plt.cm.binary)
    plt.show()

file = 'perturbed/t10k-images-idx3-ubyte'
arr = idx2numpy.convert_from_file(file)
# arr is now a np.ndarray type of object of shape 60000, 28, 28

for i in range(10):
    plt.imshow(arr[i], cmap=plt.cm.binary)
    plt.show()
