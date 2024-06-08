import matplotlib.pyplot as plt
from PIL import Image
from cnn.cifar10_classifier import CIFAR10Classifier

def test(classifier, image_path):

    image = Image.open(image_path)
    plt.imshow(image)
    predicted_class = classifier.predict_image(image_path)
    plt.title(f'Predicted class: {predicted_class}')
    plt.show()

image_path = '/content/frog-2_ver_1.jpg'
classifier = CIFAR10Classifier()
test(classifier, image_path)