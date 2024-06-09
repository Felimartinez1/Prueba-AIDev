import matplotlib.pyplot as plt
from PIL import Image
from cifar10_image_classifier.cifar10_classifier import CIFAR10Classifier
import yaml

with open ('cifar10_config.yml', 'r') as ymlfile:
    config = yaml.safe_load(ymlfile)
    test_config = config['test']


def test(classifier, image_path):

    image = Image.open(image_path)
    plt.imshow(image)
    predicted_class = classifier.predict_image(image_path)
    plt.title(f'Predicted class: {predicted_class}')
    plt.show()

image_path = test_config['image_path']
classifier = CIFAR10Classifier()
test(classifier, image_path)