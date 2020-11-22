import argparse
import numpy
import PIL
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import random

image = None
image_size = None
max_percent_changed = 0.1 # 10%
max_pixels_changed = image_size * max_percent_changed
change_pixel_by = 1.1

class GA:
    def __init__(self, model, image, threshold):
        self.population = []
        self.population_size = 100
        pass
    
    def get_true_label(self):
        """TODO: zhanto and aza
        identify and store the original image classification
        """
        pass

    def generate_random_modified_image(self):
        """ TODO: mukha
        generates new copy of an image with random pixels modified
        """
        new_image = numpy.copy(image)
        for change_count in range(max_pixels_changed):
            x_coordinate = random.randint(0, 31) # TODO: change 31 to some other variable. 31 is hard coded.
            y_coordinate = random.randint(0, 31) # TODO: change 31 to some other variable. 31 is hard coded.
            red = min(new_image[x_coordinate][y_coordinate][0] * change_pixel_by, 1.0)
            green = min(new_image[x_coordinate][y_coordinate][1] * change_pixel_by, 1.0)
            blue = min(new_image[x_coordinate][y_coordinate][2] * change_pixel_by, 1.0)
            new_pixel = [red, green, blue]
            new_image[x_coordinate][y_coordinate][:] = new_pixel
        return new_image

    def initialize_population(self):
        """TODO: mukha
        generates POPULATION_SIZE number of parents.
        For each new parent, we only change at most
        fixed number of pixels, maxPixelsChanged.
        Each modified pixel would differ from the
        original image's pixel by some constant
        multiple.
        """
        for member_count in range(self.population_size):
            self.population.append(generate_random_modified_image)
        pass

    def compute_fitness(self):
        """TODO: zhanto and aza
        fitness function
        """
        pass
    
    def crossover(self, image1, image2):
        """TODO: mukha
        generates 1 offspring from 2 parents
        uniformal selection of pixels
        image: numpy array

        We first create copy of unmodified
        original image.
        We select at most maxPixelsChanged
        number of modified pixels from either
        parent and add to the offspring.
        Afterwards, apply mutation and add
        the offspring to the current population.
        """
        offspring = numpy.copy(image)
        self.mutate(offspring)
        return offspring
    
    def mutate(self, image):
        """TODO: zhans and mukha
        randomly change certain pixels
        image: numpy array
        """
        pass

    def next_generation(self):
        """TODO: zhans and mukha
        design selection and crossover, then implement
        """
        pass
    
    def get_perturbations(self):
        """TODO: zhanto and aza
        combine all functions together
        """
        pass
    
class Model:
    def __init__(self, model, perturbations):
        """TODO: zhanto and aza
        1) only perturbations are used for retraining
        2) train data + perturbations are used for retraining
        """
        pass
    
    def retrain_model(self):
        """TODO: zhanto and aza
        retrain the model with new images
        """
        pass
    
    def evaluate_model(self):
        """TODO: zhanto and aza
        Evaluated on whole test data
        """
        pass

if __name__ == "__main__":
    """TODO: zhanto and aza
    uncomment and implement the flow
    """
    # parser = argparse.ArgumentParser(description='GA-based adversarial perturbation generator')
    # parser.add_argument('model_name', type=str,
    #                 help='Required model name argument')
    # parser.add_argument('image_name', type=str,
    #                 help='Required image name argument')
    # parser.add_argument('threshold', type=float, default=0.1,
    #                 help='Required threshold for percentage of pixels\
    #                     that can be changed')
    # parser.add_argument('-p', '--population_size', type=int, default=25,
    #                 help="Optional population size argument")

    # args = parser.parse_args()

    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

    # Open image as numpy array
    image = numpy.asarray(PIL.Image.open('automobile10.png'))
    image = image/255.0
    image = image.reshape(-1, 32, 32, 3)

    model = tf.keras.models.load_model('cifar10')
    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
    
    # Get prediction
    pred = model.predict(image)
    print(pred)

    # Get predicted class name
    index = tf.math.argmax(pred, axis=1).numpy()[0]
    print("Predicted class is {}".format(class_names[index]))