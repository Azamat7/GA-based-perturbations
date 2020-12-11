import argparse
import numpy
import numpy.matlib
import PIL
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import random

class GA:
    def __init__(self, model, image, threshold, max_iterations):
        # image: numpy array
        self.original_image = image
        self.image_shape = image.shape
        self.width = self.image_shape[1]
        self.height = self.image_shape[2]

        self.max_percent_changed = 0.1 # 10%
        self.total_pixel_number = self.width * self.height
        self.max_pixels_changed = int(self.total_pixel_number * self.max_percent_changed)

        self.change_pixel_by_list = [0.1, 0.2, 1.8, 1.9]

        self.population = []
        self.population_size = threshold
        self.max_iterations = max_iterations

        self.model = model
        self.set_true_label()

    def get_change_pixel_by(self):
        index = numpy.random.choice(len(self.change_pixel_by_list))
        return self.change_pixel_by_list[index]
    
    def set_true_label(self):
        """
        Identifies and stores the original image classification
        """
        index, prob = self.predict(self.original_image)
        self.true_label = index

    def generate_random_modified_image(self, prob):
        """
        generates new copy of an image with random pixels modified
        """
        l = len(self.change_pixel_by_list)
        sample = self.change_pixel_by_list + [1.0]
        probs = [prob/l]*l + [1-prob]
        temp = numpy.random.choice(sample, size=(1, self.width, self.height), 
                                replace=True, p=probs)
        prod = numpy.multiply(self.original_image, temp[..., numpy.newaxis])
        prod[prod>=1.0] = 1.0
        return prod

    def initialize_population(self):
        """
        generates POPULATION_SIZE number of parents.
        For each new parent, we only change at most
        fixed number of pixels, maxPixelsChanged.
        Each modified pixel would differ from the
        original image's pixel by some constant
        multiple.
        """
        initial_prob = 0.03
        for member_count in range(self.population_size):
            new_member = self.generate_random_modified_image(initial_prob)
            self.population.append(new_member)

    def normalize(self, pred):
        """
        Normalize model prediction result
        Returns normalized array
        """
        min_pred = numpy.amin(pred)
        pred = pred - min_pred if min_pred < 0 else pred
        return tf.keras.utils.normalize(pred, axis=-1, order=1)

    def predict(self, image):
        """
        Returns index with highest probabiltiy and probability
        ex: (1, 0.33)
        """
        pred = self.model.predict(image)
        normalized = self.normalize(pred)
        index = tf.math.argmax(normalized, axis=1).numpy()[0]
        return index, normalized[0][index]

    def compute_fitness(self, image):
        """
        Returns if the highest probability corresponds to the original index
        and the probability
        ex: (False, 0.15)
        """
        index, probability = self.predict(image)
        if index != self.true_label:
            return False, probability
        return True, probability
    
    def population_fitness(self):
        self.fitnesses = [self.compute_fitness(image) for image in self.population]

    def check_population(self):
        for i in range(self.population_size):
            if not self.fitnesses[i][0]:
                self.adversarial = self.population[i]
                return True
        return False

    def count_changed_pixels(self, image):
        comparison = (image != self.original_image).any(axis=3)
        return numpy.count_nonzero(comparison)
    
    def crossover(self, image1, image2):
        """
        Randomly pick pixels from 2 images
        """
        temp = numpy.random.randint(0, 2, (1, self.width, self.height))
        temp_invert = 1 - temp
        offspring = numpy.multiply(image1, temp[..., numpy.newaxis]) + numpy.multiply(image2, temp_invert[..., numpy.newaxis])
        return offspring
    
    def mutate(self, image):
        """
        randomly change certain pixels
        image: numpy array
        """
        prob=0.01
        l = len(self.change_pixel_by_list)
        sample = self.change_pixel_by_list + [1.0]
        probs = [prob/l]*l + [1-prob]
        temp = numpy.random.choice(sample, size=(1, self.width, self.height), 
                                replace=True, p=probs)
        image = numpy.multiply(image, temp[..., numpy.newaxis])
        image[image>=1.0] = 1.0
        return image

    def next_generation(self):
        """
        Assumes that self.population is sorted by fitness function
        """
        indexes = numpy.random.choice(self.population_size // 2, self.population_size // 2)
        it = iter(indexes)
        for i in it:
            parent1 = self.population[i]
            parent2 = self.population[next(it)]
            offspring = self.crossover(parent1, parent2)
            if self.count_changed_pixels(offspring) > self.max_pixels_changed:
                continue
            offspring = self.mutate(offspring)
            self.population.append(offspring)
            self.fitnesses.append(self.compute_fitness(offspring))

        generation = sorted(zip(self.population, self.fitnesses), key=lambda x: x[1][1])
        self.population = [x[0] for x in generation[:self.population_size]]
        print('Confidence:', generation[0][1][1])
    
    def get_perturbations(self):
        """
        combine all functions together
        """
        self.initialize_population()

        for i in range(self.max_iterations):
            print("Iteration {}".format(i+1))
            self.population_fitness()
            if self.check_population():
                break
            self.next_generation()
        
        if hasattr(self, 'adversarial'):
            im = PIL.Image.fromarray((self.adversarial*255.0).astype(numpy.uint8)[0])
            im.save("adversarial.png")
        else:
            print("Could not find any adversarial :(")

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

    ga = GA(model, image, 100, 20)
    ga.get_perturbations()

    # random_image = ga.generate_random_modified_image(0.03)
    # print(ga.count_changed_pixels(random_image))
    # print(ga.changed_pixels(random_image))
    # print(len(ga.get_changed_pixel_coordinates(random_image)))
    # print(len(ga.get_changed_pixel_coordinates(ga.original_image)))
    # im = PIL.Image.fromarray((random_image*255.0).astype(numpy.uint8)[0])
    # im.save("random_image.png")
    # # print(ga.get_changed_pixel_coordinates(random_image))

    # ga.initialize_population()
    # print(len(ga.get_changed_pixel_coordinates(ga.population[0])))
    # print(len(ga.get_changed_pixel_coordinates(ga.population[1])))
    # offspring = ga.crossover(ga.population[0], ga.population[1])
    # print(len(ga.get_changed_pixel_coordinates(offspring)))
    # offspringImage = PIL.Image.fromarray((offspring*255.0).astype(numpy.uint8)[0])
    # offspringImage.save("random_image_crossover.png")
    # print(ga.get_changed_pixel_coordinates(offspring))