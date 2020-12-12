import numpy
import numpy.matlib
import PIL
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import random

from parser import Parser

class GA:
    def __init__(self, model, image, population_size, max_iterations, threshold):
        # image: numpy array
        self.original_image = image
        self.width = image.shape[1]
        self.height = image.shape[2]

        self.max_percent_changed = threshold
        self.total_pixel_number = self.width * self.height
        self.max_pixels_changed = int(self.total_pixel_number * self.max_percent_changed)

        self.change_pixel_by_list = [0.1, 0.2]
        self.increment_pixel_by_list = [0.3, 0.4]

        self.population = []
        self.population_size = population_size
        self.max_iterations = max_iterations

        self.model = model
        self.set_true_label()
    
    def set_true_label(self):
        """
        Identifies and stores the true label of image
        """
        index, prob = self.predict(self.original_image)
        self.true_label = index

    def generate_random_modified_image(self, prob):
        """
        Generates a new copy of an image with random pixels modified
        """
        l1 = len(self.change_pixel_by_list)
        sample1 = self.change_pixel_by_list + [1.0]
        probs1 = [prob/l1]*l1 + [1-prob]

        l2 = len(self.increment_pixel_by_list)
        sample2 = self.increment_pixel_by_list + [0.0]
        probs2 = [prob/l2]*l2 + [1-prob]

        prod = numpy.random.choice(sample1, size=(1, self.width, self.height), 
                                replace=True, p=probs1)
        inc = numpy.random.choice(sample2, size=(1, self.width, self.height), 
                                replace=True, p=probs2)

        ans = numpy.multiply(self.original_image, prod[..., numpy.newaxis])
        ans = numpy.add(ans, inc[..., numpy.newaxis])
        ans[ans>=1.0] = 1.0
        ans[ans<=0.0] = 0.0
        return ans

    def initialize_population(self):
        """
        generates POPULATION_SIZE number of parents.
        For each new parent, we only change at most
        fixed number of pixels, maxPixelsChanged.
        Each modified pixel would differ from the
        original image's pixel by some constant
        multiple.
        """
        initial_prob = 0.05
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
        Randomly pick pixels from 2 images (uniform)
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
        prob=0.03
        l1 = len(self.change_pixel_by_list)
        sample1 = self.change_pixel_by_list + [1.0]
        probs1 = [prob/l1]*l1 + [1-prob]

        l2 = len(self.increment_pixel_by_list)
        sample2 = self.increment_pixel_by_list + [0.0]
        probs2 = [prob/l2]*l2 + [1-prob]

        prod = numpy.random.choice(sample1, size=(1, self.width, self.height), 
                                replace=True, p=probs1)
        inc = numpy.random.choice(sample2, size=(1, self.width, self.height), 
                                replace=True, p=probs2)

        ans = numpy.multiply(image, prod[..., numpy.newaxis])
        ans = numpy.add(ans, inc[..., numpy.newaxis])
        ans[ans>=1.0] = 1.0
        ans[ans<=0.0] = 0.0
        return ans

    def next_generation(self):
        """
        Assumes that self.population is sorted by fitness function
        """
        if (self.population_size // 2)%2 == 1:
            indexes = numpy.random.choice(self.population_size // 2, self.population_size // 2 + 1)
        else:
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
    
    def get_perturbation(self):
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
            im = PIL.Image.fromarray((self.adversarial*255.0).astype(numpy.uint8).reshape((28,28)))
            im.save("adversarial.png")
        else:
            print("Could not find any adversarial :(")

if __name__ == "__main__":
    parser = Parser()
    args = parser.parse_args()

    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

    # Open image as numpy array
    image = numpy.asarray(PIL.Image.open(args.image_name))
    image = image/255.0
    image = image.reshape(-1, 28, 28, 1)

    model = tf.keras.models.load_model(args.model_name)
    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

    ga = GA(model, image, args.population_size, args.iterations, args.threshold)
    ga.get_perturbation()