from argparse import ArgumentParser

class Parser:
    def __init__(self):
        self.parser = ArgumentParser(description='GA-based adversarial perturbation generator')
        self.parser.add_argument('model_name', type=str,
            help='Required model name argument')
        self.parser.add_argument('image_name', type=str,
            help='Required image name argument')
        self.parser.add_argument('threshold', type=float, default=0.1,
            help='Required threshold for percentage of pixels that can be changed')
        self.parser.add_argument('-p', '--population_size', type=int, default=25,
            help="Optional population size argument")
    
    def parse_args(self):
        return self.parser.parse_args()