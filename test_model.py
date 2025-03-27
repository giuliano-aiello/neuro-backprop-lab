from tester.tester import Tester
from utils.load_model import load_model
from utils.config_loader import load_config_testing
from loader_dataset.loader_dataset import LoaderDataset

model = load_model()
criterion = load_config_testing()
loader_test_set = LoaderDataset.get_mnist_loader_dataset(False, 500, 5000)

Tester.test(model, criterion, loader_test_set)
