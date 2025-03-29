from tester.tester import Tester
from utils.load_model import load_model
from utils.config_loader import load_config_testing
from loader_dataset.loader_dataset import LoaderDataset


model = load_model()
criterion, loader_test_set = load_config_testing()

Tester.test(model, criterion, loader_test_set)
