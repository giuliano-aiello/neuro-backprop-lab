from tester.tester import Tester
from utils.load_model import load_model
from utils.config_loader import load_config_testing

model, optimizer = load_model()
criterion, loader_test_set = load_config_testing(optimizer)

Tester.test(model, criterion, loader_test_set)
