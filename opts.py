import argparse

def parse_opts():
    parser = argparse.ArgumentParser(description="attention-based deep ensembles with uncertainty 1.0")
    parser.add_argument('--x_train', default='data/train_data.p', type=str, help='train image (num*192*192*1) path')
    parser.add_argument('--y_train', default='data/train_label.p', type=str, help='train label (gestational age: num*1) path')
    parser.add_argument('--x_validation', default='data/validation_data.p', type=str, help='validation image path')
    parser.add_argument('--y_validation', default='data/validation_label.p', type=str, help='validation label path')
    parser.add_argument('--x_test', default='data/test_data.p', type=str, help='test image path')
    parser.add_argument('--y_test', default='data/test_label.p', type=str, help='test label path')
    parser.add_argument('--batch_size', default=16, type=int, help='batch size')
    parser.add_argument('--model_name', default='RAN4BAE', type=str, help='model name')
    parser.add_argument('--epochs', default=50, type=int, help='training epoch')
    parser.add_argument('--action', type=str, required=True, choices=['train', 'predict'], help='choose train or predict')
    parser.add_argument('--verbose', action='store_true', help='')
    parser.set_defaults(verbose=False)

    args = parser.parse_args()

    return args
