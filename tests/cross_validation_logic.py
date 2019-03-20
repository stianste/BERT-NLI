training_examples = [x for x in range(100)]

fold_size = int(len(training_examples) / 10)

# for fold_number in range(10):
#     dev_set = training_examples[fold_number * fold_size : (fold_number + 1) * fold_size]
#     print(dev_set)

def get_dev_and_training_for_fold(fold):
    dev_set = training_examples[fold_number * fold_size : (fold_number + 1) * fold_size]
    train_set = training_examples[0:fold_number * fold_size] + training_examples[(fold_number + 1) * fold_size : ]
    return dev_set, train_set

fold_number = 0
dev_set, train_set = get_dev_and_training_for_fold(fold_number)

assert dev_set == [x for x in range(10)]
assert train_set == [x for x in range(10, 100)]

fold_number = 5
dev_set, train_set = get_dev_and_training_for_fold(fold_number)

expected = [x for x in range(50, 60)]
assert dev_set == expected, f'{dev_set} is not equal {expected}'
expected = [x for x in range(50)] + [x for x in range(60, 100)]
assert train_set == expected, f'{train_set} is not equal {expected}'

print("All tests passed!")
