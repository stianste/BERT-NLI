training_examples = [x for x in range(100)]

fold_size = int(len(training_examples) / 10)

for fold_number in range(10):
    fold = training_examples[fold_number * fold_size : (fold_number + 1) * fold_size]
    print(fold)