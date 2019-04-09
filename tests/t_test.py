from scipy import stats

bert_base_accuracies = [
    0.636,
    0.642,
    0.638,
    0.649,
    0.648
]

bert_large_accuracies = [
    0.643,
    0.647,
    0.649,
    0.662,
    0.650
]

ind_res = stats.ttest_ind(bert_base_accuracies, bert_large_accuracies) # 0.102
print(f'Independent results {ind_res}')

ind_rel = stats.ttest_rel(bert_base_accuracies, bert_large_accuracies) # 0.102
print(f'Paired results {ind_rel}') # 0.019

# If we observe a large p-value, for example larger than 0.05 or 0.1,
# then we cannot reject the null hypothesis of identical average scores.
# If the p-value is smaller than the threshold, e.g. 1%, 5% or 10%,
# then we reject the null hypothesis of equal averages.

