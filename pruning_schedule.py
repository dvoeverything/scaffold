pruning_schedule = [0.05, 0.07, 0.1, 0.12, 0.15]  # Different pruning ratios for each round

for iter_prune_round in range(len(pruning_schedule)):
    amount = pruning_schedule[iter_prune_round]  # Pick pruning ratio for this round
    print(f"\n\nIterative Global pruning round {iter_prune_round + 1} with {amount*100:.1f}% pruning")

    prune.ln_structured(best_model.conv1, name="weight", amount=amount, n=2, dim=0)
    prune.ln_structured(best_model.conv2, name="weight", amount=amount, n=2, dim=0)
    prune.ln_structured(best_model.conv3, name="weight", amount=amount, n=2, dim=0)
    prune.ln_structured(best_model.op, name="weight", amount=amount, n=2, dim=0)
