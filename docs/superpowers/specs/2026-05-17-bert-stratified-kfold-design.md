# BERT Stratified 5-Fold Cross-Validation Design

## Goal

Extend `src/bert_classification_final.ipynb` with a rigorous 5-fold cross-validation workflow for the BBC news dataset. The workflow must preserve label ratios in each fold, fine-tune BERT independently per fold, report classification metrics, analyze errors with confusion matrices, and save the best checkpoint for each fold.

## Chosen Approach

Use a notebook-based implementation with clear helper functions. This keeps the project easy to present while preventing the cross-validation section from becoming one long, hard-to-read training block.

The notebook will keep the existing single train/validation/test pipeline intact unless the user later asks to replace it. The new cross-validation section will be added after the existing evaluation/checkpointing flow as an additional robustness check.

## Cross-Validation Flow

Use `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` over the full cleaned dataset and `label_id` column.

For each fold:

1. Split indices into train and validation subsets while preserving class proportions.
2. Build fold-specific `BBCDataset` and `DataLoader` objects.
3. Initialize a fresh `BertForSequenceClassification` from `bert-base-uncased`.
4. Attach `label2id` and `id2label` to the model config before training.
5. Fine-tune for the configured number of epochs.
6. Evaluate after every epoch on the validation fold.
7. Save the best checkpoint for that fold based on validation macro F1.

Each fold trains from the same base BERT checkpoint. Fold N must not continue training from Fold N-1.

## Metrics

For every fold and epoch, collect:

- `val_loss`
- `accuracy`
- `precision_macro`
- `recall_macro`
- `f1_macro`
- `precision_weighted`
- `recall_weighted`
- `f1_weighted`

After all folds finish, create a `cv_results_df` with one row per fold using the best epoch for that fold. Then compute mean and standard deviation for Accuracy, Precision, Recall, and F1.

Macro F1 is the primary selection metric because it treats all BBC categories evenly despite mild class imbalance.

## Error Analysis

Store validation predictions and true labels from the best epoch of each fold.

Generate:

- A confusion matrix per fold.
- One aggregated out-of-fold confusion matrix using predictions from all five validation folds.
- A classification report per fold.
- An aggregated classification report across all out-of-fold predictions.

The aggregated confusion matrix is the main error-analysis artifact because each article appears exactly once as validation data.

## Checkpointing

Save fold checkpoints under:

```text
../models/bert_bbc_cv/fold_1_best
../models/bert_bbc_cv/fold_2_best
../models/bert_bbc_cv/fold_3_best
../models/bert_bbc_cv/fold_4_best
../models/bert_bbc_cv/fold_5_best
```

Each checkpoint includes:

- model weights
- tokenizer files
- HuggingFace config with real BBC labels
- `label_mapping.json`
- `fold_metrics.json`

After all folds, save:

```text
../models/bert_bbc_cv/cv_summary.json
```

The summary records per-fold best metrics, mean/std metrics, and the best overall fold by macro F1.

## Notebook Organization

Add these sessions after the current BERT evaluation section:

1. `26. Configure Stratified K-Fold Cross-Validation`
2. `27. Define Cross-Validation Helper Functions`
3. `28. Run 5-Fold BERT Fine-Tuning`
4. `29. Summarize Cross-Validation Metrics`
5. `30. Analyze Cross-Validation Confusion Matrices`
6. `31. Save Cross-Validation Summary`

## Error Handling

The implementation should:

- Create checkpoint directories if missing.
- Avoid overwriting existing fold checkpoints unless the notebook cell is rerun intentionally.
- Keep `random_state=42` for reproducibility.
- Clear model references between folds and call `torch.cuda.empty_cache()` when CUDA is available.
- Fall back cleanly across `cuda`, `mps`, and `cpu` as the existing notebook does.

## Verification

Minimum verification after implementation:

- The notebook JSON loads successfully.
- The new cross-validation cells can be inspected in order.
- `StratifiedKFold` produces five folds and each fold contains all five BBC labels.
- Saved checkpoint configs contain real labels: `business`, `entertainment`, `politics`, `sport`, `tech`.
- If dependencies are available, run at least one short smoke test fold with reduced epochs before launching full 5-fold training.
