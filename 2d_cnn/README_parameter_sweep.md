# Parameter Sweep for Regularization Parameters

This directory contains scripts to perform comprehensive parameter sweeps for regularization parameters in the 2D CNN model. The system tests different combinations of weight decay, entropy regularization, dropout rates, and attention dropout rates to find optimal regularization settings.

## Files Overview

- `parameter_sweep.py` - Full parameter sweep (320 combinations)
- `focused_parameter_sweep.py` - Focused parameter sweep (81 combinations)
- `analyze_sweep_results.py` - Analysis and visualization of results
- `run_parameter_sweep.py` - Easy-to-use runner script
- `main.py` - Original training script (for reference)

## Regularization Parameters Tested

### 1. Weight Decay (L2 Regularization)
- **Purpose**: Prevents overfitting by penalizing large weights
- **Values tested**: `[0.0, 0.0001, 0.001, 0.01]` (full sweep)
- **Values tested**: `[0.0, 0.0001, 0.001]` (focused sweep)

### 2. Entropy Regularization
- **Purpose**: Encourages uniform attention distribution across slices
- **Values tested**: `[0.0, 0.0001, 0.0005, 0.001, 0.005]` (full sweep)
- **Values tested**: `[0.0, 0.0005, 0.001]` (focused sweep)

### 3. Dropout Rate
- **Purpose**: Prevents overfitting by randomly zeroing activations
- **Values tested**: `[0.0, 0.1, 0.2, 0.3]` (full sweep)
- **Values tested**: `[0.1, 0.2, 0.3]` (focused sweep)

### 4. Attention Dropout Rate
- **Purpose**: Applies dropout specifically to attention weights
- **Values tested**: `[0.0, 0.1, 0.2, 0.3]` (full sweep)
- **Values tested**: `[0.0, 0.1, 0.2]` (focused sweep)

## Quick Start

### Option 1: Run Focused Sweep (Recommended)
```bash
python run_parameter_sweep.py --sweep-type focused
```

### Option 2: Run Full Sweep
```bash
python run_parameter_sweep.py --sweep-type full
```

### Option 3: Analyze Existing Results Only
```bash
python run_parameter_sweep.py --analyze-only
```

## Manual Execution

### Run Focused Parameter Sweep
```bash
python focused_parameter_sweep.py
```

### Run Full Parameter Sweep
```bash
python parameter_sweep.py
```

### Analyze Results
```bash
python analyze_sweep_results.py
```

## Output Structure

After running a parameter sweep, you'll find the following structure:

```
focused_sweep_output/
└── focused_sweep_YYYYMMDD_HHMMSS/
    ├── run_000/
    │   ├── parameters.json
    │   ├── training_log.json
    │   └── best_model.pt
    ├── run_001/
    │   ├── parameters.json
    │   ├── training_log.json
    │   └── best_model.pt
    ├── ...
    ├── focused_parameter_sweep_summary.csv
    ├── best_parameters.json
    └── analysis_plots/
        ├── overall_performance.png
        ├── parameter_impact.png
        ├── parameter_heatmaps.png
        ├── top_configurations.png
        └── analysis_summary.json
```

## Analysis Outputs

The analysis script generates several visualizations:

1. **Overall Performance**: Distribution of F1 scores, AUC scores, and best epochs
2. **Parameter Impact**: Individual parameter effects on F1 scores
3. **Parameter Heatmaps**: Interaction effects between parameter pairs
4. **Top Configurations**: Bar chart of the best performing parameter combinations
5. **Statistical Summary**: Detailed statistics for each parameter value

## Expected Results

### Focused Sweep (81 combinations)
- **Time**: ~2-4 hours
- **Memory**: ~4-8 GB GPU memory
- **Expected best F1**: 0.75-0.85 (depending on data)

### Full Sweep (320 combinations)
- **Time**: ~8-16 hours
- **Memory**: ~4-8 GB GPU memory
- **Expected best F1**: 0.75-0.85 (depending on data)

## Interpreting Results

### Key Metrics to Look For:
1. **Best F1 Score**: Primary metric for model performance
2. **Final AUC**: Secondary metric for model performance
3. **Best Epoch**: Indicates training stability
4. **Parameter Trends**: Which parameter values consistently perform well

### Common Patterns:
- **Weight Decay**: Values around 0.0001-0.001 often work well
- **Entropy Regularization**: Values around 0.0005-0.001 often improve attention uniformity
- **Dropout**: Values around 0.1-0.2 often provide good regularization
- **Attention Dropout**: Lower values (0.0-0.1) often work better

## Customizing the Sweep

To modify the parameter ranges, edit the `PARAMETER_COMBINATIONS` or `FOCUSED_PARAMETER_COMBINATIONS` dictionaries in the respective sweep scripts:

```python
FOCUSED_PARAMETER_COMBINATIONS = {
    'weight_decay': [0.0, 0.0001, 0.001],  # Add/remove values
    'entropy_reg_weight': [0.0, 0.0005, 0.001],  # Modify ranges
    'dropout_rate': [0.1, 0.2, 0.3],  # Change values
    'attn_dropout_rate': [0.0, 0.1, 0.2],  # Adjust ranges
}
```

## Troubleshooting

### Common Issues:

1. **Out of Memory**: Reduce batch size or use fewer parameter combinations
2. **Long Training Time**: Use the focused sweep instead of full sweep
3. **Poor Results**: Check data paths and ensure datasets are loaded correctly
4. **Analysis Errors**: Ensure matplotlib and seaborn are installed

### Dependencies:
```bash
pip install torch torchvision
pip install numpy pandas scikit-learn
pip install matplotlib seaborn
pip install SimpleITK tqdm
```

## Best Practices

1. **Start with Focused Sweep**: Use the focused sweep first to get a quick understanding
2. **Monitor Progress**: Check the console output for training progress
3. **Save Results**: Results are automatically saved, but backup important findings
4. **Iterate**: Use results to refine parameter ranges for subsequent sweeps
5. **Validate**: Always validate the best parameters on a separate test set

## Example Workflow

1. Run focused sweep: `python run_parameter_sweep.py --sweep-type focused`
2. Review results in the generated plots
3. Identify promising parameter ranges
4. Optionally run full sweep with refined ranges
5. Apply best parameters to your main training script

## Integration with Main Training

Once you find the best parameters, update your main training script:

```python
# Update these values in main.py based on sweep results
WEIGHT_DECAY = 0.0001  # From sweep results
ENTROPY_REG_WEIGHT = 0.0005  # From sweep results
DROPOUT_RATE = 0.2  # From sweep results
ATTN_DROPOUT_RATE = 0.1  # From sweep results
``` 