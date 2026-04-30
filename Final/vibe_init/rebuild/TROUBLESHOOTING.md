# Troubleshooting Kernel Crashes

## Symptom: Kernel Crashes After Data Loading

**Error Pattern:**
```
✓ Data loaded successfully
The Kernel crashed while executing code in the current cell or a previous cell.
```

The kernel crashes silently after data loads, but before results are printed.

## Root Causes (in order of likelihood)

### 1. **PyTorch Not Installed (GRU Model)**

**Symptom:** Works with LogisticGLM, crashes with GRU

**Fix:**
```bash
pip install torch
```

Then run `test_utility_models_minimal.ipynb` again.

**To check if PyTorch is installed:**
```python
import torch
print(torch.__version__)
```

---

### 2. **Out of Memory (Large Dataset)**

**Symptom:** Crash happens during model training or evaluation

The full PhysioNet dataset is large (~20K patients, 790K rows). With GRU + multiple iterations, memory usage can exceed available RAM.

**Fixes:**
1. **Use LogisticGLM only** (no GRU)
   - LogisticGLM: ~50 MB per iteration
   - GRU: ~500 MB per iteration
   
2. **Reduce data size**
   - Edit notebook: limit to first N patients
   - Edit the data loader or filter before bootstrap
   
3. **Reduce iterations**
   - Use `N_ITER = 2` instead of `5`
   - Use `TRAIN_SIZES = [100, 200]` instead of `[50, 100, 200, 300]`

---

### 3. **Incompatible PyTorch Version**

**Symptom:** PyTorch is installed but crashes on `torch.cuda` or tensor operations

**Fix:** Check version compatibility:
```bash
pip install torch --upgrade
```

Or use CPU-only PyTorch (recommended for this analysis):
```bash
pip install torch --force-reinstall
```

---

### 4. **Missing Gender Column in Bootstrap Samples**

**Symptom:** Crash when computing per-group metrics

Some bootstrap samples might not have the Gender column or might have missing values.

**Fix:** The `test_utility_models_quick.ipynb` handles this gracefully. If it still crashes, the issue is elsewhere.

---

## Diagnostic Steps

### Step 1: Run the Quick Test (LogisticGLM Only)

```bash
cd /Users/vrose/ClaudeContainer/PUBH8475/Final/vibe_init
jupyter notebook rebuild/test_utility_models_quick.ipynb
```

This notebook:
- Tests each component separately (data loading, splitting, model creation, evaluation)
- Prints exactly which step fails
- Shows full error traceback

**Expected runtime:** 30-60 seconds for 4 evaluations

If this works → your environment is OK; the issue is with GRU (PyTorch)
If this crashes → let me know the error message from Step 6 or 7

---

### Step 2: Check PyTorch

```bash
python3 << 'EOF'
try:
    import torch
    print(f"✓ PyTorch version: {torch.__version__}")
    print(f"✓ CUDA available: {torch.cuda.is_available()}")
except ImportError:
    print("✗ PyTorch NOT installed")
    print("  Install with: pip install torch")
EOF
```

---

### Step 3: Test GRU Model Directly

```bash
cd /Users/vrose/ClaudeContainer/PUBH8475/Final/vibe_init/rebuild
python3 << 'EOF'
import sys
sys.path.insert(0, '.')
import numpy as np

print("Testing GRU model...")
try:
    from models import GRUModel
    
    model = GRUModel(hidden_size=64, num_layers=1, dropout=0.2, epochs=2)
    print("✓ GRU model created")
    
    # Dummy data
    X_train = np.random.randn(100, 30).astype(float)
    y_train = np.random.randint(0, 2, 100)
    
    print("Fitting on dummy data...")
    model.fit(X_train, y_train)
    print("✓ GRU model fitted")
    
    print("Making predictions...")
    proba = model.predict_proba(X_train[:10])
    print(f"✓ Predictions shape: {proba.shape}")
    
except ImportError as e:
    print(f"✗ PyTorch import failed: {e}")
except Exception as e:
    print(f"✗ GRU model failed: {e}")
    import traceback
    traceback.print_exc()
EOF
```

---

## Recommended Approaches

### For Quick Verification (No GRU)
Use **`test_utility_models_quick.ipynb`**
- Model: LogisticGLM only
- Config: 2 train × 1 boot × 2 iter = 4 evaluations
- Runtime: ~30 seconds
- No PyTorch needed

### For Full LogisticGLM Evaluation
Use **`test_utility_at_scales.ipynb`**
- Model: LogisticGLM only
- Config: 4 train × 3 boot × 10 iter = 120 evaluations
- Runtime: ~5-10 minutes
- No PyTorch needed
- Production-quality results

### For Multi-Model Comparison (with GRU)
Use **`test_utility_models_minimal.ipynb`**
- Models: LogisticGLM + GRU (skip XGBoost)
- Config: 2 train × 1 boot × 3 iter = 12 evaluations
- Runtime: ~2-3 minutes
- Requires PyTorch (installed)

### For Full Evaluation (All Models)
Use **`test_utility_model_comparison.ipynb`**
- Models: LogisticGLM, XGBoost, GRU
- Config: 3 train × 2 boot × 5 iter = 90 evaluations
- Runtime: 10-15 minutes
- Requires: PyTorch + XGBoost (both installed)

---

## Quick Fix Checklist

1. ✅ Try `test_utility_models_quick.ipynb` first (30 sec)
   - If works → environment is OK
   - If crashes → check error message from diagnostic output

2. ✅ If LogisticGLM works but GRU fails:
   ```bash
   pip install torch
   ```

3. ✅ If memory issues (large dataset):
   - Use smaller training sizes: `[50, 100]` instead of `[50, 100, 200, 300]`
   - Use fewer iterations: `N_ITER = 2` instead of `5`
   - Use LogisticGLM only (smaller memory footprint)

4. ✅ If still crashing:
   - Share the full error output from the `test_utility_models_quick.ipynb` diagnostic section
   - The step-by-step test will pinpoint the exact failure

---

## Contact Points

If `test_utility_models_quick.ipynb` crashes:

**Step to report:**
- Which step fails? (Steps 1-7 in the "SINGLE CONFIG TEST" section)
- Full error traceback (shown in notebook or Jupyter log)
- Jupyter log location: check Jupyter terminal output

The step-by-step diagnostics will make debugging much faster.
