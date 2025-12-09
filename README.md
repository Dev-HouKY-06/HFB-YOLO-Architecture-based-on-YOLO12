# 📚 HFB-YOLO-Architecture-Based-On-YOLO12
HFB (Hierarchical Fusion Block) is a novel feature fusion module designed to replace the standard concatenation operation in one-stage object detectors (like YOLO12).

## ❓ How To Use

To integrate the **HFB module** into the official Ultralytics YOLO framework, please follow the steps below:

### 1. File Placement
Place the `HFB.py` file into the neural network attention modules directory.

```text
Ultralytics
└── ultralytics
    └── nn
        └── attention
            └── HFB.py
```

### 2. Register the Module
To make the model recognize the new class, you need to import it in `ultralytics/nn/tasks.py`.

```python
# In ultralytics/nn/tasks.py

from ultralytics.nn.attention.HFB import HierarchicalFusionBlock  # Import your module

```

### 3. Modify YAML Configuration
In the model configuration file (e.g., `yolo12.yaml` or `yolo12n-hfb.yaml`), replace the original layer in the Neck network with the `HFB` module.

```yaml
# yolo12n-hfb.yaml
# ... (Backbone)

head:
  # ... (Insert Here)
  - [-1, 1, HierarchicalFusionBlock, [* , * , *]] # 14
  # ... (Rest of the head)
```

# ⚖️ License & Copyright
[License & Copyright](https://github.com/Dev-HouKY-06/HFB-YOLO-Architecture-based-on-YOLO12/blob/main/licence) © 2025-Present, Kunyan Hou
