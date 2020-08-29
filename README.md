- run test

```bash
python -m pytest tests/test_autodiff.py -m task1_4
```

- import minitorch

```python
import sys
sys.path.append("..")

import minitorch
```

- import other package:
as we run in two modes: interactive in notebook and script mode

```python
try:
    from .strategies import tensor_data, indices
except:
    from strategies import tensor_data, indices
```