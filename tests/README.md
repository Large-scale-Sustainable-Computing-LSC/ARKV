# AKCB Test Suite

This directory contains the test suite for the AKCB (Adaptive KV Caches under Budget) framework.

## Structure

```
tests/
├── __init__.py                    # Test package initialization
├── conftest.py                    # Shared pytest fixtures
├── unit/                          # Unit tests
│   ├── __init__.py
│   ├── test_calculator.py        # Tests for heavy-hitter calculation
│   └── test_cache.py             # Tests for cache implementations
└── integration/                   # Integration tests
    ├── __init__.py
    └── test_experiments.py       # Tests for experiment setup
```

## Running Tests

### Run All Tests

```bash
# From the project root
./run_tests.sh
```

Or using pytest directly:

```bash
pytest tests/ -v
```

### Run Specific Test Categories

**Unit tests only:**
```bash
pytest tests/unit/ -v
```

**Integration tests only:**
```bash
pytest tests/integration/ -v
```

### Run Specific Test Files

```bash
pytest tests/unit/test_calculator.py -v
pytest tests/unit/test_cache.py -v
pytest tests/integration/test_experiments.py -v
```

### Run with Coverage

```bash
pytest tests/ --cov=akcb --cov-report=html
```

This will generate a coverage report in `htmlcov/index.html`.

## Test Categories

### Unit Tests

#### `test_calculator.py`
- Tests for `calculate_heavy_hitter()` function
- Tests for `calculate_entropy()` function
- Edge cases: empty windows, single queries, uniform/deterministic distributions

#### `test_cache.py`
- Tests for `ADCacheConfig` configuration class
- Tests for `AdaptiveCache` factory
- Tests for different cache types: mix, quant, origin, window

### Integration Tests

#### `test_experiments.py`
- Tests for experiment configuration files
- Tests for `Telemetry` tracking
- Tests for `FirstTokenTimer`
- Tests for cache configuration integration

## Writing New Tests

### Test Naming Convention

- Test files: `test_*.py`
- Test classes: `Test*`
- Test functions: `test_*`

### Example Test

```python
import pytest
from akcb.config import ADCacheConfig

class TestMyFeature:
    """Test suite for my feature"""
    
    def test_basic_functionality(self):
        """Test basic functionality"""
        # Arrange
        config = create_test_config()
        
        # Act
        result = my_function(config)
        
        # Assert
        assert result.is_valid()
```

### Using Fixtures

Fixtures are defined in `conftest.py` and can be used by any test:

```python
def test_with_fixture(device, cache_config_args):
    """Test using shared fixtures"""
    assert device is not None
    assert cache_config_args.cache_size == 1024
```

## Requirements

Install test dependencies:

```bash
pip install pytest pytest-cov
```

## CI/CD Integration

These tests can be integrated into CI/CD pipelines:

```yaml
# Example GitHub Actions
- name: Run tests
  run: |
    pip install pytest pytest-cov
    pytest tests/ --cov=akcb --cov-report=xml
```

## Troubleshooting

### Import Errors

If you encounter import errors, make sure to install the package in development mode:

```bash
pip install -e .
```

### CUDA Tests

Some tests may require CUDA. They will automatically skip if CUDA is not available:

```python
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_feature():
    ...
```

## Contributing

When adding new features to AKCB:

1. Write unit tests for the new functionality
2. Add integration tests if the feature interacts with multiple components
3. Ensure all tests pass before submitting PR
4. Maintain test coverage above 80%

## Contact

For questions about the test suite, please open an issue on GitHub.
