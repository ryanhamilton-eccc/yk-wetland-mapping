# YK Wetland Mapping

This project focuses on mapping wetlands in the Yukon using various data sources and machine learning techniques.

## Installation

### Prerequisites

- [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)
- Python 3.9 or higher

### Setting up the environment

1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/yk-wetland-mapping.git
    cd yk-wetland-mapping
    ```

2. Create a new conda environment:

    ```bash
    conda create --name yk-wetland-mapping python=3.9
    conda activate yk-wetland-mapping
    ```

3. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

```python
# to run pipeline with all steps
from ykwmp import run_pipeline

if __name__ == '__main__':
    import ee

    ee.Initialize(project="yk-wetland-mapping")

    run_pipeline(
        project_id='yk-wetland-mapping',
        workspace="test-workspace",
        region_path=r"D:\Yukon-Wetland-Mapping-Data\YK_Seperated_AOI\165.shp",
        trainval_root=r"D:\Yukon-Wetland-Mapping-Data\YK_GeeReady\165",
        gdrive_foldername="165"
    )
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.