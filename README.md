# Water Quality Management

A toolkit for water quality data analysis, preprocessing, and prediction.  
This project provides scripts and notebooks for handling sensor and weather data, building predictive models, and managing water quality datasets.

## Features

- Data preprocessing and cleaning
- Integration of sensor and weather data
- Predictive modeling for water quality
- Jupyter notebooks for exploratory analysis

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/lewashby/waterquality.git
   cd waterquality
   ```

2. **Install Poetry (if not already installed):**  
   See the [Poetry documentation](https://python-poetry.org/docs/).

3. **Install dependencies:**
   ```bash
   poetry install
   ```

## Usage

- **Run the main CLI:**
  ```bash
  poetry run waterquality
  ```
  This will display available commands and usage instructions.

- **Jupyter Notebooks:**  
  Notebooks for data exploration and preprocessing are in the `notebooks/` directory.  

## Project Structure

```
waterquality/
│
├── data/           # Raw and processed datasets (CSV, XLSX)
├── notebooks/      # Jupyter notebooks for analysis and preprocessing
├── waterquality/   # Main Python package
│   ├── DataHandler.py   # Data loading and preprocessing
│   ├── Model.py         # Model definitions
│   ├── ModelBuilder.py  # Model building utilities
│   ├── Pipeline.py      # Data processing pipelines
│   ├── Utils.py         # Utility functions
│   └── main.py          # CLI entry point
├── tests/          # Unit tests
├── README.md
├── pyproject.toml
└── poetry.lock
```

### Default Data Sources

- **Sensors data:** `data/Fitterizzi_output.csv`
- **Weather data:** `data/Meteo Agosto 22 - Luglio 23.xlsx`

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.
