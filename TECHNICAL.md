# Technical Documentation - CTC Risk Innovation Analysis

## Overview
This document provides technical details about the implementation of the loan portfolio risk analysis system.

## System Components

### 1. Data Preprocessing (`clean_data.py`)
- **Purpose**: Ensures data quality and consistency
- **Key Functions**:
  - `load_and_clean_data()`: Main data cleaning pipeline
  - Handles missing values
  - Standardizes formats
  - Removes duplicates
  - Performs data quality checks

### 2. Risk Analysis (`app.py`)
- **Purpose**: Performs comprehensive risk analysis
- **Key Components**:
  - Portfolio composition analysis
  - Credit quality assessment
  - Industry concentration analysis
  - Expected loss calculations
  - Value at Risk (VaR) estimation

## Implementation Details

### Data Cleaning Process
```python
def load_and_clean_data():
    # Load raw data
    # Standardize formats
    # Handle missing values
    # Perform quality checks
    return cleaned_df, cleanup_report
```

### Risk Calculations

1. **Expected Loss Calculation**
```python
Expected Loss = PD × LGD × EAD
where:
- PD: Rating-based probability of default
- LGD: Industry-specific loss given default
- EAD: Drawn amount (current exposure)
```

2. **Value at Risk (VaR) Calculation**
```python
def calculate_var(df, confidence_level, n_simulations):
    # Monte Carlo simulation
    # Generate default scenarios
    # Calculate loss distribution
    return var_estimate
```

## Data Dependencies

### Input Data Requirements
- CSV file with specific columns:
  - Client_ID
  - Client_Short_Name
  - Loan_Number
  - Total_Loan_Amount
  - Drawn_Amount
  - Client_Location
  - Client_Industry
  - Client_Internal_Rating
  - Client_External_Rating
  - Time_to_Maturity_Years

### Reference Data
1. **PD Mapping Table**
   - Internal ratings to default probabilities

2. **LGD Table**
   - Industry-specific loss given default rates

## Output Files

1. **Visualizations**
   - `graph_internal_rating.png`: Credit quality distribution
   - `graph_drawn_industry.png`: Industry exposure analysis
   - `graph_industry_loss.png`: Expected loss by industry
   - `var_analysis.png`: Portfolio loss distribution and VaR analysis
     - Shows the empirical loss distribution
     - Marks 95% and 99% VaR levels
     - Includes kernel density estimate for smooth visualization

2. **Data Files**
   - `CTC_Risk_Innovation_Loans_Dataset_cleaned.csv`: Cleaned dataset

## System Requirements

### Python Dependencies
```
pandas>=1.3.0
numpy>=1.20.0
matplotlib>=3.4.0
seaborn>=0.11.0
scipy>=1.7.0
```

### Hardware Requirements
- Minimum 8GB RAM recommended for large portfolios
- SSD storage for faster data processing

## Performance Considerations

1. **Data Processing**
   - Efficient pandas operations
   - Vectorized calculations where possible
   - Optimized memory usage

2. **Monte Carlo Simulation**
   - Parallel processing capabilities
   - Configurable simulation parameters

## Error Handling

1. **Data Validation**
   - Input data format checking
   - Missing value detection
   - Data type validation

2. **Risk Calculation Safeguards**
   - Non-negative value enforcement
   - Range validation for probabilities
   - Error logging

## Future Enhancements

1. **Planned Features**
   - Correlation-based VaR calculation
   - Stress testing scenarios
   - Rating migration analysis

2. **Technical Improvements**
   - GPU acceleration for Monte Carlo
   - Real-time monitoring capabilities
   - API integration

## Maintenance

### Regular Tasks
1. Update PD and LGD tables
2. Validate risk calculations
3. Monitor performance metrics

### Troubleshooting
1. Check input data quality
2. Verify calculation parameters
3. Review log files

## Contact Information
For technical support or questions about the implementation, please contact the development team.

## Version History
- v1.0: Initial implementation
- v1.1: Added VaR calculation
- v1.2: Enhanced visualization capabilities
