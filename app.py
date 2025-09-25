"""
Advanced Risk Analysis Script for CTC Risk Innovation Loan Portfolio - Case Study Implementation

This script performs comprehensive risk analysis on a cleaned loan portfolio dataset,
including credit quality assessment, industry concentration analysis, and expected loss
calculations using industry-standard methodologies.

The analysis includes:
1. Portfolio composition analysis
2. Credit quality distribution
3. Industry concentration assessment
4. Expected loss calculations using PD/LGD approach
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from scipy.stats import norm
import matplotlib.ticker as ticker

def load_and_analyze_portfolio():
    """
    Load the cleaned dataset and perform comprehensive risk analysis.
    
    Returns:
        tuple: (total_expected_loss, industry_loss_breakdown)
    """
    # Load the cleaned dataset
    print("Loading cleaned dataset...")
    df = pd.read_csv('CTC_Risk_Innovation_Loans_Dataset_cleaned.csv')
    df.columns = [c.strip() for c in df.columns]

    # Portfolio Overview Statistics
    print("\nPortfolio Overview:")
    print("-" * 50)
    print(f"Total Number of Loans: {len(df):,}")
    print(f"Total Portfolio Exposure: ${df['Total_Loan_Amount'].sum():,.2f}")
    print(f"Total Drawn Amount: ${df['Drawn_Amount'].sum():,.2f}")
    
    # Credit Quality Analysis - Graph 1
    plt.figure(figsize=(12,7))
    sns.countplot(x='Client_Internal_Rating', data=df, 
                 order=sorted(df['Client_Internal_Rating'].unique()),
                 palette='viridis')
    plt.title('Credit Quality Distribution', fontsize=12, pad=15)
    plt.xlabel('Internal Rating', fontsize=10)
    plt.ylabel('Number of Loans', fontsize=10)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('graph_internal_rating.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Industry Concentration Analysis - Graph 2
    plt.figure(figsize=(14,7))
    industry_drawn = df.groupby('Client_Industry')['Drawn_Amount'].sum().sort_values(ascending=True)
    industry_drawn.plot(kind='barh', color='navy')
    plt.title('Industry Concentration Analysis', fontsize=12, pad=15)
    plt.xlabel('Drawn Amount (USD)', fontsize=10)
    plt.ylabel('Industry', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('graph_drawn_industry.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Risk Parameters based on internal rating and industry
    internal_pd = {
        'AAA': 0.02, 'AA': 0.05, 'A': 0.10, 'A-': 0.20,
        'BBB': 0.50, 'BB': 2.00, 'B': 6.00, 'CCC': 20.00
    }
    lgd = {
        'Manufacturing': 50, 'Automobile': 55, 'TMT': 60,
        'Pharmacy': 40, 'Finance': 45, 'Insurance': 35
    }

    def compute_expected_loss(row):
        """
        Calculate expected loss using the formula: EL = PD × LGD × EAD
        where EAD is represented by the Drawn Amount
        """
        pd_val = internal_pd.get(row['Client_Internal_Rating'], 0.5) / 100
        lgd_val = lgd.get(row['Client_Industry'], 0.5) / 100
        ead = row['Drawn_Amount']
        return ead * pd_val * lgd_val

    # Calculate Expected Loss for each loan
    df['ExpectedLoss'] = df.apply(compute_expected_loss, axis=1)
    total_expected_loss = df['ExpectedLoss'].sum()
    
    # Industry-wise Loss Analysis
    industry_loss = df.groupby('Client_Industry').agg({
        'ExpectedLoss': 'sum',
        'Drawn_Amount': 'sum'
    }).sort_values('ExpectedLoss', ascending=True)
    
    industry_loss['Loss_Rate'] = (industry_loss['ExpectedLoss'] / 
                                industry_loss['Drawn_Amount'] * 100)

    # Expected Loss Visualization - Graph 3
    plt.figure(figsize=(14,7))
    industry_loss['ExpectedLoss'].plot(kind='barh', color='crimson')
    plt.title('Expected Loss Distribution by Industry', fontsize=12, pad=15)
    plt.xlabel('Expected Loss (USD)', fontsize=10)
    plt.ylabel('Industry', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('graph_industry_loss.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Print Risk Metrics
    print("\nRisk Metrics Summary:")
    print("-" * 50)
    print(f"Total Expected Loss: ${total_expected_loss:,.2f}")
    print(f"Expected Loss Rate: {(total_expected_loss/df['Drawn_Amount'].sum()*100):.2f}%")
    
    print("\nIndustry-wise Risk Profile:")
    print("-" * 50)
    print(industry_loss['Loss_Rate'].round(2).to_string())

    # Additional Visualizations

    # 4. Rating vs Drawn Amount Box Plot
    plt.figure(figsize=(14, 7))
    sns.boxplot(x='Client_Internal_Rating', y='Drawn_Amount', data=df,
                order=sorted(df['Client_Internal_Rating'].unique()))
    plt.title('Loan Size Distribution by Rating', fontsize=12, pad=15)
    plt.xlabel('Internal Rating', fontsize=10)
    plt.ylabel('Drawn Amount (USD)', fontsize=10)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('rating_amount_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 5. Geographic Exposure Heatmap
    plt.figure(figsize=(15, 8))
    geo_industry = pd.pivot_table(df, 
                                values='Drawn_Amount',
                                index='Client_Location',
                                columns='Client_Industry',
                                aggfunc='sum',
                                fill_value=0)
    sns.heatmap(geo_industry, cmap='YlOrRd', annot=True, fmt='.0f', 
                cbar_kws={'label': 'Drawn Amount (USD)'})
    plt.title('Geographic and Industry Risk Heatmap', fontsize=12, pad=15)
    plt.xlabel('Industry', fontsize=10)
    plt.ylabel('Location', fontsize=10)
    plt.tight_layout()
    plt.savefig('geo_industry_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 6. Maturity Profile Visualization
    plt.figure(figsize=(12, 6))
    df['Maturity_Bucket'] = pd.cut(df['Time_to_Maturity_Years'], 
                                  bins=[0, 1, 3, 5, 10, float('inf')],
                                  labels=['0-1Y', '1-3Y', '3-5Y', '5-10Y', '>10Y'])
    maturity_exposure = df.groupby('Maturity_Bucket')['Drawn_Amount'].sum()
    maturity_exposure.plot(kind='bar', color='teal')
    plt.title('Maturity Profile of Portfolio', fontsize=12, pad=15)
    plt.xlabel('Time to Maturity', fontsize=10)
    plt.ylabel('Drawn Amount (USD)', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('maturity_profile.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 7. Risk Contribution Analysis
    plt.figure(figsize=(12, 6))
    risk_contribution = (df.groupby('Client_Internal_Rating')
                        .agg({
                            'ExpectedLoss': 'sum',
                            'Drawn_Amount': 'sum'
                        }))
    risk_contribution['Risk_Ratio'] = (risk_contribution['ExpectedLoss'] / 
                                     risk_contribution['Drawn_Amount'] * 100)
    risk_contribution['Risk_Ratio'].plot(kind='bar', color='purple')
    plt.title('Risk/Return Ratio by Rating', fontsize=12, pad=15)
    plt.xlabel('Internal Rating', fontsize=10)
    plt.ylabel('Risk Ratio (%)', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('risk_return_profile.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 8. Correlation Analysis
    plt.figure(figsize=(10, 10))
    industry_returns = df.pivot_table(
        values='ExpectedLoss',
        index=df.index,
        columns='Client_Industry',
        aggfunc='sum',
        fill_value=0
    )
    correlation_matrix = industry_returns.corr()
    sns.heatmap(correlation_matrix, 
                annot=True, 
                cmap='coolwarm', 
                center=0,
                fmt='.2f')
    plt.title('Industry Risk Correlation Matrix', fontsize=12, pad=15)
    plt.tight_layout()
    plt.savefig('industry_correlation.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 9. Portfolio Concentration Analysis
    plt.figure(figsize=(12, 6))
    df['Size_Category'] = pd.qcut(df['Drawn_Amount'], q=5, labels=['Very Small', 'Small', 'Medium', 'Large', 'Very Large'])
    concentration_data = df.groupby('Size_Category')['Drawn_Amount'].sum()
    plt.pie(concentration_data, labels=concentration_data.index, autopct='%1.1f%%', 
            colors=sns.color_palette("Set3"))
    plt.title('Portfolio Concentration by Loan Size', fontsize=12, pad=15)
    plt.savefig('portfolio_concentration.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 10. Loss Distribution by Rating and Industry
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
    # Loss by Rating
    rating_loss = df.groupby('Client_Internal_Rating')['ExpectedLoss'].sum().sort_values()
    rating_loss.plot(kind='barh', ax=ax1, color='indianred')
    ax1.set_title('Expected Loss by Rating', fontsize=10)
    ax1.set_xlabel('Expected Loss (USD)', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Loss by Industry
    industry_loss['ExpectedLoss'].plot(kind='barh', ax=ax2, color='seagreen')
    ax2.set_title('Expected Loss by Industry', fontsize=10)
    ax2.set_xlabel('Expected Loss (USD)', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('loss_distribution_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Calculate Value at Risk (VaR)
    def calculate_var(df, confidence_level=0.95, n_simulations=10000):
        """
        Calculate portfolio Value at Risk using Monte Carlo simulation
        
        Parameters:
        -----------
        df : pandas DataFrame
            Portfolio data
        confidence_level : float
            VaR confidence level (e.g., 0.95 for 95% VaR)
        n_simulations : int
            Number of Monte Carlo simulations
        
        Returns:
        --------
        float : VaR value
        """
        portfolio_value = df['Drawn_Amount'].sum()
        
        # Initialize simulation array
        losses = np.zeros(n_simulations)
        
        for i in range(n_simulations):
            # Generate random default indicators
            default_indicators = np.random.random(len(df)) < df['ExpectedLoss']/df['Drawn_Amount']
            
            # Calculate losses for this simulation
            simulation_losses = df['Drawn_Amount'] * default_indicators
            losses[i] = simulation_losses.sum()
        
        # Calculate VaR
        var = np.percentile(losses, confidence_level * 100)
        
        return var/portfolio_value * 100  # Return as percentage of portfolio value

    # Calculate VaR and create visualization
    def plot_var_distribution(df, n_simulations=10000):
        """
        Create visualization of loss distribution and VaR levels
        """
        portfolio_value = df['Drawn_Amount'].sum()
        losses = np.zeros(n_simulations)
        
        for i in range(n_simulations):
            default_indicators = np.random.random(len(df)) < df['ExpectedLoss']/df['Drawn_Amount']
            simulation_losses = df['Drawn_Amount'] * default_indicators
            losses[i] = simulation_losses.sum() / portfolio_value * 100  # Convert to percentage
        
        # Calculate VaR values
        var_95 = np.percentile(losses, 95)
        var_99 = np.percentile(losses, 99)
        
        # Create the visualization
        plt.figure(figsize=(12, 7))
        
        # Plot the histogram of losses
        sns.histplot(losses, bins=50, stat='density', alpha=0.6, color='skyblue')
        
        # Add kernel density estimate
        sns.kdeplot(losses, color='navy', linewidth=2)
        
        # Add vertical lines for VaR levels
        plt.axvline(var_95, color='orange', linestyle='--', 
                   label=f'95% VaR: {var_95:.2f}%')
        plt.axvline(var_99, color='red', linestyle='--', 
                   label=f'99% VaR: {var_99:.2f}%')
        
        # Customize the plot
        plt.title('Portfolio Loss Distribution and Value at Risk', fontsize=12, pad=15)
        plt.xlabel('Portfolio Loss (%)', fontsize=10)
        plt.ylabel('Density', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        # Save the plot
        plt.savefig('var_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return var_95, var_99

    # Generate VaR visualization and get VaR values
    var_95, var_99 = plot_var_distribution(df)
    
    print("\nValue at Risk (VaR) Analysis:")
    print("-" * 50)
    print(f"95% VaR: {var_95:.2f}% of portfolio value")
    print(f"99% VaR: {var_99:.2f}% of portfolio value")

    # Geographic Analysis
    print("\nGeographic Concentration:")
    print("-" * 50)
    geo_exposure = df.groupby('Client_Location')['Drawn_Amount'].sum().sort_values(ascending=False)
    print(geo_exposure.head().to_string())

    # Maturity Profile Analysis
    print("\nMaturity Profile:")
    print("-" * 50)
    df['Maturity_Bucket'] = pd.cut(df['Time_to_Maturity_Years'], 
                                  bins=[0, 1, 3, 5, 10, float('inf')],
                                  labels=['0-1Y', '1-3Y', '3-5Y', '5-10Y', '>10Y'])
    maturity_profile = df.groupby('Maturity_Bucket')['Drawn_Amount'].sum()
    print(maturity_profile.to_string())

    return total_expected_loss, industry_loss, var_95, var_99

if __name__ == "__main__":
    total_loss, industry_breakdown, var_95, var_99 = load_and_analyze_portfolio()
