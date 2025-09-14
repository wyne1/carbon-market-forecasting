#!/usr/bin/env python3
"""
Automated Carbon Market Analysis Pipeline
=========================================

This script automates the entire carbon market analysis process:
- Data preprocessing and loading
- Model training
- Backtesting with preconfigured parameters
- Prediction generation
- Plot generation and saving
- MongoDB storage of predictions

Usage: python automated_pipeline.py
"""

import os
import sys
import warnings
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from pymongo import MongoClient

# Import your utility modules
from utils.dataset import MarketData, DataPreprocessor
from utils.data_processing import prepare_data
from utils.model_utils import create_model, train_model, generate_model_predictions
from utils.mongodb_utils import setup_mongodb_connection, save_recent_predictions
from utils.backtesting import backtest_model_with_metrics
from utils.smart_preprocessing import SmartAuctionPreprocessor
from utils.data_processing import reverse_normalize
from utils.lseg_data_loader import LSEGDataLoader
# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

class AutomatedPipeline:
    """
    Automated pipeline for carbon market analysis
    """
    
    def __init__(self, config=None):
        """
        Initialize the pipeline with configuration
        
        Args:
            config (dict): Configuration parameters
        """
        # Default configuration
        self.config = {
            'data_file': 'data/latest_data_jul.xlsx',
            'output_dir': 'output_plots',
            'initial_balance': 10000.0,
            'take_profit': 0.04,  # 4%
            'stop_loss': 0.03,    # 3%
            'position_size_fraction': 1.0,  # 100%
            'risk_free_rate': 0.01,  # 1%
            'input_width': 7,
            'out_steps': 7,
            'max_epochs': 40,
            'save_to_mongodb': True,
            'plot_style': 'seaborn-v0_8',
            'figure_size': (12, 8),
            'dpi': 300
        }
        
        # Update with user config if provided
        if config:
            self.config.update(config)
            
        # Create output directory
        self.output_dir = Path(self.config['output_dir'])
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.model = None
        self.preprocessor = None
        self.test_df = None
        self.predictions_df = None
        self.recent_preds = None
        self.trend = None
        self.ensemble_results = None  # Store ensemble results
        self.auction_loader = LSEGDataLoader()
            
        
        print("🚀 Automated Carbon Market Analysis Pipeline Initialized")
        print(f"📁 Output directory: {self.output_dir.absolute()}")

    def train_model(self, merged_df):
        """
        Train the prediction model
        """
        print("\n🤖 Training model...")
        
        try:
            # Prepare data
            merged_df = merged_df.drop(['Auction_Type', 'Day of Week'], axis=1, errors='ignore')
            train_df, test_df, val_df, preprocessor = prepare_data(merged_df)
            
            # Clean data
            train_df = train_df.dropna(axis=1)
            test_df = test_df.dropna(axis=1)
            val_df = val_df.dropna(axis=1)
            
            print(f"📊 Training set: {train_df.shape}")
            print(f"📊 Test set: {test_df.shape}")
            print(f"📊 Validation set: {val_df.shape}")
            
            # Create and train model
            num_features = len(test_df.columns)
            model = create_model(num_features, self.config['out_steps'])
            
            history = train_model(model, train_df, val_df, test_df, preprocessor, 
                                max_epochs=self.config['max_epochs'])
            
            # Generate predictions
            predictions_df, recent_preds, trend = generate_model_predictions(model, test_df)
            
            # Store results
            self.model = model
            self.preprocessor = preprocessor
            self.test_df = test_df
            self.predictions_df = predictions_df
            self.recent_preds = recent_preds
            self.trend = trend
            
            print(f"✅ Model trained successfully")
            print(f"📈 Trend detected: {trend}")
            
            return history
            
        except Exception as e:
            print(f"❌ Error training model: {e}")
            raise

    def run_backtest(self):
        """
        Run backtesting with configured parameters
        """
        print("\n📈 Running backtest...")
        
        try:
            trade_log_df, performance_metrics, balance_history_df = backtest_model_with_metrics(
                self.model,
                self.test_df,
                self.config['input_width'],
                self.config['out_steps'],
                self.config['initial_balance'],
                self.config['take_profit'],
                self.config['stop_loss'],
                position_size_fraction=self.config['position_size_fraction'],
                risk_free_rate=self.config['risk_free_rate'],
                preprocessor=self.preprocessor
            )
            
            print("✅ Backtest completed successfully")
            print(f"📊 Total trades: {len(trade_log_df)}")
            print(f"💰 Total return: {performance_metrics.get('Total Return (%)', 0):.2f}%")
            print(f"📉 Max drawdown: {performance_metrics.get('Maximum Drawdown (%)', 0):.2f}%")
            
            return trade_log_df, performance_metrics, balance_history_df
            
        except Exception as e:
            print(f"❌ Error running backtest: {e}")
            raise

    def save_performance_metrics(self, performance_metrics):
        """
        Save performance metrics to CSV and text files
        """
        print("\n💾 Saving performance metrics...")
        
        try:
            # Save as CSV
            metrics_df = pd.DataFrame.from_dict(performance_metrics, orient='index', columns=['Value'])
            metrics_df.reset_index(inplace=True)
            metrics_df.rename(columns={'index': 'Metric'}, inplace=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_path = self.output_dir / f"performance_metrics_{timestamp}.csv"
            metrics_df.to_csv(csv_path, index=False)
            
            # Save as formatted text
            txt_path = self.output_dir / f"performance_summary_{timestamp}.txt"
            with open(txt_path, 'w') as f:
                f.write("Carbon Market Trading Strategy Performance Summary\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                f.write("Configuration:\n")
                f.write("-" * 20 + "\n")
                for key, value in self.config.items():
                    f.write(f"{key}: {value}\n")
                
                f.write("\nPerformance Metrics:\n")
                f.write("-" * 20 + "\n")
                for metric, value in performance_metrics.items():
                    f.write(f"{metric}: {value}\n")
            
            print(f"✅ Metrics saved to: {csv_path}")
            print(f"✅ Summary saved to: {txt_path}")
            
        except Exception as e:
            print(f"❌ Error saving metrics: {e}")

    def plot_equity_curve(self, balance_history_df):
        """
        Generate and save equity curve plot
        """
        print("\n📈 Generating equity curve plot...")
        
        try:
            plt.style.use('default')  # Use default style for compatibility
            fig, ax1 = plt.subplots(figsize=self.config['figure_size'])
            
            # Prepare data
            balance_history_df = balance_history_df.reset_index()
            balance_history_df['Date'] = pd.to_datetime(balance_history_df['Date'])
            balance_history_df['Returns'] = balance_history_df['Balance'].pct_change().fillna(0)
            balance_history_df['Cumulative Return'] = (1 + balance_history_df['Returns']).cumprod() - 1
            balance_history_df['Cumulative Max'] = balance_history_df['Balance'].cummax()
            balance_history_df['Drawdown'] = balance_history_df['Balance'] / balance_history_df['Cumulative Max'] - 1
            
            # Plot equity curve
            ax1.plot(balance_history_df['Date'], balance_history_df['Balance'], 
                    color='blue', linewidth=2, label='Equity Curve')
            
            # Shade drawdown areas
            ax1.fill_between(balance_history_df['Date'], 
                           balance_history_df['Balance'], 
                           balance_history_df['Cumulative Max'],
                           where=balance_history_df['Balance'] < balance_history_df['Cumulative Max'],
                           interpolate=True, color='red', alpha=0.3, label='Drawdown')
            
            # Secondary axis for cumulative returns
            ax2 = ax1.twinx()
            ax2.plot(balance_history_df['Date'], balance_history_df['Cumulative Return'], 
                    color='green', linewidth=1, alpha=0.7, label='Cumulative Return (%)')
            ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
            ax2.set_ylabel('Cumulative Return (%)')
            
            # Formatting
            ax1.set_title('Equity Curve with Drawdowns and Cumulative Returns', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Account Balance ($)')
            ax1.grid(True, alpha=0.3)
            ax1.legend(loc='upper left')
            
            plt.tight_layout()
            
            # Save plot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_path = self.output_dir / f"equity_curve_{timestamp}.png"
            plt.savefig(plot_path, dpi=self.config['dpi'], bbox_inches='tight')
            plt.close()
            
            print(f"✅ Equity curve saved to: {plot_path}")
            
        except Exception as e:
            print(f"❌ Error generating equity curve: {e}")

    def plot_trades_on_price_chart(self, trade_log_df):
        """
        Generate and save trades visualization on price chart
        """
        print("\n📊 Generating trades visualization...")
        
        try:
            fig, ax = plt.subplots(figsize=self.config['figure_size'])
            
            # Prepare data
            test_df_denorm = reverse_normalize(self.test_df.copy(), 
                                             self.preprocessor.train_mean['Auc Price'], 
                                             self.preprocessor.train_std['Auc Price'])
            
            # Plot price data
            ax.plot(test_df_denorm.index, test_df_denorm['Auc Price'], 
                   label='Auction Price', color='blue', linewidth=1)
            
            # Plot trades
            for idx, trade in trade_log_df.iterrows():
                entry_date = pd.to_datetime(trade['Entry Date'])
                exit_date = pd.to_datetime(trade['Exit Date'])
                signal = trade['Signal']
                entry_price = trade['Entry Price']
                exit_price = trade['Exit Price']
                return_pct = trade['Return (%)']
                
                # Skip if dates not in index
                if entry_date not in test_df_denorm.index or exit_date not in test_df_denorm.index:
                    continue
                
                # Determine success and colors
                is_successful = return_pct > 0
                success_color = 'limegreen' if is_successful else 'red'
                
                # Plot entry signals
                if signal == 'Buy':
                    ax.scatter(entry_date, entry_price, color='green', marker='^', s=100, zorder=5)
                    ax.text(entry_date, entry_price, "Buy", fontsize=8, 
                           verticalalignment='bottom', color='green')
                else:
                    ax.scatter(entry_date, entry_price, color='red', marker='v', s=100, zorder=5)
                    ax.text(entry_date, entry_price, "Sell", fontsize=8, 
                           verticalalignment='top', color='red')
                
                # Add return annotation
                mid_date = entry_date + (exit_date - entry_date) / 2
                mid_price = (entry_price + exit_price) / 2
                ax.text(mid_date, mid_price, f"{return_pct:.1f}%", fontsize=8, 
                       color=success_color, ha='center', va='center',
                       bbox=dict(facecolor='white', edgecolor=success_color, 
                               alpha=0.7, boxstyle='round,pad=0.3'))
            
            # Formatting
            ax.set_title('Carbon Auction Price with Trading Signals', fontsize=14, fontweight='bold')
            ax.set_xlabel('Date')
            ax.set_ylabel('Price (€)')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            plt.tight_layout()
            
            # Save plot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_path = self.output_dir / f"trades_visualization_{timestamp}.png"
            # plt.savefig(plot_path, dpi=self.config['dpi'], bbox_inches='tight')
            plt.show()
            plt.close()
            
            print(f"✅ Trades visualization saved to: {plot_path}")
            
        except Exception as e:
            print(f"❌ Error generating trades visualization: {e}")

    def plot_predictions(self):
        """
        Generate and save predictions plot
        """
        print("\n🔮 Generating predictions plot...")
        
        try:
            fig, ax = plt.subplots(figsize=self.config['figure_size'])
            
            # Prepare data
            recent_preds = self.recent_preds.copy()
            test_df = reverse_normalize(self.test_df.copy(), 
                                      self.preprocessor.train_mean['Auc Price'], 
                                      self.preprocessor.train_std['Auc Price'])
            
            # Plot historical data (last 60 days)
            plot_df = test_df.tail(60)
            ax.plot(plot_df.index, plot_df['Auc Price'], 
                   label='Historical Price', color='blue', marker='o', markersize=3)
            
            # Denormalize predictions
            recent_preds['Auc Price'] = (recent_preds['Auc Price'] * 
                                       self.preprocessor.train_std['Auc Price']) + \
                                      self.preprocessor.train_mean['Auc Price']
            
            # Calculate prediction difference
            pred_diff = np.mean(recent_preds.iloc[1:]['Auc Price'].values) - \
                       recent_preds.iloc[1]['Auc Price']
            
            # Plot predictions with trend indication
            if pred_diff > 0:
                ax.scatter(recent_preds.index[0], recent_preds.iloc[0]['Auc Price'], 
                          color='green', marker='^', s=150, zorder=5, label='Buy Signal')
                ax.plot(recent_preds.index, recent_preds['Auc Price'], 
                       color='green', linestyle='--', linewidth=2, alpha=0.8, label='Prediction (Bullish)')
                trend_text = "BUY"
                trend_color = 'green'
            else:
                ax.scatter(recent_preds.index[0], recent_preds.iloc[0]['Auc Price'], 
                          color='red', marker='v', s=150, zorder=5, label='Sell Signal')
                ax.plot(recent_preds.index, recent_preds['Auc Price'], 
                       color='red', linestyle='--', linewidth=2, alpha=0.8, label='Prediction (Bearish)')
                trend_text = "SELL"
                trend_color = 'red'
            
            # Add trend annotation
            ax.text(recent_preds.index[0], recent_preds.iloc[0]['Auc Price'], 
                   trend_text, fontsize=12, fontweight='bold',
                   verticalalignment='bottom', color=trend_color,
                   bbox=dict(facecolor='white', edgecolor=trend_color, alpha=0.8))
            
            # Formatting
            ax.set_title('Carbon Auction Price Predictions (7-Day Forecast)', 
                        fontsize=14, fontweight='bold')
            ax.set_xlabel('Date')
            ax.set_ylabel('Price (€)')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Rotate x-axis labels for better readability
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Save plot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_path = self.output_dir / f"predictions_{timestamp}.png"
            plt.savefig(plot_path, dpi=self.config['dpi'], bbox_inches='tight')
            plt.close()
            
            print(f"✅ Predictions plot saved to: {plot_path}")
            
        except Exception as e:
            print(f"❌ Error generating predictions plot: {e}")

    def save_to_mongodb(self):
        """
        Save predictions to MongoDB
        """
        if not self.config['save_to_mongodb']:
            print("\n⏭️  Skipping MongoDB save (disabled in config)")
            return
            
        print("\n💾 Saving predictions to MongoDB...")
        
        try:
            collection = setup_mongodb_connection()
            save_message = save_recent_predictions(collection, self.recent_preds, self.preprocessor)
            print(f"✅ {save_message}")
            
        except Exception as e:
            print(f"❌ Error saving to MongoDB: {e}")
            print("💡 Make sure MongoDB is running locally on port 27017")

    def save_trade_log(self, trade_log_df):
        """
        Save detailed trade log to CSV
        """
        print("\n📝 Saving trade log...")
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_path = self.output_dir / f"trade_log_{timestamp}.csv"
            
            # Format dates for CSV
            trade_log_formatted = trade_log_df.copy()
            trade_log_formatted['Entry Date'] = trade_log_formatted['Entry Date'].dt.date
            trade_log_formatted['Exit Date'] = trade_log_formatted['Exit Date'].dt.date
            
            trade_log_formatted.to_csv(csv_path, index=False)
            print(f"✅ Trade log saved to: {csv_path}")
            
        except Exception as e:
            print(f"❌ Error saving trade log: {e}")

    def generate_summary_report(self, performance_metrics, trade_log_df):
        """
        Generate a comprehensive summary report
        """
        print("\n📄 Generating summary report...")
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = self.output_dir / f"analysis_summary_{timestamp}.md"
            
            with open(report_path, 'w') as f:
                f.write("# Carbon Market Analysis Summary Report\n\n")
                f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                f.write("## Configuration\n\n")
                f.write("| Parameter | Value |\n")
                f.write("|-----------|-------|\n")
                for key, value in self.config.items():
                    f.write(f"| {key.replace('_', ' ').title()} | {value} |\n")
                
                f.write("\n## Performance Summary\n\n")
                f.write("| Metric | Value |\n")
                f.write("|--------|-------|\n")
                for metric, value in performance_metrics.items():
                    if isinstance(value, float):
                        f.write(f"| {metric} | {value:.2f} |\n")
                    else:
                        f.write(f"| {metric} | {value} |\n")
                
                f.write("\n## Trading Activity\n\n")
                f.write(f"- **Total Trades:** {len(trade_log_df)}\n")
                f.write(f"- **Winning Trades:** {len(trade_log_df[trade_log_df['Profit/Loss'] > 0])}\n")
                f.write(f"- **Losing Trades:** {len(trade_log_df[trade_log_df['Profit/Loss'] < 0])}\n")
                
                if len(trade_log_df) > 0:
                    avg_trade_duration = (trade_log_df['Exit Date'] - trade_log_df['Entry Date']).dt.days.mean()
                    f.write(f"- **Average Trade Duration:** {avg_trade_duration:.1f} days\n")
                
                f.write(f"\n## Model Prediction\n\n")
                f.write(f"- **Predicted Trend:** {self.trend.title()}\n")
                f.write(f"- **Signal:** {'BUY' if self.trend == 'positive' else 'SELL'}\n")
                
                f.write("\n## Files Generated\n\n")
                f.write("This analysis generated the following files:\n\n")
                for file_path in self.output_dir.glob(f"*{timestamp}*"):
                    f.write(f"- `{file_path.name}`\n")
            
            print(f"✅ Summary report saved to: {report_path}")
            
        except Exception as e:
            print(f"❌ Error generating summary report: {e}")

    def train_ensemble_models(self, merged_df):
        """
        Train ensemble of models for improved prediction accuracy
        """
        if not self.config.get('include_ensemble_analysis', False):
            print("\n⏭️  Skipping ensemble training (disabled in config)")
            return None
            
        print(f"\n🤖 Training ensemble of {self.config['ensemble_num_models']} models...")
        
        try:
            from utils.model_utils import train_ensemble_models
            
            ensemble_predictions = []
            num_models = self.config['ensemble_num_models']
            max_epochs = self.config['ensemble_max_epochs']
            
            print(f"📊 Training {num_models} models with {max_epochs} epochs each...")
            
            for i, (preds, trend, preprocessor, test_df) in enumerate(train_ensemble_models(
                merged_df, num_models, max_epochs)):
                
                ensemble_predictions.append((preds, trend))
                print(f"✅ Model {i+1}/{num_models} trained - Trend: {trend}")
                
                # Store the preprocessor and test_df from the last model (they should be the same)
                if i == 0:  # Store from first model
                    self.ensemble_preprocessor = preprocessor
                    self.ensemble_test_df = test_df
            
            # Store ensemble results
            self.ensemble_results = {
                'predictions': ensemble_predictions,
                'num_models': num_models,
                'preprocessor': self.ensemble_preprocessor,
                'test_df': self.ensemble_test_df
            }
            
            print(f"✅ Ensemble training completed successfully!")
            print(f"📈 Models trained: {len(ensemble_predictions)}")
            
            # Calculate consensus
            buy_votes = sum(1 for _, trend in ensemble_predictions if trend == 'positive')
            sell_votes = len(ensemble_predictions) - buy_votes
            consensus = 'BUY' if buy_votes > sell_votes else 'SELL'
            strength = max(buy_votes, sell_votes) / len(ensemble_predictions) * 100
            
            print(f"🎯 Ensemble consensus: {consensus} ({strength:.1f}% agreement)")
            
            return self.ensemble_results
            
        except Exception as e:
            print(f"❌ Error training ensemble models: {e}")
            return None

    def save_ensemble_results(self):
        """
        Save ensemble results to file for later PDF generation
        """
        if not self.ensemble_results:
            return
            
        try:
            import pickle
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            ensemble_file = self.output_dir / f"ensemble_results_{timestamp}.pkl"
            
            with open(ensemble_file, 'wb') as f:
                pickle.dump(self.ensemble_results, f)
                
            print(f"💾 Ensemble results saved to: {ensemble_file}")
            
        except Exception as e:
            print(f"❌ Error saving ensemble results: {e}")

    def generate_pdf_report(self, performance_metrics, trade_log_df):
        """
        Generate comprehensive PDF report
        """
        print("\n📄 Generating PDF report...")
        
        try:
            from pdf_report_generator import PDFReportGenerator
            
            generator = PDFReportGenerator(str(self.output_dir))
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Store current analysis data for PDF generation
            self.current_performance_metrics = performance_metrics
            self.current_trade_log_df = trade_log_df
            
            # Pass ensemble results to PDF generator
            pdf_path = generator.generate_pdf_report(
                timestamp=timestamp,
                ensemble_results=self.ensemble_results
            )
            print(f"✅ PDF report generated: {pdf_path}")
            
            return pdf_path
            
        except ImportError:
            print("❌ PDF generation requires reportlab. Install with: pip install reportlab")
            return None
        except Exception as e:
            print(f"❌ Error generating PDF: {e}")
            return None

    def run_full_pipeline(self):
        """
        Run the complete automated pipeline
        """
        start_time = datetime.now()
        print(f"🎯 Starting Carbon Market Analysis Pipeline at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)
        
        try:
            # Step 1: Load and preprocess data
            merged_df = self.auction_loader.load_auction_data()
            
            # Step 2: Train model
            history = self.train_model(merged_df)
            
            # Step 3: Train ensemble models (if enabled)
            ensemble_results = self.train_ensemble_models(merged_df)
            
            # Step 4: Run backtest
            trade_log_df, performance_metrics, balance_history_df = self.run_backtest()
            
            # Step 4: Generate and save all plots
            self.plot_equity_curve(balance_history_df)
            self.plot_trades_on_price_chart(trade_log_df)
            self.plot_predictions()
            
            # Step 5: Save data and reports
            self.save_performance_metrics(performance_metrics)
            self.save_trade_log(trade_log_df)
            self.save_ensemble_results()  # Save ensemble results for later use
            self.generate_summary_report(performance_metrics, trade_log_df)
            
            # Step 6: Generate PDF report
            pdf_path = self.generate_pdf_report(performance_metrics, trade_log_df)
            
            # Step 7: Save to MongoDB (optional)
            # self.save_to_mongodb()
            
            # Completion summary
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            print("\n" + "=" * 70)
            print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
            print(f"⏱️  Total execution time: {duration:.1f} seconds")
            print(f"📁 All outputs saved to: {self.output_dir.absolute()}")
            print(f"📊 Total trades executed: {len(trade_log_df)}")
            print(f"💰 Final return: {performance_metrics.get('Total Return (%)', 0):.2f}%")
            print(f"🎯 Trading signal: {'BUY' if self.trend == 'positive' else 'SELL'}")
            
            # Add ensemble results summary
            if self.ensemble_results:
                ensemble_predictions = self.ensemble_results['predictions']
                buy_votes = sum(1 for _, trend in ensemble_predictions if trend == 'positive')
                sell_votes = len(ensemble_predictions) - buy_votes
                consensus = 'BUY' if buy_votes > sell_votes else 'SELL'
                strength = max(buy_votes, sell_votes) / len(ensemble_predictions) * 100
                print(f"🤖 Ensemble consensus: {consensus} ({strength:.1f}% agreement)")
            
            if pdf_path:
                print(f"📄 PDF report: {pdf_path}")
            
            return True
            
        except Exception as e:
            print(f"\n❌ PIPELINE FAILED: {e}")
            print(f"💡 Check the error details above and ensure all dependencies are installed")
            return False


def main():
    """
    Main function to run the automated pipeline
    """
    # Custom configuration (modify as needed)
    custom_config = {
        'initial_balance': 50000.0,     # $50,000 starting capital
        'take_profit': 0.05,            # 5% take profit
        'stop_loss': 0.025,             # 2.5% stop loss
        'position_size_fraction': 0.8,  # Use 80% of capital per trade
        'max_epochs': 50,               # More training epochs
        'save_to_mongodb': True,        # Save predictions to database
        'dpi': 300                      # High resolution plots
    }
    
    # Initialize and run pipeline
    pipeline = AutomatedPipeline(config=custom_config)
    success = pipeline.run_full_pipeline()
    
    if success:
        print("\n✨ Analysis complete! Check the output_plots folder for results.")
    else:
        print("\n💥 Analysis failed! Please check the error messages above.")
        sys.exit(1)


if __name__ == "__main__":
    main()