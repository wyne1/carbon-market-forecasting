#!/usr/bin/env python3
"""
PDF Report Generator for Carbon Market Analysis
==============================================

This module generates professional PDF reports from the analysis results.
"""

import os
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, 
    Image, PageBreak, KeepTogether
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
from reportlab.platypus.tableofcontents import TableOfContents
from reportlab.lib.colors import HexColor

class PDFReportGenerator:
    """
    Generates comprehensive PDF reports from carbon market analysis results
    """
    
    def __init__(self, output_dir="output_plots"):
        self.output_dir = Path(output_dir)
        self.report_elements = []
        self.styles = getSampleStyleSheet()
        self.setup_custom_styles()
        
    def setup_custom_styles(self):
        """Setup custom styles for the PDF report"""
        # Title style
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Title'],
            fontSize=24,
            spaceAfter=30,
            textColor=colors.darkblue,
            alignment=TA_CENTER
        ))
        
        # Subtitle style
        self.styles.add(ParagraphStyle(
            name='CustomSubtitle',
            parent=self.styles['Heading1'],
            fontSize=16,
            spaceAfter=20,
            textColor=colors.darkgreen,
            alignment=TA_CENTER
        ))
        
        # Section header style
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceAfter=12,
            spaceBefore=20,
            textColor=colors.darkblue,
            borderWidth=1,
            borderColor=colors.darkblue,
            borderPadding=5
        ))
        
        # Metric style for key numbers
        self.styles.add(ParagraphStyle(
            name='MetricStyle',
            parent=self.styles['Normal'],
            fontSize=12,
            textColor=colors.darkgreen,
            fontName='Helvetica-Bold'
        ))
        
        # Summary box style
        self.styles.add(ParagraphStyle(
            name='SummaryBox',
            parent=self.styles['Normal'],
            fontSize=11,
            backgroundColor=HexColor('#f0f8ff'),
            borderWidth=1,
            borderColor=colors.lightblue,
            borderPadding=10,
            spaceAfter=15
        ))

    def add_title_page(self, config):
        """Add title page to the report"""
        # Main title
        title = Paragraph("Carbon Market Analysis Report", self.styles['CustomTitle'])
        self.report_elements.append(title)
        self.report_elements.append(Spacer(1, 0.5*inch))
        
        # Subtitle with date
        subtitle = Paragraph(
            f"Generated on {datetime.now().strftime('%B %d, %Y at %H:%M:%S')}", 
            self.styles['CustomSubtitle']
        )
        self.report_elements.append(subtitle)
        self.report_elements.append(Spacer(1, 0.5*inch))
        
        # Configuration summary box
        config_text = f"""
        <b>Analysis Configuration:</b><br/>
        • Initial Balance: ${config.get('initial_balance', 0):,.2f}<br/>
        • Take Profit: {config.get('take_profit', 0)*100:.1f}%<br/>
        • Stop Loss: {config.get('stop_loss', 0)*100:.1f}%<br/>
        • Position Size: {config.get('position_size_fraction', 0)*100:.0f}%<br/>
        • Training Epochs: {config.get('max_epochs', 0)}<br/>
        • Prediction Window: {config.get('out_steps', 7)} days
        """
        
        config_box = Paragraph(config_text, self.styles['SummaryBox'])
        self.report_elements.append(config_box)
        
        # Add page break
        self.report_elements.append(PageBreak())

    def add_executive_summary(self, performance_metrics, trade_log_df, trend):
        """Add executive summary section"""
        # Section header
        header = Paragraph("Executive Summary", self.styles['SectionHeader'])
        self.report_elements.append(header)
        
        # Key metrics
        total_return = performance_metrics.get('Total Return (%)', 0)
        max_drawdown = performance_metrics.get('Maximum Drawdown (%)', 0)
        sharpe_ratio = performance_metrics.get('Sharpe Ratio', 0)
        win_rate = performance_metrics.get('Win Rate (%)', 0)
        
        # Color code the return
        return_color = 'green' if total_return > 0 else 'red'
        
        summary_text = f"""
        The carbon market analysis has been completed with the following key findings:
        <br/><br/>
        <b>Performance Highlights:</b><br/>
        • <font color="{return_color}"><b>Total Return: {total_return:.2f}%</b></font><br/>
        • Maximum Drawdown: {max_drawdown:.2f}%<br/>
        • Sharpe Ratio: {sharpe_ratio:.2f}<br/>
        • Win Rate: {win_rate:.1f}%<br/>
        • Total Trades: {len(trade_log_df)}<br/>
        <br/>
        <b>Current Market Signal:</b><br/>
        """

        # • Model Prediction: <font color="{'green' if trend == 'positive' else 'red'}"><b>{trend.upper()}</b></font><br/>
        # • Trading Recommendation: <font color="{'green' if trend == 'positive' else 'red'}"><b>{'BUY' if trend == 'positive' else 'SELL'}</b></font>
        
        summary_para = Paragraph(summary_text, self.styles['Normal'])
        self.report_elements.append(summary_para)
        self.report_elements.append(Spacer(1, 0.3*inch))

    def add_performance_metrics_table(self, performance_metrics):
        """Add performance metrics as a formatted table"""
        header = Paragraph("Performance Metrics", self.styles['SectionHeader'])
        self.report_elements.append(header)
        
        # Prepare data for table
        data = [['Metric', 'Value']]
        
        for metric, value in performance_metrics.items():
            if isinstance(value, float):
                if 'Ratio' in metric:
                    formatted_value = f"{value:.3f}"
                elif '%' in metric:
                    formatted_value = f"{value:.2f}%"
                else:
                    formatted_value = f"{value:.2f}"
            else:
                formatted_value = str(value)
            
            data.append([metric, formatted_value])
        
        # Create table
        table = Table(data, colWidths=[4*inch, 2*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
        ]))
        
        self.report_elements.append(table)
        self.report_elements.append(Spacer(1, 0.3*inch))

    def add_trade_summary(self, trade_log_df):
        """Add trade summary statistics"""
        if len(trade_log_df) == 0:
            return
            
        header = Paragraph("Trading Activity Summary", self.styles['SectionHeader'])
        self.report_elements.append(header)
        
        # Calculate trade statistics
        winning_trades = trade_log_df[trade_log_df['Profit/Loss'] > 0]
        losing_trades = trade_log_df[trade_log_df['Profit/Loss'] < 0]
        
        avg_win = winning_trades['Profit/Loss'].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['Profit/Loss'].mean() if len(losing_trades) > 0 else 0
        
        # Trade duration analysis
        trade_log_df['Duration'] = (trade_log_df['Exit Date'] - trade_log_df['Entry Date']).dt.days
        avg_duration = trade_log_df['Duration'].mean()
        
        # Buy vs Sell analysis
        buy_trades = trade_log_df[trade_log_df['Signal'] == 'Buy']
        sell_trades = trade_log_df[trade_log_df['Signal'] == 'Sell']
        
        summary_text = f"""
        <b>Trade Distribution:</b><br/>
        • Total Trades: {len(trade_log_df)}<br/>
        • Winning Trades: {len(winning_trades)} ({len(winning_trades)/len(trade_log_df)*100:.1f}%)<br/>
        • Losing Trades: {len(losing_trades)} ({len(losing_trades)/len(trade_log_df)*100:.1f}%)<br/>
        <br/>
        <b>Trade Performance:</b><br/>
        • Average Winning Trade: ${avg_win:.2f}<br/>
        • Average Losing Trade: ${avg_loss:.2f}<br/>
        • Average Trade Duration: {avg_duration:.1f} days<br/>
        <br/>
        <b>Signal Distribution:</b><br/>
        • Buy Signals: {len(buy_trades)} ({len(buy_trades)/len(trade_log_df)*100:.1f}%)<br/>
        • Sell Signals: {len(sell_trades)} ({len(sell_trades)/len(trade_log_df)*100:.1f}%)<br/>
        """
        
        summary_para = Paragraph(summary_text, self.styles['Normal'])
        self.report_elements.append(summary_para)
        self.report_elements.append(Spacer(1, 0.3*inch))

    def add_recent_trades_table(self, trade_log_df, num_trades=10):
        """Add table of recent trades"""
        if len(trade_log_df) == 0:
            return
            
        header = Paragraph(f"Recent Trades (Last {min(num_trades, len(trade_log_df))})", self.styles['SectionHeader'])
        self.report_elements.append(header)
        
        # Get recent trades
        recent_trades = trade_log_df.tail(num_trades)
        
        # Prepare data for table
        data = [['Entry Date', 'Signal', 'Entry Price', 'Exit Price', 'Return %', 'P&L']]
        
        for _, trade in recent_trades.iterrows():
            return_pct = trade['Return (%)']
            pnl = trade['Profit/Loss']
            
            data.append([
                trade['Entry Date'].strftime('%Y-%m-%d'),
                trade['Signal'],
                f"€{trade['Entry Price']:.2f}",
                f"€{trade['Exit Price']:.2f}",
                f"{return_pct:.2f}%",
                f"${pnl:.2f}"
            ])
        
        # Create table
        table = Table(data, colWidths=[1.2*inch, 0.8*inch, 1*inch, 1*inch, 1*inch, 1*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
        ]))
        
        self.report_elements.append(table)
        self.report_elements.append(Spacer(1, 0.3*inch))

    def add_chart_image(self, image_path, title, max_width=6*inch, max_height=4*inch):
        """Add a chart image to the report"""
        if not image_path.exists():
            return
            
        # Section header
        header = Paragraph(title, self.styles['SectionHeader'])
        self.report_elements.append(header)
        
        # Add image
        img = Image(str(image_path), width=max_width, height=max_height)
        img.hAlign = 'CENTER'
        self.report_elements.append(img)
        self.report_elements.append(Spacer(1, 0.2*inch))

    def add_risk_analysis(self, performance_metrics, trade_log_df):
        """Add risk analysis section"""
        header = Paragraph("Risk Analysis", self.styles['SectionHeader'])
        self.report_elements.append(header)
        
        max_drawdown = performance_metrics.get('Maximum Drawdown (%)', 0)
        sharpe_ratio = performance_metrics.get('Sharpe Ratio', 0)
        sortino_ratio = performance_metrics.get('Sortino Ratio', 0)
        
        # Risk assessment
        risk_level = "Low" if max_drawdown < 5 else "Medium" if max_drawdown < 15 else "High"
        sharpe_assessment = "Excellent" if sharpe_ratio > 2 else "Good" if sharpe_ratio > 1 else "Poor"
        
        risk_text = f"""
        <b>Risk Assessment:</b><br/>
        • Maximum Drawdown: {max_drawdown:.2f}% ({risk_level} Risk)<br/>
        • Sharpe Ratio: {sharpe_ratio:.2f} ({sharpe_assessment})<br/>
        • Sortino Ratio: {sortino_ratio:.2f}<br/>
        <br/>
        <b>Risk Recommendations:</b><br/>
        """
        
        if max_drawdown > 10:
            risk_text += "• Consider reducing position size to limit drawdowns<br/>"
        if sharpe_ratio < 1:
            risk_text += "• Strategy may not be providing adequate risk-adjusted returns<br/>"
        if len(trade_log_df) < 30:
            risk_text += "• Limited sample size - consider longer backtesting period<br/>"
        
        risk_para = Paragraph(risk_text, self.styles['Normal'])
        self.report_elements.append(risk_para)
        self.report_elements.append(Spacer(1, 0.3*inch))

    def add_model_insights(self, trend, ensemble_results=None):
        """Add model insights and predictions"""
        header = Paragraph("Model Insights & Predictions", self.styles['SectionHeader'])
        self.report_elements.append(header)
        
        # Base model insights
        insights_text = f"""
        <b>Primary Model Prediction:</b><br/>
        • Market Trend: <font color="{'green' if trend == 'positive' else 'red'}"><b>{trend.upper()}</b></font><br/>
        • Trading Signal: <font color="{'green' if trend == 'positive' else 'red'}"><b>{'BUY' if trend == 'positive' else 'SELL'}</b></font><br/>
        • Prediction Horizon: 7 days<br/>
        <br/>
        """
        
        # # Add ensemble insights if available
        # if ensemble_results:
        #     ensemble_predictions = ensemble_results['predictions']
        #     buy_votes = sum(1 for _, trend in ensemble_predictions if trend == 'positive')
        #     sell_votes = len(ensemble_predictions) - buy_votes
        #     consensus = 'BUY' if buy_votes > sell_votes else 'SELL'
        #     consensus_strength = max(buy_votes, sell_votes) / len(ensemble_predictions) * 100
            
        #     insights_text += f"""
        #     <br/>
        #     <b>Ensemble Model Analysis:</b><br/>
        #     • Models Trained: {len(ensemble_predictions)}<br/>
        #     • Agreement Strength: {consensus_strength:.1f}%<br/>
        #     """
        
        # insights_text += """
        # <br/>
        # <b>Important Notes:</b><br/>
        # • Predictions are based on historical patterns and may not reflect future market conditions<br/>
        # • Always consider fundamental analysis alongside technical predictions<br/>
        # • Risk management is crucial regardless of signal strength<br/>
        # """
        
        # # Add ensemble-specific warnings if consensus is weak
        # if ensemble_results:
        #     if consensus_strength < 60:
        #         insights_text += """
        #         • <font color="red"><b>Warning:</b></font> Ensemble consensus is weak - consider additional analysis<br/>
        #         """
        
        insights_para = Paragraph(insights_text, self.styles['Normal'])
        self.report_elements.append(insights_para)
        self.report_elements.append(Spacer(1, 0.3*inch))

    def add_disclaimer(self):
        """Add disclaimer section"""
        header = Paragraph("Disclaimer", self.styles['SectionHeader'])
        self.report_elements.append(header)
        
        disclaimer_text = """
        <b>Important Risk Warning:</b><br/>
        This report is for informational purposes only and should not be considered as financial advice. 
        Carbon market trading involves substantial risk and may not be suitable for all investors. 
        Past performance is not indicative of future results. 
        <br/><br/>
        The predictions and analysis contained in this report are based on historical data and 
        machine learning models, which may not accurately predict future market movements. 
        Always conduct your own research and consult with qualified financial professionals 
        before making investment decisions.
        <br/><br/>
        The authors and developers of this analysis system disclaim any liability for 
        trading losses that may result from the use of this information.
        """
        
        disclaimer_para = Paragraph(disclaimer_text, self.styles['Normal'])
        self.report_elements.append(disclaimer_para)

    def add_ensemble_analysis(self, ensemble_results):
        """
        Add ensemble analysis section to the report
        """
        if not ensemble_results:
            return
            
        header = Paragraph("Ensemble Model Analysis", self.styles['SectionHeader'])
        self.report_elements.append(header)
        
        ensemble_predictions = ensemble_results['predictions']
        preprocessor = ensemble_results['preprocessor']
        test_df = ensemble_results['test_df']
        num_models = ensemble_results['num_models']
        
        # Calculate ensemble statistics
        buy_votes = sum(1 for _, trend in ensemble_predictions if trend == 'positive')
        sell_votes = len(ensemble_predictions) - buy_votes
        consensus = 'BUY' if buy_votes > sell_votes else 'SELL'
        consensus_strength = max(buy_votes, sell_votes) / len(ensemble_predictions) * 100
        
        # Ensemble summary text
        ensemble_text = f"""
        <b>Ensemble Overview:</b><br/>
        • Number of Models: {num_models}<br/>
        • Consensus Signal: <font color="{'green' if consensus == 'BUY' else 'red'}"><b>{consensus}</b></font><br/>
        • Agreement Strength: {consensus_strength:.1f}%<br/>
        • Buy Votes: {buy_votes} models<br/>
        • Sell Votes: {sell_votes} models<br/>
        <br/>
        <b>Model Performance:</b><br/>
        • Individual models trained independently on the same dataset<br/>
        • Predictions averaged to reduce overfitting and improve robustness<br/>
        • Consensus voting provides confidence measure for trading signals<br/>
        """
        
        ensemble_para = Paragraph(ensemble_text, self.styles['Normal'])
        self.report_elements.append(ensemble_para)
        self.report_elements.append(Spacer(1, 0.2*inch))
        
        # Add ensemble consensus table
        consensus_data = [['Metric', 'Value']]
        consensus_data.append(['Total Models', str(num_models)])
        consensus_data.append(['Consensus Signal', consensus])
        consensus_data.append(['Agreement Strength', f"{consensus_strength:.1f}%"])
        consensus_data.append(['Models Voting BUY', str(buy_votes)])
        consensus_data.append(['Models Voting SELL', str(sell_votes)])
        
        # Assess consensus strength
        if consensus_strength >= 80:
            strength_assessment = "Very Strong"
        elif consensus_strength >= 60:
            strength_assessment = "Strong"
        elif consensus_strength >= 55:
            strength_assessment = "Moderate"
        else:
            strength_assessment = "Weak"
            
        consensus_data.append(['Consensus Quality', strength_assessment])
        
        table = Table(consensus_data, colWidths=[3*inch, 2*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
        ]))
        
        self.report_elements.append(table)
        self.report_elements.append(Spacer(1, 0.3*inch))

    def create_ensemble_predictions_chart(self, ensemble_results, output_path):
        """
        Create ensemble predictions visualization for PDF
        """
        if not ensemble_results:
            return None
            
        try:
            import matplotlib.pyplot as plt
            from utils.data_processing import reverse_normalize
            
            ensemble_predictions = ensemble_results['predictions'] 
            preprocessor = ensemble_results['preprocessor']
            test_df = ensemble_results['test_df']
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot historical data
            test_df_denorm = reverse_normalize(test_df.copy(), 
                                             preprocessor.train_mean['Auc Price'], 
                                             preprocessor.train_std['Auc Price'])
            plot_df = test_df_denorm.tail(60)
            ax.plot(plot_df.index, plot_df['Auc Price'], 
                   label='Historical Price', color='black', marker='o', markersize=2, linewidth=1.5)
            
            # Collect all predictions for averaging
            first_pred = ensemble_predictions[0][0]
            all_predictions = np.zeros((len(ensemble_predictions), len(first_pred)))
            
            # Plot individual model predictions
            colors = plt.cm.cool(np.linspace(0, 1, len(ensemble_predictions)))
            for i, (preds, trend) in enumerate(ensemble_predictions):
                normalized_preds = (preds['Auc Price'] * preprocessor.train_std['Auc Price']) + preprocessor.train_mean['Auc Price']
                all_predictions[i] = normalized_preds
                ax.plot(preds.index, normalized_preds, 
                       color=colors[i],
                       linestyle='--',
                       alpha=0.4,
                       linewidth=1)
            
            # Plot average prediction
            avg_predictions = np.mean(all_predictions, axis=0)
            ax.plot(first_pred.index, avg_predictions,
                   label='Ensemble Average',
                   color='red',
                   linewidth=2,
                   linestyle='-')
            
            # Add ensemble signal
            buy_votes = sum(1 for _, trend in ensemble_predictions if trend == 'positive')
            sell_votes = len(ensemble_predictions) - buy_votes
            signal = 'BUY' if buy_votes > sell_votes else 'SELL'
            signal_color = 'green' if signal == 'BUY' else 'red'
            marker = '^' if signal == 'BUY' else 'v'
            
            ax.scatter(first_pred.index[0], avg_predictions[0], 
                      color=signal_color, marker=marker, s=100, zorder=5,
                      label=f'Ensemble Signal: {signal}')
            
            ax.set_title('Ensemble Model Predictions (7-Day Forecast)', fontsize=14, fontweight='bold')
            ax.set_xlabel('Date')
            ax.set_ylabel('Price (€)')
            ax.legend(loc='upper left')
            ax.grid(True, alpha=0.3)
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return output_path
            
        except Exception as e:
            print(f"❌ Error creating ensemble predictions chart: {e}")
            return None

    def create_ensemble_statistics_chart(self, ensemble_results, output_path):
        """
        Create ensemble statistics visualization (box plot + consensus) for PDF
        Fixed version that correctly transposes the data for box plot
        """
        if not ensemble_results:
            return None
            
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            from utils.data_processing import reverse_normalize
            
            ensemble_predictions = ensemble_results['predictions']
            preprocessor = ensemble_results['preprocessor']
            
            # Create figure with subplots
            fig = plt.figure(figsize=(12, 5))
            gs = fig.add_gridspec(1, 1)
            # gs = fig.add_gridspec(1, 2, width_ratios=[7, 3])
            
            # Prepare data
            first_pred = ensemble_predictions[0][0]
            all_predictions = np.zeros((len(ensemble_predictions), len(first_pred)))
            
            # Collect predictions from each model
            for i, (preds, trend) in enumerate(ensemble_predictions):
                normalized_preds = (preds['Auc Price'] * preprocessor.train_std['Auc Price']) + preprocessor.train_mean['Auc Price']
                all_predictions[i] = normalized_preds
            
            # all_predictions shape: (num_models, num_days)
            # We need to pass each day's predictions separately to violinplot
            
            # Box plot (70% width)
            ax_box = fig.add_subplot(gs[0])
            
            # Create data list where each element is predictions for one day across all models
            data_for_violin = []
            for day in range(len(first_pred)):
                data_for_violin.append(all_predictions[:, day])
            
            # Create violin plot with box plot
            parts = ax_box.violinplot(data_for_violin, positions=range(1, len(first_pred) + 1), 
                                    showmeans=True, showmedians=True)
            bp = ax_box.boxplot(data_for_violin, positions=range(1, len(first_pred) + 1), 
                                patch_artist=True)
            
            # Style violin plot
            for pc in parts['bodies']:
                pc.set_facecolor('lightblue')
                pc.set_alpha(0.3)
            if 'cmeans' in parts:
                parts['cmeans'].set_color('red')
            if 'cmedians' in parts:
                parts['cmedians'].set_color('darkblue')
            
            # Style box plot with different colors for each day
            colors = plt.cm.Set3(np.linspace(0, 1, len(first_pred)))
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
            
            # Set labels for x-axis (prediction days)
            ax_box.set_xticks(range(1, len(first_pred) + 1))
            ax_box.set_xticklabels([f'Day {i}' for i in range(1, len(first_pred) + 1)])
            
            ax_box.set_title('Price Distribution by Prediction Day', fontweight='bold')
            ax_box.set_ylabel('Price (€)')
            ax_box.set_xlabel('Prediction Day')
            ax_box.grid(True, alpha=0.3)
            
            # Add statistics text below the plot
            mean_pred = np.mean(all_predictions, axis=0)
            std_pred = np.std(all_predictions, axis=0)
            cv = (std_pred/mean_pred) * 100
            
            stats_text = f"Mean Final Price: €{mean_pred[-1]:.2f} | Std Dev: €{std_pred[-1]:.2f} | CV: {cv[-1]:.1f}%"
            ax_box.text(0.5, -0.15, stats_text, transform=ax_box.transAxes, 
                    ha='center', fontsize=9, style='italic')
            
            # Consensus pie chart (30% width)
            # ax_consensus = fig.add_subplot(gs[1])
            # buy_votes = sum(1 for _, trend in ensemble_predictions if trend == 'positive')
            # sell_votes = len(ensemble_predictions) - buy_votes
            
            # colors = ['lightgreen' if buy_votes > sell_votes else 'lightcoral', 
            #         'lightcoral' if buy_votes > sell_votes else 'lightgreen']
            # sizes = [buy_votes, sell_votes]
            # labels = [f'BUY\n{buy_votes} votes', f'SELL\n{sell_votes} votes']
            
            # # Highlight the winning consensus
            # explode = (0.1, 0) if buy_votes > sell_votes else (0, 0.1)
            
            # ax_consensus.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
            #             startangle=90, pctdistance=0.85, explode=explode,
            #             wedgeprops=dict(width=0.5))
            # ax_consensus.set_title('Model Consensus', fontweight='bold')
            
            # # Add consensus strength indicator
            # consensus_strength = max(buy_votes, sell_votes) / len(ensemble_predictions) * 100
            # strength_text = "Strong" if consensus_strength > 70 else "Moderate" if consensus_strength > 60 else "Weak"
            # ax_consensus.text(0, -1.3, f"Consensus: {strength_text}", 
            #                 transform=ax_consensus.transAxes, ha='center', fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return output_path
        
        except Exception as e:
            print(f"❌ Error creating ensemble statistics chart: {e}")
            import traceback
            traceback.print_exc()
            return None

    def generate_pdf_report(self, timestamp=None, ensemble_results=None):
        """Generate the complete PDF report"""
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print("📄 Generating PDF report...")
        
        # Find the most recent analysis files
        try:
            performance_file = max(self.output_dir.glob("performance_metrics_*.csv"))
            trade_log_file = max(self.output_dir.glob("trade_log_*.csv"))
            
            # Load data
            performance_df = pd.read_csv(performance_file)
            performance_metrics = dict(zip(performance_df['Metric'], performance_df['Value']))
            
            trade_log_df = pd.read_csv(trade_log_file)
            trade_log_df['Entry Date'] = pd.to_datetime(trade_log_df['Entry Date'])
            trade_log_df['Exit Date'] = pd.to_datetime(trade_log_df['Exit Date'])
            
            # Get config (this would need to be passed in or stored)
            config = {
                'initial_balance': 10000.0,
                'take_profit': 0.04,
                'stop_loss': 0.03,
                'position_size_fraction': 1.0,
                'max_epochs': 40,
                'out_steps': 7
            }
            
            # Determine trend from summary file
            summary_files = list(self.output_dir.glob("analysis_summary_*.md"))
            trend = "positive"  # Default
            if summary_files:
                with open(max(summary_files), 'r') as f:
                    content = f.read()
                    if "SELL" in content:
                        trend = "negative"
            
        except Exception as e:
            print(f"❌ Error loading analysis files: {e}")
            return False
        
        # Create PDF
        pdf_path = self.output_dir / f"comprehensive_report_{timestamp}.pdf"
        doc = SimpleDocTemplate(str(pdf_path), pagesize=A4, 
                              rightMargin=72, leftMargin=72, 
                              topMargin=72, bottomMargin=18)
        
        # Build report content
        self.report_elements = []
        
        # Add all sections
        self.add_title_page(config)
        self.add_executive_summary(performance_metrics, trade_log_df, trend)
        # self.add_performance_metrics_table(performance_metrics)
        # self.add_trade_summary(trade_log_df)
        # self.add_recent_trades_table(trade_log_df)
        
        # Add charts
        # equity_chart = max(self.output_dir.glob("equity_curve_*.png"), default=None)
        # if equity_chart:
        #     self.add_chart_image(equity_chart, "Equity Curve Analysis")
        
        predictions_chart = max(self.output_dir.glob("predictions_*.png"), default=None)
        if predictions_chart:
            self.add_chart_image(predictions_chart, "Price Predictions & Signals")
        
        trades_chart = max(self.output_dir.glob("trades_visualization_*.png"), default=None)
        if trades_chart:
            self.add_chart_image(trades_chart, "Trade Visualization")
        
        # Add ensemble analysis if available
        if ensemble_results:
            # self.add_ensemble_analysis(ensemble_results)
            
            # Create and add ensemble charts
            ensemble_pred_chart = self.output_dir / f"ensemble_predictions_{timestamp}.png"
            if self.create_ensemble_predictions_chart(ensemble_results, ensemble_pred_chart):
                self.add_chart_image(ensemble_pred_chart, "Ensemble Model Predictions")
            
            ensemble_stats_chart = self.output_dir / f"ensemble_statistics_{timestamp}.png"
            if self.create_ensemble_statistics_chart(ensemble_results, ensemble_stats_chart):
                self.add_chart_image(ensemble_stats_chart, "Ensemble Statistical Analysis")
        
        # Add analysis sections
        # self.add_risk_analysis(performance_metrics, trade_log_df)
        # self.add_model_insights(trend, ensemble_results)
        # self.add_disclaimer()
        
        # Build PDF
        doc.build(self.report_elements)
        
        print(f"✅ PDF report generated: {pdf_path}")
        return pdf_path

def generate_pdf_report(output_dir="output_plots"):
    """
    Convenience function to generate PDF report
    """
    generator = PDFReportGenerator(output_dir)
    
    # Try to load ensemble results if they exist (from a potential ensemble file)
    ensemble_results = None
    try:
        # Check if there are any ensemble result files
        import pickle
        ensemble_files = list(Path(output_dir).glob("ensemble_results_*.pkl"))
        if ensemble_files:
            latest_ensemble_file = max(ensemble_files)
            with open(latest_ensemble_file, 'rb') as f:
                ensemble_results = pickle.load(f)
            print(f"📊 Loaded ensemble results from {latest_ensemble_file}")
    except Exception as e:
        print(f"⚠️ No ensemble results found: {e}")
    
    return generator.generate_pdf_report(ensemble_results=ensemble_results)

if __name__ == "__main__":
    # Generate report for the most recent analysis
    pdf_path = generate_pdf_report()
    if pdf_path:
        print(f"📄 PDF report generated successfully: {pdf_path}")
    else:
        print("❌ Failed to generate PDF report")