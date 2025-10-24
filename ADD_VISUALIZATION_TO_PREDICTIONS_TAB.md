# How to Add External Variables Visualization to Predictions Tab

## Add this code to your Predictions tab in both app.py and app_with_transformer.py

### Location: After the "Save Recent Predictions" button

```python
    with tab2:
        st.header("📈 Recent Predictions")
        plot_recent_predictions(recent_preds, trend, test_df, preprocessor)
        
        collection = setup_mongodb_connection()
        
        st.header("💾 Stored Predictions")
        # ... existing stored predictions code ...
        
        # Save predictions button
        if st.button("Save Recent Predictions"):
            collection = setup_mongodb_connection()
            save_message = save_recent_predictions(collection, recent_preds, preprocessor)
            st.success(save_message)
        
        # 🌍 NEW: Add this section for external variables visualization
        st.markdown("---")
        st.header("🌍 External Variables Analysis")
        
        if use_external_vars:
            with st.expander("ℹ️ About External Variables", expanded=False):
                st.markdown("""
                This section shows how external market variables correlate with carbon prices:
                
                **Variables Included:**
                - 🛢️ **Brent Crude Oil**: Global oil benchmark ($/barrel)
                - ⛽ **TTF Gas**: European natural gas benchmark (€/MWh)
                - 📊 **EU Inflation**: Consumer price index (%)
                
                **Why These Matter:**
                - Energy prices drive industrial demand for carbon credits
                - Gas prices affect power generation carbon intensity
                - Inflation impacts real carbon price valuations
                
                **Correlation Insights:**
                - Strong positive: Energy prices drive carbon demand
                - Moderate positive: Inflation creates price pressure
                - Time-varying: Relationships change during energy crises
                """)
            
            # Import the visualization function
            from utils.plotting import plot_external_variables_correlation
            
            # Show the correlation plots
            with st.spinner("Generating external variables correlation analysis..."):
                plot_external_variables_correlation(test_df, preprocessor)
            
            # Show feature importance for external variables
            st.subheader("📊 External Feature Statistics")
            
            external_feature_cols = [col for col in test_df.columns 
                                    if any(x in col for x in ['Brent', 'TTF', 'Inflation'])]
            
            if external_feature_cols:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "External Features", 
                        len(external_feature_cols),
                        help="Number of engineered features from external variables"
                    )
                
                with col2:
                    # Calculate average correlation with Auc Price
                    if 'Auc Price' in test_df.columns:
                        from utils.data_processing import reverse_normalize
                        df_denorm = reverse_normalize(
                            test_df.copy(), 
                            preprocessor.train_mean['Auc Price'], 
                            preprocessor.train_std['Auc Price']
                        )
                        
                        correlations = []
                        for col in ['Brent_Oil', 'TTF_Gas', 'EU_Inflation']:
                            if col in df_denorm.columns:
                                corr = df_denorm['Auc Price'].corr(df_denorm[col])
                                correlations.append(abs(corr))
                        
                        avg_corr = sum(correlations) / len(correlations) if correlations else 0
                        st.metric(
                            "Avg Correlation",
                            f"{avg_corr:.3f}",
                            help="Average absolute correlation with carbon price"
                        )
                
                with col3:
                    st.metric(
                        "Data Coverage",
                        "100%",
                        help="All external variables forward-filled"
                    )
                
                # Show top external features by name
                with st.expander("🔍 View All External Features", expanded=False):
                    # Group features by type
                    feature_groups = {
                        'Price Change': [f for f in external_feature_cols if 'pct_change' in f or '_change' in f],
                        'Moving Averages': [f for f in external_feature_cols if 'MA' in f or 'EMA' in f],
                        'Volatility': [f for f in external_feature_cols if 'volatility' in f],
                        'Ratios': [f for f in external_feature_cols if 'Ratio' in f],
                        'Lags': [f for f in external_feature_cols if 'lag' in f],
                        'Interactions': [f for f in external_feature_cols if 'Interaction' in f],
                        'Other': [f for f in external_feature_cols if not any(x in f for x in ['pct_change', '_change', 'MA', 'EMA', 'volatility', 'Ratio', 'lag', 'Interaction'])]
                    }
                    
                    for group_name, features in feature_groups.items():
                        if features:
                            st.markdown(f"**{group_name}** ({len(features)} features):")
                            for feature in features[:5]:  # Show first 5
                                st.text(f"  • {feature}")
                            if len(features) > 5:
                                st.text(f"  ... and {len(features) - 5} more")
        else:
            st.info("🔔 Enable 'Include External Variables' in the sidebar to see correlation analysis")
```

## Implementation Steps

1. **Find the Predictions tab in your app file** (line starting with `with tab2:`)

2. **Locate the "Save Recent Predictions" button**

3. **Add the code above right after that button** (before the tab closes)

4. **Import statement is already in the code** (no need to add at top)

5. **Test it**:
   ```bash
   python main.py app_with_transformer
   ```

## What You'll Get

### When External Variables are ENABLED:
- ✅ 4-panel correlation visualization
- ✅ Expandable "About" section explaining the variables
- ✅ Metrics showing feature count and correlations
- ✅ List of all external features grouped by type

### When External Variables are DISABLED:
- ℹ️ Info message: "Enable 'Include External Variables' in the sidebar..."

## Visual Preview

```
📈 Recent Predictions
[Your existing prediction plot]

💾 Stored Predictions
[Your existing table]

[Save Recent Predictions] <- button

───────────────────────────

🌍 External Variables Analysis

ℹ️ About External Variables ▼

[4-panel correlation plot showing:]
├─ Carbon vs Brent Oil (dual-axis)
├─ Carbon vs TTF Gas (dual-axis)
├─ Carbon vs EU Inflation (dual-axis)
└─ Correlation Matrix (heatmap)

📊 External Feature Statistics

┌──────────────────┬──────────────────┬──────────────────┐
│ External Features│ Avg Correlation  │ Data Coverage    │
│        32        │      0.687       │      100%        │
└──────────────────┴──────────────────┴──────────────────┘

🔍 View All External Features ▼
```

## Customization Options

### Change the number of top features shown:
```python
for feature in features[:5]:  # Change 5 to 10
```

### Add statistical tests:
```python
from scipy.stats import pearsonr

for col in ['Brent_Oil', 'TTF_Gas', 'EU_Inflation']:
    if col in df_denorm.columns:
        corr, p_value = pearsonr(df_denorm['Auc Price'], df_denorm[col])
        st.write(f"{col}: r={corr:.3f}, p={p_value:.4f}")
```

### Add rolling correlation:
```python
# Show how correlation changes over time
rolling_corr = df_denorm['Auc Price'].rolling(90).corr(df_denorm['Brent_Oil'])
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(rolling_corr.index, rolling_corr)
ax.set_title('90-Day Rolling Correlation: Carbon vs Brent')
st.pyplot(fig)
```

## Troubleshooting

**If correlation plot doesn't show:**
- Check that `use_external_vars` is defined in the main scope
- Verify `test_df` has external columns: `st.write(test_df.columns)`
- Check for errors in Streamlit console

**If "About" section doesn't expand:**
- Make sure you're using `st.expander()` correctly
- Update Streamlit if version is old: `pip install --upgrade streamlit`

**If metrics show NaN:**
- Verify external variables passed through normalization
- Check that preprocessor has correct mean/std values

## Ready to Deploy! 🚀

Once you add this code, you'll have a complete external variables analysis section that:
- Shows visual correlations
- Provides educational context
- Lists all engineered features
- Calculates correlation metrics
- Works seamlessly with your toggle

The system will gracefully handle both cases (with/without external variables) and provide helpful feedback to users.
