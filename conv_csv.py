import pandas as pd

# Read CSV file as DataFrame
csv_names = [
    'lstm_err_metric.csv',
    'lstm_err_metric_no_volatility.csv',
    'lstm_err_metric_no_momentum.csv',
    'lstm_err_metric_no_market_context.csv'
]

def format_csvs():
    for csv_name in csv_names:
        df = pd.read_csv(csv_name)

        # Convert accuracy column to percentage and round to 2 decimal places
        df['accuracy'] = (df['accuracy'] * 100).round(2)
        df['f1_score'] = (df['f1_score']).round(3)
        df['auc'] = (df['auc']).round(3)

        # Rename accuracy column to accuracy(%)
        df.rename(columns={'ticker': 'Ticker'}, inplace=True)
        df.rename(columns={'accuracy': 'Accuracy(%)'}, inplace=True)
        df.rename(columns={'f1_score': 'F1-Score'}, inplace=True)
        df.rename(columns={'auc': 'AUC'}, inplace=True)

        # Display the first few rows
        print(df.head())

        # Write DataFrame to CSV file
        df.to_csv(f"conv_{csv_name}.csv", index=False)

def mean_csvs():
    df_mean_f1 = pd.DataFrame(columns=["Feature Set", "Mean F1"])
    for i, csv_name in enumerate(csv_names):
        df = pd.read_csv(csv_name)
        mean_f1_score = (df['f1_score']).mean().round(3)

        feature_set = csv_name.replace('lstm_err_metric', '').replace('.csv', '').replace('_', ' ').strip()
        if feature_set == '':
            feature_set = 'Full'

        row = {"Feature Set": feature_set, "Mean F1": mean_f1_score}
        df_mean_f1 = pd.concat([df_mean_f1, pd.DataFrame([row])], ignore_index=True)
    print(df_mean_f1)
    df_mean_f1.to_csv("lstm_ablation_mean_f1_scores.csv", index=False)

if __name__ == "__main__":
    # format_csvs()
    mean_csvs()