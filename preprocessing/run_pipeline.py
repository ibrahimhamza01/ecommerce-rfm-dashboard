from preprocessing.data_loader import run_full_pipeline

results = run_full_pipeline("data/online_retail_II.csv")

results["features"].to_csv("data/processed_transactions.csv", index=False)
results["rfm_no_outliers"].to_csv("data/rfm_table.csv", index=False)
results["pca_df"].to_csv("data/rfm_pca.csv", index=False)

print("All processed data saved successfully.")