import argparse
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results-file', type=str, default='results/results.csv')
    args = ap.parse_args()

    df = pd.read_csv(args.results_file)
    group_cols = ['dataset', 'mixup_alpha']
    metrics = ['test_acc', 'test_loss', 'ece_test']
    summary = df.groupby(group_cols)[metrics].agg(['mean', 'std']).round(4)
    print(summary)

if __name__ == "__main__":
    main()
