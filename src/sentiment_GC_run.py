import sys
import argparse
import pickle
import pandas as pd
sys.path.append("../../generating-reviews-discovering-sentiment/")
from encoder import Model


def main():
    parser = argparse.ArgumentParser(description="Run sentiment analysis with trained LSTM model",
                                     usage="python3 sentiment_GC_run.py --input path_to_articles.csv --path path_to_model")

    parser.add_argument('--input', default="/home/celsadiaz/NLP_politics_spiegel/data/330_selected_art_translated.csv",
                        help='path/330_selected_art_translated.csv')
    parser.add_argument('--path', default="/home/celsadiaz/generating-reviews-discovering-sentiment/",
                        help='path_to_root_model/')
    parser.add_argument('--n', default="all",
                        help='option to calculate sentiment of a "test" sample of "all"')
    print("yeah")
    # Load translated articles to perform the sentiment analysis to translated texts
    print("start")
    args = parser.parse_args()
    articles = pd.read_csv(args.input)
    print(articles.head())
    if args.n == 'test':
        articles = articles.loc[:3]

    # Initialize an instance of the model
    model = Model(root_path=args.path)

    results = []
    for i, text in articles["text"].iteritems():
        print("start transforming text")
        # Run LSTM model to predict final hidden units' values
        text_features = model.transform(text)
        print("text transformed")
        # Extract content from sentiment hidden unit 2388
        results.append(text_features[:, 2388])
        print(f"text {i} analyzed")
        pickle.dump(results, open("../data/sentiment_analysis_scores_test.pkl", "wb"))

    pickle.dump(results, open("../data/sentiment_analysis_scores.pkl", "wb"))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run sentiment analysis with trained LSTM model",
                                     usage="python3 sentiment_GC_run.py --input path_to_articles.csv --path path_to_model")

    parser.add_argument('--input', default="/home/celsadiaz/NLP_politics_spiegel/data/330_selected_art_translated.csv",
                        help='path/330_selected_art_translated.csv')
    parser.add_argument('--path', default="/home/celsadiaz/generating-reviews-discovering-sentiment/",
                        help='path_to_root_model/')
    parser.add_argument('--n', default="all",
                        help='option to calculate sentiment of a "test" sample of "all"')
    args = parser.parse_args()

    main(args)
