import argparse
from subprocess import call

import pandas as pd


def main(args):
    articles = pd.read_csv(args.input)
    list_texts = articles['text_en'].tolist()

    # If the number of articles is too long it will not accept it as parameter and we will specify the csv file
    if len(list_texts) > 10:
        list_texts = args.input
        print(list_texts)
    save_file = "../data/sentiment_selection_pytorch.pkl"

    # Sentences mentioned in Sutskever et al. 2017 to check if results are reproducible
    control_texts = [
        "I found this to be a charming adaptation, very lively and full of fun. With the exception of a couple of major errors, the cast is wonderful. I have to echo some of the earlier comments -- Chynna Phillips is horribly miscast as a teenager. At 27, she's just too old (and, yes, it DOES show), and lacks the singing 'chops' for Broadway-style music. Vanessa Williams is a decent-enough singer and, for a non-dancer, she's adequate. However, she is NOT Latina, and her character definitely is. She's also very STRIDENT throughout, which gets tiresome. The girls of Sweet Apple's Conrad Birdie fan club really sparkle -- with special kudos to Brigitta Dau and Chiara Zanni. I also enjoyed Tyne Daly's performance, though I'm not generally a fan of her work. Finally, the dancing Shriners are a riot, especially the dorky three in the bar. The movie is suitable for the whole family, and I highly recommend it.",
        "Just what I was looking for. Nice fitted pants, exactly matched seam to color contrast with other pants I own. Highly recommended and also very happy!",
        "The package received was blank and has no barcode. A waste of time and money."]

    if args.n == 'test':
        list_texts = list_texts[:2]
    elif args.n == 'control':
        save_file = "../data/sentiment_control_pytorch.pkl"
        list_texts = control_texts

    call(["python", args.path + "visualize.py", "-seq_length", "0", "-load_model", args.path + "mlstm_ns.pt",
          "-temperature", "0.8", "-neuron", "2388", "-init", str(list_texts), "-mode", "save_set", "-pickle",
          save_file])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run sentiment analysis with trained LSTM model",
                                     usage="python3 pytorch_sentiment.py --input path_to_articles.csv --path path_to_model")

    parser.add_argument('--input',
                        default="/Users/celsadiaz/github/NLP_politics_spiegel/data/330_selected_art_translated_correct.csv",
                        help='path/330_selected_art_translated.csv')
    parser.add_argument('--path', default="/Users/celsadiaz/github/pytorch-sentiment-neuron/",
                        help='path_to_root_model/')
    parser.add_argument('--n', default="all",
                        help='option to calculate sentiment of a "test" sample of "all"')
    args = parser.parse_args()

    main(args)
