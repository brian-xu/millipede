from argparse import ArgumentParser

import flask

from ml_model.segmenter import Segmenter

app = flask.Flask(__name__)


def start_app(app):
    parser = ArgumentParser()
    parser.add_argument(
        '--debug',
        help="Enable debug mode",
        action="store_true", default=False)
    parser.add_argument(
        '--port',
        help="Which port to serve content on",
        type=int, default=5000)
    parser.add_argument(
        '--word2vec',
        help='Word2vec model path',
        ype=str)
    parser.add_argument(
        '--model',
        help='Segmentation model path',
        type=str)
    parser.add_argument(
        '--seg_threshold',
        help='Threshold for binary classification',
        type=float, default=0.4)
    args = parser.parse_args()

    app.segmenter = Segmenter(args)
    app.run(debug=True, host='127.0.0.1', port=args.port)


if __name__ == '__main__':
    start_app(app)
