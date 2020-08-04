import io
from argparse import ArgumentParser

import flask

from ml_model.segmenter import Segmenter

app = flask.Flask(__name__)


@app.route('/')
def index():
    return flask.render_template('index.html')


@app.route('/segment_text', methods=['GET'])
def classify_url():
    title = flask.request.args.get('title', '')
    paragraph = flask.request.args.get('paragraph', '')
    threshold = float(flask.request.args.get('threshold', ''))

    result = app.segmenter.segment(paragraph, threshold)
    return flask.send_file(
        io.BytesIO(bytes(flask.render_template('website.html', result=result, title=title), 'utf-8')),
        mimetype='text/html',
        as_attachment=True,
        attachment_filename=f'{title}.html')


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
        type=str)
    parser.add_argument(
        '--model',
        help='Segmentation model path',
        type=str)
    parser.add_argument(
        '--seg_threshold',
        help='Default threshold for binary classification',
        type=float, default=0.4)
    args = parser.parse_args()

    app.segmenter = Segmenter(args)
    app.run(debug=args.debug, host='127.0.0.1', port=args.port)


if __name__ == '__main__':
    start_app(app)
