import io

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


app.segmenter = Segmenter("Model.pth", "GoogleNews-vectors-negative300.bin.gz", 0.4)

if __name__ == '__main__':
    app.run(debug=False, host='127.0.0.1', port=5000)
