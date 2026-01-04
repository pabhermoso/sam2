# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Any, Generator

from app_conf import (
    GALLERY_PATH,
    GALLERY_PREFIX,
    POSTERS_PATH,
    POSTERS_PREFIX,
    UPLOADS_PATH,
    UPLOADS_PREFIX,
)
from data.loader import preload_data
from data.schema import schema
from data.store import set_videos
from flask import Flask, make_response, Request, request, Response, send_from_directory
from flask_cors import CORS
from inference.data_types import PropagateDataResponse, PropagateInVideoRequest
from inference.image_segmentor import ImageSegmentor
from inference.multipart import MultipartResponseBuilder
from inference.predictor import InferenceAPI
from strawberry.flask.views import GraphQLView

logger = logging.getLogger(__name__)

app = Flask(__name__)
cors = CORS(app, supports_credentials=True)

videos = preload_data()
set_videos(videos)

inference_api = InferenceAPI()
image_segmentor = ImageSegmentor()


@app.route("/healthy")
def healthy() -> Response:
    return make_response("OK", 200)


@app.route(f"/{GALLERY_PREFIX}/<path:path>", methods=["GET"])
def send_gallery_video(path: str) -> Response:
    try:
        return send_from_directory(
            GALLERY_PATH,
            path,
        )
    except:
        raise ValueError("resource not found")


@app.route(f"/{POSTERS_PREFIX}/<path:path>", methods=["GET"])
def send_poster_image(path: str) -> Response:
    try:
        return send_from_directory(
            POSTERS_PATH,
            path,
        )
    except:
        raise ValueError("resource not found")


@app.route(f"/{UPLOADS_PREFIX}/<path:path>", methods=["GET"])
def send_uploaded_video(path: str):
    try:
        return send_from_directory(
            UPLOADS_PATH,
            path,
        )
    except:
        raise ValueError("resource not found")


# TOOD: Protect route with ToS permission check
@app.route("/propagate_in_video", methods=["POST"])
def propagate_in_video() -> Response:
    data = request.json
    args = {
        "session_id": data["session_id"],
        "start_frame_index": data.get("start_frame_index", 0),
    }

    boundary = "frame"
    frame = gen_track_with_mask_stream(boundary, **args)
    return Response(frame, mimetype="multipart/x-savi-stream; boundary=" + boundary)


@app.route("/segment_image", methods=["POST"])
def segment_image() -> Response:
    """
    Segment a single image using click prompts.
    Body: {
      "image": "<base64 or data URL>",
      "points": [{ "x": float, "y": float, "label": 1|0 }, ...]  # 1=fg, 0=bg
      # legacy: "point": { "x": float, "y": float }
    }
    """
    data = request.json or {}
    image_b64 = data.get("image")
    points = data.get("points")
    if points is None and data.get("point"):
        # Backward compatibility with single point payload
        pt = data.get("point")
        points = [{"x": pt.get("x"), "y": pt.get("y"), "label": 1}]
    if points is None:
        points = []
    if not image_b64 or not isinstance(points, list) or len(points) == 0:
        return make_response({"error": "image and point(s) are required"}, 400)

    try:
        coords = []
        labels = []
        for pt in points:
            if not isinstance(pt, dict):
                continue
            if "x" in pt and "y" in pt and "label" in pt:
                coords.append((float(pt["x"]), float(pt["y"])))
                labels.append(int(pt["label"]))
            elif "x" in pt and "y" in pt:
                coords.append((float(pt["x"]), float(pt["y"])))
                labels.append(1)

        if len(coords) == 0:
            return make_response({"error": "no valid points provided"}, 400)

        png_b64, bbox = image_segmentor.segment(
            image_b64=image_b64,
            points=tuple(coords),
            labels=tuple(labels),
            pad=10,
        )
        return make_response(
            {
                "png_base64": png_b64,
                "bbox": {
                    "min_x": bbox[0],
                    "min_y": bbox[1],
                    "max_x": bbox[2],
                    "max_y": bbox[3],
                },
            },
            200,
        )
    except Exception as exc:
        logger.exception("segment_image failed")
        return make_response({"error": str(exc)}, 500)


def gen_track_with_mask_stream(
    boundary: str,
    session_id: str,
    start_frame_index: int,
) -> Generator[bytes, None, None]:
    with inference_api.autocast_context():
        request = PropagateInVideoRequest(
            type="propagate_in_video",
            session_id=session_id,
            start_frame_index=start_frame_index,
        )

        for chunk in inference_api.propagate_in_video(request=request):
            yield MultipartResponseBuilder.build(
                boundary=boundary,
                headers={
                    "Content-Type": "application/json; charset=utf-8",
                    "Frame-Current": "-1",
                    # Total frames minus the reference frame
                    "Frame-Total": "-1",
                    "Mask-Type": "RLE[]",
                },
                body=chunk.to_json().encode("UTF-8"),
            ).get_message()


class MyGraphQLView(GraphQLView):
    def get_context(self, request: Request, response: Response) -> Any:
        return {"inference_api": inference_api}


# Add GraphQL route to Flask app.
app.add_url_rule(
    "/graphql",
    view_func=MyGraphQLView.as_view(
        "graphql_view",
        schema=schema,
        # Disable GET queries
        # https://strawberry.rocks/docs/operations/deployment
        # https://strawberry.rocks/docs/integrations/flask
        allow_queries_via_get=False,
        # Strawberry recently changed multipart request handling, which now
        # requires enabling support explicitly for views.
        # https://github.com/strawberry-graphql/strawberry/issues/3655
        multipart_uploads_enabled=True,
    ),
)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
