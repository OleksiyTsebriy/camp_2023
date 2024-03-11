import io

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, UploadFile, status
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from ultralytics import YOLO
import gradio as gr


app = FastAPI()


@app.get("/")
def index():
    return {"message": "OK"}


# Example of ML part

yolo_detect = YOLO()
names_to_int = {name: id_ for id_, name in yolo_detect.names.items()}


@app.post('/ml/get_yolo_objects')
def get_yolo_objects(image: UploadFile) -> StreamingResponse:
    """Inferences YOLO model on the input image

    Parameters
    ----------
    image : UploadFile
        Input image as bytes

    Returns
    -------
    StreamingResponse
        Image with plotted objects
    """
    contents = image.file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    res_img = yolo_detect(image)[0].plot()
    _, res_img = cv2.imencode(".jpg", res_img)

    return StreamingResponse(content=io.BytesIO(res_img.tobytes()), media_type='image/jpg')


def detect_image(image):
    res_img = yolo_detect(image)[0].plot()
    return cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB)


with gr.Blocks(title="YOLO detection inference") as demo:
    gr.HTML("""<div style="font-family:'Times New Roman', 'Serif'; font-size:16pt; font-weight:bold; text-align:center; color:royalblue;">YOLO</div>""")

    input_image = gr.Image(type="pil", image_mode="RGB", shape=(640, 640))
    output_image = gr.Image(type="numpy", image_mode="RGB", shape=(640, 640))

    send_btn = gr.Button("Infer")
    send_btn.click(fn=detect_image, inputs=input_image, outputs=output_image)


app = gr.mount_gradio_app(app, demo, path='/gradio')


if __name__ == "__main__":
    uvicorn.run(app)
