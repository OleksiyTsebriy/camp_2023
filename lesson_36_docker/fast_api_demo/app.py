import io
from datetime import datetime, timezone
from typing import Dict, List, Union

import cv2
import gradio as gr
import numpy as np
import uvicorn
from fastapi import FastAPI, UploadFile, status
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, computed_field
from ultralytics import YOLO

from view import router as view_router


class Customer(BaseModel):
    name: Union[str, None] = None
    gender: Union[str, None] = None
    birth_date: Union[datetime, None] = None
    description: Union[str, None] = None

    @computed_field
    @property
    def age(self) -> int:
        return (datetime.now(timezone.utc) - self.birth_date).days // 365 if self.birth_date is not None else 0


app = FastAPI()
CUSTOMERS: Dict[int, Customer] = {0: Customer(name='admin', gender='M',
                                              birth_date=datetime(1970, 1, 1, tzinfo=timezone.utc))}


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/customer/{customer_id}")
def get_customer(customer_id: int):
    data = CUSTOMERS.get(customer_id, {})
    return data


@app.get("/customer")
def get_customers():
    return CUSTOMERS


@app.post("/customer")
def create_customer(customer: Customer):
    customer_id = max(CUSTOMERS.keys()) + 1
    CUSTOMERS[customer_id] = customer
    return {"message": "Created", "customer_id": customer_id}


@app.post("/customer/{customer_id}")
def rewrite_customer(customer_id: int, customer: Customer):
    CUSTOMERS[customer_id] = customer
    return {"message": "Created", "customer_id": customer_id}


@app.put("/customer/{customer_id}")
def update_customer(customer_id: int, customer: Customer):
    if customer_id not in CUSTOMERS:
        return JSONResponse(content={"message": f"Customer with ID {customer_id} not found"},
                            status_code=status.HTTP_404_NOT_FOUND)

    update_data = customer.model_dump(exclude_unset=True)
    CUSTOMERS[customer_id] = CUSTOMERS[customer_id].model_copy(update=update_data)
    return {"message": "Updated"}


@app.delete("/customer/{customer_id}")
def remove_customer(customer_id: int):
    if customer_id not in CUSTOMERS:
        return JSONResponse(content={"message": f"Customer with ID {customer_id} not found"},
                            status_code=status.HTTP_404_NOT_FOUND)
    CUSTOMERS.pop(customer_id)
    return {"message": "Removed"}


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


app.include_router(view_router)


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
    uvicorn.run('app:app')
