import io
from datetime import datetime, timezone
from typing import Dict, List, Union

import cv2
import numpy as np
from pydantic import BaseModel, computed_field
from ultralytics import YOLO
from flask import Flask, request, send_file

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


app = Flask(__name__)
CUSTOMERS: Dict[int, Customer] = {0: Customer(name='admin', gender='M', 
                                              birth_date=datetime(1970, 1, 1, tzinfo=timezone.utc))}


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/customer/<customer_id>")
def get_customer(customer_id=None):
    return CUSTOMERS if customer_id is None else CUSTOMERS.get(customer_id, {})


@app.post("/customer")
def create_customer(customer: Customer):
    req_json = request.get_json()
    birth_date = req_json.get('birth_date', None)
    if birth_date is not None:
        birth_date = datetime.fromisoformat(birth_date)
        
    customer = Customer(name=req_json.get('name', None), gender=req_json.get('name', None),
                        birth_date=birth_date, description=req_json.get('description', None))
    customer_id = max(CUSTOMERS.keys()) + 1
    CUSTOMERS[customer_id] = customer
    return {"message": "Created", "customer_id": customer_id}


@app.post("/customer/<customer_id>")
def rewrite_customer(customer_id: int):
    req_json = request.get_json()
    birth_date = req_json.get('birth_date', None)
    if birth_date is not None:
        birth_date = datetime.fromisoformat(birth_date)
        
    customer = Customer(name=req_json.get('name', None), gender=req_json.get('name', None),
                        birth_date=birth_date, description=req_json.get('description', None))

    CUSTOMERS[customer_id] = customer
    return {"message": "Created", "customer_id": customer_id}


@app.put("/customer/<customer_id>")
def update_customer(customer_id: int):
    if customer_id not in CUSTOMERS:
        return {"message": f"Customer with ID {customer_id} not found"}, 404
    
    req_json = request.get_json()
    birth_date = req_json.get('birth_date', None)
    if birth_date is not None:
        birth_date = datetime.fromisoformat(birth_date)
        
    customer = Customer(name=req_json.get('name', None), gender=req_json.get('name', None),
                        birth_date=birth_date, description=req_json.get('description', None))
    
    update_data = customer.model_dump(exclude_unset=True)
    CUSTOMERS[customer_id] = CUSTOMERS[customer_id].model_copy(update=update_data)
    return {"message": "Updated"}


@app.delete("/customer/<customer_id>")
def remove_customer(customer_id: int):
    if customer_id not in CUSTOMERS:
        return {"message": f"Customer with ID {customer_id} not found"}, 404
    
    CUSTOMERS.pop(customer_id)
    return {"message": "Removed"}


# Example of ML part

yolo_detect = YOLO()
names_to_int = {name: id_ for id_, name in yolo_detect.names.items()}


@app.post('/ml/get_yolo_objects')
def get_yolo_objects():
    contents = request.files['image'].read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    res_img = yolo_detect(image)[0].plot()
    _, res_img = cv2.imencode(".jpg", res_img)
    
    return send_file(io.BytesIO(res_img.tobytes()), mimetype='image/jpg')


app.register_blueprint(view_router)


if __name__ == "__main__":
    # Needs to be replaced with WSGI deployment in production
    app.run()
