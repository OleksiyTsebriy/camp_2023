from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pathlib import Path
from datetime import date


ROOT_PATH = Path(__file__).parent
router = APIRouter(prefix='/view', tags=['view'])
templates = Jinja2Templates(directory=ROOT_PATH / "templates")


@router.get('/', response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request=request, name='index.html', context={'time': date.today()})
