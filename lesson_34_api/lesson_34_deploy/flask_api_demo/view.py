from flask import Blueprint, render_template
from pathlib import Path
from datetime import date


ROOT_PATH = Path(__file__).parent
router = Blueprint('view', __name__, url_prefix='/view')


@router.get('/')
def index():
    return render_template('index.html', time=date.today())
