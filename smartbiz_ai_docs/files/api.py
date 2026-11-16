"""
SmartBiz Assistant as a web service, so your mobile app or WhatsApp bot can call it.

Install:  pip install fastapi uvicorn pandas
Run:      SMARTBIZ_DATA=smartbiz_daily_business_metrics.csv uvicorn api:app --reload
Try:      curl -X POST localhost:8000/ask -H 'Content-Type: application/json' \
               -d '{"business_id": "B001", "message": "how did we do yesterday"}'
"""
import os

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from smartbiz_assistant import Assistant

DATA = os.environ.get('SMARTBIZ_DATA', 'smartbiz_daily_business_metrics.csv')
app = FastAPI(title='SmartBiz Assistant')
_frames = {bid: g for bid, g in pd.read_csv(DATA).groupby('business_id')}
_sessions = {}          # one conversation per business (use a database or cache in production)


class Question(BaseModel):
    business_id: str
    message: str
    owner: str | None = None


@app.get('/health')
def health():
    return {'status': 'ok', 'businesses': len(_frames)}


@app.post('/ask')
def ask(q: Question):
    if q.business_id not in _frames:
        raise HTTPException(404, 'Unknown business_id')
    bot = _sessions.setdefault(q.business_id, Assistant(_frames[q.business_id], owner=q.owner))
    return {'reply': bot.ask(q.message)}
