from datetime import datetime
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def _get_db():
    from app import db
    return db

class User:
    pass

# Rebuild as real model lazily
def _build(db):
    class UserModel(db.Model):
        __tablename__ = 'users'
        id = db.Column(db.Integer, primary_key=True)
        username = db.Column(db.String(80), nullable=False)
        email = db.Column(db.String(120), unique=True, nullable=False)
        password_hash = db.Column(db.String(256), nullable=False)
        role = db.Column(db.String(20), default='user')
        created_at = db.Column(db.DateTime, default=datetime.utcnow)
        predictions = db.relationship('Prediction', backref='user', lazy=True)
        def to_dict(self):
            return {'id':self.id,'username':self.username,'email':self.email,'role':self.role,'created_at':self.created_at.isoformat(),'prediction_count':len(self.predictions)}
    return UserModel

import importlib
_mod = sys.modules[__name__]
_db_ref = [None]

class _LazyUser:
    _real = None
    def __init_subclass__(cls, **kw): super().__init_subclass__(**kw)

# Simple fix: just define User using current_app pattern
from flask_sqlalchemy import SQLAlchemy as _SA
from flask import current_app

# Actually the cleanest fix - define User without importing db at module load time
# Use a module-level placeholder that gets replaced
class _Base:
    pass

User = _Base  # temporary

def _init_models(db):
    global User
    class UserReal(db.Model):
        __tablename__ = 'users'
        __table_args__ = {'extend_existing': True}
        id = db.Column(db.Integer, primary_key=True)
        username = db.Column(db.String(80), nullable=False)
        email = db.Column(db.String(120), unique=True, nullable=False)
        password_hash = db.Column(db.String(256), nullable=False)
        role = db.Column(db.String(20), default='user')
        created_at = db.Column(db.DateTime, default=datetime.utcnow)
        predictions = db.relationship('Prediction', backref='user', lazy=True)
        def to_dict(self):
            return {'id':self.id,'username':self.username,'email':self.email,'role':self.role,'created_at':self.created_at.isoformat(),'prediction_count':len(self.predictions)}
    User = UserReal
    sys.modules[__name__].User = UserReal
    return UserReal
