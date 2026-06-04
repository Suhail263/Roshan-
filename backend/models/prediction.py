from datetime import datetime
from app import db
import json

class Prediction(db.Model):
    __tablename__ = 'predictions'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    result = db.Column(db.String(10), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    risk_level = db.Column(db.String(20), nullable=False)
    input_data = db.Column(db.Text, nullable=False)
    recommendations = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    def to_dict(self):
        return {'id':self.id,'user_id':self.user_id,'username':self.user.username if self.user else 'Unknown','result':self.result,'confidence':self.confidence,'risk_level':self.risk_level,'input_data':json.loads(self.input_data),'recommendations':json.loads(self.recommendations) if self.recommendations else [],'created_at':self.created_at.isoformat()}
