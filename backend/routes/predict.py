from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from app import db
from models.prediction import Prediction
from models.user import User
from utils.predictor import predict, get_model_metrics
import json

predict_bp = Blueprint('predict', __name__)

@predict_bp.route('/predict', methods=['POST'])
@jwt_required()
def make_prediction():
    user_id = int(get_jwt_identity())
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No input data'}), 400

    try:
        result = predict(data)
        pred = Prediction(
            user_id=user_id,
            result=result['result'],
            confidence=result['confidence'],
            risk_level=result['risk_level'],
            input_data=json.dumps(data),
            recommendations=json.dumps(result['recommendations'])
        )
        db.session.add(pred)
        db.session.commit()
        result['prediction_id'] = pred.id
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@predict_bp.route('/predictions', methods=['GET'])
@jwt_required()
def user_predictions():
    user_id = int(get_jwt_identity())
    preds = Prediction.query.filter_by(user_id=user_id).order_by(Prediction.created_at.desc()).all()
    return jsonify({'predictions': [p.to_dict() for p in preds]})

@predict_bp.route('/predictions/<int:pid>', methods=['GET'])
@jwt_required()
def get_prediction(pid):
    user_id = int(get_jwt_identity())
    pred = Prediction.query.get_or_404(pid)
    if pred.user_id != user_id:
        user = User.query.get(user_id)
        if not user or user.role != 'admin':
            return jsonify({'error': 'Unauthorized'}), 403
    return jsonify(pred.to_dict())

@predict_bp.route('/predictions/<int:pid>', methods=['DELETE'])
@jwt_required()
def delete_prediction(pid):
    user_id = int(get_jwt_identity())
    pred = Prediction.query.get_or_404(pid)
    if pred.user_id != user_id:
        user = User.query.get(user_id)
        if not user or user.role != 'admin':
            return jsonify({'error': 'Unauthorized'}), 403
    db.session.delete(pred)
    db.session.commit()
    return jsonify({'message': 'Deleted'})

@predict_bp.route('/model/metrics', methods=['GET'])
def model_metrics():
    return jsonify(get_model_metrics())
