from flask import Blueprint, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from app import db
from models.prediction import Prediction
from models.user import User
from sqlalchemy import func
from datetime import datetime, timedelta

analytics_bp = Blueprint('analytics', __name__)

@analytics_bp.route('/user', methods=['GET'])
@jwt_required()
def user_analytics():
    uid = int(get_jwt_identity())
    preds = Prediction.query.filter_by(user_id=uid).order_by(Prediction.created_at).all()
    
    timeline = [{'date': p.created_at.strftime('%Y-%m-%d'), 'result': p.result,
                 'confidence': p.confidence, 'risk': p.risk_level} for p in preds]
    
    ckd_count = sum(1 for p in preds if p.result == 'CKD')
    
    # Risk distribution
    risk_dist = {}
    for p in preds:
        risk_dist[p.risk_level] = risk_dist.get(p.risk_level, 0) + 1
    
    return jsonify({
        'total_tests': len(preds),
        'ckd_count': ckd_count,
        'healthy_count': len(preds) - ckd_count,
        'timeline': timeline,
        'risk_distribution': risk_dist,
        'latest_result': preds[-1].result if preds else None,
        'latest_confidence': preds[-1].confidence if preds else None
    })

@analytics_bp.route('/global', methods=['GET'])
@jwt_required()
def global_analytics():
    uid = int(get_jwt_identity())
    user = User.query.get(uid)
    if not user or user.role != 'admin':
        return jsonify({'error': 'Admin only'}), 403

    # Weekly trend (last 30 days)
    thirty_days_ago = datetime.utcnow() - timedelta(days=30)
    recent = Prediction.query.filter(Prediction.created_at >= thirty_days_ago).all()
    
    daily = {}
    for p in recent:
        day = p.created_at.strftime('%Y-%m-%d')
        if day not in daily:
            daily[day] = {'ckd': 0, 'healthy': 0}
        daily[day]['ckd' if p.result == 'CKD' else 'healthy'] += 1
    
    return jsonify({
        'daily_trend': daily,
        'total_predictions': Prediction.query.count(),
        'total_users': User.query.filter_by(role='user').count()
    })
