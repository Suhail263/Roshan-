from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from app import db
from models.user import User
from models.prediction import Prediction
from functools import wraps

admin_bp = Blueprint('admin', __name__)

def admin_required(fn):
    @wraps(fn)
    @jwt_required()
    def wrapper(*args, **kwargs):
        uid = int(get_jwt_identity())
        user = User.query.get(uid)
        if not user or user.role != 'admin':
            return jsonify({'error': 'Admin access required'}), 403
        return fn(*args, **kwargs)
    return wrapper

@admin_bp.route('/stats', methods=['GET'])
@admin_required
def stats():
    total_users = User.query.filter_by(role='user').count()
    total_preds = Prediction.query.count()
    ckd_count = Prediction.query.filter_by(result='CKD').count()
    healthy_count = Prediction.query.filter_by(result='Healthy').count()
    return jsonify({
        'total_users': total_users,
        'total_predictions': total_preds,
        'ckd_predictions': ckd_count,
        'healthy_predictions': healthy_count,
        'ckd_rate': round(ckd_count / total_preds * 100, 1) if total_preds else 0
    })

@admin_bp.route('/users', methods=['GET'])
@admin_required
def list_users():
    q = request.args.get('q', '')
    query = User.query.filter_by(role='user')
    if q:
        query = query.filter(User.username.ilike(f'%{q}%') | User.email.ilike(f'%{q}%'))
    users = query.order_by(User.created_at.desc()).all()
    return jsonify({'users': [u.to_dict() for u in users]})

@admin_bp.route('/users/<int:uid>', methods=['DELETE'])
@admin_required
def delete_user(uid):
    user = User.query.get_or_404(uid)
    if user.role == 'admin':
        return jsonify({'error': 'Cannot delete admin'}), 403
    Prediction.query.filter_by(user_id=uid).delete()
    db.session.delete(user)
    db.session.commit()
    return jsonify({'message': 'User deleted'})

@admin_bp.route('/predictions', methods=['GET'])
@admin_required
def all_predictions():
    page = int(request.args.get('page', 1))
    per_page = int(request.args.get('per_page', 20))
    result_filter = request.args.get('result')
    query = Prediction.query
    if result_filter:
        query = query.filter_by(result=result_filter)
    preds = query.order_by(Prediction.created_at.desc()).paginate(page=page, per_page=per_page, error_out=False)
    return jsonify({
        'predictions': [p.to_dict() for p in preds.items],
        'total': preds.total,
        'pages': preds.pages,
        'current_page': page
    })

@admin_bp.route('/predictions/<int:pid>', methods=['DELETE'])
@admin_required
def admin_delete_prediction(pid):
    pred = Prediction.query.get_or_404(pid)
    db.session.delete(pred)
    db.session.commit()
    return jsonify({'message': 'Deleted'})
