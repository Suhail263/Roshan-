from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta
from functools import wraps
import os, json, pickle, numpy as np, pandas as pd

app = Flask(__name__)
app.config['SECRET_KEY'] = 'ckd-nephroai-2024-secret'
app.config['JWT_SECRET_KEY'] = 'ckd-nephroai-2024-jwt-secret-longkey-here'
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=24)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///ckd.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
CORS(app, resources={r"/api/*": {"origins": "*"}})
db = SQLAlchemy(app)
jwt = JWTManager(app)

# ── MODELS ────────────────────────────────────────────────────────────────────
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80))
    email = db.Column(db.String(120), unique=True)
    password_hash = db.Column(db.String(256))
    role = db.Column(db.String(20), default='user')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    predictions = db.relationship('Prediction', backref='user', lazy=True)
    def to_dict(self):
        return {'id':self.id,'username':self.username,'email':self.email,'role':self.role,'created_at':self.created_at.isoformat(),'prediction_count':len(self.predictions)}

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    result = db.Column(db.String(10))
    confidence = db.Column(db.Float)
    risk_level = db.Column(db.String(20))
    input_data = db.Column(db.Text)
    recommendations = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    def to_dict(self):
        return {'id':self.id,'user_id':self.user_id,'username':self.user.username if self.user else '?','result':self.result,'confidence':self.confidence,'risk_level':self.risk_level,'input_data':json.loads(self.input_data),'recommendations':json.loads(self.recommendations or '[]'),'created_at':self.created_at.isoformat()}

# ── ML ────────────────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'ml', 'saved', 'ckd_model.pkl')
_model_cache = None

def load_model():
    global _model_cache
    if _model_cache is None:
        with open(MODEL_PATH, 'rb') as f:
            _model_cache = pickle.load(f)
    return _model_cache

FEATURES = ['age','bp','sg','al','su','bgr','bu','sc','sod','pot','hemo','pcv','wc','rc','htn','dm','appet','pe','ane']

def get_recommendations(result, conf, data):
    r = []
    if result == 'CKD':
        r.append("⚠️ Consult a nephrologist immediately for comprehensive kidney evaluation.")
        if data.get('sc',0) > 1.2: r.append("🔬 High serum creatinine. Kidney filtration may be compromised.")
        if data.get('bu',0) > 40: r.append("🧪 Elevated blood urea. Restrict protein intake.")
        if data.get('htn',0): r.append("💊 Manage hypertension aggressively.")
        if data.get('dm',0): r.append("🩺 Diabetic nephropathy risk. Control blood glucose strictly.")
        if data.get('hemo',15) < 12: r.append("💉 Anemia detected. Iron supplements may be required.")
        r.append("🥗 Follow a low-sodium, low-potassium, kidney-friendly diet.")
        if conf > 85: r.append("🚨 High confidence result. Seek immediate medical attention.")
    else:
        r += ["✅ No CKD indicators in current assessment.","💧 Stay hydrated — drink 2-3 liters daily.","🥦 Balanced diet with fruits and vegetables.","🏃 30 min daily exercise supports kidney health.","🩺 Annual kidney function tests recommended."]
        if data.get('htn',0): r.append("⚠️ Monitor blood pressure — hypertension is a CKD risk factor.")
    return r

def do_predict(data):
    arts = load_model()
    vals = [float(data.get(f, np.nan) or np.nan) for f in FEATURES]
    X = pd.DataFrame([vals], columns=FEATURES)
    Xi = arts['imputer'].transform(X)
    Xi = pd.DataFrame(Xi, columns=FEATURES)
    pred = arts['model'].predict(Xi)[0]
    probs = arts['model'].predict_proba(Xi)[0]
    conf = round(float(max(probs))*100, 2)
    result = 'CKD' if pred == 1 else 'Healthy'
    ckd_prob = round(float(probs[1])*100, 2)
    if result == 'CKD':
        risk = 'Critical' if conf>=85 else 'High' if conf>=70 else 'Medium'
    else:
        risk = 'Low' if conf>=90 else 'Medium'
    return {'result':result,'confidence':conf,'ckd_probability':ckd_prob,'risk_level':risk,'recommendations':get_recommendations(result,conf,data),'feature_importance':arts['feature_importance']}

# ── HELPERS ───────────────────────────────────────────────────────────────────
def admin_required(fn):
    @wraps(fn)
    @jwt_required()
    def wrapper(*args, **kw):
        u = User.query.get(int(get_jwt_identity()))
        if not u or u.role != 'admin': return jsonify({'error':'Admin only'}), 403
        return fn(*args, **kw)
    return wrapper

# ── AUTH ROUTES ───────────────────────────────────────────────────────────────
@app.route('/api/auth/register', methods=['POST'])
def register():
    d = request.get_json()
    if User.query.filter_by(email=d['email']).first(): return jsonify({'error':'Email taken'}), 409
    u = User(username=d['username'], email=d['email'], password_hash=generate_password_hash(d['password']))
    db.session.add(u); db.session.commit()
    return jsonify({'token':create_access_token(identity=str(u.id)),'user':u.to_dict()}), 201

@app.route('/api/auth/login', methods=['POST'])
def login():
    d = request.get_json()
    u = User.query.filter_by(email=d.get('email')).first()
    if not u or not check_password_hash(u.password_hash, d.get('password','')): return jsonify({'error':'Invalid credentials'}), 401
    return jsonify({'token':create_access_token(identity=str(u.id)),'user':u.to_dict(),'message':'Login successful'})

@app.route('/api/auth/me', methods=['GET'])
@jwt_required()
def me():
    u = User.query.get(int(get_jwt_identity()))
    return jsonify({'user':u.to_dict()})

# ── PREDICTION ROUTES ─────────────────────────────────────────────────────────
@app.route('/api/predict', methods=['POST'])
@jwt_required()
def predict():
    data = request.get_json()
    try:
        r = do_predict(data)
        p = Prediction(user_id=int(get_jwt_identity()), result=r['result'], confidence=r['confidence'],
            risk_level=r['risk_level'], input_data=json.dumps(data), recommendations=json.dumps(r['recommendations']))
        db.session.add(p); db.session.commit()
        r['prediction_id'] = p.id
        return jsonify(r)
    except Exception as e:
        return jsonify({'error':str(e)}), 500

@app.route('/api/predictions', methods=['GET'])
@jwt_required()
def predictions():
    uid = int(get_jwt_identity())
    return jsonify({'predictions':[p.to_dict() for p in Prediction.query.filter_by(user_id=uid).order_by(Prediction.created_at.desc()).all()]})

@app.route('/api/predictions/<int:pid>', methods=['DELETE'])
@jwt_required()
def del_pred(pid):
    uid = int(get_jwt_identity()); p = Prediction.query.get_or_404(pid)
    u = User.query.get(uid)
    if p.user_id != uid and u.role != 'admin': return jsonify({'error':'Unauthorized'}), 403
    db.session.delete(p); db.session.commit(); return jsonify({'message':'Deleted'})

@app.route('/api/model/metrics', methods=['GET'])
def metrics():
    import json as _j
    mp = os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','ml','saved','model_metrics.json')
    with open(mp) as f: return jsonify(_j.load(f))

# ── ANALYTICS ROUTES ──────────────────────────────────────────────────────────
@app.route('/api/analytics/user', methods=['GET'])
@jwt_required()
def user_analytics():
    uid = int(get_jwt_identity())
    preds = Prediction.query.filter_by(user_id=uid).order_by(Prediction.created_at).all()
    ckd = sum(1 for p in preds if p.result=='CKD')
    rd = {}
    for p in preds: rd[p.risk_level] = rd.get(p.risk_level,0)+1
    return jsonify({'total_tests':len(preds),'ckd_count':ckd,'healthy_count':len(preds)-ckd,'risk_distribution':rd,
        'timeline':[{'date':p.created_at.strftime('%Y-%m-%d'),'result':p.result,'confidence':p.confidence,'risk':p.risk_level} for p in preds],
        'latest_result':preds[-1].result if preds else None,'latest_confidence':preds[-1].confidence if preds else None})

@app.route('/api/analytics/global', methods=['GET'])
@admin_required
def global_analytics():
    return jsonify({'total_predictions':Prediction.query.count(),'total_users':User.query.filter_by(role='user').count()})

# ── ADMIN ROUTES ──────────────────────────────────────────────────────────────
@app.route('/api/admin/stats', methods=['GET'])
@admin_required
def admin_stats():
    t = Prediction.query.count(); c = Prediction.query.filter_by(result='CKD').count()
    return jsonify({'total_users':User.query.filter_by(role='user').count(),'total_predictions':t,'ckd_predictions':c,'healthy_predictions':t-c,'ckd_rate':round(c/t*100,1) if t else 0})

@app.route('/api/admin/users', methods=['GET'])
@admin_required
def admin_users():
    q = request.args.get('q','')
    users = User.query.filter_by(role='user')
    if q: users = users.filter(User.username.ilike(f'%{q}%')|User.email.ilike(f'%{q}%'))
    return jsonify({'users':[u.to_dict() for u in users.order_by(User.created_at.desc()).all()]})

@app.route('/api/admin/users/<int:uid>', methods=['DELETE'])
@admin_required
def admin_del_user(uid):
    u = User.query.get_or_404(uid)
    if u.role=='admin': return jsonify({'error':'Cannot delete admin'}), 403
    Prediction.query.filter_by(user_id=uid).delete()
    db.session.delete(u); db.session.commit(); return jsonify({'message':'Deleted'})

@app.route('/api/admin/predictions', methods=['GET'])
@admin_required
def admin_predictions():
    page = int(request.args.get('page',1)); pp = int(request.args.get('per_page',50))
    paged = Prediction.query.order_by(Prediction.created_at.desc()).paginate(page=page,per_page=pp,error_out=False)
    return jsonify({'predictions':[p.to_dict() for p in paged.items],'total':paged.total,'pages':paged.pages})

@app.route('/api/admin/predictions/<int:pid>', methods=['DELETE'])
@admin_required
def admin_del_pred(pid):
    p = Prediction.query.get_or_404(pid)
    db.session.delete(p); db.session.commit(); return jsonify({'message':'Deleted'})

# ── STARTUP ───────────────────────────────────────────────────────────────────
with app.app_context():
    db.create_all()
    if not User.query.filter_by(email='roshan@suhail26.com').first():
        db.session.add(User(username='Suhail', email='roshan@suhail26.com', password_hash=generate_password_hash('Suhailroshan@2361'), role='admin'))
        db.session.commit()
        print("✅ Admin created: roshan@suhail26.com / Suhailroshan@2361")
    else:
        print("✅ Admin ready: roshan@suhail26.com / Suhailroshan@2361")

if __name__ == '__main__':
    print("🚀 NephroAI Backend starting on http://localhost:5000")
    app.run(debug=False, port=5000, host='0.0.0.0')
