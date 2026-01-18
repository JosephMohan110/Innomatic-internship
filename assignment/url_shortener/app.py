from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
import string
import random
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key_here' # Change this in production
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///urls.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

class URL(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    original_url = db.Column(db.String(500), nullable=False)
    short_id = db.Column(db.String(10), unique=True, nullable=False)
    created_at = db.Column(db.DateTime, server_default=db.func.now())

    def __repr__(self):
        return f'<URL {self.short_id}>'

def generate_short_id(length=6):
    characters = string.ascii_letters + string.digits
    while True:
        short_id = ''.join(random.choices(characters, k=length))
        if not URL.query.filter_by(short_id=short_id).first():
            return short_id

@app.route('/', methods=['GET', 'POST'])
def index():
    short_url = None
    if request.method == 'POST':
        original_url = request.form.get('original_url')
        if original_url:
            # Check if already exists to avoid duplicates (optional, strictly speaking we might want new short links for same URL)
            # For this assignment, let's create a new one every time or reuse if we want. 
            # Let's create a new one to be simple and allow multiple people to shorten same link.
            
            short_id = generate_short_id()
            new_url = URL(original_url=original_url, short_id=short_id)
            db.session.add(new_url)
            db.session.commit()
            
            short_url = request.host_url + short_id
            
    return render_template('index.html', short_url=short_url)

@app.route('/<short_id>')
def redirect_to_url(short_id):
    url_entry = URL.query.filter_by(short_id=short_id).first_or_404()
    return redirect(url_entry.original_url)

@app.route('/history')
def history():
    urls = URL.query.order_by(URL.created_at.desc()).all()
    return render_template('history.html', urls=urls)

@app.route('/test_create_db')
def test_create_db():
    with app.app_context():
        db.create_all()
    return "Database created!"

if __name__ == '__main__':
    if not os.path.exists('urls.db'):
        with app.app_context():
            db.create_all()
    app.run(debug=True, port=5001)
