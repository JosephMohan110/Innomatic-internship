from flask import Flask, request, render_template_string
import random
import datetime

app = Flask(__name__)

# Basic HTML template for better presentation
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Flask Assignment App</title>
    <style>
        body { font-family: sans-serif; text-align: center; padding: 50px; background-color: #f0f2f5; }
        .container { background: white; padding: 30px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); max-width: 600px; margin: auto; }
        h1 { color: #2c3e50; }
        .result { font-size: 24px; color: #e74c3c; margin: 20px 0; }
        .bonus { margin-top: 30px; padding-top: 20px; border-top: 1px solid #eee; }
        a { color: #3498db; text-decoration: none; margin: 0 10px; }
        a:hover { text-decoration: underline; }
    </style>
</head>
<body>
    <div class="container">
        {{ content|safe }}
        <div class="bonus">
            <h3>Try Cool Functions:</h3>
            <a href="/flip_coin">Flip Coin</a>
            <a href="/mystery">Mystery Box</a>
            <a href="/time_machine">Time Machine</a>
        </div>
    </div>
</body>
</html>
"""

# Requirement: Convert username to UPPER CASE from query parameter
@app.route('/')
def home():
    username = request.args.get('username')
    
    if username:
        message = f"<h1>Hello, <span class='result'>{username.upper()}</span>!</h1>"
    else:
        message = """
            <h1>Welcome!</h1>
            <p>Please provide a username in the URL query parameter to see the magic.</p>
            <p>Example: <a href="/?username=innomatic">/?username=innomatic</a></p>
        """
    
    return render_template_string(HTML_TEMPLATE, content=message)

# Bonus Function 1: Flip a Coin
@app.route('/flip_coin')
def flip_coin():
    result = random.choice(["Heads", "Tails"])
    color = "green" if result == "Heads" else "blue"
    content = f"<h1>Coin Flip Result:</h1><h2 style='color:{color}'>{result}</h2><a href='/'>Back Home</a>"
    return render_template_string(HTML_TEMPLATE, content=content)

# Bonus Function 2: Mystery Box (Random Fact/Joke)
@app.route('/mystery')
def mystery():
    surprises = [
        "Did you know? Honey never spoils.",
        "A group of flamingos is called a 'flamboyance'.",
        "Python was named after Monty Python, not the snake.",
        "The first computer bug was an actual moth.",
        "Bananas are berries, but strawberries aren't."
    ]
    surprise = random.choice(surprises)
    content = f"<h1>Mystery Box Open! 🎁</h1><p style='font-size:20px; font-style:italic;'>{surprise}</p><a href='/'>Back Home</a>"
    return render_template_string(HTML_TEMPLATE, content=content)

# Bonus Function 3: Time Machine (Current Time in weird format)
@app.route('/time_machine')
def time_machine():
    now = datetime.datetime.now()
    weird_time = now.strftime("Year: %Y | Day: %j | Hour: %H")
    content = f"<h1>Welcome to the Future (or Present?) ⏳</h1><p class='result'>{weird_time}</p><p>Day of year: {now.strftime('%j')}</p><a href='/'>Back Home</a>"
    return render_template_string(HTML_TEMPLATE, content=content)

if __name__ == '__main__':
    app.run(debug=True)
