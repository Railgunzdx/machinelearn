import random
from flask import Flask, request, render_template, jsonify #สร้างเว็บแอปพลิเคชัน
import json
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.metrics.pairwise import cosine_similarity
#ใช้สำหรับการแปลงแนวเกมเป็นเวกเตอร์และคำนวณความคล้ายคลึงระหว่างเกม


app = Flask(__name__)

with open('games.json', 'r') as f:
    games = json.load(f)

#เก็บในรูปแบบลิสต์ เช่น ["Action", "Adventure"] ให้กลายเป็นสตริงเดียว เช่น "Action Adventure" เพื่อใช้ในการสร้างโมเดล TF-IDF

def preprocess_games():
    for game in games:
        game['genre_str'] = ' '.join(game['genre'])

#  ค้นหาเกมและแนะนำเกมที่คล้ายกัน
def search_games(query):
    preprocess_games()

    # โค้ดนี้เริ่มต้นด้วยการเรียก preprocess_games() เพื่อเตรียมข้อมูลแนวเกมก่อน
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform([game['genre_str'] for game in games])

    # สร้างโมเดล TF-IDF โดยแปลงแนวเกมของทุกเกมเป็นเวกเตอร์เพื่อคำนวณความคล้ายคลึงกัน
    filtered_games = [game for game in games if query.lower() in game['name'].lower()]
    
    # ทำการค้นหาเกมที่มีชื่อเกมตรงกับคำที่ผู้ใช้ค้นหา (query)
    recommendations = []
    if filtered_games:
        query_game = filtered_games[0] 
        query_index = games.index(query_game)
        cosine_sim = cosine_similarity(tfidf_matrix[query_index], tfidf_matrix).flatten()
        
        # เมื่อค้นพบเกมที่ตรงกับคำค้นหา จะใช้เกมนั้นเป็นเกณฑ์ในการคำนวณความคล้ายคลึงกับเกมอื่นๆ โดยใช้ cosine similarity
        similar_indices = cosine_sim.argsort()[-6:-1]
        recommendations = [games[i] for i in similar_indices]

    return filtered_games, recommendations

# โค้ดนี้ทำการโหลดหน้าแรกของเว็บ 
@app.route('/')
def index():
    return render_template('index.html')

# Route นี้ใช้รับค่าค้นหา (query) จากผู้ใช้ แล้วใช้ฟังก์ชัน search_games() เพื่อค้นหาและส่งผลลัพธ์เกมที่เจอและเกมแนะนำในรูปแบบ JSON กลับไป
@app.route('/search', methods=['GET'])
def search():
    query = request.args.get('query')
    if not query:
        return jsonify({'error': 'Query parameter is missing'}), 400

    filtered_games, recommendations = search_games(query)
    
    return jsonify({
        'results': filtered_games,
        'recommendations': recommendations
    })

# Route นี้สุ่มเลือก 5 เกมจากไฟล์ games.json 
@app.route('/popular', methods=['GET'])
def popular_games():
    popular_games = random.sample(games, 5)
    return jsonify({'games': popular_games})

if __name__ == '__main__':
    app.run(debug=True)
#ส่วนนี้คือการรันแอปพลิเคชัน Flask ในโหมด debug เพื่อให้สามารถตรวจสอบข้อผิดพลาดได้