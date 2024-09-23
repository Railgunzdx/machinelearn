import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# โดยจะค้นหาเกมตามชื่อ และแนะนำเกมที่มีแนวเกม (genre) คล้ายกัน โค้ดใช้ TF-IDF (Term Frequency-Inverse Document Frequency)
#  และ cosine similarity เพื่อหาความคล้ายคลึงของเกมในเชิงของแนวเกม

# โหลดข้อมูลเกมทั้งหมดเข้ามาในตัวแปร games โดยแต่ละเกมจะเก็บข้อมูล
with open('games.json', 'r') as f:
    games = json.load(f)

# ฟังก์ชันนี้สร้างการเตรียมข้อมูลเบื้องต้นโดยการนำแนวเกม (genres) ของแต่ละเกมที่เป็นลิสต์ของคำ 
# เช่น ["Action", "Adventure"] มารวมเป็นสตริงเดียว เช่น "Action Adventure" และเก็บไว้ในฟิลด์ใหม่ชื่อ genre_str ในแต่ละเกม
def preprocess_games():
    for game in games:
        game['genre_str'] = ' '.join(game['genre'])

# เตรียมข้อมูลแนวเกม
def search_games(query):
    preprocess_games()

    #ใช้ในการแปลงข้อความเป็นเวกเตอร์เพื่อหาความคล้ายคลึงระหว่างข้อความ
    tfidf = TfidfVectorizer(stop_words='english') #ถูกใช้ในการแปลงแนวเกมของเกมแต่ละเกมให้เป็นเวกเตอร์
    tfidf_matrix = tfidf.fit_transform([game['genre_str'] for game in games]) #แมทริกซ์ที่เก็บข้อมูลของเกมในรูปแบบของเวกเตอร์จากแนวเกม
 
    # ค้นหาเกมที่ชื่อเกมมีคำที่ตรงกับคำค้นหา query โดยไม่สนใจตัวพิมพ์เล็กพิมพ์ใหญ่
    filtered_games = [game for game in games if query.lower() in game['name'].lower()]
    
    # หากเจอเกมที่ตรงกับคำค้นหา จะเลือกเกมแรกจากเกมที่เจอ (query_game)
    recommendations = []
    if filtered_games:
        query_game = filtered_games[0]  #เลือกเกมที่เจอเกมแรก
        query_index = games.index(query_game) 
        cosine_sim = cosine_similarity(tfidf_matrix[query_index], tfidf_matrix).flatten() #ถูกใช้เพื่อคำนวณความคล้ายคลึงของแนวเกมของเกมนี้กับเกมอื่นๆ ในระบบ
        #ของค่าความคล้ายคลึงระหว่างเกมนี้กับเกมอื่นๆ ทั้งหมด
        
        # เลือกเกมที่มีค่าความคล้ายคลึงมากที่สุด 5 เกม
        similar_indices = cosine_sim.argsort()[-6:-1]
        recommendations = [games[i] for i in similar_indices] #ผลลัพธ์คือเกมที่คล้ายกับเกมที่ผู้ใช้ค้นหา

    return filtered_games, recommendations #เกมที่ตรงกับคำค้นหา #เกมที่มีแนวเกมคล้ายกับเกมที่ค้นพบ
