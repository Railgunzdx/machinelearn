from flask import Flask, request, jsonify, render_template_string
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Sample game data
games = [
    { "id": 1, "name": "The Witcher 3: Wild Hunt", "genre": ["RPG", "Adventure"] },
    { "id": 2, "name": "Minecraft", "genre": ["Sandbox", "Adventure"] },
    { "id": 3, "name": "Grand Theft Auto V", "genre": ["Action", "Adventure"] },
    { "id": 4, "name": "Red Dead Redemption 2", "genre": ["Action", "Adventure"] },
    { "id": 5, "name": "Cyberpunk 2077", "genre": ["RPG", "Action"] },
    { "id": 6, "name": "Assassin’s Creed Valhalla", "genre": ["Action", "RPG"] },
    { "id": 7, "name": "Call of Duty: Warzone", "genre": ["Shooter", "Battle Royale"] },
    { "id": 8, "name": "Fortnite", "genre": ["Battle Royale", "Shooter"] },
    { "id": 9, "name": "Apex Legends", "genre": ["Battle Royale", "Shooter"] },
    { "id": 10, "name": "Valorant", "genre": ["Shooter", "Strategy"] },
    { "id": 11, "name": "Overwatch 2", "genre": ["Shooter", "Strategy"] },
    { "id": 12, "name": "Battlefield 2042", "genre": ["Shooter", "Action"] },
    { "id": 13, "name": "FIFA 22", "genre": ["Sports"] },
    { "id": 14, "name": "NBA 2K22", "genre": ["Sports"] },
    { "id": 15, "name": "Forza Horizon 5", "genre": ["Racing", "Sports"] },
    { "id": 16, "name": "Need for Speed Heat", "genre": ["Racing", "Action"] },
    { "id": 17, "name": "The Legend of Zelda: Breath of the Wild", "genre": ["Adventure", "RPG"] },
    { "id": 18, "name": "Super Mario Odyssey", "genre": ["Adventure"] },
    { "id": 19, "name": "Animal Crossing: New Horizons", "genre": ["Simulation"] },
    { "id": 20, "name": "Resident Evil Village", "genre": ["Horror", "Action"] },
    { "id": 21, "name": "Blue Archive", "genre": ["RPG", "Action"] },
    { "id": 22, "name": "Elden Ring", "genre": ["RPG", "Fantasy", "Souls-like"] },
    { "id": 23, "name": "Monster Hunter Wilds", "genre": ["Action", "Adventure"] },
    { "id": 24, "name": "DARK SOULS: REMASTERED", "genre": ["Fantasy", "Action", "Souls-like"] },
    { "id": 25, "name": "Black Myth: Wukong", "genre": ["Adventure", "Action", "Souls-like"] },
    { "id": 26, "name": "Mush Dash", "genre": ["Rhythm", "Anime"] },
    { "id": 27, "name": "Until Then", "genre": ["Visual Novel", "Drama", "Romance"] },
    { "id": 28, "name": "Persona 5 Royal", "genre": ["Visual Novel", "Anime", "RPG"] },
    { "id": 29, "name": "Hades", "genre": ["Rogue-like", "Action"] },
    { "id": 30, "name": "Genshin Impact", "genre": ["RPG", "Adventure", "Fantasy"] },
    { "id": 31, "name": "Hollow Knight", "genre": ["Metroidvania", "Action", "Adventure"] },
    { "id": 32, "name": "Stardew Valley", "genre": ["Simulation", "RPG", "Farming"] },
    { "id": 33, "name": "League of Legends", "genre": ["MOBA", "Strategy"] },
    { "id": 34, "name": "Among Us", "genre": ["Party", "Strategy"] },
    { "id": 35, "name": "Celeste", "genre": ["Platformer", "Adventure"] },
    { "id": 36, "name": "Tetris Effect", "genre": ["Puzzle", "Rhythm"] },
    { "id": 37, "name": "Slay the Spire", "genre": ["Card Game", "Rogue-like"] },
    { "id": 38, "name": "Dead by Daylight", "genre": ["Horror", "Multiplayer", "Survival"] },
    { "id": 39, "name": "The Last of Us Part II", "genre": ["Action", "Adventure", "Horror"] },
    { "id": 40, "name": "Sekiro: Shadows Die Twice", "genre": ["Action", "Adventure", "Souls-like"] },
    { "id": 41, "name": "Horizon Zero Dawn", "genre": ["RPG", "Adventure", "Action"] },
    { "id": 42, "name": "Fire Emblem: Three Houses", "genre": ["Strategy", "RPG"] },
    { "id": 43, "name": "Splatoon 2", "genre": ["Shooter", "Multiplayer"] },
    { "id": 44, "name": "Cuphead", "genre": ["Platformer", "Action"] },
    { "id": 45, "name": "Ghost of Tsushima", "genre": ["Action", "Adventure", "Open World"] },
    { "id": 46, "name": "Disco Elysium", "genre": ["RPG", "Adventure"] },
    { "id": 47, "name": "Death Stranding", "genre": ["Adventure", "Action", "Sci-Fi"] },
    { "id": 48, "name": "Returnal", "genre": ["Action", "Rogue-like", "Sci-Fi"] },
    { "id": 49, "name": "NieR: Automata", "genre": ["RPG", "Action", "Adventure"] },
    { "id": 50, "name": "Doom Eternal", "genre": ["Shooter", "Action"] },
    { "id": 51, "name": "Terraria", "genre": ["Sandbox", "Adventure"] },
    { "id": 52, "name": "Metal Gear Solid V: The Phantom Pain", "genre": ["Action", "Stealth", "Adventure"] },
    { "id": 53, "name": "Divinity: Original Sin 2", "genre": ["RPG", "Strategy", "Fantasy"] },
    { "id": 54, "name": "Xenoblade Chronicles 3", "genre": ["RPG", "Adventure", "Fantasy"] },
    { "id": 55, "name": "Bayonetta 3", "genre": ["Action", "Adventure"] },
    { "id": 56, "name": "Metroid Dread", "genre": ["Action", "Adventure", "Metroidvania"] },
    { "id": 57, "name": "It Takes Two", "genre": ["Co-op", "Adventure", "Puzzle"] },
    { "id": 58, "name": "Kirby and the Forgotten Land", "genre": ["Platformer", "Adventure"] },
    { "id": 59, "name": "Hollow Knight: Silksong", "genre": ["Metroidvania", "Action", "Adventure"] },
    { "id": 60, "name": "Sifu", "genre": ["Action", "Martial Arts"] }
]

# Preprocess game data
def preprocess_games():
    for game in games:
        game['genre_str'] = ' '.join(game['genre'])

# Search and recommend similar games
def search_games(query):
    preprocess_games()
    
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform([game['genre_str'] for game in games])
    
    filtered_games = [game for game in games if query.lower() in game['name'].lower()]
    recommendations = []
    
    if filtered_games:
        query_game = filtered_games[0]
        query_index = games.index(query_game)
        cosine_sim = cosine_similarity(tfidf_matrix[query_index], tfidf_matrix).flatten()
        similar_indices = cosine_sim.argsort()[-6:-1]
        recommendations = [games[i] for i in similar_indices]
    
    return filtered_games, recommendations

# Serve the main page with embedded HTML and CSS
@app.route('/')
def index():
    html_template = '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Game Recommendation System</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                background-color: #f4f4f4;
                margin: 0;
                padding: 0;
            }
            .container {
                max-width: 800px;
                margin: 50px auto;
                padding: 20px;
                background-color: white;
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
                border-radius: 8px;
            }
            h1 {
                text-align: center;
                color: #333;
            }
            .search-section {
                display: flex;
                justify-content: center;
                margin-bottom: 30px;
            }
            .search-section input {
                width: 60%;
                padding: 10px;
                font-size: 16px;
                border: 1px solid #ccc;
                border-radius: 4px;
            }
            .search-section button {
                padding: 10px 20px;
                margin-left: 10px;
                background-color: #007BFF;
                color: white;
                border: none;
                border-radius: 4px;
                cursor: pointer;
            }
            .search-section button:hover {
                background-color: #0056b3;
            }
            h2 {
                margin-bottom: 10px;
                color: #007BFF;
            }
            .game-list {
                display: flex;
                flex-wrap: wrap;
                gap: 20px;
            }
            .game-item {
                flex: 1 1 calc(50% - 20px);
                background-color: #f9f9f9;
                padding: 15px;
                border-radius: 6px;
                box-shadow: 0 1px 4px rgba(0, 0, 0, 0.1);
            }
            .game-item h3 {
                margin: 0 0 10px 0;
                color: #333;
            }
            .game-item p {
                margin: 0;
                color: #555;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Game Recommendation System</h1>
            <div class="search-section">
                <input type="text" id="searchInput" placeholder="Enter a game name or genre" />
                <button onclick="searchGame()">Search</button>
            </div>
            <div id="popularGames">
                <h2>Random 5 Games:</h2>
                <div id="popularGamesList" class="game-list"></div>
            </div>
            <div id="searchResults">
                <h2>Search Results:</h2>
                <div id="gameList" class="game-list"></div>
            </div>
            <div id="recommendations">
                <h2>Recommended Games:</h2>
                <div id="recommendedGames" class="game-list"></div>
            </div>
        </div>
        
        <script>
            // Load popular games when the page loads
            document.addEventListener("DOMContentLoaded", function() {
                fetchPopularGames();
            });

            function fetchPopularGames() {
                fetch('/random')
                    .then(response => response.json())
                    .then(data => displayPopularGames(data.games));
            }

            function searchGame() {
                const query = document.getElementById('searchInput').value;
                fetch(`/search?query=${query}`)
                    .then(response => response.json())
                    .then(data => {
                        displayResults(data.results);
                        displayRecommendations(data.recommendations);
                    });
            }

            function displayResults(games) {
                const gameListDiv = document.getElementById('gameList');
                gameListDiv.innerHTML = '';
                if (games.length === 0) {
                    gameListDiv.innerHTML = `<p>No games found.</p>`;
                } else {
                    games.forEach(game => {
                        const gameDiv = document.createElement('div');
                        gameDiv.classList.add('game-item');
                        gameDiv.innerHTML = `<h3>${game.name}</h3><p>Genres: ${game.genre.join(', ')}</p>`;
                        gameListDiv.appendChild(gameDiv);
                    });
                }
            }

            function displayRecommendations(games) {
                const recommendationsDiv = document.getElementById('recommendedGames');
                recommendationsDiv.innerHTML = '';
                if (games.length === 0) {
                    recommendationsDiv.innerHTML = `<p>No recommendations available.</p>`;
                } else {
                    games.forEach(game => {
                        const gameDiv = document.createElement('div');
                        gameDiv.classList.add('game-item');
                        gameDiv.innerHTML = `<h3>${game.name}</h3><p>Genres: ${game.genre.join(', ')}</p>`;
                        recommendationsDiv.appendChild(gameDiv);
                    });
                }
            }

            function displayPopularGames(games) {
                const popularGamesDiv = document.getElementById('popularGamesList');
                popularGamesDiv.innerHTML = '';
                games.forEach(game => {
                    const gameDiv = document.createElement('div');
                    gameDiv.classList.add('game-item');
                    gameDiv.innerHTML = `<h3>${game.name}</h3><p>Genres: ${game.genre.join(', ')}</p>`;
                    popularGamesDiv.appendChild(gameDiv);
                });
            }
        </script>
    </body>
    </html>
    '''
    return render_template_string(html_template)

# Search and recommend games
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

# Get random popular games
@app.route('/random', methods=['GET'])
def popular_games():
    import random
    popular_games = random.sample(games, 5)
    return jsonify({'games': popular_games})

if __name__ == '__main__':
    app.run(debug=True)
