from flask import Flask, request, jsonify, render_template_string
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Sample game data #"image_url":""
games = [
    { "id": 1, "name": "The Witcher 3: Wild Hunt", "genre": ["RPG", "Adventure"],"image_url": "https://image.api.playstation.com/vulcan/ap/rnd/202211/0711/kh4MUIuMmHlktOHar3lVl6rY.png" },
    { "id": 2, "name": "Minecraft", "genre": ["Sandbox", "Adventure"],"image_url": "https://assets.nintendo.com/image/upload/ar_16:9,c_lpad,w_1240/b_white/f_auto/q_auto/ncom/software/switch/70010000000964/a28a81253e919298beab2295e39a56b7a5140ef15abdb56135655e5c221b2a3a" },
    { "id": 3, "name": "Grand Theft Auto V", "genre": ["Action", "Adventure"],"image_url":"https://image.api.playstation.com/vulcan/ap/rnd/202203/0911/VIB0SeEj9vT6DTv7P4thJLZi.jpg?w=440" },
    { "id": 4, "name": "Red Dead Redemption 2", "genre": ["Action", "Adventure"],"image_url":"https://cdn1.epicgames.com/b30b6d1b4dfd4dcc93b5490be5e094e5/offer/RDR2476298253_Epic_Games_Wishlist_RDR2_2560x1440_V01-2560x1440-2a9ebe1f7ee202102555be202d5632ec.jpg" },
    { "id": 5, "name": "Cyberpunk 2077", "genre": ["RPG", "Action"],"image_url":"https://www.ggkeystore.com/cdn-cgi/image/fit=scale-down,w=1920,q=85,f=auto,anim=false,sharpen=0,onerror=redirect,metadata=none/storage/articles/26AUXHge3QBPbngvvON54CeUdyg6DX5OkoAgVGiC.jpeg?lg-1616829532" },
    { "id": 6, "name": "Assassin’s Creed Valhalla", "genre": ["Action", "RPG"],"image_url":"https://image.api.playstation.com/vulcan/ap/rnd/202008/0723/i2ICFMr0Ius6qtYYD9GNrY68.jpg" },
    { "id": 7, "name": "Call of Duty: Warzone", "genre": ["Shooter", "Battle Royale"],"image_url":"https://image.api.playstation.com/vulcan/ap/rnd/202306/2400/ac505d57a46e24dd96712263d89a150cb443af288c025ff2.jpg" },
    { "id": 8, "name": "Fortnite", "genre": ["Battle Royale", "Shooter"],"image_url":"https://cdn2.unrealengine.com/social-image-chapter4-s3-3840x2160-d35912cc25ad.jpg" },
    { "id": 9, "name": "Apex Legends", "genre": ["Battle Royale", "Shooter"],"image_url":"https://gmedia.playstation.com/is/image/SIEPDC/apex-legends-listing-thumb-01-ps4-en-29oct20?$facebook$" },
    { "id": 10, "name": "Valorant", "genre": ["Shooter", "Strategy"],"image_url":"https://cdn.oneesports.co.th/cdn-data/sites/3/2023/04/valorant-1.jpeg" },
    { "id": 11, "name": "Overwatch 2", "genre": ["Shooter", "Strategy"],"image_url":"https://blog.ttt-website.com/wp-content/uploads/2023/08/Overwatch-2-01.jpg" },
    { "id": 12, "name": "Battlefield 2042", "genre": ["Shooter", "Action"],"image_url":"https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTlBc7r5q-6V2Kwideaw2vy5-qRChwR3sgWiA&s" },
    { "id": 13, "name": "FIFA 22", "genre": ["Sports"],"image_url":"https://inwfile.com/s-dl/0gp5xq.jpg" },
    { "id": 14, "name": "NBA 2K22", "genre": ["Sports"],"image_url":"https://shared.steamstatic.com/store_item_assets/steam/apps/1644960/header.jpg?t=1704736929" },
    { "id": 15, "name": "Forza Horizon 5", "genre": ["Racing", "Sports"],"image_url":"https://shared.akamai.steamstatic.com/store_item_assets/steam/apps/1551360/capsule_616x353.jpg?t=1725950938" },
    { "id": 16, "name": "Need for Speed Heat", "genre": ["Racing", "Action"],"image_url":"https://image.api.playstation.com/vulcan/ap/rnd/202209/2915/iunURDnLX6vFDVdX1B06mdpF.jpg" },
    { "id": 17, "name": "The Legend of Zelda: Breath of the Wild", "genre": ["Adventure", "RPG"],"image_url":"https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRjYLoYERymCvKtHd8BzV5xK7PFpJCVcmceVA&s" },
    { "id": 18, "name": "Super Mario Odyssey", "genre": ["Adventure"],"image_url":"https://assets.nintendo.com/image/upload/c_fill,w_1200/q_auto:best/f_auto/dpr_2.0/ncom/software/switch/70010000001130/c42553b4fd0312c31e70ec7468c6c9bccd739f340152925b9600631f2d29f8b5" },
    { "id": 19, "name": "Animal Crossing: New Horizons", "genre": ["Simulation"],"image_url":"https://assets.nintendo.com/image/upload/c_fill,w_1200/q_auto:best/f_auto/dpr_2.0/ncom/software/switch/70010000027619/9989957eae3a6b545194c42fec2071675c34aadacd65e6b33fdfe7b3b6a86c3a" },
    { "id": 20, "name": "Resident Evil Village", "genre": ["Horror", "Action"],"image_url":"https://shared.akamai.steamstatic.com/store_item_assets/steam/apps/1196590/capsule_616x353.jpg?t=1719197703" },
    { "id": 21, "name": "Blue Archive", "genre": ["RPG", "Action"],"image_url":"https://mega.com.vn/media/news/2207_tong-quan-ve-blue-archive-1.jpg" },
    { "id": 22, "name": "Elden Ring", "genre": ["RPG", "Fantasy", "Souls-like"],"image_url":"https://image.api.playstation.com/vulcan/ap/rnd/202108/0410/UAnLUUMdxA9cow8TEe8IfhuC.png" },
    { "id": 23, "name": "Monster Hunter Wilds", "genre": ["Action", "Adventure"],"image_url":"https://shared.akamai.steamstatic.com/store_item_assets/steam/apps/2246340/capsule_616x353.jpg?t=1717124069" },
    { "id": 24, "name": "DARK SOULS: REMASTERED", "genre": ["Fantasy", "Action", "Souls-like"],"image_url":"https://image.api.playstation.com/cdn/HP0700/CUSA08526_00/Vx3DJTmQKYtoUsGMJQ0gPHOlcjvHQC8EM6teA1bbxFQVSXenyMVDPSaEgarNMIUp.png" },
    { "id": 25, "name": "Black Myth: Wukong", "genre": ["Adventure", "Action", "Souls-like"],"image_url":"https://shared.akamai.steamstatic.com/store_item_assets/steam/apps/2358720/f40ef565c380c617020e559b4b4b089edd93ec09/capsule_616x353.jpg?t=1725007201" },
    { "id": 26, "name": "Mush Dash", "genre": ["Rhythm", "Anime"],"image_url":"https://shared.akamai.steamstatic.com/store_item_assets/steam/apps/774171/capsule_616x353.jpg?t=1726048226" },
    { "id": 27, "name": "Until Then", "genre": ["Visual Novel", "Drama", "Romance"],"image_url":"https://shared.akamai.steamstatic.com/store_item_assets/steam/apps/1574820/capsule_616x353.jpg?t=1725485534" },
    { "id": 28, "name": "Persona 5 Royal", "genre": ["Visual Novel", "Anime", "RPG"],"image_url":"https://shared.akamai.steamstatic.com/store_item_assets/steam/apps/1687950/capsule_616x353.jpg?t=1722428154" },
    { "id": 29, "name": "Hades", "genre": ["Rogue-like", "Action"],"image_url":"https://cdn1.epicgames.com/min/offer/2560x1440-2560x1440-5e710b93049cbd2125cf0261dcfbf943.jpg" },
    { "id": 30, "name": "Genshin Impact", "genre": ["RPG", "Adventure", "Fantasy"],"image_url":"https://img.asmedia.epimg.net/resizer/v2/75F26RKWPZK3RNTTYK7C4FFDBI.jpg?auth=240bfeb8c6b05fb3d5de934cd173426d23a1907b9043d0aacc2bac8333a33ba7&width=1472&height=828&smart=true" },
    { "id": 31, "name": "Hollow Knight", "genre": ["Metroidvania", "Action", "Adventure"],"image_url":"https://assets.nintendo.com/image/upload/ar_16:9,c_lpad,w_1240/b_white/f_auto/q_auto/ncom/software/switch/70010000003208/4643fb058642335c523910f3a7910575f56372f612f7c0c9a497aaae978d3e51" },
    { "id": 32, "name": "Stardew Valley", "genre": ["Simulation", "RPG", "Farming"],"image_url":"https://shared.akamai.steamstatic.com/store_item_assets/steam/apps/413150/capsule_616x353.jpg?t=1711128146" },
    { "id": 33, "name": "League of Legends", "genre": ["MOBA", "Strategy"],"image_url":"https://i0.wp.com/highschool.latimes.com/wp-content/uploads/2021/09/league-of-legends.jpeg?fit=1607%2C895&ssl=1" },
    { "id": 34, "name": "Among Us", "genre": ["Party", "Strategy"],"image_url":"https://cdn1.epicgames.com/salesEvent/salesEvent/amoguslandscape_2560x1440-3fac17e8bb45d81ec9b2c24655758075" },
    { "id": 35, "name": "Celeste", "genre": ["Platformer", "Adventure"],"image_url":"https://assets.nintendo.com/image/upload/ar_16:9,c_lpad,w_1240/b_white/f_auto/q_auto/ncom/software/switch/70010000006442/691ba3e0801180a9864cc8a7694b6f98097f9d9799bc7e3dc6db92f086759252" },
    { "id": 36, "name": "Tetris Effect", "genre": ["Puzzle", "Rhythm"],"image_url":"https://upload.wikimedia.org/wikipedia/en/a/ae/Tetris_Effect_cover.jpg" },
    { "id": 37, "name": "Slay the Spire", "genre": ["Card Game", "Rogue-like"],"image_url":"https://shared.akamai.steamstatic.com/store_item_assets/steam/apps/646570/capsule_616x353.jpg?t=1712773871" },
    { "id": 38, "name": "Dead by Daylight", "genre": ["Horror", "Multiplayer", "Survival"],"image_url":"https://image.api.playstation.com/vulcan/ap/rnd/202203/1520/e6VbPQ56CBzVPgKmWdMMrICz.png" },
    { "id": 39, "name": "The Last of Us Part II Remastered", "genre": ["Action", "Adventure", "Horror"],"image_url":"https://image.api.playstation.com/vulcan/ap/rnd/202312/0117/da083fa5e19458dd750aa8a6ea30ba10bac6f87074693df5.jpg" },
    { "id": 40, "name": "Sekiro: Shadows Die Twice", "genre": ["Action", "Adventure", "Souls-like"],"image_url":"https://shared.akamai.steamstatic.com/store_item_assets/steam/apps/814380/capsule_616x353.jpg?t=1726158438" },
    { "id": 41, "name": "Horizon Zero Dawn", "genre": ["RPG", "Adventure", "Action"],"image_url":"https://image.api.playstation.com/vulcan/ap/rnd/202010/0221/vC7trMorHJgbImp8PCQvpI0p.png" },
    { "id": 42, "name": "Fire Emblem: Three Houses", "genre": ["Strategy", "RPG"],"image_url":"https://assets.nintendo.com/image/upload/ar_16:9,c_lpad,w_1240/b_white/f_auto/q_auto/ncom/software/switch/70010000058802/88b8db494399e659522af36331b7f588439bc4312af5c0ee57035c063651ab4a" },
    { "id": 43, "name": "Splatoon 2", "genre": ["Shooter", "Multiplayer"],"image_url":"https://assets.nintendo.com/image/upload/c_fill,w_1200/q_auto:best/f_auto/dpr_2.0/ncom/software/switch/70010000000529/b0b183a9860296016649fadb03b929411e7e5e0809af241e2e9652ebf0c5a715" },
    { "id": 44, "name": "Cuphead", "genre": ["Platformer", "Action"],"image_url":"https://assets.nintendo.com/image/upload/ar_16:9,c_lpad,w_1240/b_white/f_auto/q_auto/ncom/software/switch/70010000016330/d94d2186ef03c930392253c83c84af0c73b7e57cd902a526b09b4155a25930fe" },
    { "id": 45, "name": "Ghost of Tsushima", "genre": ["Action", "Adventure", "Open World"],"image_url":"https://image.api.playstation.com/vulcan/ap/rnd/202010/0222/b3iB2zf2xHj9shC0XDTULxND.png" },
    { "id": 46, "name": "Disco Elysium", "genre": ["RPG", "Adventure"],"image_url":"https://shared.akamai.steamstatic.com/store_item_assets/steam/apps/632470/capsule_616x353.jpg?t=1721654573" },
    { "id": 47, "name": "Death Stranding", "genre": ["Adventure", "Action", "Sci-Fi"],"image_url":"https://shared.akamai.steamstatic.com/store_item_assets/steam/apps/1850570/header.jpg?t=1725036030" },
    { "id": 48, "name": "Returnal", "genre": ["Action", "Rogue-like", "Sci-Fi"],"image_url":"https://image.api.playstation.com/vulcan/ap/rnd/202011/1621/fYZQHZ42eXXUt7c6D5YjLrq5.png" },
    { "id": 49, "name": "NieR: Automata The End of YoRHa", "genre": ["RPG", "Action", "Adventure"],"image_url":" https://fyre.cdn.sewest.net/neir-automata-nsw/627d1a967867700012c90b8b/package_us-hRh4ZporN.png" },
    { "id": 50, "name": "Doom Eternal", "genre": ["Shooter", "Action"],"image_url":"https://cdn1.epicgames.com/offer/b5ac16dc12f3478e99dcfea07c13865c/EGS_DOOMEternal_idSoftware_S1_2560x1440-06b46993a4b6c19a9e614f2dd1202215" },
    { "id": 51, "name": "Terraria", "genre": ["Sandbox", "Adventure"],"image_url":"https://www.online-station.net/wp-content/uploads/2023/07/Terraria_videogioco.jpg" },
    { "id": 52, "name": "Metal Gear Solid V: The Phantom Pain", "genre": ["Action", "Stealth", "Adventure"],"image_url":"https://image.api.playstation.com/vulcan/ap/rnd/202010/0205/dyvo9eGUf7WTZx49eTpQyDuL.png" },
    { "id": 53, "name": "Divinity: Original Sin 2", "genre": ["RPG", "Strategy", "Fantasy"],"image_url":"https://shared.akamai.steamstatic.com/store_item_assets/steam/apps/715950/capsule_616x353.jpg?t=1548949321" },
    { "id": 54, "name": "Xenoblade Chronicles 3", "genre": ["RPG", "Adventure", "Fantasy"],"image_url":"https://www.nintendo.com/th/switch/az3h/img/sns.png" },
    { "id": 55, "name": "Bayonetta 3", "genre": ["Action", "Adventure"],"image_url":"https://www.nintendo.com/eu/media/images/10_share_images/games_15/nintendo_switch_4/2x1_NSwitch_Bayonetta3_image1600w.jpg" },
    { "id": 56, "name": "Metroid Dread", "genre": ["Action", "Adventure", "Metroidvania"],"image_url":"https://www.nintendo.com/th/switch/ayl8/img/hero.jpg" },
    { "id": 57, "name": "It Takes Two", "genre": ["Co-op", "Adventure", "Puzzle"],"image_url":"https://shared.akamai.steamstatic.com/store_item_assets/steam/apps/1426210/header.jpg?t=1725585631" },
    { "id": 58, "name": "Kirby and the Forgotten Land", "genre": ["Platformer", "Adventure"],"image_url":"https://www.nintendo.com/th/switch/arzg/img/hero_sp.jpg" },
    { "id": 59, "name": "Hollow Knight: Silksong", "genre": ["Metroidvania", "Action", "Adventure"],"image_url":"https://shared.akamai.steamstatic.com/store_item_assets/steam/apps/1030300/capsule_616x353.jpg?t=1695443850" },
    { "id": 60, "name": "Sifu", "genre": ["Action", "Martial Arts"],"image_url":"https://cdn1.epicgames.com/offer/b7b42e2078524ab386a8b2a9856ef557/EGS_SIFUStandardEdition_Sloclap_G1A_03_1920x1080-1df3ab45b895adfee06b011673df20aa_1920x1080-1df3ab45b895adfee06b011673df20aa" }
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
            flex-direction: column; /* Switch to column layout */
            gap: 20px;
        }
        .game-item {
            display: flex;
            flex-direction: row;
            align-items: center;
            background-color: #f9f9f9;
            padding: 15px;
            border-radius: 6px;
            box-shadow: 0 1px 4px rgba(0, 0, 0, 0.1);
        }
        .game-item img {
            width: 150px; /* Adjust width for column layout */
            height: auto;
            border-radius: 4px;
            margin-right: 20px;
        }
        .game-info {
            display: flex;
            flex-direction: column;
            justify-content: center;
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
                    gameDiv.innerHTML = `
                        <img src="${game.image_url}" alt="${game.name}">
                        <div class="game-info">
                            <h3>${game.name}</h3>
                            <p>Genres: ${game.genre.join(', ')}</p>
                        </div>`;
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
                    gameDiv.innerHTML = `
                        <img src="${game.image_url}" alt="${game.name}">
                        <div class="game-info">
                            <h3>${game.name}</h3>
                            <p>Genres: ${game.genre.join(', ')}</p>
                        </div>`;
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
                gameDiv.innerHTML = `
                    <img src="${game.image_url}" alt="${game.name}">
                    <div class="game-info">
                        <h3>${game.name}</h3>
                        <p>Genres: ${game.genre.join(', ')}</p>
                    </div>`;
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
