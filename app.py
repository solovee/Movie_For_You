import pandas as pd
import numpy as np
import pickle
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

# Crie sua instância do app Flask
app = Flask(__name__)
CORS(app)  # Permite requisições do frontend

# ---------------------------
# Carregamento de dados
# ---------------------------
try:
    movies_df = pd.read_csv("ml-latest/movies.csv")
    ratings_df = pd.read_csv("ml-latest-small/ratings.csv")
    print("CSV de filmes e ratings carregados com sucesso.")
except FileNotFoundError as e:
    print(f"Erro: arquivo não encontrado - {e}")
    movies_df = None
    ratings_df = None

# Filtra ratings para manter apenas filmes que existem em movies.csv
if movies_df is not None and ratings_df is not None:
    ratings_df = ratings_df[ratings_df['movieId'].isin(movies_df['movieId'])].copy()
    movie_ids_com_ratings = ratings_df['movieId'].unique()
    movies_usados_df = movies_df[movies_df['movieId'].isin(movie_ids_com_ratings)].reset_index(drop=True)
    print(f"Número de filmes disponíveis: {len(movies_usados_df)}")
else:
    movies_usados_df = None

try:
    movie_embs = np.load("movie_embeddings.npy")  # shape (n_movies, emb_dim)
    print(f"Embeddings de filmes carregados: shape {movie_embs.shape}")
except FileNotFoundError:
    print("Erro: arquivo 'movie_embeddings.npy' não encontrado.")
    movie_embs = None

try:
    with open("movie2idx.pkl", "rb") as f:
        movie2idx = pickle.load(f)
    print(f"Mapa movie2idx carregado: {len(movie2idx)} filmes")
except FileNotFoundError:
    print("Erro: arquivo 'movie2idx.pkl' não encontrado.")
    movie2idx = None

# Cria o mapeamento inverso (idx -> movieId)
if movie2idx is not None:
    idx2movie = {idx: mid for mid, idx in movie2idx.items()}
else:
    idx2movie = None

# ---------------------------

def gerar_usuario_embedding(meus_ratings):
    """Gera embedding do usuário a partir de suas avaliações"""
    if movie2idx is None or movie_embs is None:
        return None
    
    # Filtra apenas filmes que existem no mapeamento
    rated_movies = [mid for mid in meus_ratings if mid in movie2idx]
    
    if not rated_movies:
        print("Nenhum filme avaliado está no dataset de embeddings.")
        return None
    
    # Obtém os índices dos filmes avaliados
    indices = [movie2idx[mid] for mid in rated_movies]
    
    # Obtém as notas normalizadas
    ratings = np.array([meus_ratings[mid] for mid in rated_movies], dtype=np.float32).reshape(-1, 1)
    
    # Obtém os embeddings dos filmes avaliados
    emb_filmes = movie_embs[indices]  # shape (n_rated, emb_dim)
    
    # Calcula embedding do usuário como média ponderada
    u_emb = (ratings * emb_filmes).sum(axis=0) / ratings.sum()
    
    return u_emb

def recomendar_top_filmes(u_emb, meus_ratings, top_n=3):
    """Retorna os títulos dos top_n filmes mais similares ao usuário"""
    if movie_embs is None or movie2idx is None or idx2movie is None:
        return []
    
    # Calcula similaridade via produto escalar (cosine similarity se embeddings normalizados)
    scores = movie_embs @ u_emb  # shape (n_movies,)
    
    # Cria máscara para filmes já avaliados
    avaliados_idx = set([movie2idx[mid] for mid in meus_ratings if mid in movie2idx])
    
    # Zera scores dos filmes já avaliados
    scores_filtered = scores.copy()
    for idx in avaliados_idx:
        scores_filtered[idx] = -np.inf
    
    # Obtém top_n filmes com maior score
    top_indices = scores_filtered.argsort()[::-1][:top_n]
    
    # Converte índices para movieIds
    top_movieIds = [idx2movie[idx] for idx in top_indices if idx in idx2movie]
    
    # Busca títulos dos filmes - RETORNA APENAS OS TÍTULOS
    filmes_recomendados = []
    for mid in top_movieIds:
        filme = movies_usados_df[movies_usados_df['movieId'] == mid]
        if not filme.empty:
            filmes_recomendados.append(filme.iloc[0]['title'])
    
    return filmes_recomendados


@app.route('/')
def home():
    """Rota principal que renderiza a página HTML"""
    return render_template('index.html')

@app.route('/api/filmes', methods=['GET'])
def get_filmes():
    """Rota que retorna todos os filmes do CSV"""
    if movies_usados_df is not None:
        return movies_usados_df.to_json(orient='records'), 200
    return jsonify({"error": "Filmes não disponíveis"}), 500

@app.route('/api/recomendar', methods=['POST'])
def processar_recomendacao():
    """Rota que recebe ratings do usuário e retorna recomendações"""
    if movies_usados_df is None or movie_embs is None or movie2idx is None:
        return jsonify({"error": "Base de dados não carregada."}), 500

    try:
        meus_ratings = request.json.get('ratings', {})
        # Converte chaves para int e valores para float
        meus_ratings = {int(k): float(v) for k, v in meus_ratings.items()}
        
        if len(meus_ratings) < 3:
            return jsonify({"error": "Avalie pelo menos 3 filmes para receber recomendações."}), 400

        # Gera embedding do usuário
        u_emb = gerar_usuario_embedding(meus_ratings)
        if u_emb is None:
            return jsonify({"error": "Nenhum filme avaliado está no dataset de embeddings."}), 400
        
        # Gera recomendações (retorna lista de strings com títulos)
        filmes_recomendados = recomendar_top_filmes(u_emb, meus_ratings, top_n=3)
        
        if not filmes_recomendados:
            return jsonify({"error": "Não foi possível gerar recomendações."}), 400
        
        # Garante que está retornando uma lista simples de strings
        return jsonify({
            "status": "success",
            "filmes_recomendados": list(filmes_recomendados)  # força conversão para lista
        }), 200
    
    except Exception as e:
        print(f"Erro na recomendação: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Ocorreu um erro interno: {str(e)}"}), 500

@app.route('/api/status', methods=['GET'])
def get_status():
    """Rota para verificar o status do sistema"""
    status = {
        "movies_loaded": movies_usados_df is not None,
        "embeddings_loaded": movie_embs is not None,
        "movie2idx_loaded": movie2idx is not None,
        "num_movies": len(movies_usados_df) if movies_usados_df is not None else 0,
        "embedding_dim": movie_embs.shape[1] if movie_embs is not None else 0
    }
    return jsonify(status), 200

# ---------------------------
# Inicialização do app
# ---------------------------
if __name__ == '__main__':
    app.run(debug=True, port=5000)