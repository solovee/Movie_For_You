import os
import pandas as pd
import numpy as np
import pickle
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer

# ---------------------------
# Inicialização do modelo de embeddings
# ---------------------------
model_st = SentenceTransformer("all-MiniLM-L6-v2")
print("Modelo SentenceTransformer carregado.")

# ---------------------------
# Inicialização Flask
# ---------------------------
app = Flask(__name__)
CORS(app)

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

# ---------------------------
# Carregamento de embeddings pré-existentes
# ---------------------------
try:
    movie_embs = np.load("movie_embeddings.npy")  # shape (n_movies, emb_dim)
    print(f"Embeddings de filmes carregados: shape {movie_embs.shape}")
except FileNotFoundError:
    print("Aviso: arquivo 'movie_embeddings.npy' não encontrado.")
    movie_embs = None

try:
    with open("movie2idx.pkl", "rb") as f:
        movie2idx = pickle.load(f)
    print(f"Mapa movie2idx carregado: {len(movie2idx)} filmes")
except FileNotFoundError:
    print("Aviso: arquivo 'movie2idx.pkl' não encontrado.")
    movie2idx = None

if movie2idx is not None:
    idx2movie = {idx: mid for mid, idx in movie2idx.items()}
else:
    idx2movie = None

# ---------------------------
# Carregamento de tags
# ---------------------------
try:
    tags_df = pd.read_csv("ml-latest-small/tags.csv")  # deve ter 'movieId' e 'tag'
    print(f"Tags carregadas: {len(tags_df)} registros")
except FileNotFoundError:
    print("Aviso: arquivo 'tags.csv' não encontrado.")
    tags_df = None

# Agrupa tags por filme
if tags_df is not None and movies_usados_df is not None:
    tags_agg = tags_df.groupby('movieId')['tag'].apply(lambda x: ' '.join(x.dropna().astype(str))).reset_index()
    movies_tags_df = movies_usados_df.merge(tags_agg, on='movieId', how='left')
    movies_tags_df['tag'] = movies_tags_df['tag'].fillna("")
else:
    movies_tags_df = movies_usados_df.copy()
    movies_tags_df['tag'] = ""

# ---------------------------
# Embeddings semânticos das TAGS dos filmes (SentenceTransformer)
# ---------------------------
def gerar_embeddings_tags_filmes_st(movies_tags_df):
    """
    Gera embeddings semânticos das tags agregadas por filme usando SentenceTransformer.
    """
    print("Gerando embeddings semânticos para tags de filmes com SentenceTransformer...")
    tag_texts = movies_tags_df['tag'].tolist()
    embeddings = model_st.encode(tag_texts, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    return embeddings.astype(np.float32)

try:
    movie_tag_embs = np.load("movie_tag_embs.npy")
    print(f"Embeddings de tags carregados: shape {movie_tag_embs.shape}")
except FileNotFoundError:
    if movies_tags_df is not None:
        movie_tag_embs = gerar_embeddings_tags_filmes_st(movies_tags_df)
        np.save("movie_tag_embs.npy", movie_tag_embs)
        print("Embeddings de tags gerados e salvos.")
    else:
        movie_tag_embs = None

# ---------------------------
# Funções de recomendação
# ---------------------------
def gerar_usuario_embedding(meus_ratings):
    """
    Gera embedding do usuário a partir de suas avaliações.
    Normaliza o embedding para que possa ser usado em similaridade coseno.
    """
    if movie2idx is None or movie_embs is None:
        return None

    # Filtra apenas filmes que possuem embedding
    rated_movies = [mid for mid in meus_ratings if mid in movie2idx]
    if not rated_movies:
        print("Nenhum filme avaliado possui embedding.")
        return None

    indices = [movie2idx[mid] for mid in rated_movies]
    ratings = np.array([meus_ratings[mid] for mid in rated_movies], dtype=np.float32).reshape(-1, 1)
    emb_filmes = movie_embs[indices]

    # Embedding do usuário como média ponderada pelos ratings
    u_emb = (ratings * emb_filmes).sum(axis=0) / ratings.sum()

    # Normaliza para similaridade coseno
    u_emb /= np.linalg.norm(u_emb) + 1e-8

    print(f'Filmes avaliados: {rated_movies}')
    print(f'Embedding do usuário: {u_emb[:5]} ...')  # mostra só os 5 primeiros valores para debug
    return u_emb


def recomendar_top_filmes(u_emb, meus_ratings, top_n=5):
    """
    Retorna os títulos dos top_n filmes mais similares ao usuário.
    Filtra filmes já avaliados.
    """
    if movie_embs is None or movie2idx is None or idx2movie is None:
        return []

    # Normaliza embeddings dos filmes
    movie_embs_norm = movie_embs / (np.linalg.norm(movie_embs, axis=1, keepdims=True) + 1e-8)

    # Calcula similaridade coseno
    scores = movie_embs_norm @ u_emb

    # Filtra filmes já avaliados
    avaliados_idx = set([movie2idx[mid] for mid in meus_ratings if mid in movie2idx])
    scores_filtered = scores.copy()
    for idx in avaliados_idx:
        scores_filtered[idx] = -np.inf

    # Seleciona top N
    top_indices = scores_filtered.argsort()[::-1][:top_n]
    top_movieIds = [idx2movie[idx] for idx in top_indices if idx in idx2movie]

    filmes_recomendados = []
    for mid in top_movieIds:
        filme = movies_usados_df[movies_usados_df['movieId'] == mid]
        if not filme.empty:
            filmes_recomendados.append(filme.iloc[0]['title'])

    print(f'Top {top_n} filmes recomendados: {filmes_recomendados}')
    return filmes_recomendados


def recomendar_por_tag(tag_input, top_n=5):
    """Busca filmes semanticamente similares à tag fornecida usando SentenceTransformer."""
    if movie_tag_embs is None:
        return []
    
    tag_emb = model_st.encode([tag_input], convert_to_numpy=True, normalize_embeddings=True)
    tag_emb = tag_emb[0].astype(np.float32)
    
    scores = movie_tag_embs @ tag_emb
    top_indices = scores.argsort()[::-1][:top_n]
    recomendados = movies_tags_df.iloc[top_indices]['title'].tolist()
    
    return recomendados

from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------
# Função item-to-item similarity
# ---------------------------
def recomendar_item_to_item(movie_id, top_n=5):
    """Retorna filmes mais similares a um filme específico usando embeddings."""
    if movie_embs is None or movie2idx is None or idx2movie is None:
        return []
    
    if movie_id not in movie2idx:
        return []
    
    movie_idx = movie2idx[movie_id]
    sims = cosine_similarity([movie_embs[movie_idx]], movie_embs)[0]
    
    # Exclui o próprio filme
    sims[movie_idx] = -np.inf
    
    top_indices = sims.argsort()[::-1][:top_n]
    top_movieIds = [idx2movie[idx] for idx in top_indices if idx in idx2movie]
    
    filmes_recomendados = []
    for mid in top_movieIds:
        filme = movies_usados_df[movies_usados_df['movieId'] == mid]
        if not filme.empty:
            filmes_recomendados.append(filme.iloc[0]['title'])
    
    return filmes_recomendados

# ---------------------------
# Rota Flask
# ---------------------------
@app.route('/api/recomendar_item', methods=['POST'])
def recomendar_item_api():
    """Recebe movieId e retorna top N filmes mais similares (item-to-item)."""
    data = request.get_json()
    movie_id = data.get("movieId", None)
    top_n = data.get("top_n", 5)
    
    if movie_id is None:
        return jsonify({"error": "Nenhum movieId fornecido."}), 400
    
    recomendados = recomendar_item_to_item(movie_id, top_n=top_n)
    if not recomendados:
        return jsonify({"error": "Não foi possível gerar recomendações."}), 400
    
    return jsonify({"status": "success", "recomendados": recomendados}), 200


# ---------------------------
# Rotas Flask
# ---------------------------
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/filmes', methods=['GET'])
def get_filmes():
    if movies_usados_df is not None:
        return movies_usados_df.to_json(orient='records'), 200
    return jsonify({"error": "Filmes não disponíveis"}), 500

@app.route('/api/recomendar', methods=['POST'])
def processar_recomendacao():
    if movies_usados_df is None or movie_embs is None or movie2idx is None:
        return jsonify({"error": "Base de dados não carregada."}), 500

    try:
        meus_ratings = request.json.get('ratings', {})
        meus_ratings = {int(k): float(v) for k, v in meus_ratings.items()}
        
        if len(meus_ratings) < 1:
            return jsonify({"error": "Avalie pelo menos um filme."}), 400

        u_emb = gerar_usuario_embedding(meus_ratings)
        if u_emb is None:
            return jsonify({"error": "Nenhum filme avaliado possui embedding."}), 400
        
        filmes_recomendados = recomendar_top_filmes(u_emb, meus_ratings, top_n=5)
        if not filmes_recomendados:
            return jsonify({"error": "Não foi possível gerar recomendações."}), 400
        
        return jsonify({
            "status": "success",
            "filmes_recomendados": list(filmes_recomendados)
        }), 200
    
    except Exception as e:
        print(f"Erro na recomendação: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/recomendar_tag', methods=['POST'])
def recomendar_por_tag_api():
    """Recomenda filmes semelhantes a uma tag/palavra/frase."""
    data = request.get_json()
    tag_input = data.get("tag", "")
    if not tag_input:
        return jsonify({"error": "Nenhuma tag fornecida."}), 400
    
    recomendados = recomendar_por_tag(tag_input, top_n=5)
    return jsonify({"status": "success", "recomendados": recomendados}), 200

@app.route('/api/status', methods=['GET'])
def get_status():
    status = {
        "movies_loaded": movies_usados_df is not None,
        "embeddings_loaded": movie_embs is not None,
        "tag_embs_loaded": movie_tag_embs is not None,
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
