import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

# Crie sua instância do app Flask
app = Flask(__name__)
CORS(app)  # Permite requisições do seu frontend

# Carregue os dados de popularidade e filmes usados
# A rota abaixo precisa ser a correta para seu projeto
try:
    df_pivot = pd.read_csv("df_pivot.csv", index_col="userId")
    df_pivot.columns = df_pivot.columns.astype(int)
    # Substitua NaN por 0 para evitar erros no KNN, se necessário
    #df_pivot = df_pivot.fillna(0)
    print("DataFrame pivotado carregado com sucesso.")
except FileNotFoundError:
    print("Erro: O arquivo 'df_pivot.csv' não foi encontrado. Verifique o caminho.")
    df_pivot = None

try:
    df_filmes = pd.read_csv("filmes_usados.csv")
    print("DataFrame de filmes carregado com sucesso.")
except FileNotFoundError:
    print("Erro: O arquivo 'filmes_usados.csv' não foi encontrado. Verifique o caminho.")
    df_filmes = None

# Suas funções de recomendação (copie e cole aqui)
def recomendar_usuario(df_pivot, meus_ratings, limiar_sim=0.7, random_state=42, caminho_popularidade="popularidade_aplicacao.csv"):
    # ... (cole sua função recomendar_usuario aqui) ...
    np.random.seed(random_state)
    minha_linha = pd.Series(meus_ratings, name='meu_usuario')
    meus_filmes = list(meus_ratings.keys())
    
    df_pop = pd.read_csv(caminho_popularidade)
    popularidade = df_pop.set_index("movieId")["popularidade"].to_dict()
    
    # ... (o restante da sua função) ...
    estrategias = [
        (1.0, 'todos'),
        (0.75, 'aleatorio'),
        (0.75, 'populares'),
        (0.5, 'aleatorio'),
        (0.5, 'populares'),
        (0.25, 'aleatorio'),
        (0.25, 'populares')
    ]
    
    melhor_usuario = None
    melhor_sim = -1
    vizinhos_extra = []

    for perc, modo in estrategias:
        print(f"\n🔎 Testando estratégia: {modo.upper()} ({int(perc*100)}% dos filmes)")

        if perc < 1.0:
            n_filmes = max(1, int(len(meus_filmes) * perc))
            if modo == 'aleatorio':
                filmes_selecionados = list(np.random.choice(meus_filmes, size=n_filmes, replace=False))
            elif modo == 'populares':
                filmes_ordenados = [f for f in meus_filmes if f in popularidade]
                filmes_ordenados.sort(key=lambda x: popularidade[x], reverse=True)
                filmes_selecionados = filmes_ordenados[:n_filmes]
        else:
            filmes_selecionados = meus_filmes
        
        print(f"🎬 Filmes usados nesta rodada: {filmes_selecionados}")

        df_filtrado = df_pivot.dropna(subset=filmes_selecionados, how='any')
        print(f"📊 Usuários restantes após filtro: {df_filtrado.shape[0]}")
        
        if df_filtrado.shape[0] == 0:
            print("⚠️ Nenhum usuário encontrado para essa seleção de filmes.")
            continue
        
        X = df_filtrado[filmes_selecionados].values
        minha_vetor = minha_linha[filmes_selecionados].values.reshape(1, -1)
        
        n_vizinhos = min(4, df_filtrado.shape[0]) 
        knn = NearestNeighbors(n_neighbors=n_vizinhos, metric='cosine')
        knn.fit(X)
        dist, ind = knn.kneighbors(minha_vetor, n_neighbors=n_vizinhos)
        
        dist = dist[0]
        ind = ind[0]
        
        sims = 1 - dist
        usuarios = df_filtrado.index[ind].tolist()
        
        print("👥 Vizinhos encontrados:")
        for u, s in zip(usuarios, sims):
            print(f"   - {u} (similaridade={s:.4f})")
        
        sim = sims[0]
        usuario_encontrado = usuarios[0]
        
        if sim > limiar_sim:
            print("✅ Similaridade acima do limiar, parando busca.")
            melhor_usuario = usuario_encontrado
            melhor_sim = sim
            vizinhos_extra = list(zip(usuarios[1:], sims[1:]))
            break
        else:
            print("❌ Similaridade baixa, tentando próxima estratégia...")

    return {
        'usuario_mais_similar': melhor_usuario,
        'similaridade': melhor_sim,
        'filmes_usados': meus_filmes,
        'vizinhos_extras': vizinhos_extra
    }

def recomendar_filmes(df_pivot, resultado_recomendar, meus_ratings, top_n=3, min_rating=4):
    # ... (cole sua função recomendar_filmes aqui) ...
    candidatos = [resultado_recomendar["usuario_mais_similar"]] + resultado_recomendar.get("vizinhos_extras", [])
    filmes_assistidos = set(meus_ratings.keys())
    
    for user_id in candidatos:
        if user_id is None:
            continue
        
        print(f"\n🎯 Tentando recomendar filmes do usuário {user_id}...")
        linha_usuario = df_pivot.loc[user_id]
        
        filmes_ordenados = (
            linha_usuario.dropna()
            .sort_values(ascending=False)
        )
        
        filmes_validos = [mid for mid, nota in filmes_ordenados.items()
                          if mid not in filmes_assistidos and nota >= min_rating]
        
        if len(filmes_validos) >= top_n:
            recomendados = filmes_validos[:top_n]
            print(f"✅ Filmes recomendados: {recomendados}")
            return recomendados
        else:
            print(f"⚠️ Usuário {user_id} não tem filmes suficientes (>= {min_rating}) para recomendar.")
    
    print("❌ Nenhum usuário tinha recomendações válidas.")
    return []

@app.route('/')
def home():
    """Rota principal que renderiza a página HTML."""
    return render_template('index.html')

@app.route('/api/filmes', methods=['GET'])
def get_filmes():
    """Rota que retorna a lista de todos os filmes do CSV."""
    if df_filmes is not None:
        filmes_json = df_filmes.to_json(orient='records')
        return filmes_json, 200
    return jsonify({"error": "Filmes data not available"}), 500

@app.route('/api/recomendar', methods=['POST'])
def processar_recomendacao():
    """Rota que recebe os ratings do usuário e retorna as recomendações."""
    if df_pivot is None or df_filmes is None:
        return jsonify({"error": "Database not loaded."}), 500

    meus_ratings = request.json.get('ratings', {})
    meus_ratings_int = {int(k): float(v) for k, v in meus_ratings.items()}
    
    if len(meus_ratings_int) < 7:
        return jsonify({"error": "Avalie pelo menos 7 filmes para receber recomendações."}), 400

    try:
        resultado_usuario = recomendar_usuario(df_pivot, meus_ratings_int)
        
        if resultado_usuario["usuario_mais_similar"] is None:
            return jsonify({
                "error": "Não foi possível encontrar um usuário similar o suficiente.",
                "detalhes": "Tente avaliar mais filmes, especialmente os mais populares."
            }), 404

        ids_recomendados = recomendar_filmes(df_pivot, resultado_usuario, meus_ratings_int, top_n=3)
        
        if not ids_recomendados:
            return jsonify({
                "error": "Não foi possível encontrar filmes para recomendar.",
                "detalhes": "O usuário similar pode não ter filmes com nota alta o suficiente."
            }), 404
        
        filmes_recomendados = df_filmes[df_filmes['movieId'].isin(ids_recomendados)]['title'].tolist()
        
        return jsonify({
            "status": "success",
            "filmes_recomendados": filmes_recomendados
        }), 200
        
    except Exception as e:
        print(f"Erro na recomendação: {e}")
        return jsonify({"error": f"Ocorreu um erro interno: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)