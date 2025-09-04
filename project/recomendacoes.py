import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors


def recomendar_usuario(df_pivot, meus_ratings, limiar_sim=0.7, random_state=42, caminho_popularidade="project/popularidade_aplicacao.csv"):
    """
    df_pivot: DataFrame pivotado userId x movieId com ratings (1-5, NaN se não avaliou)
    meus_ratings: dict {movieId: rating}
    limiar_sim: float, limiar mínimo de similaridade cosine
    caminho_popularidade: str, caminho para o CSV com colunas [movieId, popularidade]
    """
    np.random.seed(random_state)
    
    # Cria a linha do usuário
    minha_linha = pd.Series(meus_ratings, name='meu_usuario')
    
    # Lista de IDs dos filmes que forneceu
    meus_filmes = list(meus_ratings.keys())
    
    # 🔹 Lê o CSV de popularidade
    df_pop = pd.read_csv(caminho_popularidade)
    popularidade = df_pop.set_index("movieId")["popularidade"].to_dict()
    
    # Estratégias de filtragem: (percentual de filmes, modo)
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

        # Escolher filmes para a rodada
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

        # Filtra usuários que avaliaram todos os filmes selecionados
        df_filtrado = df_pivot.dropna(subset=filmes_selecionados, how='any')
        print(f"📊 Usuários restantes após filtro: {df_filtrado.shape[0]}")
        
        if df_filtrado.shape[0] == 0:
            print("⚠️ Nenhum usuário encontrado para essa seleção de filmes.")
            continue
        
        # Construir matriz para cálculo de similaridade
        X = df_filtrado[filmes_selecionados].values
        minha_vetor = minha_linha[filmes_selecionados].values.reshape(1, -1)
        
        # KNN cosine (pegar até 4 vizinhos: 1 principal + 3 extras)
        n_vizinhos = min(4, df_filtrado.shape[0])  
        knn = NearestNeighbors(n_neighbors=n_vizinhos, metric='cosine')
        knn.fit(X)
        dist, ind = knn.kneighbors(minha_vetor, n_neighbors=n_vizinhos)
        
        dist = dist[0]
        ind = ind[0]
        
        # Calcular similaridades
        sims = 1 - dist
        usuarios = df_filtrado.index[ind].tolist()
        
        print("👥 Vizinhos encontrados:")
        for u, s in zip(usuarios, sims):
            print(f"   - {u} (similaridade={s:.4f})")
        
        # Melhor usuário (primeiro da lista)
        sim = sims[0]
        usuario_encontrado = usuarios[0]
        
        if sim > limiar_sim:
            print("✅ Similaridade acima do limiar, parando busca.")
            melhor_usuario = usuario_encontrado
            melhor_sim = sim
            vizinhos_extra = list(zip(usuarios[1:], sims[1:]))  # próximos 3
            break
        else:
            print("❌ Similaridade baixa, tentando próxima estratégia...")

    return {
        'usuario_mais_similar': melhor_usuario,
        'similaridade': melhor_sim,
        'filmes_usados': meus_filmes,
        'vizinhos_extras': vizinhos_extra  # lista de (usuario, similaridade)
    }





def recomendar_filmes(df_pivot, resultado_recomendar, meus_ratings, top_n=3, min_rating=4):
    """
    df_pivot: DataFrame pivotado userId x movieId com ratings
    resultado_recomendar: dict retornado pela função recomendar_usuario
    meus_ratings: dict {movieId: rating} -> filmes já assistidos pelo usuário
    top_n: quantidade de filmes a recomendar
    min_rating: nota mínima para recomendar
    
    Retorna: lista de movieIds recomendados
    """
    candidatos = [resultado_recomendar["usuario_mais_similar"]] + resultado_recomendar.get("vizinhos_extras", [])
    filmes_assistidos = set(meus_ratings.keys())
    
    for user_id in candidatos:
        if user_id is None:
            continue
        
        print(f"\n🎯 Tentando recomendar filmes do usuário {user_id}...")
        linha_usuario = df_pivot.loc[user_id]
        
        # Ordena os filmes avaliados pelo usuário (maior rating primeiro)
        filmes_ordenados = (
            linha_usuario.dropna()
            .sort_values(ascending=False)
        )
        
        # Filtra apenas filmes não assistidos e com rating >= min_rating
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
