from flask import Flask, render_template, request, jsonify
import pandas as pd
from recomendacoes import recomendar_usuario, recomendar_filmes  
import numpy as np


# Carrega os filmes
df_filmes = pd.read_csv("project/filmes_usados.csv")  # colunas: movieId, title
df_pivot = pd.read_parquet("project/df_pivot.parquet", engine="fastparquet")
df_pivot = df_pivot.astype(np.float32)
df_pivot.columns = df_pivot.columns.astype(int)

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html", filmes=df_filmes.to_dict(orient="records"))

@app.route("/recomendar", methods=["POST"])
def recomendar():
    dados = request.json  # lista de {movieId, nota}
    if len(dados) < 7:
        return jsonify({"erro": "Selecione pelo menos 7 filmes"}), 400

    meus_ratings = {int(item["movieId"]): float(item["nota"]) for item in dados}

    # 🔹 Filtra apenas os filmes que realmente existem no df_pivot
    filmes_existentes = [f for f in meus_ratings.keys() if f in df_pivot.columns]

    if not filmes_existentes:
        return jsonify({"erro": "Nenhum dos filmes selecionados existe no dataset!"}), 400

    # 🔹 Passa apenas os ratings que correspondem a filmes existentes
    meus_ratings_filtrados = {f: meus_ratings[f] for f in filmes_existentes}

    resultado_usuario = recomendar_usuario(df_pivot, meus_ratings_filtrados, limiar_sim=0.7)
    recomendados_ids = recomendar_filmes(df_pivot, resultado_usuario, meus_ratings_filtrados, top_n=3)

    # Converte IDs para nomes
    recomendados = df_filmes[df_filmes["movieId"].isin(recomendados_ids)][["movieId","title"]].to_dict(orient="records")

    return jsonify({"recomendados": recomendados})

if __name__ == "__main__":
    app.run(debug=True)
