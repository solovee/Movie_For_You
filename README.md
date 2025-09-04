Sistema de recomendação User-User de Filmes baseado no dataset: https://www.kaggle.com/datasets/grouplens/movielens-latest-small.
Ele se baseia no algoritmo K-top cosine a threshold de similaridade
com diferentes estratégias de inferência sendo elas: filtragem com 100% dos filmes (participa do algoritmo apenas aqueles que viram pelo menos todos os filmes que o usuário avaliou), em caso de não similaridade ou zero usuários pós filtragem, continuo filtragens cada vez menos rígidas sendo elas:
75% dos filmes aleátorios
75% dos filmes mais populares
repete o processo para 50% e 25%
