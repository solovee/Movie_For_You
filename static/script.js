document.addEventListener('DOMContentLoaded', () => {
    const catalogBtn = document.getElementById('open-catalog-btn');
    const recommendBtn = document.getElementById('recommend-btn');
    const catalogModal = document.getElementById('catalog-modal');
    const loadingModal = document.getElementById('loading-modal');
    const closeBtn = catalogModal.querySelector('.close-btn');
    const searchBar = document.getElementById('search-bar');
    const movieCatalogList = document.getElementById('movie-catalog');
    const ratingsList = document.getElementById('ratings-list');
    const ratedCountSpan = document.getElementById('rated-count');
    const recommendationsList = document.getElementById('recommendations-list');
    const resultCard = document.getElementById('card-result');
    const toast = document.getElementById('toast');

    let allMovies = [];
    let userRatings = {}; // { movieId: rating, ... }

    // Função para buscar filmes do backend
    const fetchMovies = async () => {
        try {
            const response = await fetch('http://127.0.0.1:5000/api/filmes');
            if (!response.ok) {
                throw new Error('Falha ao buscar a lista de filmes.');
            }
            allMovies = await response.json();
            displayMovies(allMovies);
        } catch (error) {
            showToast('Erro ao carregar catálogo: ' + error.message, 'error');
            console.error(error);
        }
    };

    // Função para exibir a lista de filmes no modal
    const displayMovies = (movies) => {
        movieCatalogList.innerHTML = '';
        movies.forEach(movie => {
            const li = document.createElement('li');
            li.setAttribute('data-movie-id', movie.movieId);
            li.innerHTML = `
                <span>${movie.title}</span>
                <div class="rating-widget">
                    <i class="fa-solid fa-star" data-rating="1"></i>
                    <i class="fa-solid fa-star" data-rating="2"></i>
                    <i class="fa-solid fa-star" data-rating="3"></i>
                    <i class="fa-solid fa-star" data-rating="4"></i>
                    <i class="fa-solid fa-star" data-rating="5"></i>
                </div>
            `;
            movieCatalogList.appendChild(li);
        });
    };

    // Função para renderizar as avaliações do usuário
    const renderUserRatings = () => {
        ratingsList.innerHTML = '';
        const ratedMovies = Object.keys(userRatings).map(id => {
            const movie = allMovies.find(m => m.movieId == id);
            return {
                id: id,
                title: movie ? movie.title : `Filme ${id}`,
                rating: userRatings[id]
            };
        });

        ratedMovies.forEach(movie => {
            const li = document.createElement('li');
            const ratingDisplay = movie.rating % 1 === 0 ? movie.rating : movie.rating + '½';
            li.innerHTML = `
                <span>${movie.title}</span>
                <span class="rating-display">${ratingDisplay} <i class="fa-solid fa-star active"></i></span>
            `;
            ratingsList.appendChild(li);
        });

        ratedCountSpan.textContent = ratedMovies.length;
        recommendBtn.disabled = ratedMovies.length < 7;
    };

    // Função para enviar os dados e buscar recomendações
    const getRecommendations = async () => {
        if (Object.keys(userRatings).length < 7) {
            showToast('Por favor, avalie no mínimo 7 filmes.', 'warning');
            return;
        }

        loadingModal.style.display = 'flex';
        
        try {
            const response = await fetch('http://127.0.0.1:5000/api/recomendar', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ ratings: userRatings })
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || 'Erro na requisição.');
            }

            displayRecommendations(data.filmes_recomendados);
            showToast('Recomendações geradas com sucesso!', 'success');
        } catch (error) {
            showToast('Erro: ' + error.message, 'error');
            console.error('Erro na recomendação:', error);
            recommendationsList.innerHTML = `<li class="error-msg">${error.message}</li>`;
        } finally {
            loadingModal.style.display = 'none';
        }
    };

    // Função para exibir as recomendações
    const displayRecommendations = (movies) => {
        recommendationsList.innerHTML = '';
        if (movies.length === 0) {
            recommendationsList.innerHTML = `<li>Nenhuma recomendação encontrada.</li>`;
        } else {
            movies.forEach(title => {
                const li = document.createElement('li');
                li.textContent = title;
                recommendationsList.appendChild(li);
            });
        }
        resultCard.style.display = 'block';
    };

    // Função para mostrar notificação temporária
    const showToast = (message, type = 'info') => {
        toast.textContent = message;
        toast.className = `toast show ${type}`;
        setTimeout(() => {
            toast.className = 'toast';
        }, 3000);
    };

    // Event Listeners
    catalogBtn.addEventListener('click', () => {
        catalogModal.style.display = 'flex';
        displayMovies(allMovies);
    });

    closeBtn.addEventListener('click', () => {
        catalogModal.style.display = 'none';
    });

    window.addEventListener('click', (event) => {
        if (event.target === catalogModal) {
            catalogModal.style.display = 'none';
        }
    });

    searchBar.addEventListener('input', (event) => {
        const query = event.target.value.toLowerCase();
        const filteredMovies = allMovies.filter(movie => movie.title.toLowerCase().includes(query));
        displayMovies(filteredMovies);
    });

    movieCatalogList.addEventListener('click', (event) => {
        const target = event.target;
        if (target.closest('.rating-widget')) {
            const li = target.closest('li');
            const movieId = li.getAttribute('data-movie-id');
            const rating = parseFloat(target.getAttribute('data-rating'));
            
            // Lógica para arredondar para 0.5
            const rect = target.getBoundingClientRect();
            const clickX = event.clientX - rect.left;
            let finalRating = rating;
            if (clickX < rect.width / 2) {
                finalRating = rating - 0.5;
            }

            userRatings[movieId] = finalRating;
            renderUserRatings();
            showToast(`"${li.querySelector('span').textContent}" avaliado com nota ${finalRating}.`, 'info');
            
            // Atualizar estrelas visualmente
            const stars = li.querySelectorAll('.rating-widget i');
            stars.forEach(star => star.classList.remove('active'));
            for (let i = 0; i < finalRating; i++) {
                if (i + 1 <= Math.floor(finalRating)) {
                    stars[i].classList.add('active');
                }
            }
        }
    });

    recommendBtn.addEventListener('click', getRecommendations);

    // Início da aplicação
    fetchMovies();
});